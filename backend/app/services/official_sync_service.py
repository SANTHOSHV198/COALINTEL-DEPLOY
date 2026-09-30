"""Persistent, versioned synchronization for official documents."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.official_source import OfficialDocument, OfficialSource
from app.services.ingestion_service import calculate_sha256, process_file_ingestion, sanitize_filename
from app.services.processing_pipeline import execute_document_processing_pipeline

logger = logging.getLogger(__name__)

SUCCESSFUL_PROCESSING_STATES = {"READY", "REVIEW_RECOMMENDED", "VALIDATION_WARNING"}


def _isoformat_or_none(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _document_processing_succeeded(db: Session, document_id: Optional[int]) -> bool:
    """Return true only when the common pipeline completed validation."""
    if not document_id:
        return False
    document = db.get(Document, document_id)
    return bool(
        document
        and document.status in {"PARSED", "INDEXED"}
        and document.processing_status in SUCCESSFUL_PROCESSING_STATES
    )


def _process_document(db: Session, document_id: int) -> None:
    """Run and verify the common processing pipeline for an acquired file."""
    if not execute_document_processing_pipeline(db, document_id):
        document = db.get(Document, document_id)
        detail = (document.error_message if document else None) or (document.processing_status if document else "unknown")
        raise RuntimeError(f"Official document processing failed: {detail}")
    db.expire_all()
    if not _document_processing_succeeded(db, document_id):
        document = db.get(Document, document_id)
        detail = (document.error_message if document else None) or (document.processing_status if document else "unknown")
        raise RuntimeError(f"Official document did not reach a validated state: {detail}")


def ensure_ministry_source(db: Session) -> OfficialSource:
    source = db.query(OfficialSource).filter(OfficialSource.base_url == "https://coal.nic.in/major-statistics-page").first()
    if source:
        return source
    source = OfficialSource(name="Ministry of Coal", organization="Ministry of Coal", base_url="https://coal.nic.in/major-statistics-page", source_type="OFFICIAL_WEBSITE", sync_frequency="24h", status="CONNECTED")
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


def try_claim_source_sync(db: Session, source_id: int) -> bool:
    """Atomically claim a source for one manual or scheduled sync run."""
    source = db.query(OfficialSource).filter(OfficialSource.id == source_id).with_for_update().first()
    if not source or source.status == "SYNCING":
        db.rollback()
        return False
    source.status = "SYNCING"
    source.last_sync_at = datetime.now(timezone.utc)
    source.last_error = None
    db.commit()
    return True


def recover_interrupted_source_syncs(db: Session) -> int:
    """Surface worker claims orphaned by a backend restart as explicit errors."""
    sources = db.query(OfficialSource).filter(OfficialSource.status == "SYNCING").all()
    for source in sources:
        source.status = "ERROR"
        source.last_error = "Synchronization was interrupted by a backend restart before completion."
    if sources:
        db.commit()
    return len(sources)


def _current_record(db: Session, source_id: int, url: str) -> Optional[OfficialDocument]:
    return db.query(OfficialDocument).filter(
        OfficialDocument.source_id == source_id,
        OfficialDocument.document_url == url,
        OfficialDocument.is_current.is_(True),
    ).order_by(OfficialDocument.version.desc()).first()


def _latest_record(db: Session, source_id: int, url: str) -> Optional[OfficialDocument]:
    """Return the newest registry attempt, including a non-current failed attempt."""
    return db.query(OfficialDocument).filter(
        OfficialDocument.source_id == source_id,
        OfficialDocument.document_url == url,
    ).order_by(OfficialDocument.version.desc(), OfficialDocument.id.desc()).first()


def _metadata(connector: Any, item: Any) -> str:
    return json.dumps(connector.extract_source_metadata(item), default=str)


def _failed_record(db: Session, source: OfficialSource, item: Any, current: Optional[OfficialDocument], now: datetime, error: Exception) -> None:
    record = OfficialDocument(
        source_id=source.id, document_url=item.url, title=item.title, category=item.category,
        published_at=item.publication_date, download_status="FAILED", version=(current.version + 1) if current else 1,
        is_current=False, last_seen_at=now, last_error=str(error)[:500], source_metadata=None,
    )
    # Metadata extraction is deliberately optional: a failed URL must remain auditable even if its
    # connector cannot derive metadata for it.
    db.add(record)
    db.commit()


@dataclass
class _AcquiredDocument:
    """A Phase-A registry record and its staged bytes for Phase B."""

    record_id: int
    item: Any
    content_path: Optional[Path]
    document_id: Optional[int]
    previous_current_id: Optional[int] = None


def _mark_source_error(db: Session, source_id: int, error: Exception) -> Optional[OfficialSource]:
    """Best-effort terminalization for failures outside per-document isolation."""
    db.rollback()
    source = db.get(OfficialSource, source_id)
    if not source:
        return None
    source.status = "ERROR"
    source.last_error = str(error)[:500]
    try:
        db.commit()
    except Exception:
        db.rollback()
    return source


def _process_acquired_document(
    db: Session,
    source: OfficialSource,
    work: _AcquiredDocument,
    *,
    user_id: Optional[int],
    attempt_started_at: datetime,
) -> bool:
    """Process one acquired document without aborting the remaining queue."""
    record = db.get(OfficialDocument, work.record_id)
    if not record:
        logger.warning("Acquired official document record %s disappeared before processing", work.record_id)
        return False

    try:
        document_id = work.document_id
        if document_id is None:
            if work.content_path is None:
                raise RuntimeError("No staged source bytes were available for official document")
            content = work.content_path.read_bytes()
            candidate_name = (
                work.item.title
                if work.item.title and "." in work.item.title.rsplit("/", 1)[-1]
                else work.item.url.split("?", 1)[0].rsplit("/", 1)[-1]
            )
            document = process_file_ingestion(
                db,
                content,
                sanitize_filename(candidate_name),
                user_id,
                source_type="OFFICIAL",
                source_url=work.item.url,
                source_organization=source.organization,
                title=work.item.title,
                publication_date=work.item.publication_date,
                source_document_id=record.id,
                document_version=record.version,
            )
            if not document:
                raise RuntimeError("Official document ingestion returned no document")
            document_id = document.id
            record.document_id = document_id
            db.commit()

        _process_document(db, document_id)
        db.expire_all()
        if not _document_processing_succeeded(db, document_id):
            document = db.get(Document, document_id)
            detail = (document.error_message if document else None) or (document.processing_status if document else "unknown")
            raise RuntimeError(f"Official document did not reach a validated state: {detail}")

        record = db.get(OfficialDocument, work.record_id)
        record.document_id = document_id
        record.download_status = "INGESTED"
        record.last_error = None
        record.last_seen_at = attempt_started_at
        record.is_current = True
        if work.previous_current_id and work.previous_current_id != record.id:
            previous = db.get(OfficialDocument, work.previous_current_id)
            if previous:
                previous.is_current = False
        db.commit()
        return True
    except Exception as exc:
        logger.warning("Official source document processing failed (%s): %s", record.document_url, exc)
        db.rollback()
        failed = db.get(OfficialDocument, work.record_id)
        if failed:
            failed.download_status = "PROCESSING_FAILED"
            failed.last_error = str(exc)[:500]
            # Keep a failed first attempt current so the next sync can retry it.
            # A failed replacement never displaces an older successful version.
            failed.is_current = work.previous_current_id is None
            db.commit()
        return False


def sync_official_source(
    db: Session,
    source: OfficialSource,
    connector: Any,
    *,
    user_id: Optional[int] = None,
    already_claimed: bool = False,
) -> Dict[str, Any]:
    attempt_started_at = datetime.now(timezone.utc)
    if not already_claimed:
        if not try_claim_source_sync(db, source.id):
            current = db.get(OfficialSource, source.id)
            return {
                "source_id": source.id,
                "status": current.status if current else "SYNCING",
                "already_running": True,
                "last_sync_at": current.last_sync_at if current else attempt_started_at.isoformat(),
            }
        source = db.get(OfficialSource, source.id)
    else:
        source = db.get(OfficialSource, source.id)
        source.status = "SYNCING"
        source.last_error = None
        db.commit()
    counts: Dict[str, Any] = {
        "discovered": 0,
        "new": 0,
        "updated": 0,
        "unchanged": 0,
        "duplicates": 0,
        "failed": 0,
        "pages_checked": 0,
        "page_failures": 0,
    }

    try:
        # Phase A: discover, acquire, and commit every registry record first.
        # Bytes are spooled to bounded temporary storage so a large source set
        # does not remain resident in the sync worker's memory.
        if hasattr(connector, "discover_documents_with_report"):
            discovery = connector.discover_documents_with_report()
            discovered = discovery.documents
            counts["pages_checked"] = discovery.pages_checked
            counts["page_failures"] = len(discovery.page_failures)
        else:
            discovered = connector.discover_documents()
        counts["discovered"] = len(discovered)
        acquired: list[_AcquiredDocument] = []

        with TemporaryDirectory(prefix="coalintel-official-sync-") as staging_dir:
            staging = Path(staging_dir)
            for index, item in enumerate(discovered):
                current = _current_record(db, source.id, item.url)
                latest = _latest_record(db, source.id, item.url)
                try:
                    content = connector.download_document(item)
                    checksum = calculate_sha256(content)

                    if current and current.checksum == checksum:
                        current.last_seen_at = attempt_started_at
                        if current.document_id and _document_processing_succeeded(db, current.document_id):
                            current.download_status = "UNCHANGED"
                            current.last_error = None
                            db.commit()
                            counts["unchanged"] += 1
                            continue

                        # A previously DOWNLOADED/PROCESSING_FAILED row is
                        # recoverable. Keep its identity and queue processing;
                        # do not manufacture a new version for the same bytes.
                        current.download_status = "DOWNLOADED"
                        current.last_error = None
                        db.commit()
                        content_path = None
                        if not current.document_id:
                            content_path = staging / f"{index}-{current.id}.bin"
                            content_path.write_bytes(content)
                        acquired.append(_AcquiredDocument(
                            record_id=current.id,
                            item=item,
                            content_path=content_path,
                            document_id=current.document_id,
                        ))
                        counts["unchanged"] += 1
                        continue

                    # A processing failure from an earlier attempt is retried
                    # in place when the same checksum is discovered again.
                    if (
                        latest
                        and latest.checksum == checksum
                        and latest.download_status in {"DOWNLOADED", "PROCESSING_FAILED"}
                    ):
                        previous_current_id = current.id if current and current.id != latest.id else None
                        latest.last_seen_at = attempt_started_at
                        latest.download_status = "DOWNLOADED"
                        latest.last_error = None
                        latest.is_current = previous_current_id is None
                        db.commit()
                        content_path = None
                        if not latest.document_id:
                            content_path = staging / f"{index}-{latest.id}.bin"
                            content_path.write_bytes(content)
                        acquired.append(_AcquiredDocument(
                            record_id=latest.id,
                            item=item,
                            content_path=content_path,
                            document_id=latest.document_id,
                            previous_current_id=previous_current_id,
                        ))
                        counts["unchanged"] += 1
                        continue

                    state = "updated" if current else "new"
                    version = (current.version + 1) if current else 1
                    existing_doc = db.query(Document).filter(Document.file_hash == checksum).first()
                    record = OfficialDocument(
                        source_id=source.id,
                        document_url=item.url,
                        title=item.title,
                        category=item.category,
                        published_at=item.publication_date,
                        checksum=checksum,
                        download_status="DUPLICATE" if existing_doc else "DOWNLOADED",
                        document_id=existing_doc.id if existing_doc else None,
                        version=version,
                        is_current=not bool(existing_doc),
                        last_seen_at=attempt_started_at,
                        source_metadata=_metadata(connector, item),
                    )
                    db.add(record)
                    db.commit()  # Phase A durability boundary.
                    db.refresh(record)

                    if existing_doc:
                        if current:
                            current.is_current = False
                            db.commit()
                        counts[state] += 1
                        counts["duplicates"] += 1
                        continue

                    content_path = staging / f"{index}-{record.id}.bin"
                    content_path.write_bytes(content)
                    acquired.append(_AcquiredDocument(
                        record_id=record.id,
                        item=item,
                        content_path=content_path,
                        document_id=None,
                        previous_current_id=current.id if current else None,
                    ))
                    counts[state] += 1
                except Exception as exc:
                    logger.warning("Official source acquisition failed (%s): %s", item.url, exc)
                    db.rollback()
                    # A failed download has no successful acquisition row yet;
                    # retain an explicit audit record and leave the prior
                    # successful version current.
                    existing_failed = db.query(OfficialDocument).filter(
                        OfficialDocument.source_id == source.id,
                        OfficialDocument.document_url == item.url,
                        OfficialDocument.last_error == str(exc)[:500],
                    ).first()
                    if not existing_failed:
                        try:
                            _failed_record(db, source, item, current, attempt_started_at, exc)
                        except Exception:
                            db.rollback()
                    counts["failed"] += 1

            logger.info(
                "Official source Phase A complete: source_id=%s discovered=%s queued_for_processing=%s "
                "registry_rows=%s acquisition_failures=%s",
                source.id,
                counts["discovered"],
                len(acquired),
                db.query(OfficialDocument).filter(OfficialDocument.source_id == source.id).count(),
                counts["failed"],
            )

            # Phase B: process the already-registered acquisition queue. Each
            # item is isolated so a failure cannot abort later documents.
            for work in acquired:
                if not _process_acquired_document(
                    db,
                    source,
                    work,
                    user_id=user_id,
                    attempt_started_at=attempt_started_at,
                ):
                    counts["failed"] += 1

        source = db.get(OfficialSource, source.id)
        if counts["failed"] or counts["page_failures"]:
            source.status = "PARTIAL"
            errors = []
            if counts["page_failures"]:
                errors.append(f"{counts['page_failures']} category page(s) failed")
            if counts["failed"]:
                errors.append(f"{counts['failed']} document(s) failed")
            source.last_error = "; ".join(errors)
        else:
            source.status = "CONNECTED"
            source.last_error = None
            # last_sync_at is the attempt/claim time. last_success_at is the
            # terminal successful completion time, never request acceptance.
            source.last_success_at = datetime.now(timezone.utc)
        db.commit()
        return {
            "source_id": source.id,
            **counts,
            "status": source.status,
            "last_sync_at": _isoformat_or_none(source.last_sync_at),
            "last_success_at": _isoformat_or_none(source.last_success_at),
        }
    except Exception as exc:
        failed_source_id = source.id
        source = _mark_source_error(db, failed_source_id, exc)
        return {
            "source_id": failed_source_id,
            **counts,
            "status": "ERROR",
            "last_sync_at": _isoformat_or_none(source.last_sync_at) if source else None,
            "last_success_at": _isoformat_or_none(source.last_success_at) if source else None,
            "error": str(exc)[:500],
        }


def run_due_syncs(db: Session, connector_factory, *, user_id: Optional[int] = None, force: bool = False) -> Dict[str, Any]:
    """Scheduler-friendly entrypoint; default cadence is one run per 24 hours."""
    sources = db.query(OfficialSource).filter(OfficialSource.enabled.is_(True)).all()
    results = []
    now = datetime.now(timezone.utc)
    for source in sources:
        due = force or source.last_sync_at is None or (now - source.last_sync_at).total_seconds() >= 86400
        if not due:
            continue
        try:
            results.append(sync_official_source(db, source, connector_factory(source), user_id=user_id))
        except Exception as exc:
            logger.exception("Official source sync failed for source %s", source.id)
            results.append({"source_id": source.id, "status": "ERROR", "failed": 1, "error": str(exc)[:500]})
    return {"checked_at": now.isoformat(), "results": results, "due_count": len(results)}
