import logging
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import SessionLocal, get_db
from app.core.rbac import ADMIN_ONLY_ROLES, get_current_user, require_roles
from app.models.user import User
from app.models.official_source import OfficialSource, OfficialDocument
from app.services.official_source_connector import MinistryOfCoalConnector
from app.services.official_sync_service import ensure_ministry_source, sync_official_source, try_claim_source_sync

router = APIRouter(prefix="/sources", tags=["Official Sources"])
logger = logging.getLogger(__name__)
_SYNC_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="official-sync")


def _run_ministry_sync_job(source_id: int, user_id: int) -> None:
    """Run one claimed sync with a worker-owned database session."""
    job_db = SessionLocal()
    connector = None
    try:
        source = job_db.query(OfficialSource).filter(OfficialSource.id == source_id).first()
        if not source:
            raise RuntimeError("Official source disappeared before synchronization started")
        connector = MinistryOfCoalConnector()
        sync_official_source(job_db, source, connector, user_id=user_id, already_claimed=True)
    except Exception as exc:
        logger.exception("Official source sync worker failed for source %s", source_id)
        job_db.rollback()
        source = job_db.query(OfficialSource).filter(OfficialSource.id == source_id).first()
        if source:
            source.status = "ERROR"
            source.last_error = str(exc)[:500]
            job_db.commit()
    finally:
        if connector is not None:
            try:
                connector.close()
            except Exception:
                logger.exception("Official source connector cleanup failed for source %s", source_id)
        job_db.close()


@router.get("")
def list_sources(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    sources = db.query(OfficialSource).order_by(OfficialSource.name).all()
    return [{"id": source.id, "name": source.name, "organization": source.organization, "base_url": source.base_url, "source_type": source.source_type, "enabled": source.enabled, "sync_frequency": source.sync_frequency, "last_sync_at": source.last_sync_at, "last_success_at": source.last_success_at, "status": source.status, "last_error": source.last_error, "documents": db.query(OfficialDocument).filter(OfficialDocument.source_id == source.id, OfficialDocument.is_current.is_(True)).count()} for source in sources]


@router.post("/ministry-of-coal/sync", status_code=status.HTTP_202_ACCEPTED)
def sync_ministry_of_coal(db: Session = Depends(get_db), current_user: User = Depends(require_roles(ADMIN_ONLY_ROLES))):
    source = ensure_ministry_source(db)
    if not try_claim_source_sync(db, source.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Ministry synchronization is already running.", "source_id": source.id, "status": "SYNCING"},
        )
    job_id = str(uuid4())
    try:
        _SYNC_EXECUTOR.submit(_run_ministry_sync_job, source.id, current_user.id)
    except Exception as exc:
        source = db.get(OfficialSource, source.id)
        source.status = "ERROR"
        source.last_error = f"Unable to start synchronization worker: {str(exc)[:400]}"
        db.commit()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Unable to start Ministry synchronization") from exc
    return {"source_id": source.id, "job_id": job_id, "status": "SYNCING", "accepted": True}


@router.get("/{source_id}/documents")
def list_source_documents(source_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    source = db.query(OfficialSource).filter(OfficialSource.id == source_id).first()
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Official source not found")
    documents = db.query(OfficialDocument).filter(OfficialDocument.source_id == source_id).order_by(OfficialDocument.last_seen_at.desc()).all()
    return [{"id": item.id, "source_id": item.source_id, "document_url": item.document_url, "title": item.title, "category": item.category, "published_at": item.published_at, "first_seen_at": item.first_seen_at, "last_seen_at": item.last_seen_at, "checksum": item.checksum, "download_status": item.download_status, "document_id": item.document_id, "version": item.version, "is_current": item.is_current, "last_error": item.last_error} for item in documents]


@router.get("/sync-status")
def sync_status(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    sources = db.query(OfficialSource).all()
    return {"sources": [{"id": source.id, "name": source.name, "status": source.status, "last_sync_at": source.last_sync_at, "last_success_at": source.last_success_at, "last_error": source.last_error} for source in sources]}
