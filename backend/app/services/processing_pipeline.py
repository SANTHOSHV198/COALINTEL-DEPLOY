import logging
import time
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.extracted_metric import ExtractedMetric
from app.services.storage_service import document_binary_exists, read_document_binary, file_exists
from app.services.parsing_service import parse_document_file
from app.services.parsing_service import parse_document_result
from app.services.document_models import DocumentResult
from app.services.processing_states import transition
from app.services.chunking_service import chunk_text_by_tokens
from app.services.domain_extraction_service import (
    extract_entity_tuples_from_text,
    extract_entity_tuples_from_tables,
    classify_document_authority,
)
from app.services.vector_store_service import add_chunks_to_vector_store, delete_document_vectors
from app.services.knowledge_retrieval_service import index_document_knowledge

logger = logging.getLogger(__name__)


def _persist_step2c_structured_facts(db: Session, document: Document, result: DocumentResult) -> int:
    """Persist the additive Step 2C fact view without changing legacy metrics.

    The structured-fact layer is deliberately isolated from the Step 1
    compatibility tables.  A missing/unmigrated optional table is recorded as
    a warning rather than preventing the already-valid ingestion result from
    being committed.  Once migration 004 is applied, this path is the normal
    evidence-linked persistence path.
    """
    from app.models.document_artifacts import DocumentPage, DocumentTable
    from app.models.structured_fact import StructuredFact
    from app.services.structured_extraction_service import (
        extract_structured_fact_candidates,
        persist_structured_fact_candidates,
    )

    started = time.perf_counter()
    try:
        candidates = extract_structured_fact_candidates(result, document_id=document.id)
        table_ids = {
            (table.page_number, table.table_number): table.id
            for table in db.query(DocumentTable)
            .filter(DocumentTable.document_id == document.id)
            .all()
        }
        page_ids = {
            page.page_number: page.id
            for page in db.query(DocumentPage)
            .filter(DocumentPage.document_id == document.id)
            .all()
        }
        # Structured facts are derived from the current parse, just like the
        # generic page/table artifacts.  Reprocessing replaces this document's
        # derived view without creating a new source document/version.
        db.query(StructuredFact).filter(StructuredFact.document_id == document.id).delete(
            synchronize_session=False
        )
        inserted = persist_structured_fact_candidates(
            db,
            document.id,
            candidates,
            table_ids_by_number=table_ids,
            page_ids_by_number=page_ids,
        )
        db.commit()
        logger.info(
            "Document #%s Step 2C structured facts persisted candidates=%s inserted=%s tables_linked=%s seconds=%.3f",
            document.id,
            len(candidates),
            inserted,
            len(table_ids),
            time.perf_counter() - started,
        )
        return inserted
    except Exception as exc:
        db.rollback()
        # Do not turn an additive Step 2C persistence problem into a false
        # Step 1 ingestion failure.  The warning is explicit and visible to
        # operators; the legacy authoritative artifacts remain committed.
        warning = "STRUCTURED_FACT_PERSISTENCE_UNAVAILABLE"
        current_warnings = list(document.processing_warnings or [])
        if warning not in current_warnings:
            current_warnings.append(warning)
        document.processing_warnings = current_warnings
        db.commit()
        logger.warning(
            "Document #%s Step 2C structured-fact persistence skipped type=%s error=%s",
            document.id,
            type(exc).__name__,
            str(exc)[:300],
        )
        return 0


def _bounded_metric_text(value, limit: int, default: str = "UNKNOWN") -> str:
    """Keep extracted metric persistence within the legacy schema limits."""
    text = str(value or default).strip()
    return text[:limit] or default[:limit]


def _persist_common_artifacts(db: Session, document: Document, result: DocumentResult) -> None:
    """Persist the generic result before any optional domain extraction."""
    from app.models.document_artifacts import DocumentPage, DocumentTable, DocumentImage
    from dataclasses import asdict

    db.query(DocumentPage).filter(DocumentPage.document_id == document.id).delete(synchronize_session=False)
    db.query(DocumentTable).filter(DocumentTable.document_id == document.id).delete(synchronize_session=False)
    db.query(DocumentImage).filter(DocumentImage.document_id == document.id).delete(synchronize_session=False)
    for page in result.pages:
        db.add(DocumentPage(document_id=document.id, page_number=page.page_number, text=page.text or "", extraction_method=page.extraction_method, extraction_confidence=page.confidence, classification=page.classification, width=page.width, height=page.height, blocks_json=[asdict(block) for block in page.blocks], metadata_json=page.metadata))
    for table in result.tables:
        db.add(DocumentTable(document_id=document.id, page_number=table.page_number, table_number=table.table_number, title=table.title, headers_json=table.headers, rows_json=table.rows, bounding_box_json=table.bounding_box, extraction_confidence=table.extraction_confidence, extraction_method=table.extraction_method, sheet_name=table.sheet_name, cells_json=table.cells, merged_cells_json=table.merged_cells, formulas_json=table.formulas, displayed_values_json=table.displayed_values, warnings_json=table.warnings))
    for image in result.images:
        db.add(DocumentImage(document_id=document.id, page_number=image.page_number, image_number=image.image_number, source=image.source, mime_type=image.mime_type, width=image.width, height=image.height, text=image.text, bounding_box_json=image.bounding_box, ocr_confidence=image.ocr_confidence, metadata_json=image.metadata))
    document.extraction_method = result.extraction_method
    document.extraction_confidence = result.extraction_confidence
    document.metadata_json = result.metadata
    document.processing_warnings = result.warnings


def execute_document_processing_pipeline(db: Session, document_id: int) -> bool:
    """
    Orchestrates the Document Ingestion & Extraction Pipeline with low-memory safety:
    1. Validates document existence and storage binary presence.
    2. Updates status -> 'PROCESSING' and commits initial state.
    3. Retrieves document binary and executes PyMuPDF / OCR page parsing.
    4. Persists total_pages immediately.
    5. Idempotently clears previous derived chunks, metrics, and Chroma vectors for this document.
    6. Splits page text into 500-token chunks and persists to document_chunks.
    7. Extracts entity metrics tuples, applies deterministic unit normalization (-> MT),
       and persists to extracted_metrics.
    8. Indexes chunk vectors into persistent ChromaDB using low-memory ONNX embeddings.
    9. Marks document status -> 'PARSED' and commits final state.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        logger.error(f"Processing pipeline failed: Document ID #{document_id} not found.")
        return False

    # Check storage binary existence before starting processing
    if not document_binary_exists(doc.file_path):
        logger.error(f"Cannot process Document #{doc.id}: source binary missing at '{doc.file_path}'.")
        doc.status = "FAILED"
        doc.processing_status = "DOWNLOAD_FAILED"
        doc.error_message = "Source document file is missing from storage. Please re-upload the document."
        db.commit()
        return False

    try:
        pipeline_started = time.perf_counter()
        stage = "state_initialization"
        logger.info(f"Document #{doc.id} background processing started ('{doc.filename}').")
        doc.status = "PROCESSING"
        doc.error_message = None
        current_state = doc.processing_status or "DISCOVERED"
        # Legacy rows created before Step 1 have no meaningful canonical
        # state; their persisted source binary is equivalent to DOWNLOADED.
        if current_state == "DISCOVERED":
            transition(doc, "DOWNLOADED")
            current_state = "DOWNLOADED"
        elif current_state in {
            "CLASSIFYING", "EXTRACTING", "OCR", "TABLE_EXTRACTION", "VALIDATING",
            "OCR_FAILED", "EXTRACTION_FAILED", "DOWNLOAD_FAILED",
            "READY", "REVIEW_RECOMMENDED", "VALIDATION_WARNING",
        }:
            # A prior attempt may have committed an intermediate/failure state
            # before the process died. Reprocessing the same persisted binary
            # must start from a clean acquisition state; otherwise the next
            # parser transition (for example TABLE_EXTRACTION -> OCR) is
            # invalid and the document can never recover in place.
            doc.processing_status = "DOWNLOADED"
            current_state = "DOWNLOADED"
        if current_state == "DOWNLOADED":
            transition(doc, "CLASSIFYING")
        db.commit()

        # Step 1: Retrieve Document Binary & Parse Document Pages (PyMuPDF / OCR)
        logger.info(f"Document #{doc.id} parsing started from storage reference '{doc.file_path}'.")
        try:
            file_bytes = read_document_binary(doc.file_path)
        except Exception as read_err:
            logger.error(f"Failed to read storage binary for Document #{doc.id}: {read_err}")
            doc.status = "FAILED"
            doc.error_message = f"Failed to retrieve document binary from storage: {str(read_err)[:300]}"
            db.commit()
            return False

        stage = "parser_and_ocr"
        parse_started = time.perf_counter()
        common_result = parse_document_result(doc.file_path, doc.file_type, file_bytes=file_bytes, filename=doc.filename)
        parse_elapsed = time.perf_counter() - parse_started
        classifications = {}
        for parsed_page in common_result.pages:
            classifications[parsed_page.classification] = classifications.get(parsed_page.classification, 0) + 1
        logger.info(
            "Document #%s parsing stage timing parse_seconds=%.3f pages=%s classifications=%s document_timings=%s",
            doc.id, parse_elapsed, len(common_result.pages), classifications,
            common_result.metadata.get("timings_ms"),
        )
        sanitization = common_result.metadata.get("text_sanitization")
        if sanitization:
            logger.warning(
                "Document #%s text sanitization applied before persistence "
                "warning_code=%s removed_count=%s changed_fields=%s",
                doc.id,
                sanitization.get("warning_code"),
                sanitization.get("removed_count"),
                sanitization.get("changed_field_count"),
            )
        stage = "common_artifact_persistence"
        artifact_started = time.perf_counter()
        _persist_common_artifacts(db, doc, common_result)
        logger.info("Document #%s artifact persistence timing seconds=%.3f pages=%s tables=%s images=%s", doc.id, time.perf_counter() - artifact_started, len(common_result.pages), len(common_result.tables), len(common_result.images))
        if any(page.metadata.get("ocr_requested") for page in common_result.pages):
            transition(doc, "OCR")
        else:
            transition(doc, "EXTRACTING")
        if common_result.tables:
            transition(doc, "TABLE_EXTRACTION")
        pages_data = [
            {
                "page_number": page.page_number,
                "text": page.text,
                "is_ocr": page.extraction_method in {"OCR", "NATIVE+OCR"},
                "tables": [
                    {"table_index": table.table_number, "bbox": table.bounding_box or [], "row_count": len(table.rows), "col_count": max((len(row) for row in table.rows), default=0), "raw_rows": table.rows, "header_names": table.headers, "title": table.title, "extraction_confidence": table.extraction_confidence, "extraction_method": table.extraction_method, "warnings": table.warnings, "sheet_name": table.sheet_name}
                    for table in common_result.tables if table.page_number == page.page_number
                ],
            }
            for page in common_result.pages
        ]
        total_pages = len(pages_data)

        if total_pages == 0:
            logger.warning(f"No pages or text extracted from Document #{doc.id}.")
            doc.status = "FAILED"
            doc.processing_status = "EXTRACTION_FAILED"
            doc.error_message = "Document parser could not extract any readable pages or text from file."
            db.commit()
            return False

        # Incremental progress persistence: persist total_pages immediately after parsing
        doc.total_pages = total_pages
        doc.title = doc.title or common_result.title
        doc.reporting_period = doc.reporting_period or common_result.reporting_period
        db.commit()
        logger.info(f"Document #{doc.id} parsing completed ({total_pages} pages).")

        # Step 2C: build an additive evidence-linked fact view from the same
        # common result that produced the persisted pages/tables.  This keeps
        # table-cell provenance available without changing the legacy
        # extracted_metrics path used by current dashboards and validation.
        stage = "structured_fact_extraction"
        _persist_step2c_structured_facts(db, doc, common_result)

        # Step 2: Idempotent Cleanup of existing derived records and Chroma vectors
        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
        db.query(ExtractedMetric).filter(ExtractedMetric.document_id == doc.id).delete()
        db.flush()
        delete_document_vectors(doc.id)

        all_chunks = []
        all_metrics = []

        # Step 3: Iterate pages -> Chunking & Metric Extraction
        doc_origin = getattr(doc, "data_origin", None) or getattr(doc, "authority", None)
        if not doc_origin:
            doc_origin = classify_document_authority(doc.filename)

        stage = "chunking_and_metric_extraction"
        normalization_started = time.perf_counter()
        for page_info in pages_data:
            page_num = page_info["page_number"]
            page_text = page_info["text"]

            # Chunking (500 tokens, 50 overlap)
            chunks = chunk_text_by_tokens(page_text, page_number=page_num)
            for c in chunks:
                all_chunks.append(DocumentChunk(
                    document_id=doc.id,
                    page_number=c["page_number"],
                    chunk_index=c["chunk_index"],
                    chunk_text=c["chunk_text"],
                    token_count=c["token_count"],
                    embedding_id=f"chunk_{doc.id}_{c['page_number']}_{c['chunk_index']}"
                ))

            # Entity Metric Extraction from text
            text_metrics = extract_entity_tuples_from_text(
                text=page_text,
                page_number=page_num,
                default_subsidiary=doc.subsidiary or "CIL HQ",
                default_year=doc.fiscal_year
            )

            # Table-aware Metric Extraction from structured tables (additive)
            page_tables = page_info.get("tables", [])
            table_metrics = []
            if page_tables:
                try:
                    table_metrics = extract_entity_tuples_from_tables(
                        tables=page_tables,
                        page_number=page_num,
                        page_text=page_text,
                        default_subsidiary=doc.subsidiary or "CIL HQ",
                        default_year=doc.fiscal_year
                    )
                except Exception as tab_ext_err:
                    logger.warning(f"Table metric extraction note on page {page_num}: {tab_ext_err}")

            # Merge and deduplicate: table metrics take precedence for matching (page, entity, metric, year, value)
            seen_page_keys = set()
            combined_page_metrics = []

            for tm in table_metrics:
                key = (
                    tm["page_number"],
                    (tm.get("subsidiary") or "").upper(),
                    (tm.get("mine_name") or "").upper(),
                    (tm.get("metric_name") or "").upper(),
                    str(tm.get("fiscal_year", "")).strip(),
                    round(float(tm["numeric_value"]), 4)
                )
                seen_page_keys.add(key)
                combined_page_metrics.append(tm)

            for m in text_metrics:
                key = (
                    m["page_number"],
                    (m.get("subsidiary") or "").upper(),
                    (m.get("mine_name") or "").upper(),
                    (m.get("metric_name") or "").upper(),
                    str(m.get("fiscal_year", "")).strip(),
                    round(float(m["numeric_value"]), 4)
                )
                if key not in seen_page_keys:
                    seen_page_keys.add(key)
                    combined_page_metrics.append(m)

            for m in combined_page_metrics:
                all_metrics.append(ExtractedMetric(
                    document_id=doc.id,
                    page_number=m["page_number"],
                    mine_name=_bounded_metric_text(m.get("mine_name"), 100),
                    subsidiary=_bounded_metric_text(m.get("subsidiary"), 100),
                    metric_name=_bounded_metric_text(m.get("metric_name"), 100),
                    numeric_value=m["numeric_value"],
                    unit=_bounded_metric_text(m.get("unit"), 30),
                    raw_unit=_bounded_metric_text(m.get("unit"), 30),
                    standard_value=m["standard_value"],
                    standard_unit=_bounded_metric_text(m.get("standard_unit"), 10),
                    fiscal_year=_bounded_metric_text(m.get("fiscal_year"), 20),
                    confidence_score=m["confidence_score"],
                    validation_status=_bounded_metric_text(m.get("validation_status"), 30),
                    raw_snippet=m["raw_snippet"],
                    data_origin=_bounded_metric_text(doc_origin, 30)
                ))

        logger.info(f"Document #{doc.id} chunking completed ({len(all_chunks)} chunks).")
        logger.info(f"Document #{doc.id} normalization completed ({len(all_metrics)} metrics).")
        logger.info("Document #%s downstream chunking/normalization timing seconds=%.3f", doc.id, time.perf_counter() - normalization_started)

        # Step 4: Bulk Persist to Database & ChromaDB Vector Store
                # Step 5: Persist authoritative extracted data first.
        # PostgreSQL is the source of truth for extracted Ministry of Coal metrics.

        stage = "authoritative_persistence_and_validation"
        persistence_started = time.perf_counter()
        if all_chunks:
            db.bulk_save_objects(all_chunks)
            logger.info(
                f"Document #{doc.id} chunks persisted to PostgreSQL "
                f"({len(all_chunks)} chunks)."
            )

        if all_metrics:
            db.bulk_save_objects(all_metrics)
            logger.info(
                f"Document #{doc.id} extracted metrics persisted to PostgreSQL "
                f"({len(all_metrics)} metrics)."
            )

        # Commit the authoritative extraction before attempting optional
        # vector/embedding operations.
        transition(doc, "VALIDATING")
        quality_warnings = list(doc.processing_warnings or [])
        if len(pages_data) != (doc.total_pages or len(pages_data)):
            quality_warnings.append("Page count changed during validation")
        if any((page.get("is_ocr") and not page.get("text", "").strip()) for page in pages_data):
            quality_warnings.append("At least one OCR page has no extracted text")
        doc.processing_warnings = quality_warnings
        doc.status = "PARSED"
        doc.error_message = None
        db.commit()
        logger.info("Document #%s authoritative persistence/validation timing seconds=%.3f", doc.id, time.perf_counter() - persistence_started)

        # READY means generic extraction and validation completed.  The
        # legacy status remains PARSED/INDEXED for existing consumers.
        if quality_warnings:
            transition(doc, "REVIEW_RECOMMENDED")
        else:
            transition(doc, "READY")
        db.commit()

        logger.info(
            f"Document #{doc.id} authoritative extraction committed successfully. "
            f"Status -> PARSED ({len(all_chunks)} chunks, {len(all_metrics)} metrics stored)."
        )

        # Step 6: Optional PostgreSQL knowledge indexing.
        # This runs only after authoritative pages, tables, structured facts,
        # legacy chunks, and metrics have been committed.  It is deliberately
        # non-fatal: the PostgreSQL extraction result remains valid when the
        # optional Step 3 table/model is unavailable or indexing fails.
        try:
            stage = "optional_knowledge_indexing"
            knowledge_started = time.perf_counter()
            knowledge_result = index_document_knowledge(db, doc.id)
            logger.info(
                "Document #%s knowledge indexing completed result=%s seconds=%.3f",
                doc.id,
                knowledge_result,
                time.perf_counter() - knowledge_started,
            )
        except Exception as knowledge_err:
            # The authoritative commit above is intentionally preserved.  A
            # failed optional indexing transaction is rolled back before the
            # existing legacy Chroma path continues.
            db.rollback()
            logger.warning(
                "Document #%s knowledge indexing failed non-fatally "
                "error_type=%s error=%s. PostgreSQL extraction remains valid.",
                doc.id,
                type(knowledge_err).__name__,
                str(knowledge_err)[:300],
            )

        # Step 7: Optional legacy Chroma vector indexing.
        # Failure here must never invalidate the authoritative PostgreSQL data.

        if all_chunks:
            logger.info(
                f"Document #{doc.id} optional embedding/indexing started."
            )

            try:
                stage = "optional_vector_indexing"
                vector_started = time.perf_counter()
                vector_success = add_chunks_to_vector_store(
                    all_chunks,
                    filename=doc.filename,
                    subsidiary=doc.subsidiary
                )

                if vector_success:
                    # Keep the legacy status PARSED for existing API filters;
                    # canonical readiness is represented by processing_status.
                    doc.status = "PARSED"
                    doc.error_message = None
                    db.commit()

                    logger.info(
                        f"Document #{doc.id} vector indexing completed. "
                        f"Status -> INDEXED."
                )
                else:
                    logger.warning(
                    f"Document #{doc.id} vector indexing was unavailable. "
                    f"PostgreSQL extraction remains valid. Status -> PARSED."
                    )
                logger.info("Document #%s optional vector indexing timing seconds=%.3f success=%s", doc.id, time.perf_counter() - vector_started, vector_success)

            except Exception as vec_err:
                logger.warning(
                    f"Document #{doc.id} optional vector indexing failed: "
                    f"{type(vec_err).__name__} - {str(vec_err)[:300]}. "
                    f"PostgreSQL extraction remains valid."
                )

        logger.info(
            f"Document #{doc.id} processing completed successfully. "
            f"Status -> PARSED ({len(all_chunks)} chunks, {len(all_metrics)} metrics stored)."
        )
        logger.info("Document #%s total processing timing seconds=%.3f", doc.id, time.perf_counter() - pipeline_started)

        return True
    except Exception as e:
        db.rollback()
        sanitized_err = str(e)[:450]
        logger.error(
            "Document #%s processing failed stage=%s error_type=%s error=%s",
            document_id,
            stage,
            type(e).__name__,
            sanitized_err,
        )
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            if doc:
                doc.status = "FAILED"
                if doc.processing_status not in {"OCR_FAILED", "EXTRACTION_FAILED", "DOWNLOAD_FAILED", "UNSUPPORTED_FORMAT"}:
                    doc.processing_status = "EXTRACTION_FAILED"
                doc.error_message = f"Processing failed: {sanitized_err}"
                db.commit()
        except Exception as fail_err:
            logger.error(f"Failed to record failure status for Document #{document_id}: {fail_err}")
        return False


def recover_stale_processing_documents(db: Session, stale_minutes: int = 15) -> int:
    """
    Startup and on-demand recovery mechanism for orphaned PROCESSING records.
    Transitions documents that are stale (> stale_minutes) AND have lost storage binaries.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=stale_minutes)

    stale_docs = db.query(Document).filter(Document.status == "PROCESSING").all()
    recovered_count = 0

    for doc in stale_docs:
        doc_time = doc.created_at
        if doc_time is not None and doc_time.tzinfo is None:
            doc_time = doc_time.replace(tzinfo=timezone.utc)

        is_stale = (doc_time is None) or (doc_time < cutoff)

        if is_stale:
            if not document_binary_exists(doc.file_path):
                logger.warning(
                    f"Recovering stale Document #{doc.id} ('{doc.filename}'): "
                    f"Created at {doc.created_at}, storage binary missing. Marking FAILED."
                )
                doc.status = "FAILED"
                doc.error_message = "Processing was interrupted during server restart and source file is unavailable in storage. Please re-upload the document."
                recovered_count += 1
            else:
                logger.info(
                    f"Stale Document #{doc.id} detected and storage binary exists. "
                    f"Starting reprocessing."
                    )
                success = execute_document_processing_pipeline(db, doc.id)
                if success:
                    recovered_count += 1
                    logger.info(f"Successfully reprocessed stale Document #{doc.id}.")
                else:
                    # Preserve the legacy PROCESSING marker for source files
                    # that still exist; the canonical processing_status and
                    # error_message carry the explicit extraction failure.
                    doc.status = "PROCESSING"
                    db.commit()
                    logger.error(f"Failed to reprocess stale Document #{doc.id}.")

    if recovered_count > 0:
        db.commit()
        logger.info(f"Recovered {recovered_count} stale PROCESSING document(s).")

    return recovered_count
