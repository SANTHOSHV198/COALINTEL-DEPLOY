import unittest
from unittest.mock import patch
import os
import sys
import tempfile
import io
import numpy as np
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import Base
from app.models.user import User
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.extracted_metric import ExtractedMetric
from app.models.structured_fact import StructuredFact
from sqlalchemy.pool import StaticPool
from app.services.storage_service import (
    save_uploaded_file,
    read_uploaded_file,
    delete_uploaded_file,
    file_exists,
    get_file_size,
)
from app.services.embedding_service import (
    get_embedding_model,
    generate_embedding,
    generate_batch_embeddings,
    OnnxEmbeddingBackend,
    EMBEDDING_DIMENSION,
)
from app.services.vector_store_service import (
    get_chroma_collection,
    add_chunks_to_vector_store,
    delete_document_vectors,
)
from app.services.processing_pipeline import (
    execute_document_processing_pipeline,
    recover_stale_processing_documents,
    _bounded_metric_text,
)
from app.core.security import get_password_hash, create_access_token


class TestDocumentProcessingReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False
        )
        cls.SessionLocal = sessionmaker(bind=cls.engine)
        Base.metadata.create_all(bind=cls.engine)

    def setUp(self):
        self.db = self.SessionLocal()
        # Seed test admin user
        admin = self.db.query(User).filter(User.username == "test_admin").first()
        if not admin:
            admin = User(
                username="test_admin",
                email="admin@test.cil",
                hashed_password=get_password_hash("AdminPass123"),
                role="Admin",
                subsidiary="CIL HQ"
            )
            self.db.add(admin)
            self.db.commit()
            self.db.refresh(admin)
        self.admin_user = admin

    def tearDown(self):
        self.db.close()

    def test_01_storage_service_abstraction(self):
        """Verify storage_service file operations save, read, check existence, and delete cleanly."""
        sample_bytes = b"COALINTEL Storage Test Ingestion Payload - Coal Production Report"
        sample_hash = "abc1234567890def"
        filename = "test_storage_doc.pdf"

        saved_path = save_uploaded_file(sample_bytes, sample_hash, filename)
        self.assertTrue(file_exists(saved_path))
        self.assertEqual(get_file_size(saved_path), len(sample_bytes))

        read_bytes = read_uploaded_file(saved_path)
        self.assertEqual(read_bytes, sample_bytes)

        deleted = delete_uploaded_file(saved_path)
        self.assertTrue(deleted)
        self.assertFalse(file_exists(saved_path))

    def test_02_embedding_singleton_and_batch_dimensions(self):
        """Verify embedding model is loaded as a singleton and produces 384-d vectors with batching."""
        model_1 = get_embedding_model()
        model_2 = get_embedding_model()
        self.assertIs(model_1, model_2)

        sample_text = "Eastern Coalfields Limited ECL Rajmahal Open Cast Mining Project"
        vec = generate_embedding(sample_text)
        self.assertEqual(len(vec), EMBEDDING_DIMENSION)
        self.assertIsInstance(vec[0], float)

        batch_texts = [
            "Mine A produced 12.5 MT coal in FY 2023-24.",
            "Mine B overburden removal 45.2 M.Cu.M.",
            "WCL target achievement 98.4%.",
        ]
        batch_vecs = generate_batch_embeddings(batch_texts, batch_size=2)
        self.assertEqual(len(batch_vecs), 3)
        for b_vec in batch_vecs:
            self.assertEqual(len(b_vec), EMBEDDING_DIMENSION)

    def test_03_pipeline_execution_success_and_idempotency(self):
        """Verify processing pipeline transitions PENDING -> PARSED, persists progress, and is idempotent."""
        sample_content = b"ECL Rajmahal Open Cast Mine Coal Production 15.5 MT in FY 2023-24\nECL Sonepur Bazari Overburden 25.0 M.Cu.M"
        file_path = save_uploaded_file(sample_content, "hash_pipeline_test_1", "pipeline_test.csv")

        doc = Document(
            filename="pipeline_test.csv",
            file_path=file_path,
            file_hash="hash_pipeline_test_1",
            file_type="CSV",
            file_size_bytes=len(sample_content),
            subsidiary="ECL",
            fiscal_year="2023-24",
            status="PENDING",
            uploaded_by=self.admin_user.id,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        # Run pipeline first time
        success = execute_document_processing_pipeline(self.db, doc.id)
        self.assertTrue(success)
        self.db.refresh(doc)
        self.assertEqual(doc.status, "PARSED")
        self.assertIsNone(doc.error_message)
        self.assertGreaterEqual(doc.total_pages, 1)

        initial_chunks_count = self.db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()
        initial_metrics_count = self.db.query(ExtractedMetric).filter(ExtractedMetric.document_id == doc.id).count()
        initial_fact_count = self.db.query(StructuredFact).filter(StructuredFact.document_id == doc.id).count()
        self.assertGreater(initial_chunks_count, 0)
        self.assertGreater(initial_fact_count, 0)

        # Run pipeline second time (idempotency verification)
        re_success = execute_document_processing_pipeline(self.db, doc.id)
        self.assertTrue(re_success)
        self.db.refresh(doc)
        self.assertEqual(doc.status, "PARSED")

        re_chunks_count = self.db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).count()
        re_metrics_count = self.db.query(ExtractedMetric).filter(ExtractedMetric.document_id == doc.id).count()
        re_fact_count = self.db.query(StructuredFact).filter(StructuredFact.document_id == doc.id).count()
        self.assertEqual(initial_chunks_count, re_chunks_count)
        self.assertEqual(initial_metrics_count, re_metrics_count)
        self.assertEqual(initial_fact_count, re_fact_count)

        # Cleanup test file
        delete_uploaded_file(file_path)

    def _make_knowledge_index_test_document(self, suffix):
        sample_content = b"ECL Rajmahal Open Cast Mine Coal Production 15.5 MT in FY 2023-24\n"
        file_path = save_uploaded_file(sample_content, f"hash_knowledge_{suffix}", f"knowledge_{suffix}.csv")
        doc = Document(
            filename=f"knowledge_{suffix}.csv",
            file_path=file_path,
            file_hash=f"hash_knowledge_{suffix}",
            file_type="CSV",
            file_size_bytes=len(sample_content),
            subsidiary="ECL",
            fiscal_year="2023-24",
            status="PENDING",
            uploaded_by=self.admin_user.id,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc, file_path

    def test_03b_pipeline_invokes_postgres_knowledge_index_after_authoritative_commit(self):
        doc, file_path = self._make_knowledge_index_test_document("success")

        def verify_authoritative_commit(db, document_id):
            current = db.get(Document, document_id)
            self.assertEqual(current.status, "PARSED")
            return {"chunk_count": 1, "embedding_status": "READY"}

        try:
            with patch(
                "app.services.processing_pipeline.index_document_knowledge",
                side_effect=verify_authoritative_commit,
            ) as knowledge_index, patch(
                "app.services.processing_pipeline.add_chunks_to_vector_store",
                return_value=True,
            ) as legacy_index:
                self.assertTrue(execute_document_processing_pipeline(self.db, doc.id))

            knowledge_index.assert_called_once_with(self.db, doc.id)
            legacy_index.assert_called_once()
        finally:
            delete_uploaded_file(file_path)

    def test_03c_knowledge_index_failure_does_not_fail_pipeline(self):
        doc, file_path = self._make_knowledge_index_test_document("failure")

        try:
            with patch(
                "app.services.processing_pipeline.index_document_knowledge",
                side_effect=RuntimeError("knowledge store unavailable"),
            ) as knowledge_index, patch(
                "app.services.processing_pipeline.add_chunks_to_vector_store",
                return_value=True,
            ) as legacy_index:
                self.assertTrue(execute_document_processing_pipeline(self.db, doc.id))

            knowledge_index.assert_called_once_with(self.db, doc.id)
            legacy_index.assert_called_once()
            self.db.refresh(doc)
            self.assertEqual(doc.status, "PARSED")
            self.assertIn(doc.processing_status, {"READY", "REVIEW_RECOMMENDED"})
        finally:
            delete_uploaded_file(file_path)

    def test_03d_legacy_chroma_indexing_remains_in_pipeline(self):
        doc, file_path = self._make_knowledge_index_test_document("legacy")

        try:
            with patch(
                "app.services.processing_pipeline.index_document_knowledge",
                return_value={"chunk_count": 1, "embedding_status": "READY"},
            ), patch(
                "app.services.processing_pipeline.add_chunks_to_vector_store",
                return_value=True,
            ) as legacy_index:
                self.assertTrue(execute_document_processing_pipeline(self.db, doc.id))

            legacy_index.assert_called_once()
        finally:
            delete_uploaded_file(file_path)

    def test_04_pipeline_missing_file_failure(self):
        """Verify pipeline safely fails and sets error_message if physical storage file is missing."""
        missing_doc = Document(
            filename="missing_file.pdf",
            file_path="./storage/uploads/non_existent_file_99999.pdf",
            file_hash="hash_non_existent_99999",
            file_type="PDF",
            file_size_bytes=1024,
            subsidiary="BCCL",
            fiscal_year="2023-24",
            status="PENDING",
            uploaded_by=self.admin_user.id,
        )
        self.db.add(missing_doc)
        self.db.commit()
        self.db.refresh(missing_doc)

        success = execute_document_processing_pipeline(self.db, missing_doc.id)
        self.assertFalse(success)
        self.db.refresh(missing_doc)
        self.assertEqual(missing_doc.status, "FAILED")
        self.assertIn("missing", missing_doc.error_message.lower())

    def test_04b_reprocesses_stale_failure_state_in_place(self):
        """A persisted extraction failure can be retried without an invalid transition."""
        sample_content = b"ECL Rajmahal production 15.5 MT in FY 2023-24"
        file_path = save_uploaded_file(sample_content, "hash_reprocess_state_1", "reprocess_state.csv")
        doc = Document(
            filename="reprocess_state.csv",
            file_path=file_path,
            file_hash="hash_reprocess_state_1",
            file_type="CSV",
            file_size_bytes=len(sample_content),
            status="FAILED",
            processing_status="EXTRACTION_FAILED",
            uploaded_by=self.admin_user.id,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        with patch("app.services.processing_pipeline.add_chunks_to_vector_store", return_value=True):
            self.assertTrue(execute_document_processing_pipeline(self.db, doc.id))
        self.db.refresh(doc)
        self.assertEqual(doc.status, "PARSED")
        self.assertIn(doc.processing_status, {"READY", "REVIEW_RECOMMENDED"})
        delete_uploaded_file(file_path)

    def test_04c_reprocesses_review_state_in_place(self):
        sample_content = b"ECL Rajmahal production 15.5 MT in FY 2023-24"
        file_path = save_uploaded_file(sample_content, "hash_reprocess_review_1", "reprocess_review.csv")
        doc = Document(
            filename="reprocess_review.csv",
            file_path=file_path,
            file_hash="hash_reprocess_review_1",
            file_type="CSV",
            file_size_bytes=len(sample_content),
            status="PARSED",
            processing_status="REVIEW_RECOMMENDED",
            uploaded_by=self.admin_user.id,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        with patch("app.services.processing_pipeline.add_chunks_to_vector_store", return_value=True):
            self.assertTrue(execute_document_processing_pipeline(self.db, doc.id))
        self.db.refresh(doc)
        self.assertEqual(doc.status, "PARSED")
        self.assertIn(doc.processing_status, {"READY", "REVIEW_RECOMMENDED"})
        delete_uploaded_file(file_path)

    def test_04d_metric_text_respects_legacy_column_limits(self):
        self.assertEqual(len(_bounded_metric_text("x" * 300, 30)), 30)
        self.assertEqual(_bounded_metric_text("", 10), "UNKNOWN")

    def test_05_stale_processing_recovery(self):
        """Verify recover_stale_processing_documents identifies orphaned PROCESSING docs without files."""
        # 1. Stale doc (> 15 mins ago) with missing file
        stale_missing_doc = Document(
            filename="stale_missing.pdf",
            file_path="./storage/uploads/missing_stale_file_888.pdf",
            file_hash="hash_stale_missing_888",
            file_type="PDF",
            file_size_bytes=2048,
            subsidiary="CCL",
            fiscal_year="2023-24",
            status="PROCESSING",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=30),
            uploaded_by=self.admin_user.id,
        )
        # 2. Fresh doc (5 mins ago)
        fresh_doc = Document(
            filename="fresh_doc.pdf",
            file_path="./storage/uploads/fresh_doc_file_777.pdf",
            file_hash="hash_fresh_doc_777",
            file_type="PDF",
            file_size_bytes=2048,
            subsidiary="MCL",
            fiscal_year="2023-24",
            status="PROCESSING",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
            uploaded_by=self.admin_user.id,
        )
        self.db.add_all([stale_missing_doc, fresh_doc])
        self.db.commit()

        recovered = recover_stale_processing_documents(self.db, stale_minutes=15)
        self.assertEqual(recovered, 1)

        self.db.refresh(stale_missing_doc)
        self.db.refresh(fresh_doc)

        self.assertEqual(stale_missing_doc.status, "FAILED")
        self.assertIn("interrupted", stale_missing_doc.error_message.lower())
        self.assertEqual(fresh_doc.status, "PROCESSING")  # Fresh processing was NOT falsely failed

    def test_06_upload_endpoint_non_blocking_fastapi(self):
        """Verify POST /api/v1/documents/upload returns HTTP 201 with PENDING without blocking."""
        from main import app
        from database import get_db

        def override_get_db():
            try:
                yield self.db
            finally:
                pass

        app.dependency_overrides[get_db] = override_get_db
        client = TestClient(app)

        token = create_access_token(
            subject=self.admin_user.username,
            role=self.admin_user.role,
            subsidiary=self.admin_user.subsidiary,
        )
        headers = {"Authorization": f"Bearer {token}"}

        test_content = b"%PDF-1.4 Mock Upload Test Content for Non-Blocking Endpoint"
        files = {"file": ("async_upload_test.pdf", test_content, "application/pdf")}
        data = {"subsidiary": "ECL", "fiscal_year": "2023-24"}

        with patch("app.api.documents.run_background_document_processing"):
            response = client.post("/api/v1/documents/upload", headers=headers, files=files, data=data)
            self.assertEqual(response.status_code, 201)
            resp_json = response.json()
            self.assertEqual(resp_json["filename"], "async_upload_test.pdf")
            self.assertEqual(resp_json["status"], "PENDING")
            self.assertEqual(resp_json["subsidiary"], "ECL")

        app.dependency_overrides.clear()

    def test_07_real_384d_embedding_contract_and_numerical_stability(self):
        """Verify 384-d vector contract, finite normalized values, and deterministic stability."""
        test_sentences = [
            "Eastern Coalfields Limited ECL Rajmahal Open Cast Mining Project produced 15.5 MT coal in FY 2023-24.",
            "Bharat Coking Coal Limited BCCL overburden removal reached 42.1 M.Cu.M.",
            "Coal India corporate headquarters reported composite performance achieving 98.2% target.",
            "SECL Gevra mega project is expanding annual capacity to 70 MT."
        ]

        embs_1 = generate_batch_embeddings(test_sentences, batch_size=2)
        embs_2 = generate_batch_embeddings(test_sentences, batch_size=2)

        self.assertEqual(len(embs_1), 4)
        for emb in embs_1:
            self.assertEqual(len(emb), EMBEDDING_DIMENSION)
            # Assert all values are finite float numbers
            for val in emb:
                self.assertFalse(np.isnan(val))
                self.assertFalse(np.isinf(val))
            # Assert L2 norm is approximately 1.0 (normalized for cosine similarity)
            norm = sum(x * x for x in emb) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=3)

        # Assert numerical stability across identical inputs
        for idx in range(len(test_sentences)):
            diff = max(abs(a - b) for a, b in zip(embs_1[idx], embs_2[idx]))
            self.assertLess(diff, 1e-5)

    def test_08_chroma_vector_deletion_idempotency(self):
        """Verify delete_document_vectors cleanly removes vectors for target document without errors."""
        res = delete_document_vectors(999999)
        self.assertTrue(res)


if __name__ == "__main__":
    unittest.main()
