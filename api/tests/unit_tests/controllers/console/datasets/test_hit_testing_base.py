import base64
from collections.abc import Callable
from unittest.mock import Mock, patch
from uuid import uuid4

import numpy as np
import pytest
import weaviate
from dify_vdb_weaviate.weaviate_vector import WeaviateVector
from flask import Flask
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from weaviate.collections.classes.internal import MetadataReturn, Object, QueryReturn
from weaviate.connect import ConnectionParams
from werkzeug.exceptions import Forbidden, InternalServerError, NotFound

import services.errors.base
from controllers.console.app.error import (
    CompletionRequestError,
    ProviderModelCurrentlyNotSupportError,
    ProviderNotInitializeError,
    ProviderQuotaExceededError,
)
from controllers.console.datasets.error import DatasetNotInitializedError
from controllers.console.datasets.hit_testing_base import (
    DatasetsHitTestingBase,
)
from core.errors.error import (
    LLMBadRequestError,
    ModelCurrentlyNotSupportError,
    ProviderTokenNotInitError,
    QuotaExceededError,
)
from core.rag.embedding import cached_embedding
from core.rag.index_processor.constant.index_type import IndexStructureType, IndexTechniqueType
from graphon.model_runtime.errors.invoke import InvokeError
from libs.helper import generate_text_hash
from models.account import Account, Tenant, TenantAccountRole
from models.dataset import Dataset, Document, DocumentSegment
from models.engine import db
from models.enums import DataSourceType, DocumentCreatedFrom, SegmentStatus
from services.hit_testing_service import HitTestingService
from services.knowledge.dataset_service import DatasetService


@pytest.fixture
def account() -> Account:
    acc = Account(name="User", email="user@example.com")
    acc.id = "account-1"
    tenant = Tenant(name="Tenant")
    tenant.id = "tenant-1"
    acc._current_tenant = tenant
    acc.role = TenantAccountRole.OWNER
    return acc


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(id="dataset-1", tenant_id="tenant-1", name="Dataset", created_by="account-1")


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("langgenius/ollama/ollama", "qwen3-embedding"),
        ("", ""),
        ("langgenius/ollama/ollama", "embeddinggemma:2-740m"),
    ],
    ids=["stale-model", "empty-fields", "matching-model"],
)
def test_weighted_hybrid_uses_knowledge_base_embedding(
    provider: str,
    model: str,
    account: Account,
    sqlite_engine: Engine,
    sqlite_session: Session,
    monkeypatch: pytest.MonkeyPatch,
    config_overrides: Callable[..., None],
) -> None:
    """Check both search branches and their fused scores for a configured knowledge base."""
    query = "embedding"
    ollama = "langgenius/ollama/ollama"
    embedding_model = "embeddinggemma:2-740m"
    dataset = Dataset(
        id=str(uuid4()),
        tenant_id="tenant-1",
        name="EmbeddingGemma knowledge base",
        created_by=account.id,
        indexing_technique=IndexTechniqueType.HIGH_QUALITY,
        embedding_model_provider=ollama,
        embedding_model=embedding_model,
    )
    document = Document(
        id=str(uuid4()),
        tenant_id=dataset.tenant_id,
        dataset_id=dataset.id,
        position=1,
        data_source_type=DataSourceType.UPLOAD_FILE,
        batch="batch",
        name="Reference",
        doc_form=IndexStructureType.PARAGRAPH_INDEX,
        created_from=DocumentCreatedFrom.API,
        created_by=account.id,
    )
    semantic, lexical = [
        DocumentSegment(
            tenant_id=dataset.tenant_id,
            dataset_id=dataset.id,
            document_id=document.id,
            index_node_id=str(uuid4()),
            position=position,
            content=content,
            word_count=1,
            tokens=1,
            status=SegmentStatus.COMPLETED,
            created_by=account.id,
        )
        for position, content in enumerate(("reference", query), start=1)
    ]
    sqlite_session.add_all([dataset, document, semantic, lexical])
    sqlite_session.commit()

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = sqlite_engine.url
    db.init_app(app)
    endpoint = "http://localhost:8080"
    config_overrides(VECTOR_STORE="weaviate", WEAVIATE_ENDPOINT=endpoint)
    client = weaviate.WeaviateClient(ConnectionParams.from_url(endpoint, grpc_port=50051))
    collection_name = Dataset.gen_collection_name_by_id(dataset.id)
    collection = client.collections.use(collection_name)
    monkeypatch.setattr(WeaviateVector, "_init_client", lambda _self, _config: client)
    monkeypatch.setattr(client.collections, "exists", lambda name: name == collection_name)
    monkeypatch.setattr(client.collections, "use", lambda _name: collection)

    def hit(
        segment: DocumentSegment, metadata: MetadataReturn, vector: dict[str, list[float] | list[list[float]]]
    ) -> Object[dict[str, str | None], None]:
        return Object(
            uuid=uuid4(),
            metadata=metadata,
            properties={"text": segment.content, "doc_id": segment.index_node_id, "document_id": document.id},
            references=None,
            vector=vector,
            collection=collection_name,
        )

    def near_vector(*, near_vector: list[float], **_kwargs: object) -> QueryReturn[dict[str, str | None], None]:
        assert near_vector == [1.0, 0.0]
        return QueryReturn(objects=[hit(semantic, MetadataReturn(distance=0.2), {})])

    monkeypatch.setattr(collection.query, "near_vector", near_vector)
    monkeypatch.setattr(
        collection.query,
        "bm25",
        lambda **_kwargs: QueryReturn(objects=[hit(lexical, MetadataReturn(), {"default": [1.0, 0.0]})]),
    )
    query_hash = generate_text_hash(query)
    embeddings = {
        f"{ollama}_{embedding_model}_{query_hash}": base64.b64encode(np.array([1.0, 0.0]).tobytes()),
        f"{ollama}_qwen3-embedding_{query_hash}": base64.b64encode(np.array([1.0, 0.0, 0.0]).tobytes()),
    }
    monkeypatch.setattr(cached_embedding.redis_client, "get", embeddings.get)

    def parse_args(search_method: str) -> dict[str, object]:
        return DatasetsHitTestingBase.parse_args(
            {
                "query": query,
                "retrieval_model": {
                    "search_method": search_method,
                    "reranking_mode": "weighted_score",
                    "reranking_enable": False,
                    "top_k": 4,
                    "score_threshold_enabled": False,
                    "weights": {
                        "vector_setting": {
                            "vector_weight": 0.75,
                            "embedding_provider_name": provider,
                            "embedding_model_name": model,
                        },
                        "keyword_setting": {"keyword_weight": 0.25},
                    },
                },
            }
        )

    def scores(args: dict[str, object]) -> dict[str, float]:
        response = DatasetsHitTestingBase.perform_hit_testing(sqlite_session, dataset, args, account, dataset.tenant_id)
        return {record["segment"]["id"]: record["score"] for record in response["records"]}

    hybrid_args = parse_args("hybrid_search")
    with app.app_context():
        assert scores(parse_args("semantic_search")) == {semantic.id: pytest.approx(0.8)}
        assert scores(parse_args("full_text_search")) == {lexical.id: 0.0}
        assert scores(hybrid_args) == {semantic.id: pytest.approx(0.6), lexical.id: pytest.approx(1.0)}
        assert scores(hybrid_args) == {semantic.id: pytest.approx(0.6), lexical.id: pytest.approx(1.0)}

    assert hybrid_args == parse_args("hybrid_search")


def hit_testing_record() -> dict[str, object]:
    return {
        "segment": {
            "id": "segment-1",
            "position": 1,
            "document_id": "document-1",
            "content": "Chunk text",
            "answer": None,
            "word_count": 2,
            "tokens": 3,
            "keywords": None,
            "index_node_id": None,
            "index_node_hash": None,
            "hit_count": 0,
            "enabled": True,
            "disabled_at": None,
            "disabled_by": None,
            "status": "completed",
            "created_by": "account-1",
            "created_at": 1_700_000_000,
            "indexing_at": None,
            "completed_at": None,
            "error": None,
            "stopped_at": None,
            "document": {
                "id": "document-1",
                "data_source_type": "upload_file",
                "name": "guide.md",
                "doc_type": None,
                "doc_metadata": None,
            },
        },
        "child_chunks": None,
        "files": None,
        "score": 0.8,
    }


class TestGetAndValidateDataset:
    def test_success(self, dataset: Dataset, account: Account) -> None:
        with (
            patch.object(
                DatasetService,
                "get_dataset",
                return_value=dataset,
            ),
            patch.object(
                DatasetService,
                "check_dataset_permission",
            ),
        ):
            result = DatasetsHitTestingBase.get_and_validate_dataset(Mock(), "dataset-1", account, "tenant-1")

        assert result == dataset

    def test_dataset_not_found(self, account: Account) -> None:
        with patch.object(
            DatasetService,
            "get_dataset",
            return_value=None,
        ):
            with pytest.raises(NotFound, match="Dataset not found"):
                DatasetsHitTestingBase.get_and_validate_dataset(Mock(), "dataset-1", account, "tenant-1")

    def test_permission_denied(self, dataset: Dataset, account: Account) -> None:
        with (
            patch.object(
                DatasetService,
                "get_dataset",
                return_value=dataset,
            ),
            patch.object(
                DatasetService,
                "check_dataset_permission",
                side_effect=services.errors.base.NoPermissionError("no access"),
            ),
        ):
            with pytest.raises(Forbidden, match="no access"):
                DatasetsHitTestingBase.get_and_validate_dataset(Mock(), "dataset-1", account, "tenant-1")


class TestHitTestingArgsCheck:
    def test_args_check_called(self) -> None:
        args = {"query": "test"}

        with patch.object(
            HitTestingService,
            "hit_testing_args_check",
        ) as check_mock:
            DatasetsHitTestingBase.hit_testing_args_check(args)

        check_mock.assert_called_once_with(args)


class TestParseArgs:
    def test_parse_args_success(self) -> None:
        payload = {"query": "hello"}

        result = DatasetsHitTestingBase.parse_args(payload)

        assert result["query"] == "hello"

    def test_parse_args_invalid(self) -> None:
        payload = {"query": "x" * 300}

        with pytest.raises(ValueError, match="1 validation error for HitTestingPayload\\\nquery"):
            DatasetsHitTestingBase.parse_args(payload)

    def test_parse_args_ignores_unknown_fields_for_compatibility(self) -> None:
        payload = {"query": "hello", "top_k": 3}

        result = DatasetsHitTestingBase.parse_args(payload)

        assert result == {"query": "hello"}


class TestPerformHitTesting:
    def test_success(self, dataset: Dataset, account: Account) -> None:
        response = {
            "query": {"content": "hello"},
            "records": list[object](),
        }

        with patch.object(
            HitTestingService,
            "retrieve",
            return_value=response,
        ):
            result = DatasetsHitTestingBase.perform_hit_testing(
                Mock(), dataset, {"query": "hello"}, account, "tenant-1"
            )

        assert result["query"] == {"content": "hello"}
        assert result["records"] == []

    def test_success_prepares_nullable_list_fields(self, dataset: Dataset, account: Account) -> None:
        response = {
            "query": {"content": "hello"},
            "records": [hit_testing_record()],
        }

        with patch.object(
            HitTestingService,
            "retrieve",
            return_value=response,
        ):
            result = DatasetsHitTestingBase.perform_hit_testing(
                Mock(), dataset, {"query": "hello"}, account, "tenant-1"
            )

        assert result["query"] == {"content": "hello"}
        record = result["records"][0]
        assert record["segment"]["keywords"] == []
        assert record["segment"]["sign_content"] is None
        assert record["child_chunks"] == []
        assert record["files"] == []
        assert record["score"] == 0.8
        assert record["tsne_position"] is None
        assert record["summary"] is None

    def test_invalid_query_response_raises_value_error(self, dataset: Dataset, account: Account) -> None:
        with (
            patch.object(
                HitTestingService,
                "retrieve",
                return_value={"query": "hello", "records": []},
            ),
            pytest.raises(ValueError, match="Invalid hit testing query response"),
        ):
            DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")

    def test_invalid_records_response_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid hit testing records response"):
            DatasetsHitTestingBase._prepare_hit_testing_records({"records": []})

    def test_invalid_record_response_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid hit testing record response"):
            DatasetsHitTestingBase._prepare_hit_testing_records(["record"])

    def test_index_not_initialized(self, dataset: Dataset, account: Account) -> None:
        with patch.object(
            HitTestingService,
            "retrieve",
            side_effect=services.errors.index.IndexNotInitializedError(),
        ):
            with pytest.raises(DatasetNotInitializedError):
                DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")

    def test_provider_token_not_init(self, dataset: Dataset, account: Account) -> None:
        with patch.object(
            HitTestingService,
            "retrieve",
            side_effect=ProviderTokenNotInitError("token missing"),
        ):
            with pytest.raises(ProviderNotInitializeError):
                DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")

    def test_quota_exceeded(self, dataset: Dataset, account: Account) -> None:
        with patch.object(
            HitTestingService,
            "retrieve",
            side_effect=QuotaExceededError(),
        ):
            with pytest.raises(ProviderQuotaExceededError):
                DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")

    def test_model_not_supported(self, dataset: Dataset, account: Account) -> None:
        with patch.object(
            HitTestingService,
            "retrieve",
            side_effect=ModelCurrentlyNotSupportError(),
        ):
            with pytest.raises(ProviderModelCurrentlyNotSupportError):
                DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")

    def test_llm_bad_request(self, dataset: Dataset, account: Account) -> None:
        with patch.object(
            HitTestingService,
            "retrieve",
            side_effect=LLMBadRequestError("bad request"),
        ):
            with pytest.raises(ProviderNotInitializeError):
                DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")

    def test_invoke_error(self, dataset: Dataset, account: Account) -> None:
        with patch.object(
            HitTestingService,
            "retrieve",
            side_effect=InvokeError("invoke failed"),
        ):
            with pytest.raises(CompletionRequestError):
                DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")

    def test_value_error(self, dataset: Dataset, account: Account) -> None:
        with patch.object(
            HitTestingService,
            "retrieve",
            side_effect=ValueError("bad args"),
        ):
            with pytest.raises(ValueError, match="bad args"):
                DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")

    def test_unexpected_error(self, dataset: Dataset, account: Account) -> None:
        with patch.object(
            HitTestingService,
            "retrieve",
            side_effect=Exception("boom"),
        ):
            with pytest.raises(InternalServerError) as exc_info:
                DatasetsHitTestingBase.perform_hit_testing(Mock(), dataset, {"query": "hello"}, account, "tenant-1")
            assert "boom" not in str(exc_info.value)


@pytest.mark.parametrize(
    "attachments",
    [
        [""],
        [" "],
        ["not-a-uuid"],
        [None],
        [1],
        [True],
        [{}],
        "not-a-list",
        ["9c603550-52be-46d6-8078-f5047c6b631c", ""],
    ],
)
def test_parse_args_rejects_invalid_attachment_ids(attachments: object) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="attachment_ids"):
        DatasetsHitTestingBase.parse_args({"query": "", "attachment_ids": attachments})


@pytest.mark.parametrize("attachments", [None, []])
def test_parse_args_preserves_optional_attachments(attachments: list[str] | None) -> None:
    parsed = DatasetsHitTestingBase.parse_args({"query": "example", "attachment_ids": attachments})
    assert parsed.get("attachment_ids") == attachments
    DatasetsHitTestingBase.hit_testing_args_check(parsed)


def test_parse_args_normalizes_attachment_only_query() -> None:
    value = "9c603550-52be-46d6-8078-f5047c6b631c"
    parsed = DatasetsHitTestingBase.parse_args({"query": "", "attachment_ids": [value.upper(), value.replace("-", "")]})
    assert parsed["attachment_ids"] == [value, value]
    DatasetsHitTestingBase.hit_testing_args_check(parsed)
