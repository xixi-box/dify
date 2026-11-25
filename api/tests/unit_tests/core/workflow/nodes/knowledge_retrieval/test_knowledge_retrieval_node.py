import time
import uuid
from unittest.mock import Mock

import pytest
from flask import Flask
from pytest_mock import MockerFixture
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core.app.entities.app_invoke_entities import InvokeFrom, UserFrom
from core.rag.datasource import retrieval_service
from core.rag.models.document import Document
from core.rag.retrieval.dataset_retrieval import DatasetRetrieval
from core.workflow.file_reference import build_file_reference
from core.workflow.nodes.knowledge_retrieval.entities import (
    Condition,
    KnowledgeRetrievalNodeData,
    MetadataFilteringCondition,
    MultipleRetrievalConfig,
    RerankingModelConfig,
    SingleRetrievalConfig,
)
from core.workflow.nodes.knowledge_retrieval.exc import RateLimitExceededError
from core.workflow.nodes.knowledge_retrieval.knowledge_retrieval_node import (
    KnowledgeRetrievalNode,
    _normalize_metadata_filter_scalar,
    _normalize_metadata_filter_sequence_item,
)
from core.workflow.nodes.knowledge_retrieval.retrieval import Source, SourceMetadata
from core.workflow.system_variables import build_system_variables
from extensions.ext_database import db
from graphon.enums import WorkflowNodeExecutionStatus
from graphon.file import File, FileTransferMethod, FileType
from graphon.model_runtime.entities.llm_entities import LLMUsage
from graphon.runtime import RuntimeState, VariablePool
from graphon.variables import ArrayObjectSegment, FileSegment, StringSegment
from models.dataset import Dataset
from models.enums import IndexingStatus
from tests.helpers.unit.model_factories import make_dataset, make_document
from tests.workflow_test_utils import build_test_graph_init_params


def _source() -> Source:
    return Source(
        metadata=SourceMetadata(
            dataset_id="dataset-id",
            dataset_name="Knowledge",
            document_id="document-id",
            document_name="Python",
            data_source_type="upload_file",
        ),
        title="Python",
        content="Python is a programming language",
    )


@pytest.fixture
def mock_graph_init_params():
    """Create mock InitParams."""
    return build_test_graph_init_params(
        workflow_id=str(uuid.uuid4()),
        graph_config={},
        tenant_id=str(uuid.uuid4()),
        app_id=str(uuid.uuid4()),
        user_id=str(uuid.uuid4()),
        user_from=UserFrom.ACCOUNT,
        invoke_from=InvokeFrom.DEBUGGER,
        call_depth=0,
    )


@pytest.fixture
def mock_graph_runtime_state():
    """Create mock RuntimeState."""
    variable_pool = VariablePool.from_bootstrap(
        system_variables=build_system_variables(user_id=str(uuid.uuid4()), files=[]),
        user_inputs={},
        environment_variables=[],
        conversation_variables=[],
    )
    return RuntimeState(workflow_id="test-workflow", variable_pool=variable_pool, start_at=time.perf_counter())


@pytest.fixture
def mock_rag_retrieval(mocker: MockerFixture):
    """Keep real usage and request metadata while isolating retrieval I/O."""
    mock_retrieval = DatasetRetrieval()
    mocker.patch.object(mock_retrieval, "knowledge_retrieval", return_value=[])
    mocker.patch(
        "core.workflow.nodes.knowledge_retrieval.knowledge_retrieval_node.DatasetRetrieval",
        return_value=mock_retrieval,
    )
    return mock_retrieval


@pytest.fixture
def sample_node_data():
    """Create sample KnowledgeRetrievalNodeData."""
    return KnowledgeRetrievalNodeData(
        title="Knowledge Retrieval",
        type="knowledge-retrieval",
        dataset_ids=[str(uuid.uuid4())],
        retrieval_mode="multiple",
        multiple_retrieval_config=MultipleRetrievalConfig(
            top_k=5,
            score_threshold=0.7,
            reranking_mode="reranking_model",
            reranking_enable=True,
            reranking_model=RerankingModelConfig(
                provider="cohere",
                model="rerank-v2",
            ),
        ),
    )


def test_metadata_filter_normalizers_preserve_numeric_scalars_and_stringify_other_values() -> None:
    assert _normalize_metadata_filter_scalar(3) == 3
    assert _normalize_metadata_filter_scalar(True) == "True"
    assert _normalize_metadata_filter_sequence_item(4) == "4"


class TestKnowledgeRetrievalNode:
    """
    Test suite for KnowledgeRetrievalNode.
    """

    @pytest.mark.parametrize(
        ("failed_datasets", "attachment_fails", "expected_status"),
        [
            pytest.param(2, None, WorkflowNodeExecutionStatus.FAILED, id="all-searches-fail"),
            pytest.param(1, None, WorkflowNodeExecutionStatus.SUCCEEDED, id="one-search-succeeds-with-no-hits"),
            pytest.param(0, None, WorkflowNodeExecutionStatus.SUCCEEDED, id="all-searches-succeed-with-no-hits"),
            pytest.param(2, False, WorkflowNodeExecutionStatus.SUCCEEDED, id="attachment-searches-succeed"),
            pytest.param(2, True, WorkflowNodeExecutionStatus.FAILED, id="text-and-attachment-searches-fail"),
        ],
    )
    def test_run_with_dataset_search_outcomes(
        self,
        sqlite_engine: Engine,
        sqlite_session: Session,
        monkeypatch: pytest.MonkeyPatch,
        mock_graph_runtime_state,
        failed_datasets: int,
        attachment_fails: bool | None,
        expected_status: WorkflowNodeExecutionStatus,
    ) -> None:
        tenant_id = str(uuid.uuid4())
        dataset_ids = [str(uuid.uuid4()) for _ in range(2)]
        for dataset_id in dataset_ids:
            dataset = make_dataset(dataset_id=dataset_id, tenant_id=tenant_id, indexing_technique="high_quality")
            dataset.is_multimodal = attachment_fails is not None
            sqlite_session.add(dataset)
            document = make_document(document_id=str(uuid.uuid4()), dataset_id=dataset_id, tenant_id=tenant_id)
            document.indexing_status = IndexingStatus.COMPLETED
            sqlite_session.add(document)
        sqlite_session.commit()
        app = Flask(__name__)
        app.config["SQLALCHEMY_DATABASE_URI"] = str(sqlite_engine.url)
        db.init_app(app)

        searched_dataset_ids: list[str] = []
        attachment_dataset_ids: list[str] = []

        class TestVector:
            def __init__(self, *, dataset: Dataset, session: Session) -> None:
                self.dataset_id = dataset.id

            def search_by_vector(self, query: str, **kwargs: object) -> list[Document]:
                searched_dataset_ids.append(self.dataset_id)
                if self.dataset_id in dataset_ids[:failed_datasets]:
                    raise RuntimeError("Vector search unavailable")
                return []

            def search_by_file(self, file_id: str, **kwargs: object) -> list[Document]:
                attachment_dataset_ids.append(self.dataset_id)
                if attachment_fails:
                    raise RuntimeError("Vector search unavailable")
                return []

        monkeypatch.setattr(retrieval_service, "Vector", TestVector)
        query_selector = ["start", "query"]
        mock_graph_runtime_state.variable_pool.add(query_selector, StringSegment(value="What is Python?"))
        attachment_selector = None
        if attachment_fails is not None:
            attachment_selector = ["start", "image"]
            mock_graph_runtime_state.variable_pool.add(
                attachment_selector,
                FileSegment(
                    value=File(
                        type=FileType.IMAGE,
                        transfer_method=FileTransferMethod.LOCAL_FILE,
                        reference=build_file_reference(record_id=str(uuid.uuid4())),
                    )
                ),
            )
        node = KnowledgeRetrievalNode(
            node_id="knowledge-retrieval",
            data=KnowledgeRetrievalNodeData(
                title="Knowledge Retrieval",
                type="knowledge-retrieval",
                dataset_ids=dataset_ids,
                query_variable_selector=query_selector,
                query_attachment_selector=attachment_selector,
                retrieval_mode="multiple",
                multiple_retrieval_config=MultipleRetrievalConfig(top_k=5, reranking_enable=False),
            ),
            init_params=build_test_graph_init_params(tenant_id=tenant_id),
            runtime_state=mock_graph_runtime_state,
        )

        with app.app_context():
            result = node._run()
            db.engine.dispose()

        assert sorted(searched_dataset_ids) == sorted(dataset_ids)
        if attachment_fails is not None:
            assert sorted(attachment_dataset_ids) == sorted(dataset_ids)
        assert result.status == expected_status
        if expected_status == WorkflowNodeExecutionStatus.FAILED:
            assert result.error == "Vector search unavailable"
        else:
            assert result.outputs["result"] == ArrayObjectSegment(value=[])

    def test_run_with_no_query_or_attachment(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
        sample_node_data,
    ):
        """Test _run returns success when no query or attachment is provided."""
        # Arrange
        sample_node_data.query_variable_selector = None
        sample_node_data.query_attachment_selector = None

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": sample_node_data.model_dump(),
        }

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        result = node._run()

        # Assert
        assert result.status == WorkflowNodeExecutionStatus.SUCCEEDED
        assert result.outputs == {}
        assert mock_rag_retrieval.knowledge_retrieval.call_count == 0

    def test_run_with_query_variable_single_mode(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
    ):
        """Test _run with query variable in single mode."""
        # Arrange
        from graphon.nodes.llm.entities import ModelConfig

        query = "What is Python?"
        query_selector = ["start", "query"]

        # Add query to variable pool
        mock_graph_runtime_state.variable_pool.add(query_selector, StringSegment(value=query))

        node_data = KnowledgeRetrievalNodeData(
            title="Knowledge Retrieval",
            type="knowledge-retrieval",
            dataset_ids=[str(uuid.uuid4())],
            retrieval_mode="single",
            query_variable_selector=query_selector,
            single_retrieval_config=SingleRetrievalConfig(
                model=ModelConfig(
                    provider="openai",
                    name="gpt-4",
                    mode="chat",
                    completion_params={"temperature": 0.7},
                )
            ),
        )

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": node_data.model_dump(),
        }

        # Mock retrieval response
        mock_source = _source()
        mock_rag_retrieval.knowledge_retrieval.return_value = [mock_source]

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        result = node._run()

        # Assert
        assert result.status == WorkflowNodeExecutionStatus.SUCCEEDED
        assert "result" in result.outputs
        assert mock_rag_retrieval.knowledge_retrieval.called
        assert result.outputs["result"] == ArrayObjectSegment(value=[mock_source.model_dump(by_alias=True)])

    def test_run_with_query_variable_multiple_mode(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
        sample_node_data,
    ):
        """Test _run with query variable in multiple mode."""
        # Arrange
        query = "What is Python?"
        query_selector = ["start", "query"]

        # Add query to variable pool
        mock_graph_runtime_state.variable_pool.add(query_selector, StringSegment(value=query))
        sample_node_data.query_variable_selector = query_selector

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": sample_node_data.model_dump(),
        }

        # Mock retrieval response
        mock_source = _source()
        mock_rag_retrieval.knowledge_retrieval.return_value = [mock_source]

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        result = node._run()

        # Assert
        assert result.status == WorkflowNodeExecutionStatus.SUCCEEDED
        assert "result" in result.outputs
        assert mock_rag_retrieval.knowledge_retrieval.called

    def test_run_with_invalid_query_variable_type(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
        sample_node_data,
    ):
        """Test _run fails when query variable is not StringSegment."""
        # Arrange
        query_selector = ["start", "query"]

        # Add non-string variable to variable pool
        mock_graph_runtime_state.variable_pool.add(query_selector, [1, 2, 3])
        sample_node_data.query_variable_selector = query_selector

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": sample_node_data.model_dump(),
        }

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        result = node._run()

        # Assert
        assert result.status == WorkflowNodeExecutionStatus.FAILED
        assert "Query variable is not string type" in result.error

    def test_run_with_invalid_attachment_variable_type(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
        sample_node_data,
    ):
        """Test _run fails when attachment variable is not FileSegment or ArrayFileSegment."""
        # Arrange
        attachment_selector = ["start", "attachments"]

        # Add non-file variable to variable pool
        mock_graph_runtime_state.variable_pool.add(attachment_selector, "not a file")
        sample_node_data.query_attachment_selector = attachment_selector

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": sample_node_data.model_dump(),
        }

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        result = node._run()

        # Assert
        assert result.status == WorkflowNodeExecutionStatus.FAILED
        assert "Attachments variable is not array file or file type" in result.error

    def test_run_with_rate_limit_exceeded(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
        sample_node_data,
    ):
        """Test _run handles RateLimitExceededError properly."""
        # Arrange
        query = "What is Python?"
        query_selector = ["start", "query"]

        mock_graph_runtime_state.variable_pool.add(query_selector, StringSegment(value=query))
        sample_node_data.query_variable_selector = query_selector

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": sample_node_data.model_dump(),
        }

        # Mock retrieval to raise RateLimitExceededError
        mock_rag_retrieval.knowledge_retrieval.side_effect = RateLimitExceededError(
            "knowledge base request rate limit exceeded"
        )

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        result = node._run()

        # Assert
        assert result.status == WorkflowNodeExecutionStatus.FAILED
        assert "rate limit" in result.error.lower()

    def test_run_with_generic_exception(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
        sample_node_data,
    ):
        """Test _run handles generic exceptions properly."""
        # Arrange
        query = "What is Python?"
        query_selector = ["start", "query"]

        mock_graph_runtime_state.variable_pool.add(query_selector, StringSegment(value=query))
        sample_node_data.query_variable_selector = query_selector

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": sample_node_data.model_dump(),
        }

        # Mock retrieval to raise generic exception
        mock_rag_retrieval.knowledge_retrieval.side_effect = Exception("Unexpected error")

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        result = node._run()

        # Assert
        assert result.status == WorkflowNodeExecutionStatus.FAILED
        assert "Unexpected error" in result.error

    def test_extract_variable_selector_to_variable_mapping(self):
        """Test _extract_variable_selector_to_variable_mapping class method."""
        # Arrange
        node_id = "knowledge_node_1"
        node_data = KnowledgeRetrievalNodeData(
            type="knowledge-retrieval",
            title="Knowledge Retrieval",
            dataset_ids=[str(uuid.uuid4())],
            retrieval_mode="multiple",
            query_variable_selector=["start", "query"],
            query_attachment_selector=["start", "attachments"],
        )
        graph_config = {}

        # Act
        mapping = KnowledgeRetrievalNode._extract_variable_selector_to_variable_mapping(
            graph_config=graph_config,
            node_id=node_id,
            node_data=node_data,
        )

        # Assert
        assert mapping[f"{node_id}.query"] == ["start", "query"]
        assert mapping[f"{node_id}.queryAttachment"] == ["start", "attachments"]


class TestFetchDatasetRetriever:
    """
    Test suite for _fetch_dataset_retriever method.
    """

    def test_fetch_dataset_retriever_single_mode(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
    ):
        """Test _fetch_dataset_retriever in single mode."""
        # Arrange
        from graphon.nodes.llm.entities import ModelConfig

        query = "What is Python?"
        variables = {"query": query}

        node_data = KnowledgeRetrievalNodeData(
            title="Knowledge Retrieval",
            type="knowledge-retrieval",
            dataset_ids=[str(uuid.uuid4())],
            retrieval_mode="single",
            single_retrieval_config=SingleRetrievalConfig(
                model=ModelConfig(
                    provider="openai",
                    name="gpt-4",
                    mode="chat",
                    completion_params={"temperature": 0.7},
                )
            ),
        )

        # Mock retrieval response
        mock_source = _source()
        mock_rag_retrieval.knowledge_retrieval.return_value = [mock_source]

        node_id = str(uuid.uuid4())
        config = {"id": node_id, "data": node_data.model_dump()}

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        results, usage = node._fetch_dataset_retriever(Mock(), node_data=node_data, variables=variables)

        # Assert
        assert len(results) == 1
        assert isinstance(usage, LLMUsage)
        assert mock_rag_retrieval.knowledge_retrieval.called

    def test_fetch_dataset_retriever_multiple_mode_with_reranking(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
        sample_node_data,
    ):
        """Test _fetch_dataset_retriever in multiple mode with reranking."""
        # Arrange
        query = "What is Python?"
        variables = {"query": query}

        # Mock retrieval response
        mock_rag_retrieval.knowledge_retrieval.return_value = []

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": sample_node_data.model_dump(),
        }

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        results, usage = node._fetch_dataset_retriever(Mock(), node_data=sample_node_data, variables=variables)

        # Assert
        assert isinstance(results, list)
        assert isinstance(usage, LLMUsage)
        assert mock_rag_retrieval.knowledge_retrieval.called

        # Verify reranking parameters via request object
        call_args = mock_rag_retrieval.knowledge_retrieval.call_args
        request = call_args[1]["request"]
        assert request.reranking_enable is True
        assert request.reranking_mode == "reranking_model"

    def test_fetch_dataset_retriever_multiple_mode_without_reranking(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
    ):
        """Test _fetch_dataset_retriever in multiple mode without reranking."""
        # Arrange
        query = "What is Python?"
        variables = {"query": query}

        node_data = KnowledgeRetrievalNodeData(
            title="Knowledge Retrieval",
            type="knowledge-retrieval",
            dataset_ids=[str(uuid.uuid4())],
            retrieval_mode="multiple",
            multiple_retrieval_config=MultipleRetrievalConfig(
                top_k=5,
                score_threshold=0.7,
                reranking_enable=False,
                reranking_mode="reranking_model",
            ),
        )

        # Mock retrieval response
        mock_rag_retrieval.knowledge_retrieval.return_value = []

        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": node_data.model_dump(),
        }

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        # Act
        results, usage = node._fetch_dataset_retriever(Mock(), node_data=node_data, variables=variables)

        # Assert
        assert isinstance(results, list)
        assert mock_rag_retrieval.knowledge_retrieval.called

        # Verify reranking is disabled
        call_args = mock_rag_retrieval.knowledge_retrieval.call_args
        request = call_args[1]["request"]
        assert request.reranking_enable is False

    def test_resolve_metadata_filtering_conditions_templates(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
    ):
        """_resolve_metadata_filtering_conditions should expand {{#...#}} and keep numbers/None unchanged."""
        # Arrange
        node_id = str(uuid.uuid4())
        config = {
            "id": node_id,
            "data": {
                "title": "Knowledge Retrieval",
                "type": "knowledge-retrieval",
                "dataset_ids": [str(uuid.uuid4())],
                "retrieval_mode": "multiple",
            },
        }
        # Variable in pool used by template
        mock_graph_runtime_state.variable_pool.add(["start", "query"], StringSegment(value="readme"))

        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        conditions = MetadataFilteringCondition(
            logical_operator="and",
            conditions=[
                Condition(name="document_name", comparison_operator="is", value="{{#start.query#}}"),
                Condition(name="tags", comparison_operator="in", value=["x", "{{#start.query#}}"]),
                Condition(name="year", comparison_operator="=", value=2025),
            ],
        )

        # Act
        resolved = node._resolve_metadata_filtering_conditions(conditions)

        # Assert
        assert resolved.logical_operator == "and"
        assert resolved.conditions[0].value == "readme"
        assert isinstance(resolved.conditions[1].value, list)
        assert resolved.conditions[1].value[1] == "readme"
        assert resolved.conditions[2].value == 2025

    def test_fetch_passes_resolved_metadata_conditions(
        self,
        mock_graph_init_params,
        mock_graph_runtime_state,
        mock_rag_retrieval,
    ):
        """_fetch_dataset_retriever should pass resolved metadata conditions into request."""
        # Arrange
        query = "hi"
        variables = {"query": query}
        mock_graph_runtime_state.variable_pool.add(["start", "q"], StringSegment(value="readme"))

        node_data = KnowledgeRetrievalNodeData(
            title="Knowledge Retrieval",
            type="knowledge-retrieval",
            dataset_ids=[str(uuid.uuid4())],
            retrieval_mode="multiple",
            multiple_retrieval_config=MultipleRetrievalConfig(
                top_k=4,
                score_threshold=0.0,
                reranking_mode="reranking_model",
                reranking_enable=True,
                reranking_model=RerankingModelConfig(provider="cohere", model="rerank-v2"),
            ),
            metadata_filtering_mode="manual",
            metadata_filtering_conditions=MetadataFilteringCondition(
                logical_operator="and",
                conditions=[
                    Condition(name="document_name", comparison_operator="is", value="{{#start.q#}}"),
                ],
            ),
        )

        node_id = str(uuid.uuid4())
        config = {"id": node_id, "data": node_data.model_dump()}
        node = KnowledgeRetrievalNode(
            node_id=node_id,
            data=KnowledgeRetrievalNodeData.model_validate(config["data"]),
            init_params=mock_graph_init_params,
            runtime_state=mock_graph_runtime_state,
        )

        mock_rag_retrieval.knowledge_retrieval.return_value = []

        # Act
        node._fetch_dataset_retriever(Mock(), node_data=node_data, variables=variables)

        # Assert the passed request has resolved value
        call_args = mock_rag_retrieval.knowledge_retrieval.call_args
        request = call_args[1]["request"]
        assert request.metadata_filtering_conditions is not None
        assert request.metadata_filtering_conditions.conditions[0].value == "readme"
