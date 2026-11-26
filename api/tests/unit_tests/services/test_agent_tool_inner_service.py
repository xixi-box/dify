"""Unit tests for the Agent tool inner invoke service."""

import asyncio
import json
import sqlite3
from collections.abc import Generator
from dataclasses import dataclass, field
from unittest.mock import MagicMock, PropertyMock, patch

import httpx
import pytest
from Crypto.PublicKey import RSA
from dify_agent.layers.dify_core_tools import DifyCoreToolsLayerConfig
from dify_agent.layers.dify_core_tools.client import DifyCoreToolsClient
from dify_agent.layers.execution_context import DifyExecutionContextLayerConfig
from dify_agent.protocol import CreateRunRequest
from flask import Flask
from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

import core.db.session_factory as session_factory_module
from clients.agent_backend import (
    DIFY_CORE_TOOLS_LAYER_ID,
    DIFY_EXECUTION_CONTEXT_LAYER_ID,
    AgentBackendAgentAppRunInput,
    AgentBackendModelConfig,
    AgentBackendRunRequestBuilder,
    AgentBackendWorkflowNodeRunInput,
)
from controllers.inner_api import bp as inner_api_bp
from core.app.entities.app_invoke_entities import InvokeFrom, UserFrom
from core.helper import encrypter
from core.mcp.types import CallToolResult, TextContent
from core.tools.__base.tool_runtime import ToolRuntime
from core.tools.entities.tool_entities import ToolInvokeFrom, ToolInvokeMessage, ToolProviderType
from core.tools.errors import (
    ToolInvokeError,
    ToolParameterValidationError,
    ToolProviderCredentialValidationError,
    ToolProviderNotFoundError,
)
from core.tools.mcp_tool.provider import MCPToolProviderController
from core.workflow.file_reference import build_file_reference, is_canonical_file_reference, resolve_file_record_id
from core.workflow.nodes.agent_v2.dify_tools_builder import WorkflowAgentDifyToolsBuilder
from extensions.ext_database import db
from graphon.file import File, FileTransferMethod, FileType
from libs import rsa
from libs.key_providers.rsa_key_provider import RSAKeyProvider
from models import ToolFile
from models.agent_config_entities import AgentSoulToolsConfig
from models.model import App, AppMode
from services.agent_tool_inner_service import AgentToolInnerService
from services.entities.agent_tool_inner import AgentToolInvokeRequest
from services.errors.agent_tool_inner import AgentToolInnerServiceError
from tests.helpers.unit.config_override import config_overrides_context
from tests.helpers.unit.core.mcp.fixtures import make_provider


def _request() -> AgentToolInvokeRequest:
    return AgentToolInvokeRequest.model_validate(
        {
            "caller": {
                "tenant_id": "tenant-1",
                "user_id": "user-1",
                "user_from": "account",
                "app_id": "app-1",
                "invoke_from": "service-api",
                "conversation_id": "conversation-1",
                "workflow_id": "workflow-1",
                "workflow_run_id": "workflow-run-1",
                "node_id": "node-1",
                "node_execution_id": "node-exec-1",
                "agent_id": "agent-1",
                "agent_config_version_id": "snapshot-1",
                "trace_session_id": "trace-session-1",
            },
            "tool": {
                "provider_type": "plugin",
                "provider_id": "langgenius/search/search",
                "tool_name": "search",
                "credential_id": "credential-1",
                "tool_parameters": {"query": "dify"},
                "runtime_parameters": {"region": "us"},
            },
        }
    )


@dataclass
class _FakeTool:
    """Stands in for the resolved agent tool; only its runtime is mutated by the service."""

    runtime: ToolRuntime = field(default_factory=lambda: ToolRuntime(tenant_id="tenant-1"))


def _persist_app(session: Session, *, tenant_id: str = "tenant-1") -> App:
    app = App(
        id="app-1",
        tenant_id=tenant_id,
        name="Agent tool gateway test app",
        description="",
        mode=AppMode.AGENT_CHAT,
        enable_site=False,
        enable_api=False,
        max_active_requests=None,
    )
    session.add(app)
    session.commit()
    return app


def _messages() -> Generator[ToolInvokeMessage]:
    yield ToolInvokeMessage(
        type=ToolInvokeMessage.MessageType.TEXT,
        message=ToolInvokeMessage.TextMessage(text="ok"),
    )


def _invoke_observation(session: Session, messages: list[ToolInvokeMessage]) -> str:
    """Exercise the public service with the resolved tool's transformed messages."""
    with (
        patch("services.agent_tool_inner_service.ToolManager.get_agent_tool_runtime", return_value=_FakeTool()),
        patch("services.agent_tool_inner_service.ToolEngine.generic_invoke", return_value=iter(messages)),
        patch(
            "services.agent_tool_inner_service.ToolFileMessageTransformer.transform_tool_invoke_messages",
            side_effect=lambda messages, **_kwargs: messages,
        ),
    ):
        return AgentToolInnerService().invoke(_request(), session=session).observation


def test_agent_v2_resolves_mcp_headers_from_original_request(app: Flask, sqlite_session: Session) -> None:
    _persist_app(sqlite_session)
    inner_app = Flask(__name__)
    inner_app.config["TESTING"] = True
    inner_app.register_blueprint(inner_api_bp)
    private_key = RSA.generate(2048)
    public_key = private_key.public_key().export_key()
    provider = make_provider({})
    provider.tenant_id = "tenant-1"
    provider.icon = "icon.svg"
    provider.tools = [{"name": "search", "inputSchema": {"type": "object"}}]
    header_templates = {
        "Authorization": "Bearer {{request.headers.x-caller-token}}",
        "X-Combined": "{{ request.header.X-Region }} / {{request.headers.X-Missing}}",
        "X-Static": "static-value",
        "X-Internal": "{{request.headers.X-Inner-Api-Key}}",
    }
    tools = AgentSoulToolsConfig.model_validate(
        {
            "dify_tools": [
                {
                    "provider_type": "mcp",
                    "provider_id": "test-server",
                    "tool_name": "search",
                    "credential_type": "unauthorized",
                }
            ]
        }
    )
    outbound_headers: list[dict[str, str]] = []

    def capture_mcp_transport(**kwargs):
        outbound_headers.append(kwargs["headers"].copy())
        return transport

    def invoke_inner(request: httpx.Request) -> httpx.Response:
        response = inner_app.test_client().open(
            request.url.path, method=request.method, headers=dict(request.headers), data=request.content
        )
        return httpx.Response(response.status_code, content=response.data)

    async def invoke_core_tool(
        config: DifyCoreToolsLayerConfig, execution_context: DifyExecutionContextLayerConfig
    ) -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(invoke_inner)) as http_client:
            client = DifyCoreToolsClient(base_url="http://api", api_key="inner-only-value", http_client=http_client)
            response = await client.invoke(
                execution_context=execution_context, tool_config=config.tools[0], tool_parameters={}
            )
        return response.observation

    transport = MagicMock()
    transport.__enter__.return_value = (MagicMock(), MagicMock(), MagicMock())
    with (
        config_overrides_context(PLUGIN_DAEMON_KEY="plugin-daemon-key", INNER_API_KEY_FOR_PLUGIN="inner-only-value"),
        patch.object(type(db), "engine", new_callable=PropertyMock, return_value=sqlite_session.get_bind()),
        patch.object(RSAKeyProvider, "encrypt", side_effect=lambda _, value: rsa.encrypt(value, public_key)),
        patch.object(
            rsa, "get_decrypt_decoding", return_value=(private_key, rsa.gmpy2_pkcs10aep_cipher.new(private_key))
        ),
        patch("core.tools.tool_manager.ToolManager.get_mcp_provider_controller") as get_controller,
        patch(
            "services.tools.mcp_tools_manage_service.MCPToolManageService.get_provider_entity_by_persisted_reference",
            return_value=provider,
        ),
        patch(
            "services.tools.mcp_tools_manage_service.MCPToolManageService.get_provider_entity_by_server_identifier",
            return_value=provider,
        ),
        patch("core.mcp.mcp_client.streamablehttp_client", side_effect=capture_mcp_transport),
        patch("core.mcp.mcp_client.ClientSession") as mcp_session,
    ):
        provider.server_url = encrypter.encrypt_token(provider.tenant_id, "https://mcp.example.test/mcp")
        provider.headers = {
            key: encrypter.encrypt_token(provider.tenant_id, value) for key, value in header_templates.items()
        }
        get_controller.return_value = MCPToolProviderController.from_entity(provider)
        mcp_session.return_value.__enter__.return_value.call_tool.return_value = CallToolResult(
            content=[TextContent(type="text", text="ok")]
        )
        for request_headers, is_workflow in (
            ({"X-Caller-Token": "first-caller", "X-Region": "eu", "X-Unrelated": "unrelated-value"}, False),
            ({"X-Caller-Token": "{{request.header.X-Region}}", "X-Region": "us"}, True),
            ({}, False),
        ):
            with app.test_request_context(headers=request_headers):
                prepared = WorkflowAgentDifyToolsBuilder().build_layers(
                    tenant_id="tenant-1",
                    app_id="app-1",
                    user_id="user-1",
                    tools=tools,
                    invoke_from=InvokeFrom.SERVICE_API,
                )
            execution_context = DifyExecutionContextLayerConfig(
                tenant_id="tenant-1",
                user_id="user-1",
                user_from="end-user",
                app_id="app-1",
                agent_mode="workflow_run" if is_workflow else "agent_app",
                workflow_id="workflow-1" if is_workflow else None,
                invoke_from="service-api",
            )
            model = AgentBackendModelConfig(plugin_id="langgenius/openai", model_provider="openai", model="gpt-test")
            builder = AgentBackendRunRequestBuilder()
            if is_workflow:
                run_request = builder.build_for_workflow_node(
                    AgentBackendWorkflowNodeRunInput(
                        model=model,
                        execution_context=execution_context,
                        backend_binding_ref="binding-ref",
                        user_prompt="Search.",
                        workflow_node_job_prompt="Run the search.",
                        core_tools=prepared.core_tools,
                    )
                )
            else:
                run_request = builder.build_for_agent_app(
                    AgentBackendAgentAppRunInput(
                        model=model,
                        execution_context=execution_context,
                        backend_binding_ref="binding-ref",
                        user_prompt="Search.",
                        core_tools=prepared.core_tools,
                    )
                )
            serialized_run = run_request.model_dump_json()
            assert "first-caller" not in serialized_run
            assert "unrelated-value" not in serialized_run
            parsed_run = CreateRunRequest.model_validate_json(serialized_run)
            layers = {layer.name: layer.config for layer in parsed_run.composition.layers}
            assert "ok" in asyncio.run(
                invoke_core_tool(
                    DifyCoreToolsLayerConfig.model_validate(layers[DIFY_CORE_TOOLS_LAYER_ID]),
                    DifyExecutionContextLayerConfig.model_validate(layers[DIFY_EXECUTION_CONTEXT_LAYER_ID]),
                )
            )

    assert outbound_headers == [
        {"Authorization": "Bearer first-caller", "X-Combined": "eu / ", "X-Static": "static-value", "X-Internal": ""},
        {
            "Authorization": "Bearer {{request.header.X-Region}}",
            "X-Combined": "us / ",
            "X-Static": "static-value",
            "X-Internal": "",
        },
        {"Authorization": "Bearer ", "X-Combined": " / ", "X-Static": "static-value", "X-Internal": ""},
    ]


@pytest.mark.parametrize("message_type", ["image_link", "binary_link", "link"])
def test_media_tool_observations_expose_persisted_canonical_handles(sqlite_session: Session, message_type: str) -> None:
    _persist_app(sqlite_session)
    record = ToolFile(
        tenant_id="tenant-1",
        user_id="user-1",
        conversation_id=None,
        file_key="tools/chart.png",
        name="chart.png",
        mimetype="image/png",
        size=42,
    )
    sqlite_session.add(record)
    sqlite_session.commit()
    message = ToolInvokeMessage(
        type=ToolInvokeMessage.MessageType(message_type),
        message=ToolInvokeMessage.TextMessage(text=f"/files/tools/{record.id}.png"),
        meta={"tool_file_id": record.id},
    )
    observation = _invoke_observation(sqlite_session, [message])
    summary = json.loads(observation.split("Tool generated media file: ")[1].split(" (call")[0])
    assert summary["transfer_method"] == "tool_file"
    assert resolve_file_record_id(summary["reference"]) == record.id
    assert (summary["filename"], summary["mime_type"], summary["size"]) == ("chart.png", "image/png", 42)
    assert "read_media" in observation
    record.tenant_id = "other"
    sqlite_session.commit()
    inaccessible = _invoke_observation(sqlite_session, [message])
    assert "no longer accessible" in inaccessible
    assert "chart.png" not in inaccessible


def test_media_tool_observations_read_tool_files_committed_in_separate_transaction_under_pinned_snapshot(
    sqlite_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a pinned outer read snapshot, invoke still resolves ToolFiles committed during transformation."""
    snapshot_engine = create_engine(sqlite_engine.url)

    @event.listens_for(snapshot_engine, "connect")
    def _configure_wal_snapshot(dbapi_connection: sqlite3.Connection, _connection_record: object) -> None:
        dbapi_connection.isolation_level = None
        dbapi_connection.execute("PRAGMA journal_mode=WAL")

    @event.listens_for(snapshot_engine, "begin")
    def _begin_snapshot(conn: Connection) -> None:
        conn.exec_driver_sql("BEGIN")

    try:
        snapshot_factory = sessionmaker(bind=snapshot_engine, expire_on_commit=False)
        monkeypatch.setattr(session_factory_module, "_session_maker", snapshot_factory)
        monkeypatch.setattr("core.tools.tool_file_manager.storage.save", lambda *_args: None)
        with snapshot_factory() as setup_session:
            _persist_app(setup_session)

        blob_message = ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.BLOB,
            message=ToolInvokeMessage.BlobMessage(blob=b"png-bytes"),
            meta={"mime_type": "image/png", "filename": "chart.png"},
        )
        with (
            snapshot_factory() as outer_session,
            patch("services.agent_tool_inner_service.ToolManager.get_agent_tool_runtime", return_value=_FakeTool()),
            patch("services.agent_tool_inner_service.ToolEngine.generic_invoke", return_value=iter([blob_message])),
        ):
            response = AgentToolInnerService().invoke(_request(), session=outer_session)
            summary = json.loads(response.observation.split("Tool generated media file: ")[1].split(" (call")[0])
            record_id = resolve_file_record_id(summary["reference"])
            assert record_id is not None
            # Prove the outer session's MVCC snapshot remained pinned before the ToolFile commit.
            assert outer_session.get(ToolFile, record_id) is None

        assert summary["transfer_method"] == "tool_file"
        assert (summary["filename"], summary["mime_type"], summary["size"]) == ("chart.png", "image/png", 9)
    finally:
        snapshot_engine.dispose()


@pytest.mark.parametrize("message_type", ["image_link", "binary_link"])
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/chart.png?token=a%2Fb&part=1#original",
        "http://example.com/chart.png?token=a%2Fb&part=1#original",
    ],
)
def test_external_media_link_observations_preserve_original_urls(
    sqlite_session: Session, message_type: str, url: str
) -> None:
    """Given an external media link, invoke exposes a copyable handle with its exact URL."""
    _persist_app(sqlite_session)
    message = ToolInvokeMessage(
        type=ToolInvokeMessage.MessageType(message_type),
        message=ToolInvokeMessage.TextMessage(text=url),
    )
    observation = _invoke_observation(sqlite_session, [message])
    summary = json.loads(observation.split("Tool generated media link: ")[1].split(" (call")[0])
    assert summary == {"transfer_method": "remote_url", "url": url}
    assert "read_media" in observation


@pytest.mark.parametrize(
    ("message_type", "expected_observation"),
    [
        (
            "image_link",
            "image has been created and sent to user already, "
            "you do not need to create it, just tell the user to check it now.",
        ),
        ("binary_link", "text='file:///workspace/chart.png'"),
    ],
)
def test_non_http_media_links_preserve_existing_observations(
    sqlite_session: Session, message_type: str, expected_observation: str
) -> None:
    """Given a non-HTTP media link, invoke retains its text fallback without creating a handle."""
    _persist_app(sqlite_session)
    message = ToolInvokeMessage(
        type=ToolInvokeMessage.MessageType(message_type),
        message=ToolInvokeMessage.TextMessage(text="file:///workspace/chart.png"),
    )
    assert _invoke_observation(sqlite_session, [message]) == expected_observation


@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize(
    ("method", "reference"),
    [
        ("remote_url", None),
        ("local_file", "record"),
        ("local_file", build_file_reference(record_id="record")),
        ("tool_file", "record"),
        ("tool_file", build_file_reference(record_id="record")),
        ("datasource_file", "record"),
        ("datasource_file", build_file_reference(record_id="record")),
    ],
)
def test_file_message_observations_preserve_copyable_locators(
    sqlite_session: Session, method: str, reference: str | None, include_metadata: bool
) -> None:
    """Given a File message, invoke preserves URLs and emits canonical references with optional metadata."""
    _persist_app(sqlite_session)
    file = File(
        filename="report.pdf" if include_metadata else None,
        mime_type="application/pdf" if include_metadata else None,
        extension=".pdf",
        file_type=FileType.DOCUMENT,
        transfer_method=FileTransferMethod(method),
        size=25,
        remote_url="https://example.com/report.pdf?key=original" if method == "remote_url" else None,
        reference=reference,
    )
    message = ToolInvokeMessage(
        type=ToolInvokeMessage.MessageType.FILE,
        message=ToolInvokeMessage.FileMessage(file_marker="file"),
        meta={"file": file},
    )
    observation = _invoke_observation(sqlite_session, [message])
    summary = json.loads(observation.split(": ", 1)[1].split(" (call")[0])
    assert summary["transfer_method"] == method
    assert ("filename" in summary) is include_metadata
    assert ("mime_type" in summary) is include_metadata
    assert all(value is not None for value in summary.values())
    assert "read_media" in observation
    if method == "remote_url":
        assert summary["url"] == file.remote_url
    else:
        assert is_canonical_file_reference(summary["reference"])
        assert resolve_file_record_id(summary["reference"]) == "record"


def test_file_message_without_file_metadata_preserves_default_formatter_output(sqlite_session: Session) -> None:
    """Given a FILE message without a File object in meta, invoke leaves it to ToolEngine formatting."""
    _persist_app(sqlite_session)
    message = ToolInvokeMessage(
        type=ToolInvokeMessage.MessageType.FILE,
        message=ToolInvokeMessage.FileMessage(file_marker="file"),
        meta={},
    )
    observation = _invoke_observation(sqlite_session, [message])
    assert "read_media" not in observation


@pytest.mark.parametrize(
    ("workflow_id", "expected_tool_invoke_from"),
    [
        ("workflow-1", ToolInvokeFrom.WORKFLOW),
        (None, ToolInvokeFrom.AGENT),
    ],
)
def test_invoke_uses_agent_tool_runtime_and_returns_observation(
    sqlite_session: Session,
    workflow_id: str | None,
    expected_tool_invoke_from: ToolInvokeFrom,
) -> None:
    fake_tool = _FakeTool()
    _persist_app(sqlite_session)

    with (
        patch(
            "services.agent_tool_inner_service.ToolManager.get_agent_tool_runtime",
            return_value=fake_tool,
        ) as mock_get_runtime,
        patch("services.agent_tool_inner_service.ToolEngine.generic_invoke", return_value=_messages()) as mock_invoke,
        patch(
            "services.agent_tool_inner_service.ToolFileMessageTransformer.transform_tool_invoke_messages",
            side_effect=lambda messages, **_kwargs: messages,
        ),
    ):
        request = _request()
        request.caller.workflow_id = workflow_id
        response = AgentToolInnerService().invoke(request, session=sqlite_session)

    assert response.observation == "ok"
    assert response.metadata == {
        "provider_type": "plugin",
        "provider_id": "langgenius/search/search",
        "tool_name": "search",
    }
    agent_tool = mock_get_runtime.call_args.kwargs["agent_tool"]
    assert agent_tool.provider_type is ToolProviderType.PLUGIN
    assert agent_tool.tool_parameters == {"region": "us"}
    assert fake_tool.runtime.dify_run_context is not None
    assert fake_tool.runtime.dify_run_context.tenant_id == "tenant-1"
    assert fake_tool.runtime.dify_run_context.app_id == "app-1"
    assert fake_tool.runtime.dify_run_context.user_id == "user-1"
    assert fake_tool.runtime.dify_run_context.user_from is UserFrom.ACCOUNT
    assert fake_tool.runtime.dify_run_context.invoke_from is InvokeFrom.SERVICE_API
    assert fake_tool.runtime.dify_run_context.trace_session_id == "trace-session-1"
    assert fake_tool.runtime.tool_invoke_from is expected_tool_invoke_from
    mock_invoke.assert_called_once()


def test_invoke_raises_app_not_found_when_session_has_no_app(sqlite_session: Session) -> None:
    with pytest.raises(AgentToolInnerServiceError) as exc_info:
        AgentToolInnerService().invoke(_request(), session=sqlite_session)

    assert exc_info.value.error_code == "app_not_found"
    assert exc_info.value.status_code == 404
    assert exc_info.value.description == "App not found."


def test_invoke_raises_app_tenant_mismatch_when_app_belongs_to_other_tenant(sqlite_session: Session) -> None:
    _persist_app(sqlite_session, tenant_id="tenant-2")

    with pytest.raises(AgentToolInnerServiceError) as exc_info:
        AgentToolInnerService().invoke(_request(), session=sqlite_session)

    assert exc_info.value.error_code == "app_tenant_mismatch"
    assert exc_info.value.status_code == 403
    assert exc_info.value.description == "App does not belong to the caller tenant."


def test_invoke_maps_tool_runtime_app_not_found_value_error_to_specific_error_code(sqlite_session: Session) -> None:
    fake_tool = _FakeTool()
    _persist_app(sqlite_session)

    with (
        patch("services.agent_tool_inner_service.ToolManager.get_agent_tool_runtime", return_value=fake_tool),
        patch("services.agent_tool_inner_service.ToolEngine.generic_invoke", side_effect=ValueError("app not found")),
    ):
        with pytest.raises(AgentToolInnerServiceError) as exc_info:
            AgentToolInnerService().invoke(_request(), session=sqlite_session)

    assert exc_info.value.error_code == "app_not_found"
    assert exc_info.value.status_code == 404
    assert exc_info.value.description == "App not found."


def test_invoke_maps_tool_invoke_error_without_private_tool_engine_helper(sqlite_session: Session) -> None:
    fake_tool = _FakeTool()
    _persist_app(sqlite_session)

    with (
        patch("services.agent_tool_inner_service.ToolManager.get_agent_tool_runtime", return_value=fake_tool),
        patch(
            "services.agent_tool_inner_service.ToolEngine.generic_invoke",
            side_effect=ToolInvokeError("workflow crashed"),
        ),
    ):
        with pytest.raises(AgentToolInnerServiceError) as exc_info:
            AgentToolInnerService().invoke(_request(), session=sqlite_session)

    assert exc_info.value.error_code == "agent_tool_invoke_failed"


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ToolProviderNotFoundError("provider gone"), "agent_tool_declaration_not_found"),
        (ToolProviderCredentialValidationError("credential invalid"), "agent_tool_credential_invalid"),
        (ToolParameterValidationError("query is required"), "tool_parameters_invalid"),
    ],
)
def test_invoke_maps_runtime_lookup_errors_to_service_error_codes(
    sqlite_session: Session, error: Exception, expected_code: str
) -> None:
    _persist_app(sqlite_session)

    with patch("services.agent_tool_inner_service.ToolManager.get_agent_tool_runtime", side_effect=error):
        with pytest.raises(AgentToolInnerServiceError) as exc_info:
            AgentToolInnerService().invoke(_request(), session=sqlite_session)

    assert exc_info.value.error_code == expected_code
