"""Store shared published graphs while keeping historical run graphs readable."""

import json

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from enums import WorkflowKind
from models.workflow import Workflow, WorkflowRun, WorkflowType


def reference_published_graph(session: Session, run: WorkflowRun) -> None:
    """Replace an inline graph only if its exact published version has the same JSON.

    The caller persists the run in this transaction. A shared lock keeps the
    source version available until the reference is committed or rolled back.
    """
    if not run.graph or not run.app_id or run.version == Workflow.VERSION_DRAFT:
        return
    if run.type not in (WorkflowType.WORKFLOW, WorkflowType.CHAT):
        return

    graph = session.scalar(
        select(Workflow.graph)
        .where(
            Workflow.id == run.workflow_id,
            Workflow.tenant_id == run.tenant_id,
            Workflow.app_id == run.app_id,
            Workflow.version == run.version,
            Workflow.type == run.type,
            Workflow.kind.is_distinct_from(WorkflowKind.SNIPPET),
        )
        .with_for_update(read=True)
    )
    if graph is None:
        return
    try:
        same_graph = json.dumps(json.loads(graph), sort_keys=True) == json.dumps(json.loads(run.graph), sort_keys=True)
    except ValueError:
        # A missing or invalid canonical payload cannot replace a run snapshot.
        return
    if same_graph:
        run.graph = WorkflowRun.GRAPH_FROM_WORKFLOW


def preserve_referenced_run_graphs(session: Session, workflow: Workflow) -> None:
    """Copy the canonical graph into dependent runs before deleting a locked version.

    The caller holds the workflow row lock and deletes the workflow in the same
    transaction. Rollback leaves both the source and its references intact.
    """
    if (
        workflow.version != Workflow.VERSION_DRAFT
        and workflow.type in (WorkflowType.WORKFLOW, WorkflowType.CHAT)
        and workflow.kind != WorkflowKind.SNIPPET
    ):
        session.execute(
            update(WorkflowRun)
            .where(
                WorkflowRun.workflow_id == workflow.id,
                WorkflowRun.tenant_id == workflow.tenant_id,
                WorkflowRun.app_id == workflow.app_id,
                WorkflowRun.version == workflow.version,
                WorkflowRun.type == workflow.type,
                WorkflowRun.graph == WorkflowRun.GRAPH_FROM_WORKFLOW,
            )
            .values(graph=workflow.graph)
        )
