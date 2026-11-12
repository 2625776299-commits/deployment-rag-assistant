"""HTTP adapter; model/retrieval dependencies must be configured explicitly."""

from collections.abc import Callable
import logging
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.rag_service import run_rag_service
from app.retrieval_contract import RetrievalRequest, SourceScope
from app.build_runtime import PinnedRag
from app.provider_errors import provider_public_reply


logger = logging.getLogger(__name__)
Question = Annotated[str, StringConstraints(
    strict=True, strip_whitespace=True, min_length=1, max_length=4000,
)]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Question


class SourceInfo(BaseModel):
    chunk_id: str
    document_id: str
    section_id: str
    source_url: str


class AskResponse(BaseModel):
    request_id: str
    status: Literal["answer", "needs_user", "insufficient", "stopped", "failed"]
    message: str
    required_information: list[str] = Field(default_factory=list)
    sources: list[SourceInfo] = Field(default_factory=list)


def to_public_response(result: dict, request_id: str) -> tuple[int, AskResponse]:
    """Select public fields; internal traces and raw model data stay internal."""
    if result["status"] == "failed":
        error = result.get("error") or {}
        provider_reply = provider_public_reply(error)
        if provider_reply is not None:
            return 503, AskResponse(request_id=request_id, status='failed', message=provider_reply['message'])
        # Only this existing failure code positively identifies a timeout.
        # Other service failures currently lack typed transient/internal causes.
        timed_out = error.get("stage") == "evaluate" and error.get("code") == "evaluation_timeout"
        return (503 if timed_out else 500), AskResponse(
            request_id=request_id, status="failed",
            message=("评价服务暂时未能完成处理，请稍后重试。" if timed_out
                     else "本轮处理未能完成，请稍后重试；如持续失败，请提供请求编号。"),
        )

    # Sources belong to an actually cited response, not every retrieved candidate.
    answer = result.get("answer") or {}
    return 200, AskResponse(
        request_id=request_id,
        status=result["status"],
        message=result["message"],
        required_information=result["required_information"],
        sources=answer.get("sources", []),
    )


def create_app(
    *,
    retrieve: Callable[[str], list[dict]] | None = None,
    evaluate: Callable[[str, list[dict]], dict] | None = None,
    generate: Callable[[str, list[dict]], dict] | None = None,
    request_retrieval: Callable[[RetrievalRequest], list[dict]] | None = None,
    initial_scope: SourceScope | None = None, scope_policy: Callable | None = None,
    bind_request: Callable[[], PinnedRag] | None = None,
    plan_queries: Callable | None = None,
    allocation_policy: str = 'unified_rrf',
    conversation_service=None, authenticate=None,
) -> FastAPI:
    dependencies = (retrieve, evaluate, generate)
    configured = all(callable(fn) for fn in dependencies)
    if any(fn is not None for fn in dependencies) and not configured:
        raise ValueError("Configure all three RAG callbacks, or leave all unconfigured")
    if bind_request is not None:
        if not callable(bind_request) or any(fn is not None for fn in (*dependencies, request_retrieval)):
            raise ValueError('Request binding must be callable and cannot mix with static callbacks')
        configured = True
    if request_retrieval is not None and (not configured or not callable(request_retrieval)):
        raise ValueError('Scoped retrieval requires configured RAG callbacks')
    if request_retrieval is None and bind_request is None and (initial_scope is not None or scope_policy is not None):
        raise ValueError('Scope requires a request-based retriever')
    if initial_scope is not None and not isinstance(initial_scope, SourceScope):
        raise ValueError('Invalid initial scope')
    if scope_policy is not None and not callable(scope_policy):
        raise ValueError('Invalid scope policy')
    if plan_queries is not None and not callable(plan_queries):
        raise ValueError('Invalid query planner')
    if allocation_policy not in ('unified_rrf','reserved_then_rrf'):
        raise ValueError('Invalid allocation policy')
    app = FastAPI(title="Deployment Troubleshooting RAG")
    if (conversation_service is None) != (authenticate is None):
        raise ValueError('Configure conversation service and authentication together')
    if conversation_service is not None:
        from app.conversation_api import install_conversation_routes
        install_conversation_routes(app, conversation_service, authenticate)

    @app.get("/health")
    def health() -> dict[str, str]:
        # Liveness only; it does not imply model/index readiness.
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse, responses={
        500: {"model": AskResponse}, 503: {"model": AskResponse},
    })
    def ask(request: AskRequest, response: Response) -> AskResponse:
        request_id = uuid4().hex
        response.headers["X-Request-ID"] = request_id
        if not configured:
            response.status_code = 503
            logger.warning("rag_unconfigured request_id=%s", request_id)
            return AskResponse(request_id=request_id, status="failed",
                               message="问答服务尚未配置完成，暂时无法处理问题。")
        try:
            # Resolve once. Publishing another build cannot replace these local references.
            bound = bind_request() if bind_request is not None else None
            if bound is not None and not isinstance(bound, PinnedRag):
                raise TypeError('request_binder_must_return_pinned_rag')
            if bind_request is not None and bound is None:
                raise TypeError('request_binder_returned_none')
            result = run_rag_service(
                request.question,
                retrieve=bound.retrieve if bound is not None else retrieve,
                evaluate=bound.evaluate if bound is not None else evaluate,
                generate=bound.generate if bound is not None else generate,
                request_retrieval=bound.search if bound is not None else request_retrieval,
                initial_scope=initial_scope, scope_policy=scope_policy,
                build_id=bound.build.build_id if bound is not None else None,
                plan_queries=plan_queries,allocation_policy=allocation_policy,
            )
            status_code, public = to_public_response(result, request_id)
            logger.info("rag_result request_id=%s status=%s stop_reason=%s counts=%s",
                        request_id, result["status"], result.get("stop_reason"), result.get("counts"))
            if bound is not None:
                logger.info('rag_build request_id=%s build_id=%s', request_id, bound.build.build_id)
        except Exception:
            # Do not turn unexpected exceptions into evidence/user-information gaps.
            logger.exception("rag_unexpected_failure request_id=%s", request_id)
            status_code = 500
            public = AskResponse(request_id=request_id, status="failed",
                                 message="服务处理异常，请提供请求编号以便排查。")
        response.status_code = status_code
        return public

    return app


# No implicit clients, provider requests, or index builds when importing the app.
app = create_app()
