"""Unit tests — ChatPipelineService wiring (fakes; no DB/LLM/network).

Verifies the pipeline runs planner → orchestrator → builder → renderer in order,
that the Planner receives only primitives (message + opaque fiori_context dict,
never a Fiori object), that the orchestrator receives the plan, and that the
RetrievalContext is passed straight into the builder.
"""
from app.models.conversation_context import Channel, ConversationContext
from app.models.llm_context import ContextItem, ContextStatistics, LLMContext
from app.models.planner import Intent, PlannerResult, RetrievalSource
from app.services.chat_context.pipeline import ChatPipelineService
from app.services.retrieval.models import RetrievalContext, RetrievalItem


class FakePlanner:
    def __init__(self):
        self.calls = []

    async def analyze(self, message, *, app_id=None, fiori_context=None, session=None):
        self.calls.append({"message": message, "app_id": app_id,
                           "fiori_context": fiori_context, "session": session})
        return PlannerResult(
            intent=Intent.DATA_QUERY, confidence=0.91, application=app_id,
            entity="SalesOrder", tool=None,
            retrieval_sources=[RetrievalSource.METADATA, RetrievalSource.PGVECTOR],
            requires_live_data=False, missing_parameters=[],
        )


class FakeOrchestrator:
    def __init__(self, rctx):
        self._rctx = rctx
        self.request = None

    async def retrieve(self, request):
        self.request = request
        return self._rctx


class FakeBuilder:
    def __init__(self, llm_ctx):
        self._llm = llm_ctx
        self.arg = None

    def build(self, rctx):
        self.arg = rctx
        return self._llm


def _llm_context():
    return LLMContext(
        application="bk",
        application_metadata=[ContextItem(source="Metadata", retriever="MetadataRetriever",
                                          title="SalesOrder", content="entity")],
        semantic_knowledge=[ContextItem(source="Pgvector", retriever="VectorRetriever",
                                        content="a doc")],
        statistics=ContextStatistics(token_estimate=25,
                                     retrievers_used=["MetadataRetriever", "VectorRetriever"]),
    )


async def test_pipeline_runs_all_stages_and_wires_correctly():
    rctx = RetrievalContext(application="bk",
                            metadata=[RetrievalItem(source=RetrievalSource.METADATA, ref="entity:SalesOrder")])
    planner, orch, builder = FakePlanner(), FakeOrchestrator(rctx), FakeBuilder(_llm_context())
    svc = ChatPipelineService(planner=planner, orchestrator=orch, builder=builder)

    cc = ConversationContext(message="show me orders", channel=Channel.EMBEDDED_FIORI,
                             app_id="bk", fiori_context={"serviceUrl": "/odata/v4/svc"})
    out = await svc.run(cc, session=None)

    assert out is not None
    assert out.intent == "DATA_QUERY"
    assert out.confidence == 0.91
    assert "MetadataRetriever" in out.retrievers_used
    assert out.token_estimate == 25
    # renderer ran → prepared_context is a string containing the section
    assert "APPLICATION METADATA" in out.prepared_context

    # Planner received primitives only (opaque fiori_context dict, not a Fiori object)
    assert planner.calls[0]["message"] == "show me orders"
    assert planner.calls[0]["app_id"] == "bk"
    assert planner.calls[0]["fiori_context"] == {"serviceUrl": "/odata/v4/svc"}
    # Orchestrator received a RetrievalRequest carrying the plan
    assert orch.request.plan.intent is Intent.DATA_QUERY
    assert orch.request.app_id == "bk"
    # Builder received the orchestrator's RetrievalContext directly
    assert builder.arg is rctx


async def test_pipeline_global_chat_no_app_context():
    rctx = RetrievalContext(application=None)
    planner, orch, builder = FakePlanner(), FakeOrchestrator(rctx), FakeBuilder(LLMContext())
    svc = ChatPipelineService(planner=planner, orchestrator=orch, builder=builder)

    cc = ConversationContext(message="what is machine learning", channel=Channel.GLOBAL)
    out = await svc.run(cc, session=None)

    assert out is not None
    assert planner.calls[0]["app_id"] is None
    assert planner.calls[0]["fiori_context"] is None
    # empty LLMContext renders to "" (pipeline ran, nothing retrieved)
    assert out.prepared_context == ""
