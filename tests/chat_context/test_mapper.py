"""Unit tests — ChatRequest → ConversationContext mapper (pure)."""
from app.models.chat import ChatRequest
from app.models.conversation_context import Channel
from app.services.chat_context.mapper import chat_request_to_conversation_context


def test_global_chat_maps_to_global_channel():
    cc = chat_request_to_conversation_context(ChatRequest(message="hi"), user_id="u", session_id="s")
    assert cc.channel is Channel.GLOBAL
    assert cc.app_id is None
    assert cc.fiori_context is None
    assert cc.user_id == "u" and cc.session_id == "s"


def test_app_id_only_is_embedded():
    cc = chat_request_to_conversation_context(ChatRequest(message="x", app_id="bk"),
                                              user_id=None, session_id=None)
    assert cc.channel is Channel.EMBEDDED_FIORI
    assert cc.app_id == "bk"


def test_fiori_context_extracts_ui_fields_both_casings():
    req = ChatRequest(
        message="show orders",
        app_id="bk",
        fiori_context={
            "current_view": "ListReport",
            "entityData": {"ID": 1},          # camelCase record
            "currentEntity": "SalesOrder",
            "uiContext": {"selected": 3},
        },
    )
    cc = chat_request_to_conversation_context(req, user_id="u", session_id="s")
    assert cc.channel is Channel.EMBEDDED_FIORI
    assert cc.current_view == "ListReport"
    assert cc.current_record == {"ID": 1}
    assert cc.current_entity == "SalesOrder"
    assert cc.ui_context == {"selected": 3}
    # fiori_context is preserved as an OPAQUE dict (passed straight to planner/retrieval)
    assert cc.fiori_context == req.fiori_context


def test_conversation_history_is_carried():
    req = ChatRequest(
        message="x", app_id="bk",
        conversation_history=[{"role": "user", "content": "prev"}],
    )
    cc = chat_request_to_conversation_context(req, user_id=None, session_id=None)
    assert cc.conversation_history == [{"role": "user", "content": "prev"}]


def test_odd_payload_types_are_ignored_not_fatal():
    # A non-dict entity_data / non-str current_view must NOT break construction —
    # the pipeline should survive odd payloads rather than fall back to legacy.
    req = ChatRequest(
        message="x", app_id="bk",
        fiori_context={"entityData": ["not", "a", "dict"], "currentView": {"weird": 1}},
    )
    cc = chat_request_to_conversation_context(req, user_id=None, session_id=None)
    assert cc.current_record is None     # non-dict ignored
    assert cc.current_view is None       # non-str ignored
    assert cc.fiori_context == req.fiori_context  # raw dict still preserved for retrieval


def test_fiori_context_present_without_app_id_is_embedded():
    cc = chat_request_to_conversation_context(
        ChatRequest(message="x", fiori_context={"serviceUrl": "/odata/v4/svc"}),
        user_id=None, session_id=None,
    )
    assert cc.channel is Channel.EMBEDDED_FIORI
