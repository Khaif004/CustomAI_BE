"""Unit tests — EntityResolver (pure, in-memory registry)."""
from app.services.planner.entity_resolver import EntityResolver
from app.services.planner.text_signals import PlanSignals
from tests.planner.fakes import FakeEntityRegistry


def resolve(message, entities, app_id="bk", aliases=None, fiori_context=None):
    reg = FakeEntityRegistry({app_id: entities}, {app_id: aliases or {}})
    resolver = EntityResolver(reg)
    signals = PlanSignals.build(message, app_id=app_id, fiori_context=fiori_context)
    return resolver.resolve(signals, app_id, fiori_context)


def test_exact_singular_match():
    r = resolve("show me all sales orders", ["SalesOrder", "Material"])
    assert r.name == "SalesOrder"
    assert r.source in ("exact", "fuzzy")


def test_compound_refinement_upgrades_to_longer_entity():
    r = resolve("show me the order items", ["Order", "OrderItems"])
    assert r.name == "OrderItems"
    assert r.source == "compound"


def test_alias_resolution():
    r = resolve("show me all po", ["PurchaseOrder"], aliases={"po": "PurchaseOrder"})
    assert r.name == "PurchaseOrder"
    assert r.source == "alias"


def test_fuzzy_typo_tolerance():
    r = resolve("list the suppliars", ["Supplier", "Material"])
    assert r.name == "Supplier"


def test_no_guess_returns_none():
    r = resolve("tell me a joke", ["SalesOrder", "Material"])
    assert r.name is None
    assert r.source is None


def test_fiori_entity_hint_wins():
    r = resolve(
        "what is its current price",
        ["Material", "SalesOrder"],
        fiori_context={"entity": "Material"},
    )
    assert r.name == "Material"
    assert r.source == "fiori"


def test_unknown_app_yields_none():
    reg = FakeEntityRegistry({"bk": ["SalesOrder"]})
    resolver = EntityResolver(reg)
    signals = PlanSignals.build("show me orders", app_id="other")
    r = resolver.resolve(signals, "other")
    assert r.name is None


def test_no_app_id_yields_none():
    reg = FakeEntityRegistry({"bk": ["SalesOrder"]})
    resolver = EntityResolver(reg)
    signals = PlanSignals.build("show me orders", app_id=None)
    r = resolver.resolve(signals, None)
    assert r.name is None
