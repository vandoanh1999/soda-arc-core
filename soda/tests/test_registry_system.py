# soda/tests/test_registry_system.py

import time
import tempfile
import os
import json
import pytest
import threading
from soda.registry import (
    Registry,
    ComponentType,
    PrimitiveCategory,
    register,
    get_registry,
    reset_registry,
)


@pytest.fixture(autouse=True)
def reset_global_registry():
    reset_registry()
    yield
    reset_registry()


def test_singleton_behavior():
    r1 = Registry()
    r2 = Registry()
    assert r1 is r2


def test_register_and_get_component():
    registry = Registry()

    def dummy(x: int) -> int:
        return x + 1

    registry.register(
        name="dummy",
        component=dummy,
        component_type=ComponentType.PRIMITIVE,
        description="test primitive",
        category=PrimitiveCategory.SPATIAL,
        tags=["unit"],
    )

    comp = registry.get("dummy", ComponentType.PRIMITIVE)
    assert comp is dummy


def test_register_duplicate_without_override():
    registry = Registry()

    def fn(x):
        return x

    registry.register("a", fn, ComponentType.PRIMITIVE)

    with pytest.raises(ValueError):
        registry.register("a", fn, ComponentType.PRIMITIVE)


def test_override_registration():
    registry = Registry()

    def f1(x):
        return x

    def f2(x):
        return x * 2

    registry.register("x", f1, ComponentType.PRIMITIVE)
    registry.register("x", f2, ComponentType.PRIMITIVE, override=True)

    comp = registry.get("x", ComponentType.PRIMITIVE)
    assert comp is f2


def test_alias_resolution():
    registry = Registry()

    def fn(x):
        return x

    registry.register("base", fn, ComponentType.PRIMITIVE)
    registry.alias("alias_base", "base", ComponentType.PRIMITIVE)

    comp = registry.get("alias_base")
    assert comp is fn


def test_enable_disable_component():
    registry = Registry()

    def fn(x):
        return x

    registry.register("test", fn, ComponentType.PRIMITIVE)
    registry.disable("test", ComponentType.PRIMITIVE)

    assert registry.get("test", ComponentType.PRIMITIVE) is None

    registry.enable("test", ComponentType.PRIMITIVE)
    assert registry.get("test", ComponentType.PRIMITIVE) is fn


def test_usage_stats_update():
    registry = Registry()

    @register(ComponentType.PRIMITIVE, name="inc")
    def inc(x: int) -> int:
        return x + 1

    fn = registry.get("inc", ComponentType.PRIMITIVE)
    fn(1)
    fn(2)

    stats = registry.get_stats("inc", ComponentType.PRIMITIVE)

    assert stats["usage_count"] == 2
    assert stats["success_count"] == 2
    assert stats["failure_count"] == 0
    assert stats["success_rate"] == 1.0


def test_failure_stats_update():
    registry = Registry()

    @register(ComponentType.PRIMITIVE, name="failer")
    def failer(x: int) -> int:
        raise RuntimeError("boom")

    fn = registry.get("failer", ComponentType.PRIMITIVE)

    with pytest.raises(RuntimeError):
        fn(1)

    stats = registry.get_stats("failer", ComponentType.PRIMITIVE)

    assert stats["usage_count"] == 1
    assert stats["failure_count"] == 1
    assert stats["success_count"] == 0
    assert stats["success_rate"] == 0.0


def test_get_all_filtered():
    registry = Registry()

    def a(x):
        return x

    def b(x):
        return x

    registry.register(
        "a",
        a,
        ComponentType.PRIMITIVE,
        category=PrimitiveCategory.SPATIAL,
        tags=["fast"],
    )
    registry.register(
        "b",
        b,
        ComponentType.PRIMITIVE,
        category=PrimitiveCategory.COLOR,
        tags=["slow"],
    )

    results = registry.get_all(
        ComponentType.PRIMITIVE,
        category=PrimitiveCategory.SPATIAL,
        tags=["fast"],
    )

    assert "a" in results
    assert "b" not in results


def test_dependency_validation():
    registry = Registry()

    def dep(x):
        return x

    def main(x):
        return x

    registry.register("dep", dep, ComponentType.PRIMITIVE)
    registry.register(
        "main",
        main,
        ComponentType.PRIMITIVE,
        dependencies=["primitive::dep"],
    )

    assert registry.validate_dependencies("main", ComponentType.PRIMITIVE)


def test_dependency_validation_failure():
    registry = Registry()

    def main(x):
        return x

    registry.register(
        "main",
        main,
        ComponentType.PRIMITIVE,
        dependencies=["primitive::missing"],
    )

    assert not registry.validate_dependencies("main", ComponentType.PRIMITIVE)


def test_save_and_load_registry_state():
    registry = Registry()

    def fn(x):
        return x

    registry.register("a", fn, ComponentType.PRIMITIVE)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "registry.json")
        registry.save(path)

        assert os.path.exists(path)

        new_registry = Registry()
        new_registry.load(path)

        with open(path, "r") as f:
            data = json.load(f)

        assert "components" in data
        assert ComponentType.PRIMITIVE.value in data["components"]


def test_clear_registry():
    registry = Registry()

    def fn(x):
        return x

    registry.register("a", fn, ComponentType.PRIMITIVE)
    registry.clear(ComponentType.PRIMITIVE)

    assert registry.get("a", ComponentType.PRIMITIVE) is None


def test_thread_safe_registration():
    registry = Registry()

    def worker(i):
        def fn(x):
            return x + i

        registry.register(
            name=f"fn_{i}",
            component=fn,
            component_type=ComponentType.PRIMITIVE,
        )

    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    components = registry.list_components(ComponentType.PRIMITIVE)
    assert len(components) == 10


def test_search_functionality():
    registry = Registry()

    def fn(x):
        return x

    registry.register(
        "searchable",
        fn,
        ComponentType.PRIMITIVE,
        description="This component handles spatial transform",
        tags=["spatial", "transform"],
    )

    results = registry.search("spatial")
    assert len(results) >= 1
    assert any(name == "searchable" for _, name, _ in results)


def test_sorted_by_priority():
    registry = Registry()

    def a(x):
        return x

    def b(x):
        return x

    registry.register("a", a, ComponentType.PRIMITIVE, priority=1)
    registry.register("b", b, ComponentType.PRIMITIVE, priority=10)

    sorted_items = registry.get_sorted_by_priority(ComponentType.PRIMITIVE)

    assert sorted_items[0][0] == "b"