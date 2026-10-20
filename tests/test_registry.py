import pytest

from service.registry.registry import ModelRegistry, RegistryError


def test_register_versions_and_aliases(tmp_path, variant_factory):
    reg = ModelRegistry(tmp_path / "registry")
    v1 = reg.register("news", variant_factory(), {"accuracy": 0.9})
    v2 = reg.register("news", variant_factory(), {"accuracy": 0.92})
    assert (v1.version, v2.version) == (1, 2)
    assert (v1.path / "allm_meta.json").exists()

    # No production alias yet -> latest version.
    assert reg.resolve("news").version == 2
    reg.set_alias("news", "production", 1)
    assert reg.resolve("news").version == 1
    assert reg.resolve("news@production").version == 1
    assert reg.resolve("news@2").version == 2


def test_errors(tmp_path, variant_factory):
    reg = ModelRegistry(tmp_path / "registry")
    reg.register("news", variant_factory())
    with pytest.raises(RegistryError):
        reg.resolve("missing")
    with pytest.raises(RegistryError):
        reg.resolve("news@9")
    with pytest.raises(RegistryError):
        reg.set_alias("news", "production", 9)
    with pytest.raises(RegistryError, match="classification model"):
        reg.register("news", variant_factory(task="summarization"))


def test_paths_are_posix(tmp_path, variant_factory):
    reg = ModelRegistry(tmp_path / "registry")
    reg.register("news", variant_factory())
    assert reg.list_models()["news"]["versions"]["1"]["path"] == "models/news/v1"
