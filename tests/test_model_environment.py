"""Offline checks for architecture-specific model locks; never import Torch."""

import tomllib
from pathlib import Path
from urllib.parse import unquote

import pytest
from packaging.markers import Marker

SKILL = Path(__file__).resolve().parents[1] / "skills/sam3-segmentation"
MODEL = SKILL / "environments/sam3"


def load_toml(path):
    return tomllib.loads(path.read_text())


def matches(marker, machine, system="linux"):
    return Marker(marker).evaluate(
        {"platform_machine": machine, "sys_platform": system, "python_version": "3.12"}
    )


@pytest.mark.parametrize(
    "machine,system,supported",
    [
        ("x86_64", "linux", True),
        ("aarch64", "linux", True),
        ("ppc64le", "linux", False),
        ("x86_64", "win32", False),
        ("aarch64", "darwin", False),
        ("x86_64", "darwin", False),
    ],
)
def test_model_platform_scope(machine, system, supported):
    config = load_toml(MODEL / "pyproject.toml")["tool"]["uv"]
    lock = load_toml(MODEL / "uv.lock")
    for markers in [
        config["environments"],
        config["required-environments"],
        lock["supported-markers"],
        lock["required-markers"],
    ]:
        assert sum(matches(m, machine, system) for m in markers) == int(supported)


@pytest.mark.parametrize("machine,cuda", [("x86_64", "cu128"), ("aarch64", "cu130")])
@pytest.mark.parametrize("name,version", [("torch", "2.10.0"), ("torchvision", "0.25.0")])
def test_model_sources_and_lock_select_matching_cuda_wheels(machine, cuda, name, version):
    config = load_toml(MODEL / "pyproject.toml")
    uv = config["tool"]["uv"]
    assert f"{name}=={version}" in config["project"]["dependencies"]
    sources = [s for s in uv["sources"][name] if matches(s["marker"], machine)]
    assert len(sources) == 1
    index = next(i for i in uv["index"] if i["name"] == sources[0]["index"])
    assert index["explicit"] is True
    assert index["url"] == f"https://download.pytorch.org/whl/{cuda}"

    packages = load_toml(MODEL / "uv.lock")["package"]
    project = next(p for p in packages if p["name"] == "sam3-segmentation-model")
    deps = [
        d for d in project["dependencies"] if d["name"] == name and matches(d["marker"], machine)
    ]
    assert len(deps) == 1
    dep = deps[0]
    assert dep["version"] == f"{version}+{cuda}"
    assert dep["source"] == {"registry": index["url"]}
    package = next(
        p
        for p in packages
        if (p["name"], p["version"], p["source"]) == (name, dep["version"], dep["source"])
    )
    assert any(
        f"{name}-{version}+{cuda}-cp312-cp312-manylinux_2_28_{machine}.whl" in unquote(wheel["url"])
        and wheel["hash"].startswith("sha256:")
        for wheel in package["wheels"]
    )
    if name == "torchvision":
        torch_dep = next(d for d in package["dependencies"] if d["name"] == "torch")
        assert torch_dep["version"] == f"2.10.0+{cuda}"
        assert torch_dep["source"] == dep["source"]
    else:
        major = "13" if cuda == "cu130" else "12"
        assert any(d["name"] == f"nvidia-cudnn-cu{major}" for d in package["dependencies"])
        triton = next(p for p in packages if p["name"] == "triton")
        assert any(
            "-cp312-cp312-" in w["url"] and w["url"].endswith(f"_{machine}.whl")
            for w in triton["wheels"]
        )
