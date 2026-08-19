"""Contrato de leitura de memória residente (S1.1).

A métrica principal do artigo. Um contador que não responde é um instrumento
quebrado, e a coerência interna entre braços não revelaria isso — todos os seis
reportariam o mesmo número errado.
"""

from __future__ import annotations

import json

import pytest

from quantptbr import corpus, memory, server, stats

SMOKE_PATH = corpus.REPO_ROOT / "runs" / "s1.1_memory_smoke.json"


def test_analytical_estimator_reproduces_the_preregistered_predictions():
    """A previsão de cada braço na matriz não é digitada: sai deste estimador."""
    from quantptbr import embedding

    dims = embedding.Envelope.load().dimensions
    for arm in stats.matrix()["arms"]:
        predicted = memory.analytical_mb(corpus.EXPECTED_PASSAGES, dims, arm["quantization"])
        assert predicted == pytest.approx(arm["predicted_resident_mb"], rel=0.01), arm["id"]


def test_estimator_refuses_an_unknown_scheme():
    with pytest.raises(ValueError, match="desconhecida"):
        memory.analytical_mb(1000, 768, {"kind": "trinary"})


def test_void_flag_needs_both_swap_and_memory_pressure():
    def snapshot(available: float, swap: float) -> memory.MemorySnapshot:
        return memory.MemorySnapshot(0, 0, 0, available, swap, "", "")

    assert snapshot(available=500, swap=1000).is_void
    assert not snapshot(available=8000, swap=1000).is_void, "swap parado não invalida nada"
    assert not snapshot(available=500, swap=0).is_void


@pytest.fixture()
def running_server():
    if not server.is_running():
        pytest.skip("servidor parado; rode `python scripts/qdrant_server.py start`")


def test_cgroup_is_the_leaf_holding_the_qdrant_process(running_server):
    """Ler o scope agregaria o que mais estiver pendurado nele."""
    leaf = memory.cgroup_path()
    assert leaf.name == "container"
    assert (leaf / "memory.stat").exists()
    assert (leaf / "cgroup.procs").read_text().split(), "cgroup folha sem processo algum"


def test_reading_memory_is_a_single_call(running_server):
    snapshot = memory.read()
    assert snapshot.anon_mb > 0
    assert snapshot.total_mb >= snapshot.anon_mb
    assert snapshot.host_available_mb > 0
    assert "container" in snapshot.cgroup
    assert set(snapshot.as_dict()) >= {
        "anon_mb",
        "file_mb",
        "total_mb",
        "host_available_mb",
        "host_swap_used_mb",
        "is_void",
    }


def test_cgroup_and_podman_stats_agree_within_unit_conversion(running_server):
    """Divergência é explicada, nunca reconciliada por média."""
    check = memory.cross_check()
    assert abs(check["difference_mb"]) < 0.05 * check["cgroup_total_mb"]


def test_rss_overstates_residency(running_server):
    """A razão de o contrato rejeitar RSS, medida em vez de argumentada."""
    assert memory.rss_mb() > memory.read().anon_mb


@pytest.mark.skipif(not SMOKE_PATH.exists(), reason="teste de fumaça não executado")
def test_instrument_validation_passes_every_check():
    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    failed = [name for name, ok in smoke["checks"].items() if not ok]
    assert not failed, f"instrumento não validado: {failed}"


@pytest.mark.skipif(not SMOKE_PATH.exists(), reason="teste de fumaça não executado")
def test_verdict_is_reproducible_from_the_recorded_states():
    """A regra do veredito é função pura dos estados medidos, não do momento da execução."""
    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    checks, _ = memory.smoke_verdict(smoke["cases"], smoke["float32_payload_mb"])
    assert checks == smoke["checks"]


@pytest.mark.skipif(not SMOKE_PATH.exists(), reason="teste de fumaça não executado")
def test_page_cache_resolves_the_vector_payload():
    """O achado que derruba `anon` como métrica principal.

    Os vetores do Qdrant são mapeados em disco em toda configuração, então a
    residência deles aparece em `file`, não em `anon`. Medido: o page cache do
    container sob busca exaustiva bate com a carga de vetores.
    """
    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8"))
    float32_ram = smoke["cases"][0]["states"]["exhaustive"]
    payload = smoke["float32_payload_mb"]
    assert float32_ram["file_mb"] == pytest.approx(payload, rel=0.2)
    assert float32_ram["anon_mb"] < 0.5 * payload, (
        "os vetores apareceram em anon; a premissa original do contrato voltaria a valer"
    )


@pytest.mark.skipif(not SMOKE_PATH.exists(), reason="teste de fumaça não executado")
def test_quantized_vectors_are_the_only_thing_in_anonymous_memory():
    """`always_ram=True` é o único caminho que coloca vetores em memória anônima.

    Distinção que o contrato precisa: o quantizado é inevictável e tem de ser
    provisionado; o original é page cache e pode ser descartado sob pressão, ao
    custo de latência. Reportar um número só apagaria essa diferença.
    """
    cases = json.loads(SMOKE_PATH.read_text())["cases"]
    payload = json.loads(SMOKE_PATH.read_text())["float32_payload_mb"]
    float32_anon = cases[0]["states"]["exhaustive"]["anon_mb"]
    int8_anon = cases[2]["states"]["exhaustive"]["anon_mb"]
    assert int8_anon - float32_anon > 0.25 * payload


@pytest.mark.skipif(not SMOKE_PATH.exists(), reason="teste de fumaça não executado")
def test_residency_is_reported_per_declared_state():
    """Residência não é um número: depende do que já foi consultado.

    Medir logo após a escrita em massa mede o rastro do build. O artigo reporta
    o estado exaustivo, que toca todos os vetores e define o teto.
    """
    for case in json.loads(SMOKE_PATH.read_text())["cases"]:
        states = case["states"]
        assert set(states) == {"cold", "warm_hnsw", "exhaustive"}, case["case"]
        assert states["exhaustive"]["total_mb"] >= states["cold"]["total_mb"], case["case"]
