#!/usr/bin/env python3
"""M3.2 / M3.3 / M3.5 — Memória residente, latência e footprint, um braço por vez.

**Por que uma passada só.** As três medições exigem a mesma condição cara: uma
coleção sozinha no servidor, em container recém-subido. Coleções vizinhas
carregadas contaminam o page cache, que é exatamente onde a residência dos
vetores aparece. Como o contrato manda derrubar as outras, cada braço precisa
ser reconstruído depois que o anterior sai — fazer memória e latência em passadas
separadas custaria cinco reconstruções a mais sem medir nada de novo.

**Estados declarados.** A residência é lida em três: frio logo após o reinício,
morno após uma carga fixa de consultas HNSW, e exaustivo após busca exata, que
toca todos os vetores e define o teto. O estado vai junto de todo número.

**A janela de paginação anula a medição.** Não o volume de swap residente — a
taxa durante a janela. Ver `memory.SwapWatch`.
"""

from __future__ import annotations

import json
import statistics
from datetime import UTC, datetime

from quantptbr import corpus, index, manifest, memory, retrieval, server, stats

OUTPUT = corpus.REPO_ROOT / "runs" / "m3_arms.json"

#: Consultas de aquecimento, idênticas em todos os braços. Define o conjunto de
#: trabalho de forma reprodutível em vez de deixá-lo depender do acaso.
WARMUP_REPEATS = 4

#: Repetições da latência, depois do aquecimento descartado. 49 × 5 dá 245
#: amostras por braço, o bastante para um p95 que não dependa de uma consulta só.
LATENCY_REPEATS = 5


def sole_collection(client, arm_id: str) -> None:
    """Garante que só a coleção deste braço existe. Verificado, não suposto."""
    wanted = retrieval.arm_config(arm_id)["collection"]
    for existing in [c.name for c in client.get_collections().collections]:
        if existing != wanted:
            print(f"    derrubando {existing}")
            client.delete_collection(existing)
    if not client.collection_exists(wanted):
        print(f"    construindo {wanted}")
        index.build(client, arm_id)
    remaining = [c.name for c in client.get_collections().collections]
    if remaining != [wanted]:
        raise RuntimeError(f"esperada só {wanted}, servidor tem {remaining}")


def latency_ms(client, arm_id: str, mode: str, repeats: int) -> list[float]:
    """Latência de cada consulta, cronometrada dentro do harness de recuperação.

    O p95 precisa de amostras por consulta: cronometrar passadas inteiras daria
    cinco números, e um p95 sobre cinco não é um p95.
    """
    samples: list[float] = []
    for _ in range(repeats):
        retrieval.retrieve(client, arm_id, mode, record_ms=samples)
    return samples


def summarise(samples: list[float]) -> dict:
    ordered = sorted(samples)
    index_p95 = min(int(0.95 * len(ordered)), len(ordered) - 1)
    return {
        "median": round(statistics.median(ordered), 2),
        "p95": round(ordered[index_p95], 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "samples": len(ordered),
    }


def measure(client, arm_id: str) -> dict:
    queries = len(retrieval.load_queries()[0])
    server.restart()
    client = server.client()
    server.require_server(client)

    with memory.SwapWatch() as swap:
        states = {"cold": memory.read().as_dict()}

        for _ in range(WARMUP_REPEATS):
            retrieval.retrieve(client, arm_id, "hnsw")
        states["warm_hnsw"] = memory.read().as_dict()

        # Latência só depois do aquecimento: a primeira passada pagina os
        # segmentos do disco e mediria I/O de carga, não custo de serviço.
        samples = latency_ms(client, arm_id, "hnsw", LATENCY_REPEATS)

        retrieval.retrieve(client, arm_id, "exact")
        states["exhaustive"] = memory.read().as_dict()

    return {
        "arm": arm_id,
        "collection": retrieval.arm_config(arm_id)["collection"],
        "states": states,
        "reported_state": "exhaustive",
        "reported_resident_mb": states["exhaustive"]["total_mb"],
        "predicted_resident_mb": retrieval.arm_config(arm_id)["predicted_resident_mb"],
        "latency_ms_per_query": summarise(samples)
        | {
            "passes": LATENCY_REPEATS,
            "queries_per_pass": queries,
            "concurrency": 1,
            "residency_state": "warm_hnsw",
        },
        "rss_mb": memory.rss_mb(),
        "swap": swap.as_dict(),
        "footprint": index.footprint(retrieval.arm_config(arm_id)["collection"]),
    }


def main() -> int:
    manifest.require_clean_tree()
    server.start()

    # O A5 compartilha a coleção do A4: medir memória dele mediria o A4 outra
    # vez. A latência, essa sim, difere — o rescoring lê os originais do disco.
    arms = [a["id"] for a in stats.matrix()["arms"]]
    buildable = [a for a in arms if not retrieval.arm_config(a).get("_shares_collection_with")]

    results = []
    for arm_id in buildable:
        print(f"\n{arm_id}")
        client = server.client()
        sole_collection(client, arm_id)
        row = measure(client, arm_id)

        # O A5 anda junto do A4: mesma coleção, mesma residência, só a busca muda.
        if arm_id == "A4":
            shared = [a for a in arms if retrieval.arm_config(a).get("_shares_collection_with")]
            for rider in shared:
                samples = latency_ms(server.client(), rider, "hnsw", LATENCY_REPEATS)
                row.setdefault("shares_residency_with", {})[rider] = summarise(samples)

        results.append(row)
        state = row["states"]["exhaustive"]
        print(
            f"    residente {row['reported_resident_mb']:8.1f} MB   "
            f"(anon {state['anon_mb']:.1f} / file {state['file_mb']:.1f})   "
            f"previsto {row['predicted_resident_mb']}   "
            f"latência mediana {row['latency_ms_per_query']['median']:.1f} ms"
        )
        if row["swap"]["is_void"]:
            print("    MEDIÇÃO NULA: o host paginou durante a janela")

    void = [r["arm"] for r in results if r["swap"]["is_void"]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "issues": ["RAF-67", "RAF-68", "RAF-70"],
                "finished_at": datetime.now(UTC).isoformat(),
                "warmup_repeats": WARMUP_REPEATS,
                "latency_repeats": LATENCY_REPEATS,
                "void_measurements": void,
                "arms": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n→ {OUTPUT}")
    if void:
        print(f"braços a remedir: {void}")
    return 1 if void else 0


if __name__ == "__main__":
    raise SystemExit(main())
