#!/usr/bin/env python3
"""M4.4 — Regenera toda tabela do artigo a partir das evidências gravadas.

**Ponto de entrada único.** Quem escrever o artigo não lê `runs/*.json` a mão
nem digita número nenhum: roda isto e copia. Todo valor sai de um arquivo de
evidência produzido por um script de medição, e a procedência de cada um está
no bloco final.

**Não mede nada.** Se um arquivo de evidência falta, a tabela correspondente
sai marcada como ausente em vez de ser preenchida com estimativa — uma lacuna
declarada é recuperável, um número inventado não.

Saídas em `paper/`: `tables.md` para colar, `numbers.json` para conferir.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from quantptbr import corpus, manifest, stats

RUNS = corpus.REPO_ROOT / "runs"
OUTPUT_DIR = corpus.REPO_ROOT / "paper"

#: Estado de residência que o artigo reporta. O exaustivo continua gravado, mas
#: o `exact` do Qdrant varre os originais e ignora os códigos, então ele mede o
#: caminho dos originais em todo braço em vez do teto do braço quantizado.
REPORTED_STATE = "warm_hnsw"

MISSING = "—"


def load(name: str) -> dict | None:
    path = RUNS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def arms() -> list[dict]:
    return stats.matrix()["arms"]


def table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown puro: o formato final é do LaTeX de quem escrever, não nosso."""
    line = "| " + " | ".join(header) + " |"
    rule = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([line, rule, *body])


def quality_rows(numbers: dict) -> list[list[str]]:
    """T1 — qualidade por braço, com o intervalo que calibra o verbo da frase."""
    bootstrap = load("m4_bootstrap.json")
    by_arm = {}
    if bootstrap:
        for row in bootstrap["comparisons"]:
            if row["mode"] == "hnsw" and row["metric"] == "nDCG@10":
                by_arm[row["arm"]] = row

    rows = []
    for arm in arms():
        evaluation = load(f"eval_{arm['id'].lower()}_hnsw.json")
        if not evaluation:
            rows.append([arm["id"], arm["label"], MISSING, MISSING, MISSING, MISSING])
            continue
        aggregate = evaluation["evaluation"]["aggregate"]
        comparison = by_arm.get(arm["id"])
        if comparison:
            interval = f"[{comparison['ci_low']:+.4f}, {comparison['ci_high']:+.4f}]"
            delta = f"{comparison['mean_difference']:+.4f}"
            verdict = comparison["verdict"]
        else:
            interval, delta = "", MISSING
            verdict = "linha de base"
        rows.append(
            [
                arm["id"],
                arm["label"],
                f"{aggregate['nDCG@10']:.4f}",
                f"{aggregate['R(rel=2)@100']:.4f}",
                f"{delta} {interval}".strip(),
                verdict,
            ]
        )
        numbers["quality"][arm["id"]] = {
            "ndcg10": aggregate["nDCG@10"],
            "recall100": aggregate["R(rel=2)@100"],
            "delta_vs_a0": comparison["mean_difference"] if comparison else None,
            "ci": [comparison["ci_low"], comparison["ci_high"]] if comparison else None,
            "verdict": verdict,
            "minimum_detectable_difference": (
                comparison["minimum_detectable_difference"] if comparison else None
            ),
        }
    return rows


def cost_rows(numbers: dict) -> tuple[list[list[str]], list[str]]:
    """T2 — a tabela que carrega a tese central do artigo.

    Comprimir reduz a memória que é preciso provisionar **e aumenta os bytes que
    é preciso armazenar**: o Qdrant guarda os originais float32 ao lado dos
    códigos. Por isso cada braço carrega as duas razões, e a nominal — a que a
    documentação e os blogs citam — fica ao lado para o contraste ser visível.
    """
    measured = load("m3_arms.json")
    if not measured:
        return [], ["`runs/m3_arms.json` ausente: rode `scripts/10_measure_arms.py`."]

    by_arm = {row["arm"]: row for row in measured["arms"]}
    baseline = by_arm.get("A0")
    notes: list[str] = []
    rows = []
    for arm in arms():
        row = by_arm.get(arm["id"])
        if not row:
            rows.append([arm["id"], arm["label"], MISSING, MISSING, MISSING, MISSING, MISSING])
            notes.append(f"{arm['id']}: não medido nesta execução.")
            continue
        state = row["states"][REPORTED_STATE]
        disk = row["footprint"]["total_mb"]
        floor = state["anon_mb"]
        storage = f"{baseline['footprint']['total_mb'] / disk:.2f}x" if baseline else MISSING
        residency = (
            f"{baseline['states'][REPORTED_STATE]['anon_mb'] / floor:.2f}x"
            if baseline and floor
            else MISSING
        )
        rows.append(
            [
                arm["id"],
                arm["label"],
                f"{arm['nominal_compression']}x",
                f"{disk:,.1f}",
                storage,
                f"{floor:,.1f}",
                residency,
            ]
        )
        numbers["cost"][arm["id"]] = {
            "nominal_compression": arm["nominal_compression"],
            "disk_mb": disk,
            "provisioning_floor_mb": floor,
            "resident_total_mb": row["reported_resident_mb"],
            "page_cache_mb": state["file_mb"],
            "residency_state": REPORTED_STATE,
            "unindexed_points": (row.get("build") or {}).get("unindexed_points"),
        }
        residual = (row.get("build") or {}).get("unindexed_points") or 0
        if residual:
            notes.append(
                f"{arm['id']}: {residual:,} pontos num segmento residual fora do HNSW, "
                "buscados por varredura exata."
            )
    if measured.get("void_measurements"):
        notes.append(
            "Medições anuladas por paginação do container: "
            f"{', '.join(measured['void_measurements'])}."
        )
    return rows, notes


def latency_rows(numbers: dict) -> list[list[str]]:
    """T3 — latência. Cache quente e consultas repetidas: não é cold start."""
    measured = load("m3_arms.json")
    if not measured:
        return []
    by_arm = {row["arm"]: row for row in measured["arms"]}
    rows = []
    for arm in arms():
        row = by_arm.get(arm["id"])
        if not row:
            rows.append([arm["id"], arm["label"], MISSING, MISSING, MISSING])
            continue
        latency = row["latency_ms_per_query"]
        rows.append(
            [
                arm["id"],
                arm["label"],
                f"{latency['median']:.1f}",
                f"{latency['p95']:.1f}",
                f"{latency['samples']}",
            ]
        )
        numbers["latency"][arm["id"]] = latency
    return rows


def decomposition_rows(numbers: dict) -> list[list[str]]:
    """T4 — de onde vem a perda: dos códigos, do grafo, ou recuperada no rescoring."""
    decomposition = load("m4_decomposition.json")
    if not decomposition:
        return []
    rows = []
    for arm_id, values in decomposition["arms"].items():
        rows.append(
            [
                arm_id,
                f"{values['quantization_error']:+.4f}",
                f"{values['graph_error_at_preregistered_ef']:+.4f}",
                f"{values['rescoring_recovery']:+.4f}",
                f"{values['reported_ndcg10']:.4f}",
            ]
        )
        numbers["decomposition"][arm_id] = values
    return rows


def geometry_lines(numbers: dict) -> list[str]:
    """T5 — por que o braço binário colapsa, e não só que ele colapsa."""
    geometry = load("m2.8_geometry.json")
    if not geometry:
        return []
    numbers["geometry"] = {
        key: geometry[key]
        for key in (
            "dimensions",
            "sample_size",
            "degeneracy_threshold",
            "mean_of_dimension_means",
            "mean_abs_offset_over_std",
            "degenerate_dimensions",
            "effective_bits",
        )
    }
    fraction = geometry["effective_bits"] / geometry["dimensions"]
    return [
        (
            f"- Média das médias por dimensão: **{geometry['mean_of_dimension_means']:+.5f}** "
            "— o espaço *parece* centrado quando olhado no agregado."
        ),
        (
            "- Deslocamento por dimensão, |média|/desvio: "
            f"**{geometry['mean_abs_offset_over_std']:.3f}** "
            "— olhado dimensão a dimensão, não está."
        ),
        (
            f"- Dimensões degeneradas em sinal (limiar {geometry['degeneracy_threshold']}, "
            f"declarado antes de calcular): **{geometry['degenerate_dimensions']} de "
            f"{geometry['dimensions']}**."
        ),
        (
            f"- Orçamento efetivo do código binário: **{geometry['effective_bits']:.1f} bits de "
            f"{geometry['dimensions']}** ({fraction:.1%})."
        ),
    ]


def calibration_lines(numbers: dict) -> list[str]:
    """T6 — a validação do aparelho, que é o que autoriza todo o resto.

    Sem ela o estudo compara seis braços entre si e não sabe se algum deles está
    perto do que a literatura obteve. Reproduz o protocolo dos autores do Quati:
    run de 1M pontuado contra os qrels de 10M.
    """
    path = RUNS / "retrieval" / "A0_exact.jsonl"
    if not path.exists():
        return []
    from collections import defaultdict

    from quantptbr import evaluation

    run: dict[str, dict[str, float]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        hit = json.loads(line)
        run[hit["query_id"]][hit["passage_id"]] = hit["score"]
    report = evaluation.calibration_report(dict(run))
    numbers["calibration"] = report
    band = report["range_check"]
    return [
        (
            f"- Alvo publicado (Bueno et al., Quati, Tabela 6, E5-base): "
            f"**{report['published']:.4f}**"
        ),
        (
            f"- Medido aqui, mesmo protocolo: **{report['preregistered_over_49']:.4f}** "
            f"(Δ {report['delta_49']:+.4f})"
        ),
        f"- Veredito: **{report['verdict']}**",
        (
            f"- Checagem de faixa: {band['floor']['system']} {band['floor']['ndcg10']:.4f} "
            f"< A0 {report['study_qrels_ndcg10']:.4f} "
            f"< {band['nearest_above']['system']} {band['nearest_above']['ndcg10']:.4f} "
            f"— dentro: **{band['inside']}**"
        ),
    ]


def provenance(numbers: dict) -> list[str]:
    """O bloco que torna a tabela auditável em vez de apenas legível."""
    reference = load("eval_a0_hnsw.json")
    lines = []
    if reference:
        controlled = reference["controlled"]
        numbers["provenance"] = {
            "corpus_sha256": controlled["corpus_sha256"],
            "qrels_sha256": controlled["qrels_sha256"],
            "envelope": controlled["envelope"],
            "server_version": controlled["server_version"],
            "image_digest": controlled["image_digest"],
            "hnsw": controlled["hnsw"],
            "distance": controlled["distance"],
            "measurement_tools": controlled["measurement_tools"],
            "judged_queries": len(controlled["judged_query_ids"]),
            "host": reference["host"],
        }
        lines += [
            (
                f"- Corpus `sha256:{controlled['corpus_sha256'][:16]}…`, "
                f"qrels `sha256:{controlled['qrels_sha256'][:16]}…`"
            ),
            (
                f"- Modelo `{controlled['envelope']['model_id']}` "
                f"@ `{controlled['envelope']['model_revision'][:12]}`, "
                f"{controlled['envelope']['dimensions']}d"
            ),
            f"- Qdrant {controlled['server_version']}, imagem `{controlled['image_digest'][:23]}…`",
            (
                f"- HNSW m={controlled['hnsw']['m']}, "
                f"ef_construct={controlled['hnsw']['ef_construct']}, "
                f"ef_search={controlled['hnsw']['ef_search']}, {controlled['distance']}"
            ),
            (
                f"- Consultas julgadas: {len(controlled['judged_query_ids'])} "
                f"(pontuáveis: {reference['evaluation']['queries']})"
            ),
            f"- Kernel `{reference['host']['kernel']}`, Python {reference['host']['python']}",
        ]
    commit = manifest.git_commit()
    numbers["regenerated_at"] = datetime.now(UTC).isoformat()
    numbers["git_commit"] = commit
    lines.append(f"- Regenerado em {numbers['regenerated_at']} no commit `{commit[:12]}`")
    return lines


def main() -> int:
    numbers: dict = {
        "quality": {},
        "cost": {},
        "latency": {},
        "decomposition": {},
    }

    cost, cost_notes = cost_rows(numbers)
    sections = [
        "# Tabelas do artigo — regeneradas, não digitadas",
        "",
        "Gerado por `scripts/11_regenerate.py`. Todo número vem de um arquivo em `runs/`.",
        "Nenhum valor aqui foi transcrito à mão; um campo `—` significa evidência ausente,",
        "não estimativa.",
        "",
        "## T1 — Qualidade por braço (busca HNSW, 49 consultas pontuáveis)",
        "",
        table(
            ["Braço", "Configuração", "nDCG@10", "R(rel≥2)@100", "Δ vs A0 (IC 95%)", "Veredito"],
            quality_rows(numbers),
        ),
        "",
        "> Um intervalo que contém zero **não** é equivalência: é ausência de evidência de",
        "> diferença neste tamanho de amostra. A diferença mínima detectável por braço está",
        "> em `numbers.json`.",
        "",
        "## T2 — Compressão: armazenamento contra residência",
        "",
        "A distinção central do artigo. O Qdrant guarda os vetores float32 originais ao",
        "lado dos códigos quantizados, então comprimir **reduz a memória a provisionar e",
        "aumenta os bytes a armazenar**. A razão nominal é a que a documentação e os posts",
        "de prática citam; as outras duas são as medidas.",
        "",
        table(
            [
                "Braço",
                "Configuração",
                "Compressão nominal",
                "Disco (MB)",
                "Razão de armazenamento",
                "Piso de provisionamento (MB)",
                "Razão de residência",
            ],
            cost,
        ),
        "",
        "> Piso de provisionamento = memória anônima do cgroup do container no estado",
        f"> `{REPORTED_STATE}`: é o que precisa existir na máquina. O page cache dos originais",
        "> é evictável e aparece em `numbers.json` como `page_cache_mb`.",
    ]
    if cost_notes:
        sections += ["", *[f"> {note}" for note in cost_notes]]

    latency = latency_rows(numbers)
    if latency:
        sections += [
            "",
            "## T3 — Latência por consulta (ms)",
            "",
            table(["Braço", "Configuração", "Mediana", "p95", "Amostras"], latency),
            "",
            "> Cache quente, consultas repetidas, concorrência 1. Não é um perfil de",
            "> cold start, e os valores são específicos deste hardware.",
        ]

    decomposition = decomposition_rows(numbers)
    if decomposition:
        sections += [
            "",
            "## T4 — Decomposição do erro (nDCG@10)",
            "",
            table(
                ["Braço", "Quantização", "Grafo (ef pré-registrado)", "Rescoring", "Reportado"],
                decomposition,
            ),
            "",
            "> Obtida varrendo `ef` até o platô, com rescoring desligado. Isola quanto da",
            "> perda vem dos códigos e quanto vem da busca aproximada.",
        ]

    geometry = geometry_lines(numbers)
    if geometry:
        sections += ["", "## T5 — Geometria dos vetores e o teto do braço binário", "", *geometry]

    calibration = calibration_lines(numbers)
    if calibration:
        sections += [
            "",
            "## T6 — Validação do aparelho",
            "",
            "Reproduz o protocolo dos autores do Quati — run sobre o corpus de 1M pontuado",
            "contra os qrels de 10M — para confrontar a linha de base com o único número de",
            "E5-base publicado sobre este benchmark. É o que autoriza ler as outras tabelas.",
            "",
            *calibration,
        ]

    sections += ["", "## Procedência", "", *provenance(numbers)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_path = OUTPUT_DIR / "tables.md"
    numbers_path = OUTPUT_DIR / "numbers.json"
    tables_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    numbers_path.write_text(
        json.dumps(numbers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n".join(sections))
    print(f"\n→ {tables_path}\n→ {numbers_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
