"""Cálculo de métricas contra o gabarito oficial, sob o protocolo de S1.2.

As métricas nunca são implementadas à mão: a convenção de desconto e a
construção do DCG ideal têm variantes defensáveis, e escolher uma por acidente é
erro de medição sem contrapartida. Aqui só se decide *o que* medir e *sobre
quais consultas* — o cálculo é do `ir_measures`.

Comportamento verificado do `ir_measures`/`pytrec_eval` (2026-08-18): uma consulta
sem nenhuma passagem relevante recebe **0,0 e entra na média**, em vez de ser
omitida. Como a consulta 2 do Quati é exatamente esse caso, incluí-la puxaria o
nDCG absoluto para baixo em ~2% — justamente a grandeza que a calibração do A0
compara contra o valor publicado. Daí a exclusão explícita, pré-registrada.
"""

from __future__ import annotations

import json
from pathlib import Path

import ir_measures
from ir_measures import R, nDCG

from quantptbr import corpus, stats

#: Um run no formato que o `ir_measures` consome: {query_id: {passage_id: score}}.
Run = dict[str, dict[str, float]]


def measures() -> list:
    """As duas métricas do artigo, com o limiar de relevância vindo da matriz."""
    threshold = stats.matrix()["evaluation"]["relevance_threshold_for_recall"]
    return [nDCG @ 10, R(rel=threshold) @ 100]


def scoreable_query_ids() -> list[str]:
    """As consultas que entram em qualquer média, conforme pré-registrado."""
    excluded = set(stats.matrix()["evaluation"]["excluded_query_ids"])
    return [q for q in corpus.judged_query_ids() if q not in excluded]


def evaluation_qrels(include_excluded: bool = False) -> dict[str, dict[str, int]]:
    qrels = corpus.load_qrels()
    if include_excluded:
        return qrels
    keep = set(scoreable_query_ids())
    return {q: v for q, v in qrels.items() if q in keep}


def evaluate(run: Run, include_excluded: bool = False) -> dict:
    """Avalia um run e devolve agregados e valores por consulta.

    Os valores por consulta não são um extra: o bootstrap pareado de S1.2
    consome exatamente eles, e uma média já agregada não permite parear nada.
    """
    qrels = evaluation_qrels(include_excluded)
    expected = set(qrels)

    # Consulta julgada mas pré-registrada como excluída pode estar no run: o
    # harness recupera as 50 e a avaliação escolhe. O que não pode é consulta
    # sem gabarito nenhum entrar na média.
    unjudged = set(run) - set(corpus.load_qrels())
    if unjudged:
        raise ValueError(f"consultas sem gabarito vazaram para o run: {sorted(unjudged)[:5]}")
    run = {q: v for q, v in run.items() if q in expected}

    missing = expected - set(run)
    if missing:
        raise ValueError(f"consultas julgadas sem resultados no run: {sorted(missing, key=int)}")

    selected = measures()
    per_query: dict[str, dict[str, float]] = {str(m): {} for m in selected}
    for result in ir_measures.iter_calc(selected, qrels, run):
        per_query[str(result.measure)][result.query_id] = float(result.value)

    aggregate = {
        str(m): float(v) for m, v in ir_measures.calc_aggregate(selected, qrels, run).items()
    }
    return {
        "queries": len(expected),
        "include_excluded": include_excluded,
        "aggregate": aggregate,
        "per_query": per_query,
    }


def save(evaluation: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def calibration_report(run: Run) -> dict:
    """Confronta o A0 com o nDCG@10 publicado, nas duas convenções de denominador.

    Reportar as duas não é indecisão: não sabemos se os autores do Quati
    incluíram a consulta 2 na média, e a diferença é da ordem da própria
    tolerância de calibração. Ver qual das duas aproxima o valor publicado é
    informação sobre a convenção deles, não um grau de liberdade nosso — a média
    de 49 continua sendo a pré-registrada, aconteça o que acontecer.
    """
    target = stats.matrix()["evaluation"]["calibration_target"]
    over_49 = evaluate(run)["aggregate"]["nDCG@10"]
    over_50 = evaluate(run, include_excluded=True)["aggregate"]["nDCG@10"]

    published = target["published_ndcg10"]
    return {
        "published": published,
        "floor_bm25": target["floor"]["ndcg10"],
        "preregistered_over_49": over_49,
        "diagnostic_over_50": over_50,
        "delta_49": over_49 - published,
        "delta_50": over_50 - published,
        "verdict": _calibration_verdict(over_49, over_50, published, target["floor"]["ndcg10"]),
    }


def _calibration_verdict(over_49: float, over_50: float, published: float, floor: float) -> str:
    closest = min(abs(over_49 - published), abs(over_50 - published))
    if max(over_49, over_50) < floor:
        return "PARE: abaixo do piso do BM25; o instrumento está quebrado"
    if closest <= 0.02:
        return "aparelho validado"
    if closest <= 0.05:
        return "investigar convenção de ganho, prefixos, max_length; registrar a causa"
    return "PARE: fora de 0,05 do valor publicado"
