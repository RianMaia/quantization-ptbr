"""Contrato do corpus congelado.

A asserção que importa é a de cobertura: os qrels do Quati indexam a coleção
inteira de 1M, então qualquer subamostragem do corpus destrói o gabarito em
silêncio e leva o nDCG a zero por construção, e não por quantização.
"""

from __future__ import annotations

import pytest

from quantptbr import corpus

pytestmark = pytest.mark.skipif(
    not corpus.MANIFEST_PATH.exists(),
    reason="corpus não baixado; rode scripts/01_fetch_quati.py",
)


@pytest.fixture(scope="module")
def passage_ids() -> list[str]:
    return corpus.load_passage_ids()


@pytest.fixture(scope="module")
def qrels() -> dict[str, dict[str, int]]:
    return corpus.load_qrels()


def test_manifest_matches_files_on_disk():
    corpus.verify_manifest()


def test_passage_count_matches_published_figure(passage_ids):
    assert len(passage_ids) == corpus.EXPECTED_PASSAGES


def test_passage_ids_are_unique(passage_ids):
    assert len(set(passage_ids)) == len(passage_ids)


def test_every_judged_passage_is_in_the_corpus(passage_ids, qrels):
    known = set(passage_ids)
    orphans = {q: sorted(set(v) - known) for q, v in qrels.items() if set(v) - known}
    assert not orphans, f"qrels referenciam passagens ausentes do corpus: {orphans}"


def test_judged_query_count_matches_published_figure(qrels):
    assert len(corpus.judged_query_ids(qrels)) == corpus.EXPECTED_JUDGED_QUERIES


def test_mean_judged_passages_matches_published_figure(qrels):
    counts = [len(v) for v in qrels.values()]
    assert sum(counts) / len(counts) == pytest.approx(corpus.EXPECTED_MEAN_JUDGED, abs=0.01)


def test_every_judged_query_has_text(qrels):
    topics = corpus.load_topics()
    missing = [q for q in corpus.judged_query_ids(qrels) if q not in topics]
    assert not missing, f"consultas julgadas sem texto: {missing}"


def test_relevance_grades_are_the_expected_ordinal_scale(qrels):
    observed = {g for v in qrels.values() for g in v.values()}
    assert observed == set(corpus.RELEVANCE_GRADES)


def test_unscoreable_queries_are_exactly_the_known_ones(qrels):
    """Trava um fato do dataset, não uma expectativa.

    A consulta 2 tem 53 julgamentos, todos grau 0, e portanto DCG ideal nulo.
    Se o conjunto mudar, alguma premissa da avaliação mudou junto e o limiar de
    relevância fixado em S1.7 precisa ser revisto antes de qualquer medição.
    """
    assert tuple(corpus.unscoreable_query_ids(qrels)) == corpus.KNOWN_UNSCOREABLE_QUERY_IDS


def test_passage_order_is_reproducible():
    """A ordem define os IDs dos pontos; duas leituras têm de coincidir."""
    first = [pid for pid, _ in zip(corpus.iter_passages(), range(1000))]
    second = [pid for pid, _ in zip(corpus.iter_passages(), range(1000))]
    assert first == second
