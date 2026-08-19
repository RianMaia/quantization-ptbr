"""Leitura da memória residente sob o contrato de S1.1.

A métrica principal do artigo é `anon` do cgroup v2 do container do Qdrant, e
não o RSS. O motivo é mensurável nesta máquina: o Qdrant mapeia os vetores em
disco, então o RSS conta páginas *file-backed* e descartáveis cujo tamanho
depende de quantas consultas já rodaram. Com o servidor vazio, `RssFile` marca
~32 MiB contra `file` de 1,2 MiB no cgroup — a discrepância não é ruído, é a
diferença entre "memória que o processo precisa" e "páginas que por acaso estão
no cache".

O processo do Qdrant vive num cgroup **folha** (`<scope>/container`), não no
scope. Ler o scope agregaria o que mais estiver pendurado nele.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from quantptbr import server

MIB = 1024**2


@dataclass(frozen=True)
class MemorySnapshot:
    """Uma linha de medição. Toda medida do estudo carrega uma destas."""

    anon_mb: float
    file_mb: float
    total_mb: float
    host_available_mb: float
    host_swap_used_mb: float
    cgroup: str
    taken_at: str

    @property
    def is_void(self) -> bool:
        """Medição tirada sob pressão de swap não vale, e precisa ser detectável depois."""
        return self.host_swap_used_mb > 0 and self.host_available_mb < 1024

    def as_dict(self) -> dict:
        return asdict(self) | {"is_void": self.is_void}


def cgroup_path() -> Path:
    """O cgroup folha que contém exatamente os processos do container."""
    scope = server.podman(
        "inspect", server.CONTAINER, "--format", "{{.State.CgroupPath}}"
    ).stdout.strip()
    if not scope:
        raise RuntimeError(f"{server.CONTAINER} não está de pé; não há cgroup para ler")
    leaf = Path("/sys/fs/cgroup") / scope.lstrip("/") / "container"
    if not (leaf / "memory.stat").exists():
        raise RuntimeError(f"{leaf}/memory.stat ausente; o contrato de S1.1 precisa de emenda")
    return leaf


def _read_stat(path: Path) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in (line.split(maxsplit=1) for line in path.read_text().splitlines())
    }


def _host() -> tuple[float, float]:
    meminfo = {
        key.rstrip(":"): int(value.split()[0])
        for key, value in (
            line.split(maxsplit=1) for line in Path("/proc/meminfo").read_text().splitlines()
        )
    }
    available = meminfo["MemAvailable"] / 1024
    swap_used = (meminfo["SwapTotal"] - meminfo["SwapFree"]) / 1024
    return available, swap_used


def read() -> MemorySnapshot:
    """As três figuras de memória do container, mais o estado do host, numa chamada."""
    leaf = cgroup_path()
    stat = _read_stat(leaf / "memory.stat")
    total = int((leaf / "memory.current").read_text())
    available, swap_used = _host()
    return MemorySnapshot(
        anon_mb=round(stat["anon"] / MIB, 1),
        file_mb=round(stat["file"] / MIB, 1),
        total_mb=round(total / MIB, 1),
        host_available_mb=round(available, 1),
        host_swap_used_mb=round(swap_used, 1),
        cgroup=str(leaf),
        taken_at=datetime.now(UTC).isoformat(),
    )


def cross_check() -> dict:
    """Confere a leitura direta contra o `podman stats`.

    Divergência é explicada, nunca reconciliada por média: as duas fontes medem
    coisas ligeiramente diferentes, e a que vale é o cgroup.
    """
    snapshot = read()
    raw = server.podman("stats", "--no-stream", "--format", "json", server.CONTAINER).stdout
    # `mem_usage` vem formatado ("209MB / 25.02GB") e em unidades SI, enquanto o
    # cgroup é lido em bytes e convertido em MiB. A diferença de ~4% entre as
    # duas escalas é esperada e não é reconciliada — o cgroup é a fonte.
    used, _, _ = json.loads(raw)[0]["mem_usage"].partition("/")
    podman_mb = _parse_si(used.strip()) / MIB
    return {
        "cgroup_total_mb": snapshot.total_mb,
        "podman_stats_mb": round(podman_mb, 1),
        "difference_mb": round(snapshot.total_mb - podman_mb, 1),
        "_note": "podman reporta em unidades SI; o cgroup é convertido em MiB",
    }


def _parse_si(value: str) -> float:
    for suffix, factor in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3), ("B", 1.0)):
        if value.endswith(suffix):
            return float(value.removesuffix(suffix)) * factor
    raise ValueError(f"unidade não reconhecida em {value!r}")


def analytical_mb(passages: int, dims: int, quantization: dict | None) -> float:
    """A previsão, para ser impressa ao lado da medição — hipótese sob teste.

    Estimativas analíticas não substituem medição em lugar nenhum do estudo. Elas
    existem para que a diferença entre previsto e medido seja um resultado.
    """
    if quantization is None:
        return passages * dims * 4 / MIB
    kind = quantization["kind"]
    if kind == "scalar":
        return passages * dims * 1 / MIB
    if kind == "binary":
        return passages * dims / 8 / MIB
    if kind == "product":
        return passages * dims * 4 / int(quantization["compression"].lstrip("xX")) / MIB
    raise ValueError(f"quantização desconhecida: {kind}")


def rss_mb() -> float:
    """RSS do processo do Qdrant. **Nunca reportado como memória residente.**

    Existe só para que o artigo possa mostrar o número que teria sido reportado
    sob a definição rejeitada, e o quanto ele difere.
    """
    pid = subprocess.run(
        ["pgrep", "-x", "qdrant"], capture_output=True, text=True, check=False
    ).stdout.split()
    if not pid:
        raise RuntimeError("processo qdrant não encontrado")
    rollup = _read_stat_kb(Path(f"/proc/{pid[0]}/smaps_rollup"))
    return round(rollup["Rss"] / 1024, 1)


def _read_stat_kb(path: Path) -> dict[str, int]:
    values = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts and parts[0].isdigit():
                values[key] = int(parts[0])
    return values


def smoke_verdict(cases: list[dict], payload_mb: float) -> tuple[dict, dict]:
    """Separa validação de instrumento de caracterização de braço.

    A primeira execução misturou as duas, e as checagens falharam por codificarem
    uma expectativa que o dado refutou — não por o instrumento estar quebrado.
    Aqui só as **checagens** são portão. Elas perguntam se o contador resolve a
    carga e distingue configurações; nada mais.

    As **observações** são comparações entre configurações. Elas são registradas,
    nunca aprovadas ou reprovadas: 400 mil vetores aleatórios com cinco consultas
    exaustivas não caracterizam braço nenhum. Isso é trabalho de M3, a 1M, com os
    vetores reais e o conjunto de consultas real.
    """
    peak = [c["states"]["exhaustive"] for c in cases]
    cold = [c["states"]["cold"] for c in cases]
    float32_ram, float32_disk, int8_ram, int8_disk = peak

    checks = {
        # A prova: o page cache do container tem de refletir o volume de vetores.
        "o contador resolve a carga de vetores": abs(float32_ram["file_mb"] - payload_mb)
        < 0.2 * payload_mb,
        # Um contador que devolve o mesmo número para tudo não mede nada.
        "o contador distingue configurações": (
            max(p["total_mb"] for p in peak) - min(p["total_mb"] for p in peak) > 0.2 * payload_mb
        ),
        # Residência não é um número: depende do que já foi consultado.
        "o estado declarado importa": any(
            abs(p["total_mb"] - c["total_mb"]) > 0.1 * payload_mb
            for p, c in zip(peak, cold, strict=True)
        ),
        # Quantizado com always_ram é memória anônima; o original é page cache.
        "always_ram aparece em anon": int8_ram["anon_mb"] - float32_ram["anon_mb"]
        > 0.25 * payload_mb,
    }
    observations = {
        "float32_on_disk_muda_a_residencia_mb": round(
            float32_disk["total_mb"] - float32_ram["total_mb"], 1
        ),
        "int8_always_ram_muda_a_residencia_mb": round(
            int8_ram["total_mb"] - float32_ram["total_mb"], 1
        ),
        "int8_com_on_disk_muda_a_residencia_mb": round(
            int8_disk["total_mb"] - float32_ram["total_mb"], 1
        ),
        "_confirmar_em_M3": "vetores aleatórios a 400k não caracterizam braço; a 1M sim",
    }
    return checks, observations
