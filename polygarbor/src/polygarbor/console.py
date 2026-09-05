"""Saida de terminal: cabecalhos, passos e tabelas simples, sem dependencias extras."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Sequence

_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def disable_color() -> None:
    global _COLOR
    _COLOR = False


def title(text: str) -> None:
    line = "=" * max(len(text), 60)
    print(f"\n{_c('1;36', line)}\n{_c('1;36', text)}\n{_c('1;36', line)}")


def section(text: str) -> None:
    print(f"\n{_c('1;34', '▸ ' + text)}")


def info(label: str, value: object = "") -> None:
    print(f"  {_c('90', label + ':'):<34} {value}" if value != "" else f"  {label}")


def ok(text: str) -> None:
    print(f"  {_c('32', '✔')} {text}")


def warn(text: str) -> None:
    print(f"  {_c('33', '!')} {text}")


def error(text: str) -> None:
    print(f"{_c('1;31', '✖ erro:')} {text}", file=sys.stderr)


def bullet(text: str) -> None:
    print(f"    {_c('90', '·')} {text}")


def table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> None:
    """Tabela alinhada em texto puro."""
    data = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(str(headers[i])), *(len(r[i]) for r in data)) if data else len(headers[i])
        for i in range(len(headers))
    ]
    head = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print("  " + _c("1", head))
    print("  " + _c("90", "  ".join("-" * w for w in widths)))
    for row in data:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))


@contextmanager
def step(text: str):
    """Cronometra um bloco e imprime o tempo gasto."""
    print(f"\n{_c('1;34', '▸ ' + text)}")
    start = time.perf_counter()
    yield
    print(f"  {_c('32', '✔')} concluido em {time.perf_counter() - start:.1f}s")
