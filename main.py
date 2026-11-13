#!/usr/bin/env python3
"""Orquestador de experimentos (punto de entrada).

Uso (con el entorno conda activado):

    python main.py olivetti       # PCA, LDA y Fisherfaces sobre Olivetti
    python main.py lfw            # mismo protocolo sobre LFW (50 identidades)
    python main.py experimentos   # Olivetti y LFW
    python main.py robustez       # repite el protocolo con varias semillas de split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import PipelineExperimentos  # noqa: E402


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimentos de embeddings faciales lineales (LAB02).",
    )
    parser.add_argument(
        "etapa",
        choices=["olivetti", "lfw", "experimentos", "robustez"],
        help=(
            "Conjunto a evaluar. 'experimentos' corre Olivetti y LFW; "
            "'robustez' repite ambos con varias semillas de split."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    pipeline = PipelineExperimentos()
    acciones = {
        "olivetti": pipeline.ejecutar_olivetti,
        "lfw": pipeline.ejecutar_lfw,
        "experimentos": pipeline.ejecutar_experimentos,
        "robustez": pipeline.ejecutar_robustez,
    }
    acciones[args.etapa]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
