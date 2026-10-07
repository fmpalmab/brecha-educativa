"""
Interfaz de línea de comandos (CLI) unificada para brecha_educativa.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

from brecha_educativa.config import CONSOLIDATED_FILE, PROCESSED_DIR, FIGURES_DIR
from brecha_educativa.pipeline import run_full_educational_pipeline
from brecha_educativa.processor import process_raw_data, generate_synthetic_educational_dataset
from brecha_educativa.spatial import plot_schools_map
from brecha_educativa.report import generate_educational_report


def configure_console_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    configure_console_output()

    parser = argparse.ArgumentParser(
        prog="brecha_educativa",
        description="Pipeline Econométrico y Espacial de Análisis de Brecha Educativa (Santiago RM)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pipeline", action="store_true", help="Ejecutar el pipeline integral de análisis")
    group.add_argument("--process", action="store_true", help="Procesar y consolidar archivos crudos de MINEDUC")
    group.add_argument("--map", action="store_true", help="Generar mapa geoespacial de distribución de colegios")
    group.add_argument("--report", action="store_true", help="Generar reporte ejecutivo de hallazgos")
    group.add_argument("--synth", action="store_true", help="Generar dataset sintético para pruebas locales")

    parser.add_argument("--schools", type=int, default=150, help="Cantidad de colegios sintéticos")

    args = parser.parse_args(argv)

    if args.synth:
        print(f"Generando {args.schools} colegios sintéticos...")
        df = generate_synthetic_educational_dataset(n_schools=args.schools)
        out = PROCESSED_DIR / "base_sintetica.csv"
        df.to_csv(out, index=False)
        print(f"Dataset sintético guardado en: {out}")
        return 0

    if args.process:
        process_raw_data()
        return 0

    if args.map:
        if not CONSOLIDATED_FILE.is_file():
            process_raw_data()
        df = pd.read_csv(CONSOLIDATED_FILE)
        out = FIGURES_DIR / "mapa_colegios_santiago.png"
        plot_schools_map(df, output_path=out)
        return 0

    if args.report:
        if not CONSOLIDATED_FILE.is_file():
            process_raw_data()
        df = pd.read_csv(CONSOLIDATED_FILE)
        text = generate_educational_report(df)
        print(text)
        return 0

    # Por defecto o con --pipeline
    res = run_full_educational_pipeline()
    print(f"\nResumen: {len(res['df'])} colegios analizados. Duncan Index: {res['duncan_index']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
