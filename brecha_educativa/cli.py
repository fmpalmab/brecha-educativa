"""
Interfaz de línea de comandos (CLI) unificada para brecha_educativa.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd

from brecha_educativa.config import CONSOLIDATED_FILE, PROCESSED_DIR, FIGURES_DIR
from brecha_educativa.pipeline import run_full_educational_pipeline
from brecha_educativa.processor import process_raw_data, generate_synthetic_educational_dataset
from brecha_educativa.spatial import plot_schools_map
from brecha_educativa.report import generate_educational_report
from brecha_educativa.longitudinal import run_longitudinal_pipeline, plot_longitudinal_comparative


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
        description="Pipeline Econométrico, Espacial y Longitudinal de Brecha Educativa (Santiago RM)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pipeline", action="store_true", help="Ejecutar el pipeline integral de análisis")
    group.add_argument("--process", action="store_true", help="Procesar y consolidar archivos crudos de MINEDUC")
    group.add_argument("--map", action="store_true", help="Generar mapa geoespacial de distribución de colegios")
    group.add_argument("--report", action="store_true", help="Generar reporte ejecutivo de hallazgos")
    group.add_argument("--synth", action="store_true", help="Generar dataset sintético para pruebas locales")
    group.add_argument("--longitudinal", action="store_true", help="Ejecutar análisis comparativo longitudinal multianual")

    parser.add_argument("--schools", type=int, default=150, help="Cantidad de colegios sintéticos")
    parser.add_argument("--years", type=str, default="2022,2023,2024,2025", help="Años a comparar separados por coma (ej: 2018,2022,2023,2024,2025)")
    parser.add_argument("--output", type=str, help="Ruta de archivo de salida")

    args = parser.parse_args(argv)

    if args.synth:
        print(f"Generando {args.schools} colegios sintéticos...")
        df = generate_synthetic_educational_dataset(n_schools=args.schools)
        out = PROCESSED_DIR / "base_sintetica.csv"
        df.to_csv(out, index=False)
        print(f"Dataset sintético guardado en: {out}")
        return 0

    if args.longitudinal:
        print("Ejecutando análisis longitudinal comparativo...")
        year_list = [int(y.strip()) for y in args.years.split(",") if y.strip().isdigit()]
        out_fig = args.output or (FIGURES_DIR / "00_evolucion_longitudinal_brecha.png")
        summary = run_longitudinal_pipeline(years=year_list, output_figure=out_fig)

        print("\n=== RESUMEN LONGITUDINAL MULTIANUAL ===")
        print(summary.metrics_by_year.to_string(index=False))
        print(f"\nDelta Brecha 4° Básico ({year_list[0]} -> {year_list[-1]}): {summary.gap_delta_first_to_last:+.2f} pts")
        print(f"Delta Índice de Duncan: {summary.duncan_delta_first_to_last:+.4f}")
        print(f"Gráfico longitudinal generado en: {out_fig}")
        return 0

    if args.process:
        process_raw_data()
        return 0

    if args.map:
        if not CONSOLIDATED_FILE.is_file():
            process_raw_data()
        df = pd.read_csv(CONSOLIDATED_FILE)
        out = args.output or (FIGURES_DIR / "mapa_colegios_santiago.png")
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
