"""
Pipeline integral de procesamiento, análisis y generación de figuras de brecha educativa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from brecha_educativa.config import (
    CONSOLIDATED_FILE,
    METRICS_SUMMARY_FILE,
    FIGURES_DIR,
    REPORTS_DIR,
    EXECUTIVE_REPORT_FILE,
)
from brecha_educativa.processor import process_raw_data
from brecha_educativa.analysis import (
    compute_duncan_segregation_index,
    compute_commune_metrics_summary,
)
from brecha_educativa.spatial import plot_schools_map
from brecha_educativa.visualizer import plot_correlation_heatmap, plot_fee_distribution
from brecha_educativa.report import generate_educational_report


def run_full_educational_pipeline() -> Dict[str, Any]:
    """Ejecuta el flujo completo de análisis de brecha educativa."""
    print("=== INICIANDO PIPELINE DE BRECHA EDUCATIVA RM ===")

    # 1. Cargar o procesar base consolidada
    if CONSOLIDATED_FILE.is_file():
        print(f"[1/4] Cargando base de datos consolidada: {CONSOLIDATED_FILE}")
        df = pd.read_csv(CONSOLIDATED_FILE)
    else:
        print("[1/4] Procesando archivos crudos de MINEDUC y SIMCE...")
        df = process_raw_data()

    # 2. Resumen comunal y métricas de segregación
    print("[2/4] Calculando índice de Duncan y agregaciones comunales...")
    duncan_val = compute_duncan_segregation_index(df)
    summary_df = compute_commune_metrics_summary(df)
    summary_df.to_csv(METRICS_SUMMARY_FILE, index=False)
    print(f"      Índice de Segregación Duncan (D): {duncan_val:.3f}")

    # 3. Generación de mapas y figuras científicas
    print("[3/4] Generando mapas espaciales y figuras estadísticas...")
    heatmap_path = FIGURES_DIR / "matriz_correlacion_general.png"
    dist_path = FIGURES_DIR / "distribucion_financiamiento.png"
    map_path = FIGURES_DIR / "mapa_colegios_santiago.png"

    plot_correlation_heatmap(df, output_path=heatmap_path)
    plot_fee_distribution(df, output_path=dist_path)
    plot_schools_map(df, output_path=map_path)

    # 4. Reporte ejecutivo
    print("[4/4] Redactando informe ejecutivo de hallazgos...")
    report_text = generate_educational_report(df, output_path=EXECUTIVE_REPORT_FILE)

    print("\n=== PIPELINE EDUCATIVO FINALIZADO CON ÉXITO ===")
    return {
        "df": df,
        "duncan_index": duncan_val,
        "summary": summary_df,
        "report": report_text,
    }
