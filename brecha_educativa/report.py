"""
Generación de reportes ejecutivos de hallazgos estadísticos y brechas educativas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import pandas as pd

from brecha_educativa.config import EXECUTIVE_REPORT_FILE
from brecha_educativa.analysis import (
    compute_duncan_segregation_index,
    compute_commune_metrics_summary,
)


def generate_educational_report(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = EXECUTIVE_REPORT_FILE,
) -> str:
    """Genera informe consolidado con métricas de segregación y brecha."""
    duncan = compute_duncan_segregation_index(df)
    summary_comunas = compute_commune_metrics_summary(df)

    lines = []
    lines.append("=" * 70)
    lines.append("REPORTE DE HALLAZGOS Y SEGREGACIÓN EDUCATIVA (SANTIAGO RM 2024)")
    lines.append(f"Total Establecimientos Analizados: {len(df)}")
    lines.append(f"Total Matrícula Estudiantil:       {int(df['MAT_TOTAL'].sum()):,}")
    lines.append(f"Índice de Disimilitud de Duncan:    {duncan:.3f} (Segregación Alta > 0.40)")
    lines.append("=" * 70 + "\n")

    # Comparativa por Tipo de Pago
    lines.append("1. COMPARATIVA SEGÚN TIPO DE PAGO (GRATUITO VS PAGADO)")
    comp = df.groupby("TIPO_PAGO").agg(
        Colegios=("RBD", "count"),
        Matricula_Promedio=("MAT_TOTAL", "mean"),
        Alumnos_por_Docente=("ratio_alumno_docente", "mean"),
        SIMCE_4B=("SIMCE_4B_AVG", "mean"),
        SIMCE_2M=("SIMCE_2M_AVG", "mean"),
    )
    lines.append(comp.to_string())
    lines.append("\n" + "-" * 50 + "\n")

    # Top comunas con mayor concentración de colegios pagados
    lines.append("2. TOP 10 COMUNAS CON MAYOR PROPORCIÓN DE MATRÍCULA PAGADA")
    lines.append(summary_comunas[["NOM_COM_RBD", "total_colegios", "matricula_total", "pct_pagados", "simce_4b_prom"]].head(10).to_string(index=False))
    lines.append("\n" + "-" * 50 + "\n")

    # Top comunas con menor puntaje SIMCE
    lines.append("3. TOP 10 COMUNAS CON MENOR DESEMPEÑO SIMCE 4° BÁSICO")
    low_simce = summary_comunas.sort_values(by="simce_4b_prom", ascending=True).head(10)
    lines.append(low_simce[["NOM_COM_RBD", "total_colegios", "simce_4b_prom", "ratio_docente_prom"]].to_string(index=False))
    lines.append("\n")

    full_text = "\n".join(lines)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Reporte guardado exitosamente en: {out_p}")

    return full_text
