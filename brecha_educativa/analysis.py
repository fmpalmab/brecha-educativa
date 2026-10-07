"""
Análisis estadístico, econométrico y métricas de segregación escolar.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_duncan_segregation_index(
    df: pd.DataFrame,
    group_col: str = "NOM_COM_RBD",
    category_col: str = "TIPO_PAGO",
    weight_col: str = "MAT_TOTAL",
) -> float:
    """
    Calcula el Índice de Disimilitud de Duncan (D) para segregación escolar entre dos grupos
    (Gratuito vs Pagado) a través de las comunas o unidades espaciales.
    
    D = 0.5 * sum | (p_i / P_total) - (g_i / G_total) |
    Varía entre 0 (integración perfecta) y 1 (segregación total).
    """
    pivot = df.pivot_table(
        index=group_col,
        columns=category_col,
        values=weight_col,
        aggfunc="sum",
        fill_value=0.0,
    )

    if "Pagado" not in pivot.columns or "Gratuito" not in pivot.columns:
        return 0.0

    total_paid = float(pivot["Pagado"].sum())
    total_free = float(pivot["Gratuito"].sum())

    if total_paid == 0 or total_free == 0:
        return 0.0

    p_prop = pivot["Pagado"] / total_paid
    g_prop = pivot["Gratuito"] / total_free

    duncan_index = float(0.5 * np.sum(np.abs(p_prop - g_prop)))
    return duncan_index


def compute_correlations_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de correlación entre variables de interés."""
    cols_map = {
        "orden_precio": "Nivel Precio",
        "MAT_TOTAL": "Matrícula Total",
        "ratio_alumno_docente": "Alumnos/Docente",
        "ratio_alumno_curso": "Alumnos/Curso",
        "SIMCE_4B_AVG": "SIMCE 4°B",
        "SIMCE_2M_AVG": "SIMCE 2°M",
    }
    available = [c for c in cols_map.keys() if c in df.columns]
    sub_df = df[available].rename(columns=cols_map)
    return sub_df.corr()


def compute_commune_metrics_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula un resumen estadístico agregado por comuna."""
    resumen = df.groupby("NOM_COM_RBD").agg(
        total_colegios=("RBD", "count"),
        matricula_total=("MAT_TOTAL", "sum"),
        ratio_docente_prom=("ratio_alumno_docente", "mean"),
        ratio_docente_std=("ratio_alumno_docente", "std"),
        ratio_curso_prom=("ratio_alumno_curso", "mean"),
        simce_4b_prom=("SIMCE_4B_AVG", "mean"),
        simce_2m_prom=("SIMCE_2M_AVG", "mean"),
        pct_pagados=("TIPO_PAGO", lambda x: (x == "Pagado").mean() * 100.0),
    ).reset_index()

    return resumen.sort_values(by="pct_pagados", ascending=False)
