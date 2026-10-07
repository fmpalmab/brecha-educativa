"""
Generación de gráficos científicos, infografías y visualizaciones de impacto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from brecha_educativa.config import FIGURES_DIR, COLOR_FREE, COLOR_PAID
from brecha_educativa.analysis import compute_correlations_matrix


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Genera matriz de correlación estilizada."""
    corr_df = compute_correlations_matrix(df)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr_df, annot=True, cmap="RdBu_r", vmin=-1.0, vmax=1.0,
        fmt=".2f", cbar_kws={"label": "Correlación de Pearson"}, ax=ax,
    )
    ax.set_title("Matriz de Correlación: Variables Educativas y SIMCE", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=130)

    plt.close(fig)
    return fig


def plot_fee_distribution(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Genera gráfico de distribución de ratio alumno/curso por tipo de pago."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    sns.boxplot(
        data=df, x="TIPO_PAGO", y="ratio_alumno_docente",
        hue="TIPO_PAGO", palette=[COLOR_FREE, COLOR_PAID], legend=False, ax=ax1,
    )
    ax1.set_title("Alumnos por Docente", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Ratio Alumnos / Docente")
    ax1.set_xlabel("Tipo de Financiamiento")
    ax1.grid(True, linestyle="--", alpha=0.4)

    if "ratio_alumno_curso" in df.columns:
        sns.boxplot(
            data=df, x="TIPO_PAGO", y="ratio_alumno_curso",
            hue="TIPO_PAGO", palette=[COLOR_FREE, COLOR_PAID], legend=False, ax=ax2,
        )
        ax2.set_title("Alumnos por Aula (Saturación)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Ratio Alumnos / Curso")
        ax2.set_xlabel("Tipo de Financiamiento")
        ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=130)

    plt.close(fig)
    return fig
