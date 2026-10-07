"""
brecha_educativa - Pipeline de ciencia de datos y econometría espacial para análisis de brecha y segregación escolar en Santiago de Chile.
"""

from brecha_educativa.config import *
from brecha_educativa.processor import (
    normalizar_texto,
    clean_coord,
    clasificar_pago_consolidado,
    process_raw_data,
    generate_synthetic_educational_dataset,
)
from brecha_educativa.analysis import (
    compute_duncan_segregation_index,
    compute_correlations_matrix,
    compute_commune_metrics_summary,
)
from brecha_educativa.spatial import load_communes_geodataframe, plot_schools_map
from brecha_educativa.visualizer import plot_correlation_heatmap, plot_fee_distribution
from brecha_educativa.report import generate_educational_report
from brecha_educativa.pipeline import run_full_educational_pipeline

__version__ = "2.0.0"
