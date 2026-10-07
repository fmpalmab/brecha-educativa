"""Pruebas unitarias para el índice de Duncan y análisis econométrico."""

import pytest
import pandas as pd
from brecha_educativa.analysis import (
    compute_duncan_segregation_index,
    compute_correlations_matrix,
    compute_commune_metrics_summary,
)


def test_duncan_segregation_index_perfect_integration():
    # Proporciones idénticas en ambas comunas -> D = 0.0
    df = pd.DataFrame(
        {
            "NOM_COM_RBD": ["COMUNA_A", "COMUNA_A", "COMUNA_B", "COMUNA_B"],
            "TIPO_PAGO": ["Gratuito", "Pagado", "Gratuito", "Pagado"],
            "MAT_TOTAL": [100, 100, 100, 100],
        }
    )
    duncan = compute_duncan_segregation_index(df)
    assert duncan == pytest.approx(0.0)


def test_duncan_segregation_index_complete_segregation():
    # Comuna A solo pagados, Comuna B solo gratuitos -> D = 1.0
    df = pd.DataFrame(
        {
            "NOM_COM_RBD": ["COMUNA_A", "COMUNA_B"],
            "TIPO_PAGO": ["Pagado", "Gratuito"],
            "MAT_TOTAL": [200, 200],
        }
    )
    duncan = compute_duncan_segregation_index(df)
    assert duncan == pytest.approx(1.0)


def test_compute_correlations_matrix():
    df = pd.DataFrame(
        {
            "orden_precio": [0, 1, 2, 3, 4, 5],
            "MAT_TOTAL": [100, 150, 200, 250, 300, 350],
            "ratio_alumno_docente": [15, 16, 17, 18, 19, 20],
        }
    )
    corr = compute_correlations_matrix(df)
    assert not corr.empty
    assert corr.shape[0] == 3
