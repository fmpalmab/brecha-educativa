import numpy as np
import pandas as pd
import pytest

from brecha_educativa.longitudinal import (
    generate_longitudinal_dataset,
    compute_longitudinal_metrics,
    plot_longitudinal_comparative,
    run_longitudinal_pipeline,
)
from brecha_educativa.processor import generate_synthetic_educational_dataset


def test_generate_longitudinal_dataset():
    synth = generate_synthetic_educational_dataset(n_schools=50)
    years = [2022, 2023, 2024]
    panel = generate_longitudinal_dataset(base_df=synth, years=years)

    assert len(panel) == 50 * 3
    assert set(panel["YEAR"].unique()) == set(years)
    assert "TIPO_PAGO" in panel.columns
    assert "SIMCE_4B_AVG" in panel.columns


def test_compute_longitudinal_metrics():
    synth = generate_synthetic_educational_dataset(n_schools=60)
    years = [2022, 2023, 2024, 2025]
    panel = generate_longitudinal_dataset(base_df=synth, years=years)
    metrics = compute_longitudinal_metrics(panel)

    assert len(metrics) == 4
    assert list(metrics["Año"]) == years
    assert "Indice_Duncan" in metrics.columns
    assert "Brecha_SIMCE_4B" in metrics.columns
    assert "YoY_Brecha_4B" in metrics.columns

    # El índice Duncan debe estar en [0, 1]
    assert (metrics["Indice_Duncan"] >= 0.0).all()
    assert (metrics["Indice_Duncan"] <= 1.0).all()

    # Los colegios pagados deben tener puntaje mayor en promedio (brecha positiva)
    assert (metrics["Brecha_SIMCE_4B"] > 0).all()


def test_plot_longitudinal_comparative(tmp_path):
    synth = generate_synthetic_educational_dataset(n_schools=40)
    panel = generate_longitudinal_dataset(base_df=synth, years=[2023, 2024])
    metrics = compute_longitudinal_metrics(panel)

    out_file = tmp_path / "longitudinal_test.png"
    fig = plot_longitudinal_comparative(metrics, output_path=out_file)

    assert out_file.is_file()
    assert out_file.stat().st_size > 1000


def test_run_longitudinal_pipeline(tmp_path):
    out_fig = tmp_path / "pipe_fig.png"
    summary = run_longitudinal_pipeline(years=[2023, 2024], output_figure=out_fig)

    assert summary.metrics_by_year is not None
    assert isinstance(summary.gap_delta_first_to_last, float)
    assert out_fig.is_file()
