"""Pruebas para el generador sintético de datos educativos."""

import pytest
from brecha_educativa.processor import generate_synthetic_educational_dataset


def test_synthetic_educational_dataset():
    df = generate_synthetic_educational_dataset(n_schools=25)
    assert len(df) == 25
    expected_cols = [
        "RBD", "NOM_RBD", "NOM_COM_RBD", "LATITUD", "LONGITUD",
        "TIPO_PAGO", "MAT_TOTAL", "ratio_alumno_docente", "SIMCE_4B_AVG",
    ]
    for col in expected_cols:
        assert col in df.columns

    assert (df["MAT_TOTAL"] > 0).all()
    assert (df["ratio_alumno_docente"] > 0).all()
