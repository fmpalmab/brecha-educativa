"""Pruebas unitarias para limpieza y normalización de datos educativos."""

import pytest
import pandas as pd
from brecha_educativa.processor import (
    normalizar_texto,
    clean_coord,
    clasificar_pago_consolidado,
)


def test_normalizar_texto():
    assert normalizar_texto("Ñuñoa") == "NUNOA"
    assert normalizar_texto("Peñalolén  ") == "PENALOLEN"
    assert normalizar_texto("Estación Central") == "ESTACION CENTRAL"
    assert normalizar_texto(None) == "DESCONOCIDO"


def test_clean_coord():
    assert clean_coord("-70,6543") == pytest.approx(-70.6543)
    assert clean_coord(-33.45) == pytest.approx(-33.45)
    assert clean_coord("invalido") is None
    assert clean_coord(None) is None


def test_clasificar_pago_consolidado():
    row_free = pd.Series({"PAGO_MENSUAL": "GRATUITO", "categoria_dependencia": "PARTICULAR_SUBV"})
    assert clasificar_pago_consolidado(row_free) == "Gratuito"

    row_paid = pd.Series({"PAGO_MENSUAL": "MAS DE $100.000", "categoria_dependencia": "PARTICULAR_PAGADO"})
    assert clasificar_pago_consolidado(row_paid) == "Pagado"

    row_municipal = pd.Series({"PAGO_MENSUAL": "SIN INFORMACION", "categoria_dependencia": "MUNICIPAL_DAEM"})
    assert clasificar_pago_consolidado(row_municipal) == "Gratuito"
