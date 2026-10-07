"""
Configuración centralizada para brecha-educativa.
Rutas de datos, límites geográficos (BBox) y paletas de colores corporativas.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
SHAPEFILE_DIR = EXTERNAL_DIR / "comunas_rm"
FIGURES_DIR = REPO_ROOT / "figures"
REPORTS_DIR = REPO_ROOT / "reports"

# Crear carpetas requeridas
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Rutas de archivos
CONSOLIDATED_FILE = PROCESSED_DIR / "base_consolidada_rm_2024_final.csv"
METRICS_SUMMARY_FILE = PROCESSED_DIR / "resumen_comunal_metricas.csv"
SHAPEFILE_PATH = SHAPEFILE_DIR / "COMUNA_C17.shp"
EXECUTIVE_REPORT_FILE = REPORTS_DIR / "hallazgos_estadisticos.txt"

# Bounding Box Urbano de Santiago [min_lon, min_lat, max_lon, max_lat]
BBOX: List[float] = [-70.872116, -33.642527, -70.469742, -33.327552]

# Criterios de limpieza y representatividad
MIN_COLEGIOS_POR_COMUNA: int = 3

# Paleta de colores consistente
COLOR_FREE: str = "#2ca02c"   # Verde (Educación gratuita / subvencionada)
COLOR_PAID: str = "#d62728"   # Rojo (Educación pagada)
COLOR_NEUTRAL: str = "#1f77b4" # Azul
