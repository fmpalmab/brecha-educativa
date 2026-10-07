"""
Módulo de procesamiento geoespacial y mapas de la Región Metropolitana.
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import box

from brecha_educativa.config import (
    SHAPEFILE_PATH,
    EXTERNAL_DIR,
    BBOX,
    COLOR_FREE,
    COLOR_PAID,
)
from brecha_educativa.processor import normalizar_texto


def load_communes_geodataframe(shp_path: Union[str, Path] = SHAPEFILE_PATH):
    """Carga el shapefile de comunas de la RM usando GeoPandas."""
    try:
        import geopandas as gpd
    except ImportError:
        print("[Aviso] geopandas no está disponible.")
        return None

    path_shp = Path(shp_path)
    if not path_shp.is_file():
        # Buscar en ubicaciones alternativas
        alt_paths = list(EXTERNAL_DIR.glob("**/COMUNA_C17.shp"))
        if alt_paths:
            path_shp = alt_paths[0]
        else:
            print(f"[Aviso] No se encontró shapefile en {path_shp}")
            return None

    try:
        gdf = gpd.read_file(path_shp)
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs(epsg=4326)

        # Normalizar nombre comuna
        name_col = "NOM_COM" if "NOM_COM" in gdf.columns else "Comuna" if "Comuna" in gdf.columns else gdf.columns[0]
        gdf["NOM_COM_NORM"] = gdf[name_col].apply(normalizar_texto)

        # Recorte al BBOX urbano
        polygon_bbox = box(BBOX[0], BBOX[1], BBOX[2], BBOX[3])
        gdf_clipped = gdf[gdf.geometry.intersects(polygon_bbox)].copy()
        return gdf_clipped
    except Exception as e:
        print(f"[Error cargando mapa]: {e}")
        return None


def plot_schools_map(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """Genera mapa con la distribución espacial de colegios gratuitos vs pagados."""
    gdf_map = load_communes_geodataframe()

    fig, ax = plt.subplots(figsize=(10, 10))

    if gdf_map is not None:
        gdf_map.boundary.plot(ax=ax, color="#7f7f7f", linewidth=0.8, alpha=0.7)

    # Filtrar coordenadas válidas
    clean_df = df.dropna(subset=["LONGITUD", "LATITUD"])
    gratuitos = clean_df[clean_df["TIPO_PAGO"] == "Gratuito"]
    pagados = clean_df[clean_df["TIPO_PAGO"] == "Pagado"]

    ax.scatter(
        gratuitos["LONGITUD"], gratuitos["LATITUD"],
        c=COLOR_FREE, s=16, alpha=0.6, label="Gratuito / Subvencionado",
    )
    ax.scatter(
        pagados["LONGITUD"], pagados["LATITUD"],
        c=COLOR_PAID, s=22, alpha=0.8, label="Particular Pagado",
    )

    ax.set_xlim(BBOX[0], BBOX[2])
    ax.set_ylim(BBOX[1], BBOX[3])
    ax.set_title("Distribución Espacial de Establecimientos Educativos en Santiago (RM)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, dpi=130)
        print(f"Mapa guardado en: {out_p}")

    if show:
        plt.show()

    plt.close(fig)
    return fig
