import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import unicodedata
import os
import requests
from shapely.geometry import box

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Bounding Box Urbano [min_lon, min_lat, max_lon, max_lat]
# Nota: Shapely usa (minx, miny, maxx, maxy) -> (Oeste, Sur, Este, Norte)
BBOX_COORDS = [-70.852116, -33.642527, -70.489742, -33.334552]

def normalizar_texto(texto):
    """Normaliza nombres para cruce (ej: 'Ñuñoa' -> 'NUNOA')."""
    if pd.isna(texto): return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def descargar_shapefile():
    """Descarga shapefile de comunas si no existe."""
    output_dir = 'data/external/comunas_hito2'
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://github.com/PLUMAS-research/visualization-course-materials/raw/master/data/comunas_rm/COMUNA_C17"
    extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    
    shp_path = f"{output_dir}/COMUNA_C17.shp"
    if os.path.exists(shp_path):
        return shp_path
        
    print("⬇️ Descargando mapa base de comunas...")
    for ext in extensions:
        try:
            r = requests.get(f"{base_url}{ext}", allow_redirects=True)
            if r.status_code == 200:
                with open(f"{output_dir}/COMUNA_C17{ext}", 'wb') as f:
                    f.write(r.content)
        except Exception as e:
            print(f"⚠️ Error descargando {ext}: {e}")
            
    return shp_path

def main():
    print(">>> INICIANDO ANÁLISIS GEOESPACIAL CONSOLIDADO (Bloque 3) <<<")
    
    # 1. Cargar Datos Consolidados
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
        print(f"Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("❌ Error: Falta 'base_consolidada_rm_2024_final.csv'. Ejecuta el Bloque 1.")
        return

    # Agrupar por comuna para el mapa (Promedio de carga docente)
    df_agg = df.groupby('NOM_COM_RBD')['ratio_alumno_docente'].mean().reset_index()
    df_agg.rename(columns={'NOM_COM_RBD': 'Comuna_Norm', 'ratio_alumno_docente': 'valor'}, inplace=True)

    # 2. Cargar Mapa Base
    shp_path = descargar_shapefile()
    try:
        gdf_comunas = gpd.read_file(shp_path)
    except Exception as e:
        print(f"❌ Error cargando shapefile: {e}")
        return

    # 3. Normalizar Nombres del Mapa
    col_nombre = 'NOM_COMUNA' if 'NOM_COMUNA' in gdf_comunas.columns else gdf_comunas.columns[0]
    gdf_comunas['nombre_norm'] = gdf_comunas[col_nombre].apply(normalizar_texto)

    # 4. Recorte Geográfico (Zoom Urbano)
    print("✂️ Aplicando Zoom Urbano (Bounding Box)...")
    if gdf_comunas.crs.to_string() != "EPSG:4326":
        gdf_comunas = gdf_comunas.to_crs(epsg=4326)
        
    zona_urbana = box(BBOX_COORDS[0], BBOX_COORDS[1], BBOX_COORDS[2], BBOX_COORDS[3])
    gdf_zoom = gpd.clip(gdf_comunas, zona_urbana)
    print(f"   -> Comunas visibles en el mapa: {len(gdf_zoom)}")

    # 5. Unión de Datos
    gdf_final = gdf_zoom.merge(df_agg, left_on='nombre_norm', right_on='Comuna_Norm', how='left')

    # 6. Visualización: Mapa Coroplético
    print("Generando mapa de calor docente...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Capa principal (Datos)
    gdf_final.plot(
        column='valor',
        cmap='magma_r', # Amarillo -> Rojo (Alto ratio es peor)
        linewidth=0.5,
        ax=ax,
        edgecolor='black',
        legend=True,
        legend_kwds={'label': "Alumnos por Docente (Promedio Comunal)", 'shrink': 0.6},
        missing_kwds={'color': 'lightgrey', 'hatch': '///', 'label': 'Sin datos'}
    )
    
    # Etiquetas inteligentes
    for _, row in gdf_final.iterrows():
        if pd.notnull(row['valor']) and row.geometry.area > 0.001: # Filtrar polígonos muy chicos
            txt = row['nombre_norm'].title()
            # Abreviaciones para limpieza
            txt = txt.replace("Pedro Aguirre Cerda", "PAC").replace("Estacion Central", "Est. Central").replace("Santiago", "Stgo")
            
            plt.annotate(
                text=txt, 
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                horizontalalignment='center',
                fontsize=9, color='black', weight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            )

    # Marco y Títulos
    ax.set_title("Mapa de Carga Docente: Gran Santiago Urbano", fontsize=18, fontweight='bold', pad=20)
    ax.set_axis_off()
    
    # REGLA: Marco visible (aunque axes off quita los ticks, podemos añadir un marco al figure si se requiere, 
    # pero en mapas suele preferirse limpio. Si quieres el recuadro negro estricto, descomenta abajo)
    # plt.box(True) 

    output_path = f'{OUTPUT_DIR}/mapa_urbano_zoom.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Mapa guardado en: {output_path}")

if __name__ == "__main__":
    # Silenciar warning de pandas sobre copias
    pd.options.mode.chained_assignment = None 
    main()