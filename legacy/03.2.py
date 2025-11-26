import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe  # <--- IMPORTANTE: Esta librería faltaba antes
import seaborn as sns
import unicodedata
import os
import requests
from shapely.geometry import box

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
sns.set_theme(style="whitegrid")

def normalizar_texto(texto):
    """Normaliza nombres para cruzar datos (ej: Ñuñoa -> NUNOA)"""
    if pd.isna(texto): return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def descargar_shapefile_hito2():
    """Descarga el shapefile específico del curso (COMUNA_C17)"""
    output_dir = 'data/external/comunas_hito2'
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://github.com/PLUMAS-research/visualization-course-materials/raw/master/data/comunas_rm/COMUNA_C17"
    extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    
    print("⬇️ Verificando mapa base...")
    for ext in extensions:
        filepath = f"{output_dir}/COMUNA_C17{ext}"
        if not os.path.exists(filepath):
            try:
                print(f"   - Descargando {ext}...")
                r = requests.get(f"{base_url}{ext}", allow_redirects=True)
                if r.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
            
    return f"{output_dir}/COMUNA_C17.shp"

def main():
    print("--- Generando Mapa Urbano (Zoom Ajustado) ---")
    
    # 1. CARGAR DATOS
    archivo_datos = 'data/processed/base_consolidada_rm_2024.csv'
    if not os.path.exists(archivo_datos):
        print("⚠️ No encuentro 'base_consolidada_rm_2024.csv'.")
        return
    df = pd.read_csv(archivo_datos)
    
    # Agregar por comuna
    df_agg = df.groupby('NOM_COM_RBD')['ratio_alumno_docente'].mean().reset_index()
    df_agg.rename(columns={'NOM_COM_RBD': 'Comuna_Norm', 'ratio_alumno_docente': 'valor'}, inplace=True)
    
    # 2. CARGAR MAPA
    shp_path = descargar_shapefile_hito2()
    try:
        gdf_comunas = gpd.read_file(shp_path)
    except:
        print("❌ No se pudo cargar el shapefile.")
        return

    # 3. NORMALIZAR NOMBRES (Para que pinte los colores)
    # Buscamos la columna del nombre
    col_nombre_mapa = 'NOM_COMUNA' if 'NOM_COMUNA' in gdf_comunas.columns else gdf_comunas.columns[0]
    print(f"Normalizando nombres del mapa usando columna: {col_nombre_mapa}...")
    
    gdf_comunas['nombre_norm'] = gdf_comunas[col_nombre_mapa].apply(normalizar_texto)
    
    # 4. APLICAR TUS NUEVAS COORDENADAS (ZOOM)
    print("✂️ Recortando mapa a tus coordenadas...")
    
    # Coordenadas que me diste:
    # Southwest: -33.642527, -70.872116
    # Northeast: -33.327552, -70.469742
    # Formato Box: [minx (Oeste), miny (Sur), maxx (Este), maxy (Norte)]
    bbox_coords = [-70.852116, -33.642527, -70.489742, -33.334552]
    
    # Aseguramos proyección
    if gdf_comunas.crs.to_string() != "EPSG:4326":
        gdf_comunas = gdf_comunas.to_crs(epsg=4326)
    
    zona_urbana = box(bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3])
    gdf_santiago_urbano = gpd.clip(gdf_comunas, zona_urbana)
    
    print(f"   -> Comunas visibles: {len(gdf_santiago_urbano)}")

    # 5. MERGE (Unir datos con mapa recortado)
    gdf_final = gdf_santiago_urbano.merge(df_agg, left_on='nombre_norm', right_on='Comuna_Norm', how='left')
    
    # 6. VISUALIZACIÓN
    print("Dibujando mapa...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Pintar
    gdf_final.plot(
        column='valor',
        cmap='YlOrRd', # Amarillo a Rojo
        linewidth=0.5,
        ax=ax,
        edgecolor='black',
        legend=True,
        legend_kwds={'label': "Alumnos por Docente (Promedio)", 'shrink': 0.6},
        missing_kwds={'color': 'lightgrey', 'hatch': '///', 'label': 'Sin datos'}
    )
    
    # Etiquetas con borde blanco (PathEffects)
    for idx, row in gdf_final.iterrows():
        if pd.notnull(row['valor']) and row.geometry.area > 0.001:
            txt = row['nombre_norm'].title()
            # Abreviar nombres largos para que quepan
            if "PEDRO AGUIRRE" in txt.upper(): txt = "PAC"
            if "ESTACION CENTRAL" in txt.upper(): txt = "Est. Central"
            if "SANTIAGO" == txt.upper(): txt = "Stgo"
            
            plt.annotate(
                text=txt, 
                xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                horizontalalignment='center',
                fontsize=9,
                color='black',
                weight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            )

    ax.set_title("Carga Docente: Zoom Gran Santiago", fontsize=18, fontweight='bold', pad=20)
    ax.set_axis_off()
    
    # Guardar
    os.makedirs('figures/finales', exist_ok=True)
    output_path = 'figures/finales/mapa_urbano_zoom.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ ¡Mapa guardado en: {output_path}!")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None 
    main()