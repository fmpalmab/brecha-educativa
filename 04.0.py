import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
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
    print("--- Generando Mapa: Gratuito vs. Pagado ---")
    
    # 1. CARGAR DATOS
    archivo_datos = 'data/processed/base_consolidada_rm_2024.csv'
    if not os.path.exists(archivo_datos):
        print("⚠️ No encuentro 'base_consolidada_rm_2024.csv'.")
        return
    df = pd.read_csv(archivo_datos)
    
    # 2. CLASIFICACIÓN POR PAGO
    print("Clasificando colegios por tipo de pago...")
    
    # Filtramos 'SIN INFORMACION' para no ensuciar el análisis
    df = df[df['PAGO_MENSUAL'] != 'SIN INFORMACION'].copy()
    
    # Creamos la categoría binaria
    df['tipo_pago_binario'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'GRATUITO' if str(x).strip().upper() == 'GRATUITO' else 'PAGADO'
    )
    
    # Agregamos por Comuna y Tipo de Pago
    map_data = df.groupby(['NOM_COM_RBD', 'tipo_pago_binario'])['ratio_alumno_docente'].mean().reset_index()
    
    # Pivoteamos para tener columnas separadas
    map_pivot = map_data.pivot(index='NOM_COM_RBD', columns='tipo_pago_binario', values='ratio_alumno_docente')
    map_pivot.reset_index(inplace=True)
    
    # 3. CARGAR Y PREPARAR MAPA
    shp_path = descargar_shapefile_hito2()
    try:
        gdf_comunas = gpd.read_file(shp_path)
    except:
        print("❌ Error cargando shapefile.")
        return

    # Normalizar nombres del mapa
    col_nombre_mapa = 'NOM_COMUNA' if 'NOM_COMUNA' in gdf_comunas.columns else gdf_comunas.columns[0]
    gdf_comunas['nombre_norm'] = gdf_comunas[col_nombre_mapa].apply(normalizar_texto)
    
    # 4. APLICAR ZOOM URBANO (Tus coordenadas)
    print("✂️ Recortando mapa...")
    # Southwest: -33.642527, -70.872116
    # Northeast: -33.327552, -70.469742
    bbox_coords = [-70.872116, -33.642527, -70.469742, -33.327552]
    
    if gdf_comunas.crs.to_string() != "EPSG:4326":
        gdf_comunas = gdf_comunas.to_crs(epsg=4326)
    
    zona_urbana = box(bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3])
    gdf_santiago_urbano = gpd.clip(gdf_comunas, zona_urbana)
    
    # 5. MERGE FINAL
    gdf_final = gdf_santiago_urbano.merge(map_pivot, left_on='nombre_norm', right_on='NOM_COM_RBD', how='left')
    
    # 6. VISUALIZACIÓN COMPARATIVA
    print("Dibujando mapas...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12), sharex=True, sharey=True)
    
    # Configuración de colores común
    cmap = 'RdYlBu_r' # Rojo=Malo, Azul=Bueno
    vmin, vmax = 10, 30 # Rango fijo para que sean comparables visualmente
    
    # Mapa 1: Colegios Gratuitos
    gdf_final.plot(
        column='GRATUITO', 
        ax=ax1, 
        cmap=cmap, 
        legend=True, 
        vmin=vmin, vmax=vmax,
        legend_kwds={'label': "Alumnos por Docente", 'shrink': 0.5},
        missing_kwds={'color': 'lightgrey', 'hatch': '///', 'label': 'Sin datos'}
    )
    ax1.set_title('Colegios GRATUITOS', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Mapa 2: Colegios Pagados (Cualquier monto)
    gdf_final.plot(
        column='PAGADO', 
        ax=ax2, 
        cmap=cmap, 
        legend=True,
        vmin=vmin, vmax=vmax,
        legend_kwds={'label': "Alumnos por Docente", 'shrink': 0.5},
        missing_kwds={'color': 'lightgrey', 'hatch': '///', 'label': 'Sin datos'}
    )
    ax2.set_title('Colegios PAGADOS (Copago o Particular)', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Etiquetas
    for ax in [ax1, ax2]:
        for idx, row in gdf_final.iterrows():
            if row.geometry.area > 0.001: # Filtro de tamaño para no saturar
                txt = row['nombre_norm'].title()
                # Abreviaciones
                if "PEDRO AGUIRRE" in txt.upper(): txt = "PAC"
                if "ESTACION CENTRAL" in txt.upper(): txt = "Est. Central"
                if "SANTIAGO" == txt.upper(): txt = "Stgo"
                
                ax.annotate(
                    text=txt, 
                    xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    horizontalalignment='center',
                    fontsize=8,
                    color='black',
                    weight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")]
                )

    plt.suptitle('La Barrera del Precio: Carga Docente en Educación Gratuita vs Pagada', fontsize=20, y=0.92)
    plt.tight_layout()
    
    os.makedirs('figures/finales', exist_ok=True)
    output_path = 'figures/finales/05_mapa_gratuidad_vs_pago.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ ¡Mapa Comparativo guardado en: {output_path}!")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None 
    main()