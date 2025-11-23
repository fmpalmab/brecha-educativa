import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import os
import requests
import numpy as np

# =============================================================================
# 1. FUNCIONES DE UTILIDAD (Descarga y Normalización)
# =============================================================================

def normalizar_texto(texto):
    """Estandariza nombres: Mayúsculas, sin tildes, sin ñ."""
    if pd.isna(texto): return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def descargar_shapefile_hito2():
    """Descarga el shapefile específico que usaste en el Hito 2 (COMUNA_C17)"""
    output_dir = 'data/external/comunas_hito2'
    os.makedirs(output_dir, exist_ok=True)
    
    base_url = "https://github.com/PLUMAS-research/visualization-course-materials/raw/master/data/comunas_rm/COMUNA_C17"
    extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
    
    print("⬇️ Descargando mapa del curso...")
    for ext in extensions:
        try:
            r = requests.get(f"{base_url}{ext}", allow_redirects=True)
            if r.status_code == 200:
                with open(f"{output_dir}/COMUNA_C17{ext}", 'wb') as f:
                    f.write(r.content)
        except Exception as e:
            print(f"Advertencia descargando {ext}: {e}")
            
    return f"{output_dir}/COMUNA_C17.shp"

# =============================================================================
# 2. PROCESO PRINCIPAL
# =============================================================================

def main():
    print("--- Generando Mapa Hito 3 (Corregido) ---")
    
    # 1. CARGAR DATOS (Tu base consolidada)
    df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
    
    # Agregar por comuna (Calculamos el promedio del ratio)
    # Asegúrate de usar los nombres de columnas que existen en tu CSV
    df_agg = df.groupby('NOM_COM_RBD')['ratio_alumno_docente'].mean().reset_index()
    df_agg.rename(columns={'NOM_COM_RBD': 'Comuna_Norm', 'ratio_alumno_docente': 'valor'}, inplace=True)
    
    # 2. CARGAR MAPA (Shapefile)
    shp_path = descargar_shapefile_hito2()
    try:
        gdf_comunas = gpd.read_file(shp_path)
    except:
        print("❌ No se pudo cargar el shapefile. Verifica la descarga.")
        return

    # 3. EL FIX: NORMALIZAR EL MAPA PARA QUE CRUCE
    # Buscamos la columna de nombre en el shapefile (suele ser NOM_COMUNA)
    col_nombre_mapa = 'NOM_COMUNA' if 'NOM_COMUNA' in gdf_comunas.columns else gdf_comunas.columns[0]
    
    print(f"Normalizando nombres del mapa (Columna: {col_nombre_mapa})...")
    gdf_comunas['nombre_norm'] = gdf_comunas[col_nombre_mapa].apply(normalizar_texto)
    
    # Filtramos solo la RM (para evitar pintar todo Chile si el shp es nacional)
    # Usamos un bounding box aproximado de Santiago o filtramos por nombres conocidos
    comunas_santiago = df_agg['Comuna_Norm'].unique()
    gdf_santiago = gdf_comunas[gdf_comunas['nombre_norm'].isin(comunas_santiago)].copy()
    
    # 4. MERGE (CRUCE)
    # Ahora cruzamos 'nombre_norm' del mapa con 'Comuna_Norm' de los datos
    gdf_final = gdf_santiago.merge(df_agg, left_on='nombre_norm', right_on='Comuna_Norm', how='left')
    
    print(f"Comunas con datos encontrados: {gdf_final['valor'].notnull().sum()} de {len(gdf_final)}")

    # 5. VISUALIZACIÓN (CLOROPLETAS)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    gdf_final.plot(
        column='valor',
        cmap='YlOrRd',      # Amarillo a Rojo (Rojo = Más alumnos por profe = Peor)
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        legend_kwds={'label': "Promedio Alumnos por Docente", 'orientation': "horizontal", 'shrink': 0.7},
        missing_kwds={'color': 'lightgrey', 'hatch': '///', 'label': 'Sin datos'}
    )
    
    ax.set_title("Distribución de la Carga Docente por Comuna", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Guardar
    os.makedirs('figures/finales', exist_ok=True)
    output_path = 'figures/finales/mapa_corregido_hito3.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ ¡Mapa generado exitosamente en: {output_path}!")

if __name__ == "__main__":
    main()