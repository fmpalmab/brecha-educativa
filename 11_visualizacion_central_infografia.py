import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
import os
import requests
from shapely.geometry import box

# Intentar importar contextily para mapa base (opcional pero recomendado)
try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("Nota: 'contextily' no instalado. Mapa sin fondo satelital.")

# --- CONFIGURACIÓN GLOBAL ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("white") # Estilo limpio para mapa

# Colores Corporativos
COLOR_FREE = '#2ca02c'  # Verde (Gratuito)
COLOR_PAID = '#d62728'  # Rojo (Pagado)

# Bounding Box Urbano
BBOX_COORDS = [-70.872116, -33.642527, -70.469742, -33.327552]

def descargar_shapefile():
    """Descarga shapefile de comunas."""
    output_dir = 'data/external/comunas_hito2'
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://github.com/PLUMAS-research/visualization-course-materials/raw/master/data/comunas_rm/COMUNA_C17"
    shp_path = f"{output_dir}/COMUNA_C17.shp"
    if not os.path.exists(shp_path):
        print("⬇️ Descargando mapa base...")
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            try:
                r = requests.get(f"{base_url}{ext}", allow_redirects=True)
                with open(f"{output_dir}/COMUNA_C17{ext}", 'wb') as f: f.write(r.content)
            except: pass
    return shp_path

def main():
    print(">>> GENERANDO VISUALIZACIÓN CENTRAL (INFOGRAFÍA) <<<")
    
    # 1. Cargar Datos Procesados
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
        print(f"Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("❌ Error: Falta el archivo de datos. Ejecuta Bloque 1.")
        return

    # 2. Preparar Datos para Visualización
    # Clasificación Binaria de Pago
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )
    # Calcular SIMCE Promedio Global (para tamaño del punto)
    # Usamos un promedio simple de los promedios disponibles
    df['SIMCE_SCORE'] = df[['SIMCE_4B_AVG', 'SIMCE_2M_AVG']].mean(axis=1)
    df_plot = df.dropna(subset=['SIMCE_SCORE', 'LATITUD', 'LONGITUD']).copy()

    # Normalizar Score para tamaño (Escalar entre 20 y 200 para que sea visible)
    min_score = df_plot['SIMCE_SCORE'].min()
    max_score = df_plot['SIMCE_SCORE'].max()
    df_plot['SIZE'] = ((df_plot['SIMCE_SCORE'] - min_score) / (max_score - min_score) * 180 + 20)

    # 3. Cargar Mapa Base (Bordes Comunales)
    shp_path = descargar_shapefile()
    try:
        gdf_comunas = gpd.read_file(shp_path)
        if gdf_comunas.crs.to_string() != "EPSG:4326":
            gdf_comunas = gdf_comunas.to_crs(epsg=4326)
        # Recorte al BBOX urbano
        zona_urbana = box(BBOX_COORDS[0], BBOX_COORDS[1], BBOX_COORDS[2], BBOX_COORDS[3])
        gdf_zoom = gpd.clip(gdf_comunas, zona_urbana)
    except Exception as e:
        print(f"❌ Error cargando shapefile: {e}")
        return

    # 4. GENERAR EL MAPA FINAL
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    
    # A. Dibujar Bordes Comunales (Fondo)
    gdf_zoom.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.8, alpha=0.5, zorder=1)

    # B. Dibujar Puntos (Colegios)
    # Separar por tipo para la leyenda
    gratuitos = df_plot[df_plot['TIPO_PAGO'] == 'Gratuito']
    pagados = df_plot[df_plot['TIPO_PAGO'] == 'Pagado']
    
    # Puntos Gratuitos (Verde)
    ax.scatter(
        gratuitos['LONGITUD'], gratuitos['LATITUD'], 
        s=gratuitos['SIZE'], c=COLOR_FREE, 
        alpha=0.7, edgecolor='white', linewidth=0.5, label='Gratuito (Público/Subv)', zorder=2
    )
    # Puntos Pagados (Rojo) - Dibujar encima para resaltar
    ax.scatter(
        pagados['LONGITUD'], pagados['LATITUD'], 
        s=pagados['SIZE'], c=COLOR_PAID, 
        alpha=0.8, edgecolor='white', linewidth=0.5, label='Pagado (Privado/Copago)', zorder=3
    )

    # C. Añadir Mapa Base de Fondo (Si contextily está disponible)
    if HAS_CTX:
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, alpha=0.6, zorder=0)
        except Exception: pass

    # D. Etiquetas de Comunas Clave (Opcional, para referencia)
    # Comunas representativas del oriente y la periferia
    comunas_clave = ['LAS CONDES', 'VITACURA', 'PROVIDENCIA', 'SANTIAGO', 'MAIPU', 'PUENTE ALTO', 'LA FLORIDA']
    for _, row in gdf_zoom.iterrows():
        nombre_norm = str(row.get('NOM_COMUNA', '')).upper()
        if any(clave in nombre_norm for clave in comunas_clave) and row.geometry.area > 0.002:
            txt = row['NOM_COMUNA'].title().replace("Santiago", "Stgo")
            ax.annotate(txt, xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                        ha='center', fontsize=10, color='black', weight='bold',
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=4)

    # E. Configuración Final y Leyenda
    ax.set_title('La Geografía de la Desigualdad Educativa en Santiago', fontsize=22, fontweight='bold', pad=20)
    ax.set_xlim(BBOX_COORDS[0], BBOX_COORDS[2])
    ax.set_ylim(BBOX_COORDS[1], BBOX_COORDS[3])
    ax.axis('off')
    
    # Leyenda de Color
    leg = ax.legend(title='Tipo de Financiamiento', title_fontsize=12, fontsize=11, loc='upper left', frameon=True)
    leg.get_frame().set_alpha(0.9)
    
    # Leyenda de Tamaño (Truco para mostrar la escala)
    # Creamos puntos "falsos" para la leyenda
    s_min = (220 - min_score) / (max_score - min_score) * 180 + 20
    s_med = (270 - min_score) / (max_score - min_score) * 180 + 20
    s_max = (320 - min_score) / (max_score - min_score) * 180 + 20
    
    l1 = plt.scatter([],[], s=s_min, c='gray', alpha=0.6, edgecolor='none')
    l2 = plt.scatter([],[], s=s_med, c='gray', alpha=0.6, edgecolor='none')
    l3 = plt.scatter([],[], s=s_max, c='gray', alpha=0.6, edgecolor='none')
    
    legend2 = ax.legend([l1, l2, l3], ['Bajo (~220)', 'Medio (~270)', 'Alto (~320)'], 
                        title='Puntaje SIMCE Promedio (Tamaño)', 
                        title_fontsize=12, fontsize=10, loc='lower left', frameon=True, labelspacing=1.2)
    legend2.get_frame().set_alpha(0.9)
    ax.add_artist(leg) # Volver a añadir la primera leyenda

    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/19_visualizacion_central_infografia.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualización central guardada en: {save_path}")

if __name__ == "__main__":
    main()