import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
import os
import requests
from shapely.geometry import box

# Intentar importar contextily para mapa base (opcional)
try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("Nota: 'contextily' no instalado. Mapa sin fondo satelital.")

# --- CONFIGURACIÓN GLOBAL ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("white")

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
    print(">>> GENERANDO VISUALIZACIÓN CENTRAL (MATRÍCULA) <<<")
    
    # 1. Cargar Datos Procesados
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
        print(f"Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("❌ Error: Falta el archivo de datos. Ejecuta Bloque 1.")
        return

    # 2. Preparar Datos
    # Clasificación Binaria
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )
    
    # Filtrar solo por matrícula válida (NO filtramos por SIMCE)
    df_plot = df.dropna(subset=['MAT_TOTAL', 'LATITUD', 'LONGITUD']).copy()
    df_plot = df_plot[df_plot['MAT_TOTAL'] > 0]

    print(f"Puntos a graficar: {len(df_plot)} (Todos con matrícula > 0)")

    # Normalizar Tamaño por Matrícula
    # Mapeamos la matrícula (ej. 50 a 4000 alumnos) a un tamaño de punto (20 a 300)
    min_mat = df_plot['MAT_TOTAL'].min()
    max_mat = df_plot['MAT_TOTAL'].max()
    
    # Fórmula de escalado lineal
    df_plot['SIZE'] = ((df_plot['MAT_TOTAL'] - min_mat) / (max_mat - min_mat) * 280 + 20)

    # 3. Cargar Mapa Base
    shp_path = descargar_shapefile()
    try:
        gdf_comunas = gpd.read_file(shp_path)
        if gdf_comunas.crs.to_string() != "EPSG:4326":
            gdf_comunas = gdf_comunas.to_crs(epsg=4326)
        zona_urbana = box(BBOX_COORDS[0], BBOX_COORDS[1], BBOX_COORDS[2], BBOX_COORDS[3])
        gdf_zoom = gpd.clip(gdf_comunas, zona_urbana)
    except Exception as e:
        print(f"❌ Error cargando shapefile: {e}")
        return

    # 4. GENERAR EL MAPA
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    
    # A. Fondo (Bordes)
    gdf_zoom.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.8, alpha=0.5, zorder=1)

    # B. Puntos (Colegios)
    # Separamos para controlar el orden de dibujo (zorder)
    gratuitos = df_plot[df_plot['TIPO_PAGO'] == 'Gratuito']
    pagados = df_plot[df_plot['TIPO_PAGO'] == 'Pagado']
    
    # Scatter Gratuito (Verde)
    ax.scatter(
        gratuitos['LONGITUD'], gratuitos['LATITUD'], 
        s=gratuitos['SIZE'], c=COLOR_FREE, 
        alpha=0.6, edgecolor='white', linewidth=0.5, label='Gratuito', zorder=2
    )
    # Scatter Pagado (Rojo)
    ax.scatter(
        pagados['LONGITUD'], pagados['LATITUD'], 
        s=pagados['SIZE'], c=COLOR_PAID, 
        alpha=0.7, edgecolor='white', linewidth=0.5, label='Pagado', zorder=3
    )

    # C. Mapa Base (Contextily)
    if HAS_CTX:
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, alpha=0.6, zorder=0)
        except: pass

    # D. Etiquetas Comunas (Para referencia)
    comunas_clave = ['LAS CONDES', 'VITACURA', 'PROVIDENCIA', 'SANTIAGO', 'MAIPU', 'PUENTE ALTO', 'LA FLORIDA', 'QUILICURA', 'SAN BERNARDO']
    for _, row in gdf_zoom.iterrows():
        nom = str(row.get('NOM_COMUNA', '')).upper()
        if any(c in nom for c in comunas_clave) and row.geometry.area > 0.002:
            txt = row['NOM_COMUNA'].title().replace("Santiago", "Stgo")
            ax.annotate(txt, xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                        ha='center', fontsize=10, color='black', weight='bold',
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")], zorder=4)

    # E. Configuración Final
    ax.set_title('Mapa de la Matrícula Escolar en Santiago (2024)', fontsize=22, fontweight='bold', pad=20)
    ax.set_xlim(BBOX_COORDS[0], BBOX_COORDS[2])
    ax.set_ylim(BBOX_COORDS[1], BBOX_COORDS[3])
    ax.axis('off')
    
    # Leyenda de Color
    leg = ax.legend(title='Tipo de Financiamiento', title_fontsize=12, fontsize=11, loc='upper left', frameon=True)
    leg.get_frame().set_alpha(0.9)
    
    # Leyenda de Tamaño (Matrícula)
    # Creamos puntos "falsos" para representar la escala de alumnos
    sizes_leg = [500, 1500, 3000]
    labels_leg = ['500 Alumnos', '1.500 Alumnos', '3.000 Alumnos']
    proxies = []
    for sz in sizes_leg:
        # Aplicamos la misma fórmula de escalado
        s_scaled = (sz - min_mat) / (max_mat - min_mat) * 280 + 20
        proxies.append(plt.scatter([],[], s=s_scaled, c='gray', alpha=0.6, edgecolor='none'))
    
    legend2 = ax.legend(proxies, labels_leg, title='Matrícula Total (Tamaño)', 
                        title_fontsize=12, fontsize=10, loc='lower left', frameon=True, labelspacing=1.5)
    legend2.get_frame().set_alpha(0.9)
    ax.add_artist(leg) # Volver a añadir la leyenda de color

    plt.tight_layout()
    save_path = f'{OUTPUT_DIR}/20_visualizacion_central_matricula.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualización guardada en: {save_path}")

if __name__ == "__main__":
    main()