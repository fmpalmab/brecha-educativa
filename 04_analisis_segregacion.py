import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import unicodedata
import os
import requests
from shapely.geometry import box

# --- CONFIGURACIÓN GLOBAL ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores Corporativos
COLOR_FREE = '#2ca02c'  # Verde (Gratuito)
COLOR_PAID = '#d62728'  # Rojo (Pagado)

# Bounding Box Urbano
BBOX_COORDS = [-70.872116, -33.642527, -70.469742, -33.327552]

def normalizar_texto(texto):
    """Normaliza nombres para cruces de datos."""
    if pd.isna(texto): return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def descargar_shapefile():
    """Gestión automática del mapa base."""
    output_dir = 'data/external/comunas_hito2'
    os.makedirs(output_dir, exist_ok=True)
    base_url = "https://github.com/PLUMAS-research/visualization-course-materials/raw/master/data/comunas_rm/COMUNA_C17"
    shp_path = f"{output_dir}/COMUNA_C17.shp"
    
    if not os.path.exists(shp_path):
        print("⬇️ Descargando mapa base...")
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
            try:
                r = requests.get(f"{base_url}{ext}", allow_redirects=True)
                if r.status_code == 200:
                    with open(f"{output_dir}/COMUNA_C17{ext}", 'wb') as f:
                        f.write(r.content)
            except Exception: pass
    return shp_path

def main():
    print(">>> INICIANDO ANÁLISIS DE SEGREGACIÓN Y MERCADO (Bloque 4) <<<")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
        print(f"Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("❌ Error: Ejecuta el Bloque 1 primero.")
        return

    # 2. Clasificación Binaria
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )

    # -------------------------------------------------------------------------
    # PARTE A: ANÁLISIS DE MERCADO (Volumen)
    # -------------------------------------------------------------------------
    print("A. Generando Gráficos de Mercado (Barras)...")
    
    mercado = df.groupby(['NOM_COM_RBD', 'TIPO_PAGO']).agg({
        'RBD': 'count', 'MAT_TOTAL': 'mean'
    }).reset_index()
    mercado.columns = ['Comuna', 'Tipo', 'Cantidad', 'Tamano_Promedio']
    orden = df['NOM_COM_RBD'].value_counts().index

    # GRÁFICO 1: Oferta (Cantidad)
    plt.figure(figsize=(16, 8))
    sns.barplot(data=mercado, x='Comuna', y='Cantidad', hue='Tipo', order=orden, 
                palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID})
    plt.title('Oferta Educativa: Cantidad de Colegios por Comuna', fontsize=18)
    plt.xticks(rotation=90, fontsize=10); plt.xlabel(''); plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/08_mercado_oferta_cantidad.png', dpi=300); plt.close()

    # GRÁFICO 2: Tamaño (Masividad)
    plt.figure(figsize=(16, 8))
    sns.barplot(data=mercado, x='Comuna', y='Tamano_Promedio', hue='Tipo', order=orden, 
                palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID})
    plt.axhline(y=800, color='gray', linestyle='--', label='Alta Masividad (>800)')
    plt.title('Fenómeno de Masividad: Tamaño Promedio de Colegios', fontsize=18)
    plt.xticks(rotation=90, fontsize=10); plt.xlabel(''); plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/09_mercado_tamano_promedio.png', dpi=300); plt.close()

    # -------------------------------------------------------------------------
    # PARTE B: EL GRÁFICO FALTANTE (Sobrerrepresentación Oferta vs Demanda)
    # -------------------------------------------------------------------------
    print("B. Generando Gráfico de Sobrerrepresentación (Oferta vs Demanda)...")
    
    # Calcular totales por comuna para sacar porcentajes
    comuna_agg = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Total_Colegios': len(x),
        'Total_Matricula': x['MAT_TOTAL'].sum(),
        'Col_Pagados': (x['TIPO_PAGO'] == 'Pagado').sum(),
        'Mat_Pagada': x.loc[x['TIPO_PAGO'] == 'Pagado', 'MAT_TOTAL'].sum()
    }))
    
    # Calcular % Pagado en Oferta (Colegios) vs Demanda (Alumnos)
    comuna_agg['% Oferta Pagada'] = comuna_agg['Col_Pagados'] / comuna_agg['Total_Colegios']
    comuna_agg['% Demanda Pagada'] = comuna_agg['Mat_Pagada'] / comuna_agg['Total_Matricula']
    
    # Calcular % Gratuito (Complemento)
    comuna_agg['% Oferta Gratuita'] = 1 - comuna_agg['% Oferta Pagada']
    comuna_agg['% Demanda Gratuita'] = 1 - comuna_agg['% Demanda Pagada']
    
    # Ordenar por % de Oferta Pagada
    comuna_agg = comuna_agg.sort_values('% Oferta Pagada')
    
    # PLOT DOBLE (Izquierda: Oferta, Derecha: Demanda)
    fig, axes = plt.subplots(1, 2, figsize=(20, 12), sharey=True)
    
    # Gráfico 1: Estructura de la Oferta (Establecimientos)
    comuna_agg[['% Oferta Gratuita', '% Oferta Pagada']].plot(
        kind='barh', stacked=True, ax=axes[0], color=[COLOR_FREE, COLOR_PAID], width=0.8
    )
    axes[0].set_title('Estructura de la OFERTA (% Colegios)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Proporción (0 a 1)')
    axes[0].legend(loc='lower left', title='Tipo')
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    # Gráfico 2: Estructura de la Demanda (Matrícula)
    comuna_agg[['% Demanda Gratuita', '% Demanda Pagada']].plot(
        kind='barh', stacked=True, ax=axes[1], color=[COLOR_FREE, COLOR_PAID], width=0.8
    )
    axes[1].set_title('Estructura de la DEMANDA (% Alumnos)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Proporción (0 a 1)')
    axes[1].legend().remove() # Leyenda redundante
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.suptitle(
    'El Desajuste: ¿Dónde se concentra la matrícula vs la infraestructura?',
    fontsize=20,
    y=1)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/10_sobrerrepresentacion_oferta_demanda.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PARTE C: MAPAS COMPARATIVOS (Geoespacial)
    # -------------------------------------------------------------------------
    print("C. Generando Mapas Comparativos...")
    
    geo_agg = df.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])[['MAT_TOTAL', 'DC_TOT']].sum().reset_index()
    geo_agg['RATIO_REAL'] = geo_agg['MAT_TOTAL'] / geo_agg['DC_TOT']
    map_pivot = geo_agg.pivot(index='NOM_COM_RBD', columns='TIPO_PAGO', values='RATIO_REAL').reset_index()
    
    shp_path = descargar_shapefile()
    if os.path.exists(shp_path):
        try:
            gdf = gpd.read_file(shp_path)
            if gdf.crs.to_string() != "EPSG:4326": gdf = gdf.to_crs(epsg=4326)
            
            col_name = 'NOM_COMUNA' if 'NOM_COMUNA' in gdf.columns else gdf.columns[0]
            gdf['nombre_norm'] = gdf[col_name].apply(normalizar_texto)
            
            zona_urbana = box(BBOX_COORDS[0], BBOX_COORDS[1], BBOX_COORDS[2], BBOX_COORDS[3])
            gdf_zoom = gpd.clip(gdf, zona_urbana)
            gdf_final = gdf_zoom.merge(map_pivot, left_on='nombre_norm', right_on='NOM_COM_RBD', how='left')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
            vmin, vmax = 15, 35 
            
            for ax, col, title, color_t in [(ax1, 'Gratuito', 'Sector GRATUITO', COLOR_FREE), 
                                            (ax2, 'Pagado', 'Sector PAGADO', COLOR_PAID)]:
                gdf_final.plot(column=col, ax=ax, cmap='magma_r', vmin=vmin, vmax=vmax, 
                              legend=True, legend_kwds={'shrink': 0.4}, edgecolor='gray', linewidth=0.5,
                              missing_kwds={'color': 'lightgrey'})
                ax.set_title(title, fontsize=16, fontweight='bold', color=color_t); ax.axis('off')
                
                for _, row in gdf_final.iterrows():
                    if row.geometry.area > 0.0015:
                        txt = row['nombre_norm'].title().replace("Santiago", "Stgo")
                        ax.annotate(txt, xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                                    ha='center', fontsize=8, path_effects=[pe.withStroke(linewidth=2, foreground="white")])

            plt.suptitle('Brecha Territorial: Carga Docente Real', fontsize=20, y=0.95); plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/11_mapa_comparativo_segregacion.png', dpi=300); plt.close()
        except Exception as e: print(f"⚠️ Error mapas: {e}")

    print(f"✅ Gráficos generados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None 
    main()