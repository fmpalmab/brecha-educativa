import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
import geopandas as gpd

# Intentar importar contextily para mapas base (opcional)
try:
    import contextily as ctx
    HAS_CTX = True
except ImportError:
    HAS_CTX = False
    print("Nota: 'contextily' no está instalado. Los mapas no tendrán imagen satelital de fondo, solo bordes comunales.")

# --- 1. CONFIGURACIÓN GLOBAL ---
OUTPUT_DIR = 'figures/presentacion_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
COLOR_FREE = '#2ca02c'  # Verde
COLOR_PAID = '#d62728'  # Rojo
PALETTE_MAPS = 'magma_r' 

# Bounding Box [min_lon, min_lat, max_lon, max_lat]
BBOX = [-70.872116, -33.642527, -70.469742, -33.327552]

def find_file(filename):
    for root, dirs, files in os.walk('.'):
        if filename in files:
            return os.path.join(root, filename)
    return None

def clean_coord(val):
    if isinstance(val, str):
        val = val.replace(',', '.')
    return pd.to_numeric(val, errors='coerce')

def load_data():
    print(">>> Cargando datos y shapefiles...")
    
    # 1. Cargar CSVs
    path_base = find_file('base_consolidada_rm_2024.csv')
    path_s4 = find_file('simce4b2024_rbd_preliminar.csv')
    path_s2 = find_file('simce2m2024_rbd_preliminar.csv')
    # Buscar shapefile de comunas
    path_shp = find_file('COMUNA_C17.shp')
    
    if not all([path_base, path_s4, path_s2]):
        raise FileNotFoundError("Faltan archivos CSV.")

    df_base = pd.read_csv(path_base)
    df_base['LATITUD'] = df_base['LATITUD'].apply(clean_coord)
    df_base['LONGITUD'] = df_base['LONGITUD'].apply(clean_coord)
    df_base['RBD'] = pd.to_numeric(df_base['RBD'], errors='coerce')

    # Procesar SIMCE
    df_s4 = pd.read_csv(path_s4, sep=';', encoding='latin-1')
    cols_4b = ['rbd', 'prom_lect4b_rbd', 'prom_mate4b_rbd']
    df_s4 = df_s4[cols_4b].rename(columns={'rbd':'RBD', 'prom_lect4b_rbd':'S4L', 'prom_mate4b_rbd':'S4M'})
    for c in ['S4L','S4M']: df_s4[c] = pd.to_numeric(df_s4[c], errors='coerce')
    df_s4['SIMCE_4B_AVG'] = df_s4[['S4L', 'S4M']].mean(axis=1)

    df_s2 = pd.read_csv(path_s2, sep=';', encoding='latin-1')
    cols_2m = ['rbd', 'prom_lect2m_rbd', 'prom_mate2m_rbd']
    df_s2 = df_s2[cols_2m].rename(columns={'rbd':'RBD', 'prom_lect2m_rbd':'S2L', 'prom_mate2m_rbd':'S2M'})
    for c in ['S2L','S2M']: df_s2[c] = pd.to_numeric(df_s2[c], errors='coerce')
    df_s2['SIMCE_2M_AVG'] = df_s2[['S2L', 'S2M']].mean(axis=1)

    df = df_base.merge(df_s4, on='RBD', how='left').merge(df_s2, on='RBD', how='left')

    # Clasificar Pago
    def classify_pago(row):
        pago = str(row['PAGO_MENSUAL']).upper()
        dep = str(row['categoria_dependencia']).upper()
        if pago == 'GRATUITO': return 'Gratuito'
        if pago == 'SIN INFORMACION' and any(x in dep for x in ['MUNICIPAL', 'SLEP', 'ADMIN']): return 'Gratuito'
        return 'Pagado'
    df['TIPO_PAGO'] = df.apply(classify_pago, axis=1)
    # Variable binaria para correlaciones (0=Gratuito, 1=Pagado)
    df['IS_PAID'] = df['TIPO_PAGO'].apply(lambda x: 1 if x == 'Pagado' else 0)

    # Filtro BBOX
    df = df.dropna(subset=['LATITUD', 'LONGITUD'])
    mask_bbox = (
        (df['LONGITUD'] >= BBOX[0]) & (df['LONGITUD'] <= BBOX[2]) & 
        (df['LATITUD'] >= BBOX[1]) & (df['LATITUD'] <= BBOX[3])
    )
    df = df[mask_bbox].copy()

    # Cargar Shapefile
    gdf_comunas = None
    if path_shp:
        try:
            gdf_comunas = gpd.read_file(path_shp)
            # Asegurar proyección Lat/Lon para coincidir con CSV
            if gdf_comunas.crs != 'EPSG:4326':
                gdf_comunas = gdf_comunas.to_crs('EPSG:4326')
            # Recortar shapefile al BBOX para que no sature el gráfico
            gdf_comunas = gdf_comunas.cx[BBOX[0]:BBOX[2], BBOX[1]:BBOX[3]]
        except Exception as e:
            print(f"Error cargando shapefile: {e}")
    
    return df, gdf_comunas

def add_basemap(ax, gdf_shape):
    """Agrega bordes de comunas y mapa base si es posible."""
    # 1. Bordes de Comunas (Prioridad)
    if gdf_shape is not None:
        gdf_shape.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.8, alpha=0.5, zorder=1)
    
    # 2. Mapa Base (Contextily)
    if HAS_CTX:
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, alpha=0.6)
        except Exception:
            pass # Fallar silenciosamente si no hay internet o error de CRS

def calc_r(df, col_x, col_y):
    """Calcula Pearson R ignorando NaNs."""
    clean = df[[col_x, col_y]].dropna()
    if len(clean) > 2:
        r, p = stats.pearsonr(clean[col_x], clean[col_y])
        return r
    return 0.0

def generate_visualizations(df, gdf_comunas):
    print(">>> Generando gráficos con bordes y valores R...")

    # --- MAPAS ---
    # Helper para mapas comunes
    def plot_map(data, col_val, title, filename, cmap=None, categorical=False):
        f, ax = plt.subplots(figsize=(12, 10))
        # Fondo
        add_basemap(ax, gdf_comunas)
        
        if categorical:
            # Gratuito
            sub_f = data[data[col_val] == 'Gratuito']
            ax.scatter(sub_f['LONGITUD'], sub_f['LATITUD'], c=COLOR_FREE, label='Gratuito', s=25, alpha=0.8, zorder=2)
            # Pagado
            sub_p = data[data[col_val] == 'Pagado']
            ax.scatter(sub_p['LONGITUD'], sub_p['LATITUD'], c=COLOR_PAID, label='Pagado', s=25, alpha=0.8, zorder=2)
            ax.legend(loc='upper right')
        else:
            # Continuo (SIMCE)
            sc = ax.scatter(data['LONGITUD'], data['LATITUD'], c=data[col_val], cmap=cmap, s=30, alpha=0.9, edgecolor='k', linewidth=0.1, zorder=2)
            plt.colorbar(sc, label='Puntaje')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlim(BBOX[0], BBOX[2])
        ax.set_ylim(BBOX[1], BBOX[3])
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=150)
        plt.close()

    # 1. Mapas SIMCE
    plot_map(df.dropna(subset=['SIMCE_4B_AVG']), 'SIMCE_4B_AVG', 'Rendimiento SIMCE 4° Básico 2024', '01_mapa_simce_4b.png', cmap=PALETTE_MAPS)
    plot_map(df.dropna(subset=['SIMCE_2M_AVG']), 'SIMCE_2M_AVG', 'Rendimiento SIMCE II Medio 2024', '02_mapa_simce_2m.png', cmap=PALETTE_MAPS)
    
    # 2. Mapa Segregación
    plot_map(df, 'TIPO_PAGO', 'Segregación Territorial: Oferta Pagada vs Gratuita', '03_mapa_segregacion.png', categorical=True)

    # --- GRÁFICOS ESTADÍSTICOS (CON R) ---
    
    # 3. Brecha Resultados (Barras)
    # Calculamos R global para cada nivel
    r_4b = calc_r(df, 'IS_PAID', 'SIMCE_4B_AVG')
    r_2m = calc_r(df, 'IS_PAID', 'SIMCE_2M_AVG')
    
    df_long = pd.melt(df, id_vars=['TIPO_PAGO'], value_vars=['SIMCE_4B_AVG', 'SIMCE_2M_AVG'], var_name='Nivel', value_name='Puntaje')
    df_long['Nivel'] = df_long['Nivel'].replace({'SIMCE_4B_AVG': '4° Básico', 'SIMCE_2M_AVG': 'II Medio'})
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_long, x='Nivel', y='Puntaje', hue='TIPO_PAGO', palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID})
    plt.ylim(200, 340)
    plt.title(f'Brecha SIMCE por Dependencia\n(Corr. Pago-Puntaje: 4°B R={r_4b:.2f}, IIM R={r_2m:.2f})')
    plt.savefig(f'{OUTPUT_DIR}/04_brecha_resultados.png', dpi=150); plt.close()

    # 4. Ranking Comunal (Dumbbell) + Correlación Comunal
    df['SIMCE_GLOBAL'] = df[['SIMCE_4B_AVG', 'SIMCE_2M_AVG']].mean(axis=1)
    
    # Crear dataset comunal para calcular R comunal
    comuna_stats = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'PCT_PAGADO': (x['IS_PAID'].mean()),
        'SIMCE_COMUNAL': x['SIMCE_GLOBAL'].mean()
    }))
    r_comunal = calc_r(comuna_stats, 'PCT_PAGADO', 'SIMCE_COMUNAL')

    # Preparar datos para plot
    stats_pivot = df.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])['SIMCE_GLOBAL'].mean().unstack()
    if 'Pagado' in stats_pivot and 'Gratuito' in stats_pivot:
        stats_pivot['Brecha'] = stats_pivot['Pagado'] - stats_pivot['Gratuito']
        top_stats = stats_pivot.sort_values('Brecha').dropna().tail(15)
        
        plt.figure(figsize=(10, 10))
        plt.hlines(y=top_stats.index, xmin=top_stats['Gratuito'], xmax=top_stats['Pagado'], color='gray', alpha=0.5)
        plt.scatter(top_stats['Gratuito'], top_stats.index, color=COLOR_FREE, s=100, label='Gratuito')
        plt.scatter(top_stats['Pagado'], top_stats.index, color=COLOR_PAID, s=100, label='Pagado')
        plt.title(f'Top 15 Brechas Comunales\n(Correlación Comunal %Pago vs SIMCE: R={r_comunal:.2f})')
        plt.xlabel('Puntaje Promedio General')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/05_ranking_brecha_comunal.png', dpi=150); plt.close()

    # 5. Saturación Aulas
    r_aulas = calc_r(df, 'IS_PAID', 'ratio_alumno_curso')
    df_clean = df[df['ratio_alumno_curso'] < 60]
    
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df_clean, x='TIPO_PAGO', y='ratio_alumno_curso', palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID}, inner='quartile')
    plt.axhline(35, color='gray', linestyle='--')
    plt.title(f'Saturación de Aulas (Alumnos/Curso)\n(Corr. Pago-Tamaño: R={r_aulas:.2f})')
    plt.savefig(f'{OUTPUT_DIR}/06_saturacion_aulas.png', dpi=150); plt.close()

    # 6. Carga Docente
    r_doc = calc_r(df, 'IS_PAID', 'ratio_alumno_docente')
    df_clean_doc = df[df['ratio_alumno_docente'] < 50]
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_clean_doc, x='TIPO_PAGO', y='ratio_alumno_docente', palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID})
    plt.title(f'Carga Docente (Alumnos/Profesor)\n(Corr. Pago-Carga: R={r_doc:.2f})')
    plt.savefig(f'{OUTPUT_DIR}/07_carga_docente.png', dpi=150); plt.close()

if __name__ == "__main__":
    try:
        df, gdf = load_data()
        generate_visualizations(df, gdf)
        print(f"\n✅ Proceso completado. Gráficos en: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n❌ Error: {e}")