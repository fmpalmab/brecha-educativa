import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# --- RUTAS DE ARCHIVOS (Ajustar según tu repo) ---
PATH_BASE = 'data/processed/base_consolidada_rm_2024.csv'
PATH_SIMCE_4B = 'data/raw/simce4b2024_rbd_preliminar.csv'
PATH_SIMCE_2M = 'data/raw/simce2m2024_rbd_preliminar.csv'
OUTPUT_DIR = 'figures/correlacion_detalle'

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

def cargar_y_procesar():
    # 1. Cargar datos
    print("Cargando datos...")
    df_base = pd.read_csv(PATH_BASE)
    df_simce_4b = pd.read_csv(PATH_SIMCE_4B, sep=';', encoding='latin-1')
    df_simce_2m = pd.read_csv(PATH_SIMCE_2M, sep=';', encoding='latin-1')

    # 2. Unificar SIMCE y hacer Merge
    cols_4b = ['rbd', 'prom_lect4b_rbd', 'prom_mate4b_rbd']
    cols_2m = ['rbd', 'prom_lect2m_rbd', 'prom_mate2m_rbd']
    
    df_4b = df_simce_4b[cols_4b].rename(columns={'rbd': 'RBD', 'prom_lect4b_rbd': 'S4L', 'prom_mate4b_rbd': 'S4M'})
    df_2m = df_simce_2m[cols_2m].rename(columns={'rbd': 'RBD', 'prom_lect2m_rbd': 'S2L', 'prom_mate2m_rbd': 'S2M'})
    
    for df in [df_base, df_4b, df_2m]:
        df['RBD'] = pd.to_numeric(df['RBD'], errors='coerce')
        
    df = df_base.merge(df_4b, on='RBD', how='left').merge(df_2m, on='RBD', how='left')
    
    # 3. Calcular Promedio SIMCE
    simce_cols = ['S4L', 'S4M', 'S2L', 'S2M']
    for col in simce_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['SIMCE_PROM'] = df[simce_cols].mean(axis=1)
    
    # 4. Definir si es Pagado (Binario 0/1)
    def is_pagado(row):
        pago = str(row['PAGO_MENSUAL']).upper()
        dep = str(row['categoria_dependencia']).upper()
        # Consideramos gratuito si dice GRATUITO o si es público sin info
        if pago == 'GRATUITO': return 0
        if pago == 'SIN INFORMACION' and any(x in dep for x in ['MUNICIPAL', 'SLEP', 'ADMIN_DELEGADA']): return 0
        return 1 # El resto es pagado (particular pagado o subvencionado con copago)
    
    df['ES_PAGADO'] = df.apply(is_pagado, axis=1)
    
    return df.dropna(subset=['SIMCE_PROM'])

def plot_comunal(df):
    # Agrupar por comuna
    comuna_stats = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'PCT_PAGADOS': (x['ES_PAGADO'].mean()) * 100, # % de colegios pagados
        'SIMCE': x['SIMCE_PROM'].mean()
    }))
    
    # Estadísticas
    r, p = stats.pearsonr(comuna_stats['PCT_PAGADOS'], comuna_stats['SIMCE'])
    print(f"Nivel Comunal -> Correlación R: {r:.4f}")

    # Plot
    plt.figure(figsize=(10, 7))
    sns.regplot(x='PCT_PAGADOS', y='SIMCE', data=comuna_stats, 
                scatter_kws={'s': 80, 'alpha': 0.6, 'color': '#1f77b4'}, 
                line_kws={'color': 'red', 'label': f'Regresión lineal (R={r:.2f})'})
    
    # Textos destacados
    for idx, row in comuna_stats.iterrows():
        if row['PCT_PAGADOS'] > 60 or row['SIMCE'] > 280 or row['SIMCE'] < 235:
            plt.text(row['PCT_PAGADOS']+1, row['SIMCE'], idx, fontsize=8, alpha=0.9)
            
    plt.title('Correlación Comunal: % Oferta Pagada vs Calidad SIMCE', fontsize=14)
    plt.xlabel('Porcentaje de Colegios Pagados en la Comuna (%)', fontsize=12)
    plt.ylabel('Puntaje SIMCE Promedio', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'scatter_comunal_oferta_simce.png'), dpi=150)
    plt.close()

def plot_establecimiento(df):
    # Estadísticas
    r, p = stats.pearsonr(df['ES_PAGADO'], df['SIMCE_PROM'])
    print(f"Nivel Colegio -> Correlación R: {r:.4f}")
    
    plt.figure(figsize=(8, 7))
    
    # Boxplot para mostrar distribución
    sns.boxplot(x='ES_PAGADO', y='SIMCE_PROM', data=df, 
                palette=['#2ca02c', '#d62728'], width=0.5)
    
    plt.title(f'Brecha de Resultados por Tipo de Financiamiento\n(Correlación Point-Biserial R={r:.2f})', fontsize=14)
    plt.xticks([0, 1], ['Gratuito\n(Municipal/SLEP/Part.Gratuito)', 'Pagado\n(Part. Pagado/Copago)'], fontsize=11)
    plt.ylabel('Puntaje SIMCE Promedio', fontsize=12)
    plt.xlabel('')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_establecimiento_simce.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    df = cargar_y_procesar()
    plot_comunal(df)
    plot_establecimiento(df)
    print(f"Gráficos generados en: {OUTPUT_DIR}")