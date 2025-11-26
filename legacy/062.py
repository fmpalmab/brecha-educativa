import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- CONFIGURACIÓN ---
PATH_BASE = 'data/processed/base_consolidada_rm_2024.csv'
PATH_SIMCE_4B = 'data/raw/simce4b2024_rbd_preliminar.csv'
PATH_SIMCE_2M = 'data/raw/simce2m2024_rbd_preliminar.csv'
OUTPUT_DIR = 'figures/estadisticas'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def cargar_datos():
    # Cargar y unir datos (similar a scripts anteriores)
    df_base = pd.read_csv(PATH_BASE)
    df_simce_4b = pd.read_csv(PATH_SIMCE_4B, sep=';', encoding='latin-1')
    df_simce_2m = pd.read_csv(PATH_SIMCE_2M, sep=';', encoding='latin-1')
    
    # Procesar y Unir
    cols_4b = ['rbd', 'prom_lect4b_rbd', 'prom_mate4b_rbd']
    cols_2m = ['rbd', 'prom_lect2m_rbd', 'prom_mate2m_rbd']
    
    df_4b_sel = df_simce_4b[cols_4b].rename(columns={'rbd': 'RBD', 'prom_lect4b_rbd': 'S_4B_L', 'prom_mate4b_rbd': 'S_4B_M'})
    df_2m_sel = df_simce_2m[cols_2m].rename(columns={'rbd': 'RBD', 'prom_lect2m_rbd': 'S_2M_L', 'prom_mate2m_rbd': 'S_2M_M'})
    
    for df in [df_base, df_4b_sel, df_2m_sel]:
        df['RBD'] = pd.to_numeric(df['RBD'], errors='coerce')
        
    df_merged = df_base.merge(df_4b_sel, on='RBD', how='left').merge(df_2m_sel, on='RBD', how='left')
    
    # Calcular Promedio SIMCE
    simce_cols = ['S_4B_L', 'S_4B_M', 'S_2M_L', 'S_2M_M']
    for col in simce_cols:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
    df_merged['SIMCE_PROM'] = df_merged[simce_cols].mean(axis=1)
    
    return df_merged

def calcular_correlaciones(df):
    # Crear variable dummy ES_PAGADO
    def is_pagado(row):
        pago = str(row['PAGO_MENSUAL']).upper()
        dep = str(row['categoria_dependencia']).upper()
        if pago == 'GRATUITO': return 0
        if pago == 'SIN INFORMACION' and any(x in dep for x in ['MUNICIPAL', 'SLEP', 'ADMIN_DELEGADA']): return 0
        return 1
    
    df['ES_PAGADO'] = df.apply(is_pagado, axis=1)
    df['ORDEN_PRECIO'] = pd.to_numeric(df['orden_precio'], errors='coerce')
    
    # 1. Correlación Nivel Establecimiento
    cols_est = ['ES_PAGADO', 'ORDEN_PRECIO', 'MAT_TOTAL', 'SIMCE_PROM']
    corr_est = df[cols_est].corr()
    
    # 2. Correlación Nivel Comunal
    comuna_stats = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'PCT_MAT_PAGADA': (x.loc[x['ES_PAGADO']==1, 'MAT_TOTAL'].sum() / x['MAT_TOTAL'].sum()),
        'PCT_COL_PAGADOS': (x['ES_PAGADO'].sum() / len(x)),
        'SIMCE_COMUNAL': x['SIMCE_PROM'].mean(),
        'TAMANO_PROM': x['MAT_TOTAL'].mean()
    }))
    
    corr_com = comuna_stats.corr()
    
    return corr_est, corr_com

def graficar_matrices(corr_est, corr_com):
    # Heatmap Establecimiento
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_est, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlación Pearson (Nivel Establecimiento)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_corr_establecimiento.png'))
    plt.close()
    
    # Heatmap Comunal
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_com, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlación Pearson (Nivel Comunal)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_corr_comunal.png'))
    plt.close()

if __name__ == "__main__":
    df = cargar_datos()
    c_est, c_com = calcular_correlaciones(df)
    
    print(">>> Matriz Establecimiento:\n", c_est)
    print("\n>>> Matriz Comunal:\n", c_com)
    
    graficar_matrices(c_est, c_com)
    print(f"\nGráficos guardados en {OUTPUT_DIR}")