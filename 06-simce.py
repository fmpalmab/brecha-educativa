# -*- coding: utf-8 -*-
"""
Script de Análisis: Oferta, Demanda y Desempeño SIMCE (2024)
------------------------------------------------------------
Este script realiza:
1. Consolidación de datos de establecimientos y resultados SIMCE preliminares.
2. Clasificación de colegios en Gratuito vs. Pagado.
3. Generación de gráficos comparativos de oferta (establecimientos) vs demanda (matrícula).
4. Análisis de brecha de resultados SIMCE.
5. Correlación entre segregación (pago) y desempeño comunal.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- CONFIGURACIÓN ---
# Ajusta estas rutas según tu estructura de carpetas
PATH_BASE = 'data/processed/base_consolidada_rm_2024.csv'
PATH_SIMCE_4B = 'data/raw/simce4b2024_rbd_preliminar.csv'
PATH_SIMCE_2M = 'data/raw/simce2m2024_rbd_preliminar.csv'
OUTPUT_DIR = 'figures/analisis_pago_simce'

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración visual
sns.set_style("whitegrid")
COLOR_GRATUITO = '#4daf4a'  # Verde
COLOR_PAGADO = '#e41a1c'    # Rojo

def cargar_y_unir_datos():
    """Carga los datasets y realiza el merge por RBD."""
    print(">>> Cargando datos...")
    
    # 1. Cargar Base Maestra
    df_base = pd.read_csv(PATH_BASE, sep=',')
    
    # 2. Cargar SIMCE (con encoding latin-1 para evitar errores)
    df_simce_4b = pd.read_csv(PATH_SIMCE_4B, sep=';', encoding='latin-1')
    df_simce_2m = pd.read_csv(PATH_SIMCE_2M, sep=';', encoding='latin-1')
    
    # 3. Seleccionar columnas relevantes SIMCE
    cols_4b = ['rbd', 'prom_lect4b_rbd', 'prom_mate4b_rbd']
    cols_2m = ['rbd', 'prom_lect2m_rbd', 'prom_mate2m_rbd']
    
    df_4b_sel = df_simce_4b[cols_4b].rename(columns={
        'rbd': 'RBD', 'prom_lect4b_rbd': 'SIMCE_4B_LECT', 'prom_mate4b_rbd': 'SIMCE_4B_MATE'
    })
    
    df_2m_sel = df_simce_2m[cols_2m].rename(columns={
        'rbd': 'RBD', 'prom_lect2m_rbd': 'SIMCE_2M_LECT', 'prom_mate2m_rbd': 'SIMCE_2M_MATE'
    })
    
    # Asegurar tipos numéricos para el merge
    for df in [df_base, df_4b_sel, df_2m_sel]:
        df['RBD'] = pd.to_numeric(df['RBD'], errors='coerce')

    # 4. Merge (Left Join)
    print(">>> Uniendo bases de datos...")
    df_merged = df_base.merge(df_4b_sel, on='RBD', how='left')\
                       .merge(df_2m_sel, on='RBD', how='left')
    
    return df_merged

def clasificar_pago(row):
    """Clasifica en 'Gratuito' o 'Pagado' basado en mensualidad y dependencia."""
    pago = str(row['PAGO_MENSUAL']).upper()
    dep = str(row['categoria_dependencia']).upper()
    
    if pago == 'GRATUITO':
        return 'Gratuito'
    elif pago == 'SIN INFORMACION':
        # Asumimos gratuito si es público, pagado si es privado sin info
        if any(x in dep for x in ['MUNICIPAL', 'SLEP', 'ADMIN_DELEGADA']):
            return 'Gratuito'
        else:
            return 'Pagado'
    else:
        return 'Pagado'

def procesar_datos(df):
    """Aplica lógica de negocio y cálculos agregados."""
    print(">>> Procesando métricas...")
    
    # Clasificar Tipo de Pago
    df['TIPO_PAGO'] = df.apply(clasificar_pago, axis=1)
    
    # Calcular Promedio General SIMCE (4B y 2M)
    simce_cols = ['SIMCE_4B_LECT', 'SIMCE_4B_MATE', 'SIMCE_2M_LECT', 'SIMCE_2M_MATE']
    for col in simce_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df['SIMCE_PROM_RBD'] = df[simce_cols].mean(axis=1)
    
    return df

def generar_estadisticas_comunales(df):
    """Agrupa los datos por comuna."""
    stats = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Total_Est': len(x),
        'Est_Gratuito': (x['TIPO_PAGO'] == 'Gratuito').sum(),
        'Est_Pagado': (x['TIPO_PAGO'] == 'Pagado').sum(),
        'Total_Mat': x['MAT_TOTAL'].sum(),
        'Mat_Gratuito': x.loc[x['TIPO_PAGO'] == 'Gratuito', 'MAT_TOTAL'].sum(),
        'Mat_Pagado': x.loc[x['TIPO_PAGO'] == 'Pagado', 'MAT_TOTAL'].sum(),
        'SIMCE_Prom_Comuna': x['SIMCE_PROM_RBD'].mean()
    }))
    
    # Calcular porcentajes
    stats['Pct_Est_Pagado'] = stats['Est_Pagado'] / stats['Total_Est']
    stats['Pct_Mat_Pagado'] = stats['Mat_Pagado'] / stats['Total_Mat']
    
    # Rellenar NaNs con 0 para cálculos
    stats = stats.fillna(0)
    
    return stats.sort_values('Pct_Est_Pagado', ascending=True)

def graficar_oferta_demanda(stats):
    """Genera gráfico comparativo de Establecimientos vs Matrícula."""
    print(">>> Generando gráfico de Oferta vs Demanda...")
    
    # Preparar datos para plot
    plot_data = stats[['Pct_Est_Pagado', 'Pct_Mat_Pagado']].copy()
    plot_data['Pct_Est_Gratuito'] = 1 - plot_data['Pct_Est_Pagado']
    plot_data['Pct_Mat_Gratuito'] = 1 - plot_data['Pct_Mat_Pagado']
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 12), sharey=True)
    
    # Gráfico 1: Establecimientos (Oferta)
    plot_data[['Pct_Est_Gratuito', 'Pct_Est_Pagado']].plot(
        kind='barh', stacked=True, ax=axes[0], 
        color=[COLOR_GRATUITO, COLOR_PAGADO], width=0.8
    )
    axes[0].set_title('Oferta: % de Establecimientos (Gratuito vs Pagado)', fontsize=14)
    axes[0].set_xlabel('Proporción', fontsize=12)
    axes[0].legend(['Gratuito', 'Pagado'], loc='lower left')
    
    # Gráfico 2: Matrícula (Demanda)
    plot_data[['Pct_Mat_Gratuito', 'Pct_Mat_Pagado']].plot(
        kind='barh', stacked=True, ax=axes[1], 
        color=[COLOR_GRATUITO, COLOR_PAGADO], width=0.8
    )
    axes[1].set_title('Demanda: % de Matrícula (Gratuito vs Pagado)', fontsize=14)
    axes[1].set_xlabel('Proporción', fontsize=12)
    axes[1].legend(['Gratuito', 'Pagado'], loc='lower left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_oferta_vs_demanda_pago.png'), dpi=150)
    plt.close()

def graficar_brecha_simce(df):
    """Genera boxplot de resultados SIMCE por tipo de pago."""
    print(">>> Generando gráfico de Brecha SIMCE...")
    
    plt.figure(figsize=(10, 8))
    
    # Boxplot
    sns.boxplot(
        x='TIPO_PAGO', y='SIMCE_PROM_RBD', data=df, 
        palette={'Gratuito': COLOR_GRATUITO, 'Pagado': COLOR_PAGADO},
        showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black"}
    )
    
    plt.title('Distribución de Puntajes SIMCE Promedio 2024 (4°B y IIM)\npor Tipo de Financiamiento', fontsize=14)
    plt.ylabel('Puntaje Promedio', fontsize=12)
    plt.xlabel('Tipo de Establecimiento', fontsize=12)
    
    # Calcular y mostrar medianas en texto
    medianas = df.groupby('TIPO_PAGO')['SIMCE_PROM_RBD'].median()
    plt.text(0, medianas['Gratuito'] + 2, f"Mediana: {medianas['Gratuito']:.0f}", ha='center', fontweight='bold')
    plt.text(1, medianas['Pagado'] + 2, f"Mediana: {medianas['Pagado']:.0f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_brecha_simce_pago.png'), dpi=150)
    plt.close()

def graficar_correlacion(stats):
    """Genera scatter plot de % Pagado vs Desempeño Comunal."""
    print(">>> Generando gráfico de Correlación Comunal...")
    
    plt.figure(figsize=(12, 9))
    
    sns.scatterplot(
        x='Pct_Mat_Pagado', y='SIMCE_Prom_Comuna', data=stats, 
        s=150, alpha=0.7, color='#2b8cbe', edgecolor='black'
    )
    
    # Etiquetas para comunas extremas o relevantes
    for idx, row in stats.iterrows():
        # Etiquetar extremos (muy pagado, muy alto simce o muy bajo simce)
        if row['Pct_Mat_Pagado'] > 0.60 or row['SIMCE_Prom_Comuna'] > 280 or row['SIMCE_Prom_Comuna'] < 240:
            plt.text(row['Pct_Mat_Pagado']+0.01, row['SIMCE_Prom_Comuna'], idx, fontsize=9, alpha=0.8)
    
    # Línea de tendencia
    sns.regplot(x='Pct_Mat_Pagado', y='SIMCE_Prom_Comuna', data=stats, scatter=False, color='gray', line_kws={'linestyle':'--'})
    
    plt.title('Correlación: Segregación (% Matrícula Pagada) vs Desempeño SIMCE Comunal', fontsize=14)
    plt.xlabel('Proporción de Estudiantes en Colegios Pagados (0.0 a 1.0)', fontsize=12)
    plt.ylabel('Promedio SIMCE Comunal', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_correlacion_pago_simce.png'), dpi=150)
    plt.close()

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    try:
        # 1. Cargar
        df_raw = cargar_y_unir_datos()
        
        # 2. Procesar
        df_processed = procesar_datos(df_raw)
        stats_comunal = generar_estadisticas_comunales(df_processed)
        
        # Guardar CSV de estadísticas comunales para revisión
        stats_comunal.to_csv(os.path.join(OUTPUT_DIR, 'resumen_estadisticas_comunales.csv'))
        
        # 3. Graficar
        graficar_oferta_demanda(stats_comunal)
        graficar_brecha_simce(df_processed)
        graficar_correlacion(stats_comunal)
        
        print(f"\n✅ Análisis completado. Gráficos guardados en: {OUTPUT_DIR}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: No se encontró algún archivo. Verifica las rutas.\nDetalle: {e}")
    except Exception as e:
        print(f"\n❌ Error inesperado:\n{e}")