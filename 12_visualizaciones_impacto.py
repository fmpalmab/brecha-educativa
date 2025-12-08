import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores Corporativos (Con transparencia para los puntos)
COLOR_FREE = '#2ca02c'  # Verde
COLOR_PAID = '#d62728'  # Rojo
PALETA = {'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID}

def plot_raincloud(df, x_col, y_col, title, ylabel, filename, y_limit=None, threshold=None):
    """
    Genera un gráfico 'Raincloud' (Violin + Box + Strip) para máximo impacto visual.
    """
    plt.figure(figsize=(12, 8))
    
    # 1. LA NUBE (Violin Plot): Muestra la densidad
    # inner=None para dejarlo limpio, alpha bajo para que sea sutil
    ax = sns.violinplot(
        x=x_col, y=y_col, data=df, 
        palette=PALETA, 
        inner=None, 
        linewidth=0, 
        saturation=0.7,
        alpha=0.3 
    )
    from matplotlib.collections import PolyCollection

    # Truco para transparencia en violines (seaborn a veces lo ignora)
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_alpha(0.2)


    # 2. LA LLUVIA (Strip Plot): Los datos crudos
    # jitter=True dispersa los puntos para ver la masa real
    sns.stripplot(
        x=x_col, y=y_col, data=df, 
        palette=PALETA, 
        size=3, 
        alpha=0.4, # Transparencia clave para ver acumulación 
        jitter=0.25,
        zorder=1
    )

    # 3. EL RESUMEN (Boxplot Minimalista): La estadística
    # width pequeño para que quede dentro del violín, sin relleno (facecolor='none')
    sns.boxplot(
        x=x_col, y=y_col, data=df, 
        width=0.15, 
        showfliers=False, # Ya mostramos los puntos con stripplot, no necesitamos outliers aquí
        boxprops={'facecolor':'none', 'edgecolor':'#333333', 'linewidth':2, 'zorder':10},
        whiskerprops={'color':'#333333', 'linewidth':2},
        capprops={'color':'#333333', 'linewidth':2},
        medianprops={'color':'#333333', 'linewidth':2},
        zorder=10
    )

    # Línea de Referencia (Umbral)
    if threshold:
        plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5, label=f'Umbral Crítico ({threshold})')
        plt.legend(loc='upper right')

    if y_limit:
        plt.ylim(0, y_limit)

    plt.title(title, fontsize=18, pad=20, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel('')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=300)
    plt.close()
    print(f"✅ Gráfico de alto impacto generado: {filename}")

def main():
    print(">>> GENERANDO VISUALIZACIONES 'RAINCLOUD' DE ALTO IMPACTO <<<")
    
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
    except FileNotFoundError:
        print("❌ Error: Ejecuta el Bloque 1 primero.")
        return

    # Clasificación
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )

    # --- GRÁFICO 1: SATURACIÓN DE AULAS (HACINAMIENTO) ---
    # Reemplaza al violin plot simple
    df_class = df[df['ratio_alumno_curso'] < 60] # Filtro visual
    plot_raincloud(
        df_class, 
        x_col='TIPO_PAGO', 
        y_col='ratio_alumno_curso',
        title='La Realidad del Aula: Hacinamiento por Tipo de Colegio',
        ylabel='Alumnos por Curso',
        filename='14_raincloud_saturacion_aulas.png',
        threshold=35, # Umbral de saturación
        y_limit=55
    )

    # --- GRÁFICO 2: CARGA DOCENTE (ATENCIÓN PERSONALIZADA) ---
    # Reemplaza al boxplot simple
    df_doc = df[df['ratio_alumno_docente'] < 50]
    plot_raincloud(
        df_doc, 
        x_col='TIPO_PAGO', 
        y_col='ratio_alumno_docente',
        title='Mito vs Realidad: Carga de Alumnos por Profesor',
        ylabel='Alumnos por Profesor',
        filename='15_raincloud_carga_docente.png',
        y_limit=45
    )

if __name__ == "__main__":
    main()