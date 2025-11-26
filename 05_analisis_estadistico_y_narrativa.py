import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats

# --- CONFIGURACIÓN GLOBAL ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilos y Colores Corporativos
sns.set_style("whitegrid")
COLOR_FREE = '#2ca02c'  # Verde (Gratuito)
COLOR_PAID = '#d62728'  # Rojo (Pagado)

def get_r(df, col_x, col_y):
    """Calcula correlación de Pearson ignorando NaNs."""
    clean = df[[col_x, col_y]].dropna()
    if len(clean) > 2:
        r, p = stats.pearsonr(clean[col_x], clean[col_y])
        return r
    return 0.0

def main():
    print(">>> INICIANDO ANÁLISIS ESTADÍSTICO FINAL (Bloque 5 Corregido) <<<")
    
    # 1. Cargar Datos Procesados
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
        print(f"Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("❌ Error: Ejecuta el Bloque 1 primero.")
        return

    # Clasificación Binaria Consistente
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )
    # Variable Dummy para Correlaciones (0=Gratuito, 1=Pagado)
    df['IS_PAID'] = df['TIPO_PAGO'].apply(lambda x: 1 if x == 'Pagado' else 0)

    # -------------------------------------------------------------------------
    # PARTE A: MATRICES DE CORRELACIÓN
    # -------------------------------------------------------------------------
    print("A. Generando Matrices de Correlación...")
    
    cols_corr = {
        'IS_PAID': 'Es Pagado',
        'orden_precio': 'Nivel Precio',
        'MAT_TOTAL': 'Matrícula Total',
        'ratio_alumno_docente': 'Alumnos/Docente',
        'ratio_alumno_curso': 'Alumnos/Curso',
        'SIMCE_4B_AVG': 'SIMCE 4°B',
        'SIMCE_2M_AVG': 'SIMCE IIM'
    }
    
    df_corr = df[cols_corr.keys()].rename(columns=cols_corr).corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlación: Variables Educativas', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/11_matriz_correlacion_general.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PARTE B: NARRATIVA DE SEGREGACIÓN (Scatter Plots por Nivel)
    # -------------------------------------------------------------------------
    print("B. Generando Scatter Plots (Separados por Nivel)...")

    # Definimos los niveles a analizar por separado
    niveles = [
        {'col': 'SIMCE_4B_AVG', 'nombre': '4° Básico', 'file_tag': '4b'},
        {'col': 'SIMCE_2M_AVG', 'nombre': 'II Medio', 'file_tag': '2m'}
    ]

    for nivel in niveles:
        col_simce = nivel['col']
        nombre_nivel = nivel['nombre']
        tag = nivel['file_tag']
        
        # Agregación Comunal Específica para este nivel
        comuna_stats = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
            'PCT_PAGADO': x['IS_PAID'].mean() * 100, # % Oferta Pagada
            'SIMCE_PROM': x[col_simce].mean()        # Promedio del nivel
        })).dropna()
        
        r_val = get_r(comuna_stats, 'PCT_PAGADO', 'SIMCE_PROM')
        
        plt.figure(figsize=(12, 8))
        sns.regplot(
            data=comuna_stats, x='PCT_PAGADO', y='SIMCE_PROM',
            scatter_kws={'s': 100, 'alpha': 0.6, 'color': '#555555'},
            line_kws={'color': COLOR_PAID, 'label': f'Regresión (R={r_val:.2f})'}
        )
        
        # Etiquetas para casos extremos
        for idx, row in comuna_stats.iterrows():
            # Criterios de etiqueta ajustados al nivel
            if row['PCT_PAGADO'] > 80 or row['SIMCE_PROM'] > 280 or row['SIMCE_PROM'] < 230:
                plt.text(row['PCT_PAGADO']+1, row['SIMCE_PROM'], idx, fontsize=9)
                
        plt.title(f'Segregación vs Calidad Comunal ({nombre_nivel})', fontsize=16)
        plt.xlabel('Porcentaje de Oferta Pagada en la Comuna (%)')
        plt.ylabel(f'Puntaje Promedio SIMCE {nombre_nivel}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/12_{tag}_scatter_segregacion_comunal.png', dpi=300)
        plt.close()

    # -------------------------------------------------------------------------
    # PARTE C: LA BRECHA EN DETALLE (Dumbbell Chart por Nivel)
    # -------------------------------------------------------------------------
    print("C. Generando Ranking de Brecha (Dumbbell Separados)...")
    
    for nivel in niveles:
        col_simce = nivel['col']
        nombre_nivel = nivel['nombre']
        tag = nivel['file_tag']

        # Calcular brecha por comuna para este nivel
        gap_stats = df.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])[col_simce].mean().unstack()
        
        if 'Pagado' in gap_stats and 'Gratuito' in gap_stats:
            gap_stats['Brecha'] = gap_stats['Pagado'] - gap_stats['Gratuito']
            # Top 15 comunas con mayor brecha (ordenadas)
            top_gap = gap_stats.sort_values('Brecha').dropna().tail(15)
            
            plt.figure(figsize=(12, 12))
            plt.hlines(y=top_gap.index, xmin=top_gap['Gratuito'], xmax=top_gap['Pagado'], color='gray', alpha=0.5, linewidth=2)
            plt.scatter(top_gap['Gratuito'], top_gap.index, color=COLOR_FREE, s=120, label='Gratuito', zorder=3)
            plt.scatter(top_gap['Pagado'], top_gap.index, color=COLOR_PAID, s=120, label='Pagado', zorder=3)
            
            plt.title(f'Top 15 Comunas: Desigualdad de Resultados SIMCE {nombre_nivel}', fontsize=16)
            plt.xlabel('Puntaje Promedio')
            plt.legend(loc='lower right')
            plt.grid(True, axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/13_{tag}_ranking_brecha_dumbbell.png', dpi=300)
            plt.close()

    # -------------------------------------------------------------------------
    # PARTE D: DISTRIBUCIÓN Y SATURACIÓN
    # -------------------------------------------------------------------------
    print("D. Generando Gráficos de Distribución...")
    
    # 1. Saturación de Aulas (Violin)
    df_clean_class = df[df['ratio_alumno_curso'] < 60]
    r_class = get_r(df, 'IS_PAID', 'ratio_alumno_curso')
    
    plt.figure(figsize=(10, 8))
    sns.violinplot(
        data=df_clean_class, x='TIPO_PAGO', y='ratio_alumno_curso',
        palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID}, inner='quartile', alpha=0.7
    )
    plt.axhline(y=35, color='gray', linestyle='--', label='Umbral Saturación (35)', linewidth=1.5)
    plt.title(f'Saturación de Aulas: Tamaño de Curso (R={r_class:.2f})', fontsize=14)
    plt.ylabel('Alumnos por Curso')
    plt.xlabel('')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/14_distribucion_saturacion_aulas.png', dpi=300)
    plt.close()
    
    # 2. Carga Docente (Boxplot)
    df_clean_doc = df[df['ratio_alumno_docente'] < 50]
    r_doc = get_r(df, 'IS_PAID', 'ratio_alumno_docente')
    
    plt.figure(figsize=(10, 8))
    sns.boxplot(
        data=df_clean_doc, x='TIPO_PAGO', y='ratio_alumno_docente',
        palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID}, linewidth=1.5
    )
    plt.title(f'Desigualdad en Carga Docente: Alumnos por Profesor (R={r_doc:.2f})', fontsize=14)
    plt.ylabel('Alumnos por Profesor')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/15_distribucion_carga_docente.png', dpi=300)
    plt.close()

    print(f"\n✅ ANÁLISIS ESTADÍSTICO FINALIZADO. Gráficos guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()