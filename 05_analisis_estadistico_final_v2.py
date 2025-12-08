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
    print(">>> INICIANDO ANÁLISIS ESTADÍSTICO FINAL (V2: Con Filtro de Outliers) <<<")
    
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
    sns.heatmap(df_corr, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Matriz de Correlación General', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/11_matriz_correlacion_general.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PARTE B: SCATTER PLOTS (NORMAL + SIN OUTLIERS)
    # -------------------------------------------------------------------------
    print("B. Generando Scatter Plots (Con y Sin Outliers)...")

    niveles = [
        {'col': 'SIMCE_4B_AVG', 'nombre': '4° Básico', 'file_tag': '4b'},
        {'col': 'SIMCE_2M_AVG', 'nombre': 'II Medio', 'file_tag': '2m'}
    ]

    for nivel in niveles:
        col_simce = nivel['col']
        nombre_nivel = nivel['nombre']
        tag = nivel['file_tag']
        
        # Agregación Comunal
        comuna_stats = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
            'PCT_PAGADO': x['IS_PAID'].mean() * 100, 
            'SIMCE_PROM': x[col_simce].mean()
        })).dropna()
        
        # --- 1. VERSIÓN COMPLETA (ORIGINAL) ---
        r_val = get_r(comuna_stats, 'PCT_PAGADO', 'SIMCE_PROM')
        
        plt.figure(figsize=(12, 8))
        sns.regplot(
            data=comuna_stats, x='PCT_PAGADO', y='SIMCE_PROM',
            scatter_kws={'s': 100, 'alpha': 0.6, 'color': '#555555'},
            line_kws={'color': COLOR_PAID, 'label': f'Tendencia Global (R={r_val:.2f})'}
        )
        for idx, row in comuna_stats.iterrows():
            if row['PCT_PAGADO'] > 80 or row['SIMCE_PROM'] > 280 or row['SIMCE_PROM'] < 230:
                plt.text(row['PCT_PAGADO']+1, row['SIMCE_PROM'], idx, fontsize=9)
                
        plt.title(f'Segregación vs Calidad Comunal ({nombre_nivel}) - Muestra Completa', fontsize=16)
        plt.xlabel('% Oferta Pagada')
        plt.ylabel('Puntaje SIMCE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/12_{tag}_scatter_segregacion_comunal.png', dpi=300)
        plt.close()

        # --- 2. VERSIÓN SIN OUTLIERS (>50% Pagado) ---
        # Filtramos comunas donde la oferta pagada sea <= 50%
        subset_no_outliers = comuna_stats[comuna_stats['PCT_PAGADO'] <= 50]
        
        if len(subset_no_outliers) > 3:
            r_val_clean = get_r(subset_no_outliers, 'PCT_PAGADO', 'SIMCE_PROM')
            
            plt.figure(figsize=(12, 8))
            sns.regplot(
                data=subset_no_outliers, x='PCT_PAGADO', y='SIMCE_PROM',
                scatter_kws={'s': 100, 'alpha': 0.7, 'color': COLOR_FREE, 'edgecolor':'k'}, # Puntos Verdes para diferenciar
                line_kws={'color': 'black', 'linestyle': '--', 'label': f'Tendencia Filtrada (R={r_val_clean:.2f})'}
            )
            
            # Etiquetas para todas las comunas del subset (son menos, caben mejor)
            for idx, row in subset_no_outliers.iterrows():
                plt.text(row['PCT_PAGADO']+0.5, row['SIMCE_PROM'], idx, fontsize=9)

            plt.title(f'¿Persiste la Brecha? Análisis en Comunas Mayoritariamente Públicas ({nombre_nivel})\n(Excluyendo comunas con >50% oferta privada)', fontsize=16)
            plt.xlabel('Porcentaje de Oferta Pagada en la Comuna (0-50%)')
            plt.ylabel(f'Puntaje Promedio SIMCE {nombre_nivel}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/12_{tag}_scatter_segregacion_SIN_OUTLIERS.png', dpi=300)
            plt.close()
            print(f"   -> Generado 12_{tag} SIN OUTLIERS (R={r_val_clean:.2f})")

    # -------------------------------------------------------------------------
    # PARTE C: RANKING DE BRECHA (DUMBBELL)
    # -------------------------------------------------------------------------
    print("C. Generando Ranking de Brecha...")
    for nivel in niveles:
        col_simce = nivel['col']
        tag = nivel['file_tag']
        gap_stats = df.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])[col_simce].mean().unstack()
        
        if 'Pagado' in gap_stats and 'Gratuito' in gap_stats:
            gap_stats['Brecha'] = gap_stats['Pagado'] - gap_stats['Gratuito']
            top_gap = gap_stats.sort_values('Brecha').dropna().tail(15)
            
            plt.figure(figsize=(12, 12))
            plt.hlines(y=top_gap.index, xmin=top_gap['Gratuito'], xmax=top_gap['Pagado'], color='gray', alpha=0.5, linewidth=2)
            plt.scatter(top_gap['Gratuito'], top_gap.index, color=COLOR_FREE, s=120, label='Gratuito', zorder=3)
            plt.scatter(top_gap['Pagado'], top_gap.index, color=COLOR_PAID, s=120, label='Pagado', zorder=3)
            plt.title(f'Top 15 Comunas: Brecha SIMCE {nivel["nombre"]}', fontsize=16)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/13_{tag}_ranking_brecha_dumbbell.png', dpi=300)
            plt.close()

    # -------------------------------------------------------------------------
    # PARTE D: DISTRIBUCIÓN
    # -------------------------------------------------------------------------
    print("D. Generando Gráficos de Distribución...")
    df_clean_class = df[df['ratio_alumno_curso'] < 60]
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=df_clean_class, x='TIPO_PAGO', y='ratio_alumno_curso', palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID})
    plt.axhline(35, color='gray', linestyle='--'); plt.title('Saturación de Aulas')
    plt.savefig(f'{OUTPUT_DIR}/14_distribucion_saturacion_aulas.png', dpi=300); plt.close()
    
    df_clean_doc = df[df['ratio_alumno_docente'] < 50]
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df_clean_doc, x='TIPO_PAGO', y='ratio_alumno_docente', palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID})
    plt.title('Carga Docente'); plt.savefig(f'{OUTPUT_DIR}/15_distribucion_carga_docente.png', dpi=300); plt.close()

    print(f"\n✅ ANÁLISIS ESTADÍSTICO V2 FINALIZADO. Gráficos en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()