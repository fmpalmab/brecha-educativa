import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats

# --- CONFIGURACIÓN GLOBAL ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores Corporativos
COLOR_FREE = '#2ca02c'  # Verde
COLOR_PAID = '#d62728'  # Rojo

def get_r(df, col_x, col_y):
    """Calcula correlación de Pearson."""
    clean = df[[col_x, col_y]].dropna()
    if len(clean) > 2:
        r, p = stats.pearsonr(clean[col_x], clean[col_y])
        return r
    return 0.0

def main():
    print(">>> GENERANDO BLOQUE 7: CONFIRMACIÓN DE DEMANDA (SATURACIÓN) <<<")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
        print(f"Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("❌ Error: Falta el archivo de datos. Ejecuta Bloque 1.")
        return

    # Clasificación Binaria
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )

    # -------------------------------------------------------------------------
    # CÁLCULO DE INDICADORES DE PRESIÓN DE DEMANDA
    # -------------------------------------------------------------------------
    print("Calculando indicadores de presión por comuna...")
    
    comuna_stats = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'N_Colegios': len(x),
        'N_Pagados': (x['TIPO_PAGO'] == 'Pagado').sum(),
        'Matricula_Total': x['MAT_TOTAL'].sum(),
        'Matricula_Pagada': x.loc[x['TIPO_PAGO'] == 'Pagado', 'MAT_TOTAL'].sum(),
        # Saturación: Alumnos por Curso Promedio
        'Saturacion_Pagada': x.loc[x['TIPO_PAGO'] == 'Pagado', 'ratio_alumno_curso'].mean(),
        'Saturacion_Gratuita': x.loc[x['TIPO_PAGO'] == 'Gratuito', 'ratio_alumno_curso'].mean()
    })).dropna()
    
    # Cálculo de Gap de Sobrerrepresentación (Demanda vs Oferta)
    comuna_stats['Pct_Oferta'] = comuna_stats['N_Pagados'] / comuna_stats['N_Colegios']
    comuna_stats['Pct_Demanda'] = comuna_stats['Matricula_Pagada'] / comuna_stats['Matricula_Total']
    comuna_stats['Presion_Demanda'] = comuna_stats['Pct_Demanda'] - comuna_stats['Pct_Oferta']
    
    # Filtrar Top 15 comunas con mayor "Presión" (Donde la gente busca más pagado de lo que hay)
    top_presion = comuna_stats.sort_values('Presion_Demanda', ascending=False).head(15)

    # -------------------------------------------------------------------------
    # GRÁFICO 1: BARRAS DE SATURACIÓN (LA EVIDENCIA DE PREFERENCIA)
    # -------------------------------------------------------------------------
    print("Generando Gráfico 1: Saturación Comparada...")
    
    # Preparar datos para seaborn (formato largo)
    plot_data = top_presion[['Saturacion_Gratuita', 'Saturacion_Pagada']].reset_index()
    plot_data = plot_data.melt(id_vars='NOM_COM_RBD', var_name='Tipo', value_name='Alumnos_Curso')
    plot_data['Tipo'] = plot_data['Tipo'].map({'Saturacion_Gratuita': 'Gratuito', 'Saturacion_Pagada': 'Pagado'})
    
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=plot_data, x='NOM_COM_RBD', y='Alumnos_Curso', hue='Tipo',
        palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID}
    )
    
    # Línea de referencia crítica
    plt.axhline(y=35, color='gray', linestyle='--', linewidth=1.5, label='Umbral de Saturación (35 alumnos)')
    
    plt.title('Prueba de Demanda: Hacinamiento en Colegios Pagados\n(En las 15 comunas con mayor déficit de oferta privada)', fontsize=16)
    plt.ylabel('Promedio de Alumnos por Curso (Aula)')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Dependencia')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/17_confirmacion_demanda_saturacion.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # GRÁFICO 2: SCATTER (ESCASEZ VS LLENADO)
    # -------------------------------------------------------------------------
    print("Generando Gráfico 2: Correlación Escasez-Llenado...")
    
    # Filtramos comunas que tengan al menos oferta pagada incipiente para ver la tendencia
    scatter_data = comuna_stats[comuna_stats['N_Pagados'] > 0]
    
    # Correlación entre % Oferta (X) y Saturación Pagada (Y)
    # Esperamos R negativo: A menor % oferta, mayor saturación.
    r_val = get_r(scatter_data, 'Pct_Oferta', 'Saturacion_Pagada')
    
    plt.figure(figsize=(10, 8))
    sns.regplot(
        data=scatter_data, x='Pct_Oferta', y='Saturacion_Pagada',
        scatter_kws={'s': scatter_data['N_Pagados']*3, 'alpha': 0.6, 'color': COLOR_PAID, 'edgecolor':'k'},
        line_kws={'color': 'black', 'linestyle': '--', 'label': f'Tendencia Lineal (R={r_val:.2f})'}
    )
    
    # Etiquetas para comunas críticas (Poca oferta, Muy llenos)
    for idx, row in scatter_data.iterrows():
        # Etiquetar cuadrante superior izquierdo (Escasez + Lleno)
        if row['Pct_Oferta'] < 0.4 and row['Saturacion_Pagada'] > 32:
            plt.text(row['Pct_Oferta']+0.01, row['Saturacion_Pagada'], idx, fontsize=9, weight='bold')
        # Etiquetar extremos de oferta (sector oriente)
        elif row['Pct_Oferta'] > 0.8:
            plt.text(row['Pct_Oferta']-0.05, row['Saturacion_Pagada'], idx, fontsize=8, alpha=0.7)

    plt.title('La Escasez genera Saturación: Oferta Disponible vs Tamaño de Curso', fontsize=16)
    plt.xlabel('Porcentaje de Colegios Pagados en la Comuna (Oferta)')
    plt.ylabel('Alumnos por Curso en Colegios Pagados (Intensidad de Demanda)')
    plt.axhline(y=35, color='red', linestyle=':', alpha=0.5, label='Zona Saturada')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/18_scatter_escasez_vs_llenado.png', dpi=300)
    plt.close()
    
    print(f"✅ Gráficos de confirmación generados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()