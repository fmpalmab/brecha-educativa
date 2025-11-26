import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats

# --- CONFIGURACIÓN ---
OUTPUT_DIR_FIG = 'figures/exploratorio'
OUTPUT_DIR_REP = 'reports'
os.makedirs(OUTPUT_DIR_FIG, exist_ok=True)
os.makedirs(OUTPUT_DIR_REP, exist_ok=True)

# Estilos y Colores
sns.set_style("whitegrid") # Whitegrid suele tener bordes, pero reforzaremos
COLOR_FREE = '#2ca02c'  # Verde
COLOR_PAID = '#d62728'  # Rojo

# Función auxiliar para Estadísticas (R)
def get_r(df, col_x, col_y):
    clean = df[[col_x, col_y]].dropna()
    if len(clean) > 2:
        r, _ = stats.pearsonr(clean[col_x], clean[col_y])
        return r
    return 0.0

def main():
    print(">>> INICIANDO ANÁLISIS EXPLORATORIO CONSOLIDADO (Bloque 2) <<<")
    
    # 1. Cargar Datos Consolidados (del Bloque 1)
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
        print(f"Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("❌ Error: No se encontró 'data/processed/base_consolidada_rm_2024_final.csv'. Ejecuta el Bloque 1 primero.")
        return

    # -------------------------------------------------------------------------
    # 2. GENERACIÓN DE REPORTE DE TEXTO UNIFICADO
    # -------------------------------------------------------------------------
    print("Generando reporte estadístico de texto...")
    
    with open(f'{OUTPUT_DIR_REP}/hallazgos_exploratorios.txt', 'w', encoding='utf-8') as f:
        f.write("REPORTE CONSOLIDADO DE HALLAZGOS - BRECHA EDUCATIVA RM\n")
        f.write("======================================================\n\n")

        # A. Desigualdad Intra-comunal
        stats_comuna = df.groupby('NOM_COM_RBD')['ratio_alumno_docente'].agg(['mean', 'std', 'count'])
        top_desigualdad = stats_comuna[stats_comuna['count'] > 10].sort_values('std', ascending=False).head(10)
        
        f.write("1. TOP 10 COMUNAS CON MAYOR DESIGUALDAD INTERNA (Std Dev Ratio Alumno/Docente):\n")
        f.write(top_desigualdad.to_string())
        f.write("\n\n")

        # B. Estadísticas por Tipo de Pago
        stats_pago = df.groupby('TIPO_PAGO')[['ratio_alumno_docente', 'ratio_alumno_curso']].describe()
        f.write("2. COMPARATIVA GRATUITO VS PAGADO:\n")
        f.write(stats_pago.to_string())
        f.write("\n\n")

        # C. Correlaciones
        corr_cols = ['ratio_alumno_docente', 'ratio_alumno_curso', 'MAT_TOTAL', 'orden_precio']
        corr_mat = df[corr_cols].corr()
        f.write("3. MATRIZ DE CORRELACIÓN GENERAL:\n")
        f.write(corr_mat.to_string())

    # -------------------------------------------------------------------------
    # 3. GRÁFICOS CONSOLIDADOS (Reglas Aplicadas)
    # -------------------------------------------------------------------------
    
    # GRÁFICO 1: Ranking de Desigualdad Comunal (Boxplot)
    # ---------------------------------------------------
    print("Generando Gráfico 1: Ranking Comunal...")
    plt.figure(figsize=(16, 10))
    # Ordenar por mediana
    order = df.groupby('NOM_COM_RBD')['ratio_alumno_docente'].median().sort_values().index
    
    sns.boxplot(data=df, x='NOM_COM_RBD', y='ratio_alumno_docente', order=order, palette="viridis", linewidth=1)
    
    plt.xticks(rotation=90, fontsize=8)
    plt.title('Distribución de Carga Docente por Comuna (Ranking Medianas)', fontsize=16)
    plt.ylabel('Estudiantes por Docente')
    plt.xlabel('')
    
    # REGLA: Marco visible
    plt.box(True) 
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR_FIG}/01_ranking_comunal_ratio.png', dpi=300)
    plt.close()

    # GRÁFICO 2: Densidad por Tipo de Pago (Rojo vs Verde)
    # ----------------------------------------------------
    print("Generando Gráfico 2: Densidad Pagado vs Gratuito...")
    plt.figure(figsize=(12, 7))
    
    sns.kdeplot(data=df, x='ratio_alumno_docente', hue='TIPO_PAGO', fill=True, common_norm=False,
                palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID}, alpha=0.3, linewidth=2)
    
    plt.title('Comparación de Densidad: Carga Docente por Financiamiento', fontsize=14)
    plt.xlim(0, 45)
    plt.xlabel('Ratio Alumnos/Docente')
    plt.box(True) # Marco
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR_FIG}/02_densidad_pago.png', dpi=300)
    plt.close()

    # GRÁFICO 3: Impacto Precio (Blues)
    # ----------------------------------------------------
    print("Generando Gráfico 3: Impacto Precio...")
    plt.figure(figsize=(12, 8))
    df_precios = df[df['orden_precio'] >= 0].copy()
    orden_precios = ['GRATUITO', '$1.000 A $10.000', '$10.001 A $25.000', '$25.001 A $50.000', '$50.001 A $100.000', 'MAS DE $100.000']
    
    sns.boxplot(data=df_precios, x='PAGO_MENSUAL', y='ratio_alumno_docente', order=orden_precios, palette="Blues")
    
    plt.xticks(rotation=30)
    plt.title('Relación Precio vs. Carga Docente', fontsize=14)
    plt.box(True) # Marco
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR_FIG}/03_impacto_precio.png', dpi=300)
    plt.close()

    # GRÁFICO 4: Matriz de Calidad (Scatter School Level)
    # ----------------------------------------------------
    # REGLA: Rojo vs Verde, Marco, R value
    print("Generando Gráfico 4: Matriz Calidad (School Level)...")
    plt.figure(figsize=(12, 8))
    
    df_clean = df[(df['ratio_alumno_docente'] < 50) & (df['ratio_alumno_curso'] < 60)].copy()
    r_val = get_r(df_clean, 'ratio_alumno_docente', 'ratio_alumno_curso')
    
    sns.scatterplot(data=df_clean, x='ratio_alumno_docente', y='ratio_alumno_curso', 
                    hue='TIPO_PAGO', palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID}, alpha=0.6)
    
    # Cuadrantes promedio
    plt.axvline(df_clean['ratio_alumno_docente'].mean(), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(df_clean['ratio_alumno_curso'].mean(), color='gray', linestyle='--', alpha=0.5)
    
    plt.title(f'Matriz de Calidad: Personificación vs Hacinamiento (R={r_val:.2f})', fontsize=14)
    plt.xlabel('Alumnos por Profesor')
    plt.ylabel('Alumnos por Curso')
    plt.box(True) # Marco
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR_FIG}/04_matriz_calidad_escuelas.png', dpi=300)
    plt.close()

    # GRÁFICO 5: Vulnerabilidad Comunal (Scatter Commune Level)
    # ---------------------------------------------------------
    print("Generando Gráfico 5: Vulnerabilidad Comunal...")
    # Agregación comunal
    df['IS_PAID'] = df['TIPO_PAGO'].apply(lambda x: 1 if x == 'Pagado' else 0)
    comunal = df.groupby('NOM_COM_RBD').agg({
        'ratio_alumno_docente': ['mean', 'std'],
        'MAT_TOTAL': 'sum',
        'IS_PAID': 'mean' # % de colegios pagados
    }).reset_index()
    comunal.columns = ['NOM_COM_RBD', 'ratio_mean', 'ratio_std', 'matricula', 'pct_paid']
    
    r_vul = get_r(comunal, 'ratio_mean', 'ratio_std')
    
    plt.figure(figsize=(14, 10))
    # Usamos un gradiente de Verde (0% pagado) a Rojo (100% pagado)
    # cmap='RdYlGn_r' (Red-Yellow-Green reversed -> Green low, Red high)
    scatter = plt.scatter(x=comunal['ratio_mean'], y=comunal['ratio_std'], 
                          s=comunal['matricula']/50, c=comunal['pct_paid'], 
                          cmap='RdYlGn_r', alpha=0.8, edgecolors='gray')
    
    plt.colorbar(scatter, label='% Oferta Pagada (Verde=0%, Rojo=100%)')
    
    # Etiquetas extremos
    for _, row in comunal.iterrows():
        if row['ratio_std'] > 6 or row['ratio_mean'] > 20 or row['matricula'] > 40000:
            plt.text(row['ratio_mean']+0.2, row['ratio_std'], row['NOM_COM_RBD'], fontsize=9)

    plt.title(f'Vulnerabilidad Comunal: Calidad vs Desigualdad (R={r_vul:.2f})', fontsize=16)
    plt.xlabel('Promedio Alumnos/Docente')
    plt.ylabel('Desigualdad Interna (Std Dev)')
    plt.box(True) # Marco
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR_FIG}/05_vulnerabilidad_comunal.png', dpi=300)
    plt.close()

    print(f"\n✅ PROCESO COMPLETADO. Gráficos en: {OUTPUT_DIR_FIG}")

if __name__ == "__main__":
    main()