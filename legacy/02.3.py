import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    print("--- Iniciando Análisis Comunal Multidimensional ---")
    
    # Configuración de directorios
    os.makedirs('figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    sns.set_theme(style="whitegrid")

    # 1. Cargar Datos (Intentamos en la raíz)
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
    except FileNotFoundError:
        # Si falla, intentamos la ruta original del repo por si acaso
        df = pd.read_csv('fmpalmab/brecha-educativa/brecha-educativa-main/data/processed/base_consolidada_rm_2024.csv')

    print(f"Datos cargados: {df.shape}")

    # 2. Ingeniería de Atributos a Nivel Comunal
    # Codificamos la dependencia para poder sumar
    df['es_municipal'] = df['categoria_dependencia'].apply(lambda x: 1 if 'Municipal' in str(x) or 'SLEP' in str(x) else 0)
    df['es_subvencionado'] = df['categoria_dependencia'].apply(lambda x: 1 if 'Subvencionado' in str(x) else 0)
    df['es_pagado'] = df['categoria_dependencia'].apply(lambda x: 1 if 'Pagado' in str(x) else 0)
    
    # Agregación
    comunal = df.groupby('NOM_COM_RBD').agg({
        'ratio_alumno_docente': ['mean', 'std', 'median'],
        'ratio_alumno_curso': ['mean'],
        'MAT_TOTAL': 'sum',
        'DC_TOT': 'sum',
        'RBD': 'count',
        'es_municipal': 'sum',
        'es_subvencionado': 'sum',
        'es_pagado': 'sum'
    })
    
    # Aplanar columnas
    comunal.columns = [
        'ratio_docente_media', 'ratio_docente_std', 'ratio_docente_mediana',
        'ratio_curso_media',
        'matricula_total',
        'docentes_total',
        'num_colegios',
        'num_municipal',
        'num_subvencionado',
        'num_pagado'
    ]
    
    # Cálculos derivados
    comunal['pct_municipal'] = (comunal['num_municipal'] / comunal['num_colegios']) * 100
    comunal['pct_pagado'] = (comunal['num_pagado'] / comunal['num_colegios']) * 100
    comunal['carga_total_real'] = comunal['matricula_total'] / comunal['docentes_total'] # Ratio macro
    
    comunal = comunal.reset_index()
    
    # -------------------------------------------------------------------------
    # VISUALIZACIÓN 1: CORRELACIONES COMUNALES
    # -------------------------------------------------------------------------
    cols_corr = ['ratio_docente_media', 'ratio_docente_std', 'ratio_curso_media', 
                 'matricula_total', 'pct_municipal', 'pct_pagado']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(comunal[cols_corr].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlación entre Variables Comunales', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/01_correlacion_comunal.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN 2: SCATTER PLOT MULTIDIMENSIONAL
    # -------------------------------------------------------------------------
    plt.figure(figsize=(14, 10))
    
    # Normalizar tamaño para el gráfico
    tamanos = comunal['matricula_total'] / comunal['matricula_total'].max() * 1000
    
    scatter = sns.scatterplot(
        data=comunal,
        x='ratio_docente_media',
        y='ratio_docente_std',
        size='matricula_total',
        hue='pct_municipal',
        sizes=(50, 800),
        palette='viridis',
        edgecolor='black',
        alpha=0.7
    )
    
    # Etiquetas para las comunas extremas
    for i in range(comunal.shape[0]):
        row = comunal.iloc[i]
        # Etiquetar si tiene alta desviación (>6) o alto ratio (>16) o es muy grande
        if row['ratio_docente_std'] > 6 or row['ratio_docente_media'] > 16 or row['matricula_total'] > 50000:
            plt.text(
                row['ratio_docente_media']+0.1, 
                row['ratio_docente_std'], 
                row['NOM_COM_RBD'], 
                fontsize=9,
                alpha=0.8
            )

    plt.title('Vulnerabilidad Comunal: Calidad Promedio vs. Desigualdad Interna', fontsize=16)
    plt.xlabel('Promedio de Alumnos por Profesor (Media Comunal)')
    plt.ylabel('Desigualdad Interna (Desviación Estándar)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/02_mapa_vulnerabilidad.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN 3: RANKING DE "SOBRECARGA" (Ratio Alumno/Curso)
    # -------------------------------------------------------------------------
    top_hacinamiento = comunal.sort_values('ratio_curso_media', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top_hacinamiento,
        x='ratio_curso_media',
        y='NOM_COM_RBD',
        palette='Reds_r'
    )
    plt.title('Top 15 Comunas con Mayor Hacinamiento en el Aula (Alumnos por Curso)', fontsize=14)
    plt.xlabel('Promedio de Alumnos por Curso')
    plt.ylabel('')
    plt.axvline(30, color='black', linestyle='--', label='Umbral Crítico (30)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/03_ranking_hacinamiento.png', dpi=300)
    plt.close()

    # Reporte de texto
    print("\nANÁLISIS A NIVEL COMUNAL - HALLAZGOS CLAVE")
    print("==========================================")
    
    print("\n1. COMUNAS CON MAYOR DESIGUALDAD INTERNA (Std Dev):")
    print(comunal.sort_values('ratio_docente_std', ascending=False).head(5)[['NOM_COM_RBD', 'ratio_docente_media', 'ratio_docente_std']].to_string(index=False))
    
    print("\n2. COMUNAS CON MAYOR HACINAMIENTO (Alumnos/Curso):")
    print(comunal.sort_values('ratio_curso_media', ascending=False).head(5)[['NOM_COM_RBD', 'ratio_curso_media']].to_string(index=False))
    
    print("\n3. Matriz de Correlación (Resumen):")
    print(comunal[cols_corr].corr().to_string())

if __name__ == "__main__":
    main()