import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    print("--- Iniciando Análisis Comunal Multidimensional ---")
    
    os.makedirs('figures/comunal', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    sns.set_theme(style="whitegrid")

    # 1. Cargar Datos
    df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
    
    # 2. Ingeniería de Atributos a Nivel Comunal
    # Primero, codificamos la dependencia para poder sumar
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
    
    # Guardar dataset comunal para uso futuro (ej. mapas)
    comunal.to_csv('data/processed/resumen_comunal_metricas.csv', index=False)
    
    # -------------------------------------------------------------------------
    # VISUALIZACIÓN 1: CORRELACIONES COMUNALES
    # -------------------------------------------------------------------------
    # ¿Qué variables se mueven juntas?
    cols_corr = ['ratio_docente_media', 'ratio_docente_std', 'ratio_curso_media', 
                 'matricula_total', 'pct_municipal', 'pct_pagado']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(comunal[cols_corr].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matriz de Correlación entre Variables Comunales', fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/comunal/01_correlacion_comunal.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN 2: SCATTER PLOT MULTIDIMENSIONAL
    # -------------------------------------------------------------------------
    # Eje X: Ratio Alumno/Docente (Media)
    # Eje Y: Desviación Estándar (Desigualdad interna)
    # Tamaño: Matrícula Total
    # Color: % de Colegios Municipales (Más azul = más público)
    
    plt.figure(figsize=(14, 10))
    scatter = sns.scatterplot(
        data=comunal,
        x='ratio_docente_media',
        y='ratio_docente_std',
        size='matricula_total',
        hue='pct_municipal',
        sizes=(100, 1000),
        palette='Blues',
        edgecolor='black',
        alpha=0.7
    )
    
    # Etiquetas para las comunas extremas
    for i in range(comunal.shape[0]):
        row = comunal.iloc[i]
        # Etiquetar si tiene alta desviación, alto ratio, o es muy grande
        if row['ratio_docente_std'] > 6 or row['ratio_docente_media'] > 20 or row['matricula_total'] > 40000:
            plt.text(
                row['ratio_docente_media']+0.2, 
                row['ratio_docente_std'], 
                row['NOM_COM_RBD'], 
                fontsize=9,
                alpha=0.8
            )

    plt.title('Mapa de Vulnerabilidad Comunal: Calidad Promedio vs. Desigualdad Interna', fontsize=16)
    plt.xlabel('Promedio de Alumnos por Profesor (Media Comunal)')
    plt.ylabel('Desigualdad Interna (Desviación Estándar)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Métricas')
    plt.tight_layout()
    plt.savefig('figures/comunal/02_mapa_vulnerabilidad.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN 3: RANKING DE "SOBRECARGA" (Ratio Alumno/Curso)
    # -------------------------------------------------------------------------
    # A veces hay pocos alumnos por profe, pero 45 alumnos por sala.
    
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
    plt.savefig('figures/comunal/03_ranking_hacinamiento.png', dpi=300)
    plt.close()

    # Reporte de texto
    with open('reports/analisis_comunal.txt', 'w', encoding='utf-8') as f:
        f.write("ANÁLISIS A NIVEL COMUNAL - HALLAZGOS CLAVE\n")
        f.write("==========================================\n\n")
        
        f.write("1. COMUNAS CON MAYOR DESIGUALDAD INTERNA (Std Dev):\n")
        f.write(comunal.sort_values('ratio_docente_std', ascending=False).head(5)[['NOM_COM_RBD', 'ratio_docente_media', 'ratio_docente_std']].to_string(index=False))
        f.write("\n\n")
        
        f.write("2. COMUNAS CON MAYOR HACINAMIENTO (Alumnos/Curso):\n")
        f.write(comunal.sort_values('ratio_curso_media', ascending=False).head(5)[['NOM_COM_RBD', 'ratio_curso_media']].to_string(index=False))
        f.write("\n\n")
        
        f.write("3. CORRELACIONES DESTACADAS:\n")
        corr_mtx = comunal[cols_corr].corr()
        f.write(corr_mtx.to_string())

    print("Análisis comunal completado. Archivos en 'figures/comunal' y 'reports/'.")

if __name__ == "__main__":
    main()