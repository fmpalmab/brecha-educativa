import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# =============================================================================
# CONFIGURACIÓN VISUAL (ESTILO Y COLORES ACCESIBLES)
# =============================================================================
# Usamos estilos limpios y paletas "Colorblind Friendly"
sns.set_theme(style="whitegrid", context="talk") 
# 'viridis' y 'cividis' son perfectas para daltonismo
palette_main = "viridis" 
palette_diverging = "RdBu_r" # Para correlaciones (Rojo-Azul es seguro si es oscuro)

# Diccionario para traducir variables de código a "Español Humano"
LABELS_MAP = {
    'ratio_docente_media': 'Promedio Alumnos por Docente',
    'ratio_docente_std': 'Desigualdad Interna (Desv. Estándar)',
    'ratio_curso_media': 'Hacinamiento (Alumnos por Curso)',
    'matricula_total': 'Matrícula Total Comunal',
    'pct_municipal': '% Oferta Municipal',
    'pct_pagado': '% Oferta Pagada',
    'NOM_COM_RBD': 'Comuna'
}

def main():
    print("--- Generando Visualizaciones Hito 3 (Modo Accesible) ---")
    
    # Crear carpetas de salida
    os.makedirs('figures/finales', exist_ok=True)

    # 1. CARGA Y PREPARACIÓN DE DATOS
    # ------------------------------------------------------
    try:
        # Intentamos cargar el procesado. Si no existe, tendrás que correr el 01_limpieza.
        df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
    except FileNotFoundError:
        print("ERROR: No se encontró 'data/processed/base_consolidada_rm_2024.csv'.")
        print("Por favor ejecuta primero el script de limpieza.")
        return

    # --- PARCHE DE SEGURIDAD: Recuperar datos de cursos si faltan ---
    if 'ratio_alumno_curso' not in df.columns:
        print("⚠️ La columna 'ratio_alumno_curso' no existe. Intentando recuperarla del raw...")
        try:
            df_mat = pd.read_csv('data/raw/Matricula_2024.csv', sep=';', usecols=['RBD', 'CUR_SIM_TOT'])
            df_mat['RBD'] = pd.to_numeric(df_mat['RBD'], errors='coerce')
            df = df.merge(df_mat, on='RBD', how='left')
            df['ratio_alumno_curso'] = df['MAT_TOTAL'] / df['CUR_SIM_TOT']
            print("✅ Recuperación exitosa.")
        except Exception as e:
            print(f"❌ No se pudo calcular ratio curso: {e}")
            df['ratio_alumno_curso'] = np.nan

    # 2. AGREGACIÓN POR COMUNA (LA MAGIA DEL ANÁLISIS)
    # ------------------------------------------------------
    # Creamos las banderas para contar tipos de colegio
    df['es_municipal'] = df['categoria_dependencia'].str.contains('Municipal|SLEP', na=False).astype(int)
    df['es_pagado'] = df['categoria_dependencia'].str.contains('Pagado', na=False).astype(int)

    comunal = df.groupby('NOM_COM_RBD').agg({
        'ratio_alumno_docente': ['mean', 'std'],
        'ratio_alumno_curso': 'mean',
        'MAT_TOTAL': 'sum',
        'RBD': 'count',
        'es_municipal': 'sum',
        'es_pagado': 'sum'
    }).reset_index()

    # Aplanar nombre de columnas
    comunal.columns = [
        'NOM_COM_RBD', 'ratio_docente_media', 'ratio_docente_std', 
        'ratio_curso_media', 'matricula_total', 'num_colegios', 
        'num_municipal', 'num_pagado'
    ]

    # Calcular porcentajes
    comunal['pct_municipal'] = (comunal['num_municipal'] / comunal['num_colegios']) * 100
    
    # Filtrar comunas muy chicas (menos de 5 colegios) para evitar ruido
    comunal = comunal[comunal['num_colegios'] >= 5]

    # 3. GENERACIÓN DE GRÁFICOS
    # ======================================================

    # GRÁFICO A: LA MATRIZ DE VULNERABILIDAD (SCATTER PLOT)
    # ------------------------------------------------------
    print("Generando Mapa de Vulnerabilidad...")
    plt.figure(figsize=(14, 10))
    
    # Usamos 'size' para matrícula y 'hue' para el tipo de dependencia predominante
    # Palette 'viridis' es segura para daltónicos.
    scatter = sns.scatterplot(
        data=comunal,
        x='ratio_docente_media',
        y='ratio_docente_std',
        size='matricula_total',
        hue='pct_municipal',
        palette='viridis', 
        sizes=(50, 600),
        alpha=0.8,
        edgecolor='gray'
    )

    # Etiquetas Inteligentes: Solo etiquetamos los casos extremos para no ensuciar
    for _, row in comunal.iterrows():
        # Criterio: Alta desigualdad (>6) O Muy mal ratio (>20) O Muy buena (>30k alumnos)
        if row['ratio_docente_std'] > 6.5 or row['ratio_docente_media'] > 19 or row['matricula_total'] > 45000:
            plt.text(
                row['ratio_docente_media'] + 0.2, 
                row['ratio_docente_std'], 
                row['NOM_COM_RBD'], 
                fontsize=10,
                fontweight='bold',
                alpha=0.9
            )

    # Títulos y Ejes traducidos
    plt.title('Mapa de Vulnerabilidad Escolar: Calidad vs. Desigualdad', fontsize=18, pad=20)
    plt.xlabel(LABELS_MAP['ratio_docente_media'], fontsize=12)
    plt.ylabel(LABELS_MAP['ratio_docente_std'], fontsize=12)
    
    # Leyenda limpia
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Métricas")
    plt.tight_layout()
    plt.savefig('figures/finales/01_mapa_vulnerabilidad_accesible.png', dpi=300)
    plt.close()


    # GRÁFICO B: RANKING DE HACINAMIENTO (BAR PLOT)
    # ------------------------------------------------------
    print("Generando Ranking de Hacinamiento...")
    plt.figure(figsize=(12, 10))
    
    # Ordenamos por hacinamiento
    top_hacinamiento = comunal.sort_values('ratio_curso_media', ascending=False).head(20)
    
    # Barplot horizontal
    sns.barplot(
        data=top_hacinamiento,
        x='ratio_curso_media',
        y='NOM_COM_RBD',
        palette='cividis_r' # Invertido: oscuro = más hacinado (intuitivo y accesible)
    )
    
    # Línea de referencia crítica (30 alumnos por sala promedio es mucho)
    plt.axvline(30, color='black', linestyle='--', linewidth=2, label='Umbral Crítico (30)')
    
    plt.title('Top 20 Comunas con Mayor Hacinamiento (Alumnos por Curso)', fontsize=16)
    plt.xlabel(LABELS_MAP['ratio_curso_media'])
    plt.ylabel('')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/finales/02_ranking_hacinamiento_accesible.png', dpi=300)
    plt.close()


    # GRÁFICO C: MATRIZ DE CORRELACIÓN SIMPLIFICADA
    # ------------------------------------------------------
    print("Generando Matriz de Correlación...")
    plt.figure(figsize=(10, 8))
    
    cols_corr = ['ratio_docente_media', 'ratio_docente_std', 'ratio_curso_media', 'pct_municipal']
    corr_matrix = comunal[cols_corr].corr()
    
    # Renombramos el índice y columnas para que se lea bonito en el gráfico
    corr_matrix.index = [LABELS_MAP[c] for c in cols_corr]
    corr_matrix.columns = [LABELS_MAP[c] for c in cols_corr]
    
    # Heatmap con anotaciones
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap=palette_diverging, # RdBu_r es estándar para divergir, seguro para deuteranopia
        vmin=-1, vmax=1,
        linewidths=1,
        linecolor='white'
    )
    
    plt.title('¿Qué variables se mueven juntas?', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/finales/03_correlacion_accesible.png', dpi=300)
    plt.close()

    print("\n¡Listo! Las imágenes aptas para el informe están en 'figures/finales/'.")

if __name__ == "__main__":
    main()