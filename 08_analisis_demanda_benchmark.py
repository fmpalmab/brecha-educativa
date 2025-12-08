import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import numpy as np
import os

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores
COLOR_FREE = '#2ca02c'  # Verde
COLOR_PAID = '#d62728'  # Rojo
COLOR_BENCHMARK = '#1f77b4' # Azul para destacar los benchmark

def normalizar_texto(texto):
    if pd.isna(texto): return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def main():
    print(">>> GENERANDO ANÁLISIS DE CONFIRMACIÓN DE DEMANDA (BENCHMARK) <<<")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
        print(f"Datos cargados: {len(df)} registros.")
    except FileNotFoundError:
        print("❌ Error: No se encuentra el archivo base.")
        return

    # 2. Procesamiento Previo
    # Normalizar nombres de comunas
    df['NOM_COM_RBD'] = df['NOM_COM_RBD'].apply(normalizar_texto)
    
    # Clasificación Pago
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )
    
    # 3. Identificar Grupos de Comunas
    # Calculamos % de Oferta Pagada por comuna
    comuna_oferta = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Total_Colegios': len(x),
        'N_Pagados': (x['TIPO_PAGO'] == 'Pagado').sum()
    }))
    comuna_oferta['Pct_Privada'] = comuna_oferta['N_Pagados'] / comuna_oferta['Total_Colegios']
    
    # GRUPO 1: Comunas con Baja Oferta (< 45%)
    # Filtramos también que tengan un mínimo de colegios para no tomar zonas rurales vacías
    comunas_baja_oferta = comuna_oferta[
        (comuna_oferta['Pct_Privada'] < 0.45) & 
        (comuna_oferta['Total_Colegios'] > 5)
    ].index.tolist()
    
    # GRUPO 2: Benchmarks (Ñuñoa y La Florida)
    benchmarks = ['NUNOA', 'LA FLORIDA']
    
    print(f"Comunas Baja Oferta encontradas: {len(comunas_baja_oferta)}")
    print(f"Benchmarks: {benchmarks}")

    # 4. Preparar Datos para el Gráfico
    # Nos interesa ver cómo se comportan los colegios PAGADOS en estos lugares
    df_pagados = df[df['TIPO_PAGO'] == 'Pagado'].copy()
    
    # Etiquetamos cada colegio según su zona
    def clasificar_zona(comuna):
        if comuna in benchmarks:
            return 'Benchmark (Ñuñoa/La Florida)'
        elif comuna in comunas_baja_oferta:
            return 'Comunas con Déficit de Oferta (<45%)'
        else:
            return 'Otros'
            
    df_pagados['Grupo_Analisis'] = df_pagados['NOM_COM_RBD'].apply(clasificar_zona)
    
    # Filtramos solo los grupos de interés
    df_plot = df_pagados[df_pagados['Grupo_Analisis'] != 'Otros'].copy()

    # 5. GRÁFICO 1: SATURACIÓN DE AULAS (Hacinamiento)
    # ¿Están más llenos los cursos en las zonas de déficit?
    plt.figure(figsize=(10, 8))
    
    # Calcular promedios para línea de referencia
    promedio_deficit = df_plot[df_plot['Grupo_Analisis'] == 'Comunas con Déficit de Oferta (<45%)']['ratio_alumno_curso'].mean()
    promedio_bench = df_plot[df_plot['Grupo_Analisis'] == 'Benchmark (Ñuñoa/La Florida)']['ratio_alumno_curso'].mean()
    
    sns.barplot(
        data=df_plot, 
        x='Grupo_Analisis', 
        y='ratio_alumno_curso', 
        palette={'Comunas con Déficit de Oferta (<45%)': COLOR_PAID, 'Benchmark (Ñuñoa/La Florida)': COLOR_BENCHMARK},
        estimator=np.mean,
        ci=95,
        capsize=0.1
    )
    
    # Anotaciones de diferencia
    diff = promedio_deficit - promedio_bench
    plt.text(0, promedio_deficit + 1, f"{promedio_deficit:.1f} Alumnos/Curso", ha='center', fontweight='bold', color=COLOR_PAID)
    plt.text(1, promedio_bench + 1, f"{promedio_bench:.1f} Alumnos/Curso", ha='center', fontweight='bold', color=COLOR_BENCHMARK)
    
    plt.title('Prueba de Demanda: Saturación de Aulas en Colegios Pagados', fontsize=14, pad=20)
    plt.ylabel('Promedio de Alumnos por Curso')
    plt.xlabel('')
    plt.ylim(0, 45)
    
    # Mensaje clave
    plt.figtext(0.5, 0.02, 
                f"Insight: En zonas de baja oferta, los cursos tienen ~{diff:.1f} alumnos más que en zonas consolidadas.\n¡La demanda presiona la capacidad al máximo!", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'{OUTPUT_DIR}/18_comparacion_saturacion_benchmark.png', dpi=300)
    plt.close()

    # 6. GRÁFICO 2: TAMAÑO DEL COLEGIO (Masividad)
    plt.figure(figsize=(10, 8))
    
    sns.boxplot(
        data=df_plot, 
        x='Grupo_Analisis', 
        y='MAT_TOTAL', 
        palette={'Comunas con Déficit de Oferta (<45%)': COLOR_PAID, 'Benchmark (Ñuñoa/La Florida)': COLOR_BENCHMARK},
        showfliers=False # Ocultar outliers extremos para ver mejor la mediana
    )
    
    plt.title('Escala de la Oferta: Tamaño de los Colegios Pagados', fontsize=14)
    plt.ylabel('Matrícula Total por Colegio')
    plt.xlabel('')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/19_comparacion_tamano_benchmark.png', dpi=300)
    plt.close()
    
    print("✅ Gráficos de cierre generados exitosamente.")

if __name__ == "__main__":
    main()