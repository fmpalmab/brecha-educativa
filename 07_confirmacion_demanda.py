import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores: Usaremos variaciones de Rojo para mostrar intensidad
COLOR_LOW_SUPPLY = '#a50f15'  # Rojo Oscuro (Escasez = Alta Presión)
COLOR_HIGH_SUPPLY = '#fb6a4a' # Rojo Claro (Abundancia = Menor Presión)

def main():
    print(">>> GENERANDO ANÁLISIS DE DEMANDA AGREGADA (<45% VS >45%) <<<")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
    except FileNotFoundError:
        print("❌ Falta archivo de datos.")
        return

    # Clasificación
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )

    # 2. Clasificar Comunas por Nivel de Oferta Privada
    print("Clasificando comunas (Baja vs Alta Oferta Privada)...")
    
    comuna_oferta = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Pct_Privada': (x['TIPO_PAGO'] == 'Pagado').sum() / len(x),
        'Total_Colegios': len(x)
    }))
    
    # Filtro de robustez: ignorar comunas con muy pocos colegios (<5)
    comuna_oferta = comuna_oferta[comuna_oferta['Total_Colegios'] > 5]
    
    def clasificar_entorno(pct):
        if pct < 0.45:
            return 'Zonas de Escasez (<45% Oferta Privada)'
        else:
            return 'Zonas de Abundancia (>45% Oferta Privada)'
            
    comuna_oferta['Tipo_Entorno'] = comuna_oferta['Pct_Privada'].apply(clasificar_entorno)
    
    # Merge para etiquetar cada colegio con el tipo de su comuna
    df = df.merge(comuna_oferta[['Tipo_Entorno']], left_on='NOM_COM_RBD', right_index=True, how='inner')

    # 3. Filtrar: Solo nos interesan los COLEGIOS PAGADOS
    # Queremos ver cómo "viven" los colegios pagados en estos dos mundos
    df_pagados = df[df['TIPO_PAGO'] == 'Pagado'].copy()
    
    print(f"Analizando {len(df_pagados)} colegios pagados...")

    # -------------------------------------------------------------------------
    # GRÁFICO 1: SATURACIÓN DE AULAS (Raincloud Plot)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Violin + Strip + Box
    ax = sns.violinplot(
        x='Tipo_Entorno', y='ratio_alumno_curso', data=df_pagados,
        palette={'Zonas de Escasez (<45% Oferta Privada)': COLOR_LOW_SUPPLY, 'Zonas de Abundancia (>45% Oferta Privada)': COLOR_HIGH_SUPPLY},
        inner=None, alpha=0.3
    )
    sns.stripplot(
        x='Tipo_Entorno', y='ratio_alumno_curso', data=df_pagados,
        color='black', alpha=0.3, size=2, jitter=0.25
    )
    sns.boxplot(
        x='Tipo_Entorno', y='ratio_alumno_curso', data=df_pagados,
        width=0.2, boxprops={'facecolor':'none', 'zorder':10}, showfliers=False
    )
    
    # Calcular medias para anotar
    medias = df_pagados.groupby('Tipo_Entorno')['ratio_alumno_curso'].mean()
    diff = medias['Zonas de Escasez (<45% Oferta Privada)'] - medias['Zonas de Abundancia (>45% Oferta Privada)']
    
    plt.axhline(35, color='gray', linestyle='--', label='Umbral Saturación (35)')
    plt.title(f'Efecto de la Escasez: Hacinamiento en el Aula\n(Diferencia: +{diff:.1f} alumnos/curso en zonas de escasez)', fontsize=15)
    plt.ylabel('Alumnos por Curso (Promedio)')
    plt.xlabel('')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/17_comparacion_agregada_saturacion.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # GRÁFICO 2: TAMAÑO DEL COLEGIO (Masividad)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(
        x='Tipo_Entorno', y='MAT_TOTAL', data=df_pagados,
        palette={'Zonas de Escasez (<45% Oferta Privada)': COLOR_LOW_SUPPLY, 'Zonas de Abundancia (>45% Oferta Privada)': COLOR_HIGH_SUPPLY},
        showfliers=False # Ocultar outliers extremos visualmente
    )
    
    # Medias
    medias_mat = df_pagados.groupby('Tipo_Entorno')['MAT_TOTAL'].mean()
    
    plt.title('La Respuesta del Mercado: Masividad Escolar', fontsize=16)
    plt.ylabel('Matrícula Total del Colegio')
    plt.xlabel('')
    
    # Anotación de Insight
    txt = f"En zonas de escasez, los colegios pagados son\nmasivos (Promedio: {medias_mat['Zonas de Escasez (<45% Oferta Privada)']:.0f} alumnos)\nvs zonas de abundancia ({medias_mat['Zonas de Abundancia (>45% Oferta Privada)']:.0f} alumnos)"
    plt.figtext(0.5, 0.02, txt, ha="center", fontsize=11, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f'{OUTPUT_DIR}/18_comparacion_agregada_tamano.png', dpi=300)
    plt.close()
    
    print(f"✅ Análisis agregado completado. Gráficos en {OUTPUT_DIR}")

if __name__ == "__main__":
    main()