import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    print("--- Generando Comparación Oferta: Pagado vs Gratuito ---")
    
    os.makedirs('figures/comparativa', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Cargar Datos
    try:
        df = pd.read_csv('fmpalmab/brecha-educativa/brecha-educativa-main/data/processed/base_consolidada_rm_2024.csv')
    except FileNotFoundError:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')

    # 2. Clasificación Binaria (Gratuito vs Pagado)
    # Filtramos 'SIN INFORMACION' para ser más precisos
    df = df[df['PAGO_MENSUAL'] != 'SIN INFORMACION'].copy()
    
    df['tipo_pago'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'GRATUITO' if str(x).strip().upper() == 'GRATUITO' else 'PAGADO'
    )

    # 3. Agrupación por Comuna
    comunal = df.groupby(['NOM_COM_RBD', 'tipo_pago']).agg({
        'RBD': 'count',          # Número de colegios
        'MAT_TOTAL': 'mean'      # Tamaño promedio (Matrícula)
    }).reset_index()
    
    # Renombrar columnas
    comunal.rename(columns={'RBD': 'num_colegios', 'MAT_TOTAL': 'tamano_promedio'}, inplace=True)

    # 4. Ordenar para el gráfico (por cantidad total de colegios en la comuna)
    orden_comunas = df.groupby('NOM_COM_RBD')['RBD'].count().sort_values(ascending=False).index
    
    # GRÁFICO 1: Número de Colegios (Oferta)
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=comunal,
        x='NOM_COM_RBD',
        y='num_colegios',
        hue='tipo_pago',
        order=orden_comunas,
        palette={'GRATUITO': '#2ecc71', 'PAGADO': '#e74c3c'} # Verde y Rojo (distinguibles)
    )
    plt.title('Cantidad de Colegios por Comuna: Gratuitos vs. Pagados', fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel('Número de Establecimientos')
    plt.xlabel('')
    plt.legend(title='Tipo de Financiamiento')
    plt.tight_layout()
    plt.savefig('figures/comparativa/01_cantidad_colegios_pago.png', dpi=300)
    plt.show()

    # GRÁFICO 2: Tamaño Promedio (Demanda/Hacinamiento potencial)
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=comunal,
        x='NOM_COM_RBD',
        y='tamano_promedio',
        hue='tipo_pago',
        order=orden_comunas,
        palette={'GRATUITO': '#2ecc71', 'PAGADO': '#e74c3c'}
    )
    plt.title('Tamaño Promedio del Colegio (Matrícula): Gratuito vs. Pagado', fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel('Promedio de Alumnos por Colegio')
    plt.xlabel('')
    plt.legend(title='Tipo de Financiamiento')
    plt.tight_layout()
    plt.savefig('figures/comparativa/02_tamano_colegios_pago.png', dpi=300)
    plt.show()

    # Resumen de texto
    print("\nResumen de Hallazgos (Top 5 Comunas por Tamaño de Colegio Pagado):")
    top_tamano_pagado = comunal[comunal['tipo_pago'] == 'PAGADO'].sort_values('tamano_promedio', ascending=False).head(5)
    print(top_tamano_pagado[['NOM_COM_RBD', 'tamano_promedio', 'num_colegios']])

if __name__ == "__main__":
    main()