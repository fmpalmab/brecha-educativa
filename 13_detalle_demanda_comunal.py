import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import unicodedata

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores
COLOR_FREE = '#2ca02c'  # Verde
COLOR_PAID = '#d62728'  # Rojo

def normalizar_texto(texto):
    if pd.isna(texto): return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def main():
    print(">>> GENERANDO DETALLE DE DEMANDA POR COMUNA (PERIFERIA VS BENCHMARK) <<<")
    
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
    df['NOM_COM_RBD'] = df['NOM_COM_RBD'].apply(normalizar_texto)

    # 2. Seleccionar Comunas de Interés
    # A. Benchmarks
    benchmarks = ['NUNOA', 'LA FLORIDA']
    
    # B. Comunas de Baja Oferta Privada (< 45%)
    comuna_oferta = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Pct_Privada': (x['TIPO_PAGO'] == 'Pagado').sum() / len(x),
        'Total_Colegios': len(x)
    }))
    
    # Filtramos comunas con oferta privada "escasa" pero que tengan suficientes colegios para ser relevantes
    periferia_critica = comuna_oferta[
        (comuna_oferta['Pct_Privada'] < 0.45) & 
        (comuna_oferta['Total_Colegios'] > 8) # Filtro para sacar comunas rurales muy chicas
    ].index.tolist()
    
    # Unir listas y asegurar que estén los benchmarks
    comunas_seleccionadas = list(set(periferia_critica + benchmarks))
    
    # Filtrar dataset
    df_zoom = df[df['NOM_COM_RBD'].isin(comunas_seleccionadas)].copy()

    # 3. Preparar Datos para el Gráfico
    # Calculamos el promedio de alumnos por curso por comuna y tipo
    plot_data = df_zoom.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])['ratio_alumno_curso'].mean().reset_index()
    
    # Identificar tipo de comuna para ordenar el gráfico
    def get_grupo(comuna):
        if comuna in benchmarks: return '1_BENCHMARK' # El 1 es para que salga arriba o abajo según orden
        return '2_PERIFERIA'
    
    plot_data['Grupo'] = plot_data['NOM_COM_RBD'].apply(get_grupo)
    
    # Ordenar: Primero por Grupo, luego por Hacinamiento Pagado descendente
    # Para ordenar, hacemos un pivot temporal
    pivot_sort = plot_data[plot_data['TIPO_PAGO'] == 'Pagado'].sort_values(['Grupo', 'ratio_alumno_curso'], ascending=[True, False])
    orden_comunas = pivot_sort['NOM_COM_RBD'].tolist()

    # 4. GRAFICAR
    plt.figure(figsize=(14, 10))
    
    # Barplot agrupado
    sns.barplot(
        data=plot_data,
        y='NOM_COM_RBD',
        x='ratio_alumno_curso',
        hue='TIPO_PAGO',
        order=orden_comunas,
        palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID},
        edgecolor='black',
        linewidth=0.5
    )
    
    # Línea de Saturación
    # plt.axvline(35, color='red', linestyle='--', linewidth=1.5, label='Saturación (35 Alumnos)')
    
    # Destacar los Benchmarks visualmente
    ax = plt.gca()
    for i, label in enumerate(ax.get_yticklabels()):
        if label.get_text() in benchmarks:
            label.set_fontweight('bold')
            label.set_color('#1f77b4') # Azul para destacar
            label.set_fontsize(11)
            # Agregar una pequeña flecha o texto indicando que es benchmark
            ax.text(-2, i, '★', color='#1f77b4', fontweight='bold', ha='right', va='center')

    plt.title('Prueba de Fuego: Hacinamiento por Comuna\n(Comparando Periferia con Baja Oferta vs Benchmarks)', fontsize=16, pad=20)
    plt.xlabel('Promedio de Alumnos por Curso (Aula)')
    plt.ylabel('')
    plt.legend(title='Tipo de Establecimiento', loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/21_detalle_demanda_comunas_nombre.png', dpi=300)
    plt.close()
    print(f"✅ Gráfico detallado generado: {OUTPUT_DIR}/21_detalle_demanda_comunas_nombre.png")

if __name__ == "__main__":
    main()