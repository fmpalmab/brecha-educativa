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
COLOR_BENCHMARK = '#1f77b4' # Azul

def normalizar_texto(texto):
    if pd.isna(texto): return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def main():
    print(">>> GENERANDO DUMBBELL CHART UNIFICADO: HACINAMIENTO <<<")
    
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

    # 2. Seleccionar Comunas
    benchmarks = ['NUNOA', 'LA FLORIDA']
    
    comuna_oferta = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Pct_Privada': (x['TIPO_PAGO'] == 'Pagado').sum() / len(x),
        'Total_Colegios': len(x)
    }))
    
    # Filtro: Baja oferta (<45%) y tamaño suficiente (>8 colegios)
    periferia_critica = comuna_oferta[
        (comuna_oferta['Pct_Privada'] < 0.45) & 
        (comuna_oferta['Total_Colegios'] > 8)
    ].index.tolist()
    
    comunas_seleccionadas = list(set(periferia_critica + benchmarks))
    df_zoom = df[df['NOM_COM_RBD'].isin(comunas_seleccionadas)].copy()

    # 3. Preparar Datos para Dumbbell (Pivotear)
    # Queremos una fila por comuna con columnas: 'Gratuito', 'Pagado'
    stats = df_zoom.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])['ratio_alumno_curso'].mean().unstack()
    
    # Limpiar nulos (por si alguna comuna no tiene de algún tipo)
    stats = stats.dropna(subset=['Pagado']) 
    
    # 4. Ordenar
    # Lógica: Benchmarks primero (o destacados), luego ordenados por saturación en Pagados
    stats['Es_Benchmark'] = stats.index.isin(benchmarks)
    stats = stats.sort_values(['Es_Benchmark', 'Pagado'], ascending=[True, True]) # Benchmarks abajo para referencia, luego por saturación

    # 5. GRAFICAR
    # Altura dinámica según cantidad de comunas
    plt.figure(figsize=(12, 14))
    
    # A. Líneas conectoras (La "barra" gris)
    plt.hlines(
        y=stats.index, 
        xmin=stats['Gratuito'], 
        xmax=stats['Pagado'], 
        color='gray', 
        alpha=0.4, 
        linewidth=2
    )
    
    # B. Puntos
    plt.scatter(stats['Gratuito'], stats.index, color=COLOR_FREE, s=100, label='Gratuito', zorder=3)
    plt.scatter(stats['Pagado'], stats.index, color=COLOR_PAID, s=100, label='Pagado', zorder=3)
    
    # C. Línea de Saturación
    #plt.axvline(35, color='red', linestyle='--', linewidth=1, label='Umbral Saturación (35)')

    # D. Destacar Benchmarks en el Eje Y
    ax = plt.gca()
    for i, label in enumerate(ax.get_yticklabels()):
        text = label.get_text()
        if text in benchmarks:
            label.set_fontweight('bold')
            label.set_color(COLOR_BENCHMARK)
            label.set_fontsize(13)
            # Flecha indicadora
            ax.annotate('BENCHMARK', xy=(0, i), xytext=(-8, i), 
                       arrowprops=dict(arrowstyle="->", color=COLOR_BENCHMARK),
                       color=COLOR_BENCHMARK, fontsize=9, va='center')

    plt.title('Prueba de Demanda: Hacinamiento en el Aula por Comuna\n(Comparación: Periferia vs Benchmarks)', fontsize=16, pad=20)
    plt.xlabel('Promedio de Alumnos por Curso')
    plt.ylabel('')
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Ajustar márgenes
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/21_detalle_demanda_unificado.png', dpi=300)
    plt.close()
    
    print(f"✅ Gráfico unificado guardado en: {OUTPUT_DIR}/21_detalle_demanda_unificado.png")

if __name__ == "__main__":
    main()