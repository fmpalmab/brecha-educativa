import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'figures/narrativa_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores para la narrativa
C_FREE = '#2ca02c' # Verde (Gratuito)
C_PAID = '#d62728' # Rojo (Pagado)

def cargar_datos():
    print(">>> Cargando y consolidando datos...")
    # Ajusta las rutas si es necesario
    df_base = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
    df_s4 = pd.read_csv('data/raw/simce4b2024_rbd_preliminar.csv', sep=';', encoding='latin-1')
    df_s2 = pd.read_csv('data/raw/simce2m2024_rbd_preliminar.csv', sep=';', encoding='latin-1')

    # Unir SIMCE
    cols_4b = ['rbd', 'prom_lect4b_rbd', 'prom_mate4b_rbd']
    cols_2m = ['rbd', 'prom_lect2m_rbd', 'prom_mate2m_rbd']
    
    df_s4 = df_s4[cols_4b].rename(columns={'rbd': 'RBD', 'prom_lect4b_rbd': 'S4L', 'prom_mate4b_rbd': 'S4M'})
    df_s2 = df_s2[cols_2m].rename(columns={'rbd': 'RBD', 'prom_lect2m_rbd': 'S2L', 'prom_mate2m_rbd': 'S2M'})
    
    for df in [df_base, df_s4, df_s2]:
        df['RBD'] = pd.to_numeric(df['RBD'], errors='coerce')

    df = df_base.merge(df_s4, on='RBD', how='left').merge(df_s2, on='RBD', how='left')
    
    # Calcular SIMCE Promedio
    simce_cols = ['S4L', 'S4M', 'S2L', 'S2M']
    for col in simce_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['SIMCE_PROM'] = df[simce_cols].mean(axis=1)

    # Definir TIPO_PAGO (Gratuito vs Pagado)
    def clasificar_pago(row):
        pago = str(row['PAGO_MENSUAL']).upper()
        dep = str(row['categoria_dependencia']).upper()
        if pago == 'GRATUITO': return 'Gratuito'
        if pago == 'SIN INFORMACION' and any(x in dep for x in ['MUNICIPAL', 'SLEP', 'ADMIN']): return 'Gratuito'
        return 'Pagado'

    df['TIPO_PAGO'] = df.apply(clasificar_pago, axis=1)
    return df

def analisis_punto_1_ratio_docente(df):
    """
    Punto 1: En comunas de menores ingresos (o generales), ¿es peor el ratio en colegios pagados?
    """
    print(">>> Generando Punto 1: Paradoja del Ratio Alumno/Docente...")
    
    # Filtramos comunas con suficiente muestra
    comunas_grandes = df['NOM_COM_RBD'].value_counts()
    top_comunas = comunas_grandes[comunas_grandes > 10].index
    df_filter = df[df['NOM_COM_RBD'].isin(top_comunas)]

    # Calcular promedio de ratio por comuna y tipo pago
    ratio_stats = df_filter.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])['ratio_alumno_docente'].mean().reset_index()
    
    # Pivotar para calcular brecha
    pivot = ratio_stats.pivot(index='NOM_COM_RBD', columns='TIPO_PAGO', values='ratio_alumno_docente')
    pivot['Brecha_Ratio'] = pivot['Pagado'] - pivot['Gratuito'] # Si es positivo, Pagado tiene PEOR ratio (más alumnos por profe)
    
    # Ordenar por donde los pagados están "peor" (mayor carga)
    pivot = pivot.sort_values('Brecha_Ratio', ascending=False).head(15) # Top 15 casos extremos
    
    plt.figure(figsize=(12, 8))
    pivot[['Gratuito', 'Pagado']].plot(kind='bar', color=[C_FREE, C_PAID], width=0.8, figsize=(12,6))
    plt.title('Punto 1: Alumnos por Docente (Mayor barra = Peor atención personalizada)', fontsize=14)
    plt.ylabel('Cantidad de Alumnos por Profesor')
    plt.xlabel('Comuna')
    plt.legend(title='Tipo Financiamiento')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_paradoja_ratio_docente.png'))
    plt.close()

def analisis_punto_3_breakdown_simce(df):
    """
    Punto 3: Breakdown por comuna. ¿Son siempre mejores los pagados?
    """
    print(">>> Generando Punto 3: Breakdown de Calidad (SIMCE) por Comuna...")
    
    stats = df.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])['SIMCE_PROM'].mean().reset_index()
    pivot = stats.pivot(index='NOM_COM_RBD', columns='TIPO_PAGO', values='SIMCE_PROM').dropna()
    
    # Calcular Delta
    pivot['Ventaja_Pagado'] = pivot['Pagado'] - pivot['Gratuito']
    pivot = pivot.sort_values('Ventaja_Pagado', ascending=True) # De menor ventaja a mayor ventaja
    
    # Gráfico de "Dumbbell" o líneas conectadas
    plt.figure(figsize=(10, 12))
    
    # Líneas grises conectando puntos
    plt.hlines(y=pivot.index, xmin=pivot['Gratuito'], xmax=pivot['Pagado'], color='gray', alpha=0.5)
    
    # Puntos
    plt.scatter(pivot['Gratuito'], pivot.index, color=C_FREE, label='Gratuito', s=60, zorder=3)
    plt.scatter(pivot['Pagado'], pivot.index, color=C_PAID, label='Pagado', s=60, zorder=3)
    
    plt.title('Punto 3: Brecha SIMCE por Comuna (Gratuito vs Pagado)', fontsize=14)
    plt.xlabel('Puntaje Promedio SIMCE')
    plt.legend()
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_breakdown_simce_comunal.png'))
    plt.close()

def analisis_punto_4_demanda_cuello_botella(df):
    """
    Punto 4 y 5: Cuello de botella.
    En comunas con POCOS colegios pagados, ¿están estos colegios más llenos (mayor matrícula promedio)?
    """
    print(">>> Generando Punto 4: Análisis de Cuello de Botella (Oferta vs Tamaño)...")
    
    # 1. Calcular % de oferta pagada por comuna
    comuna_meta = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Pct_Oferta_Pagada': (x['TIPO_PAGO'] == 'Pagado').mean(),
        'Matricula_Promedio_Pagado': x.loc[x['TIPO_PAGO'] == 'Pagado', 'MAT_TOTAL'].mean(),
        'Matricula_Promedio_Gratuito': x.loc[x['TIPO_PAGO'] == 'Gratuito', 'MAT_TOTAL'].mean()
    })).dropna()
    
    # Filtrar comunas donde hay "Poca Oferta Pagada" (ej. menos del 30% de los colegios son pagados)
    # Estas son las comunas "populares" o periféricas donde podría haber cuello de botella.
    comunas_poca_oferta = comuna_meta[comuna_meta['Pct_Oferta_Pagada'] < 0.30].sort_values('Pct_Oferta_Pagada')
    
    # Seleccionar top representativas para el gráfico
    sample = comunas_poca_oferta.head(20)
    
    plt.figure(figsize=(14, 7))
    
    # Gráfico de barras agrupadas: Tamaño Promedio del Colegio
    x = np.arange(len(sample))
    width = 0.35
    
    plt.bar(x - width/2, sample['Matricula_Promedio_Gratuito'], width, label='Tamaño Colegio Gratuito', color=C_FREE, alpha=0.7)
    plt.bar(x + width/2, sample['Matricula_Promedio_Pagado'], width, label='Tamaño Colegio Pagado', color=C_PAID, alpha=0.7)
    
    plt.xticks(x, sample.index, rotation=90)
    plt.title('Punto 4: ¿Sobredemanda? Tamaño promedio de colegios en comunas con POCA oferta privada', fontsize=14)
    plt.ylabel('Promedio de Alumnos por Colegio (Matrícula Total)')
    plt.legend()
    
    # Nota explicativa en el gráfico
    plt.figtext(0.5, 0.01, "Nota: En comunas con poca oferta privada, si la barra roja es más alta, sugiere que los pocos colegios pagados están capturando mucha matrícula (cuello de botella).", ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(OUTPUT_DIR, '03_cuello_botella_demanda.png'))
    plt.close()

if __name__ == "__main__":
    df = cargar_datos()
    
    # Ejecutar análisis
    analisis_punto_1_ratio_docente(df)
    analisis_punto_3_breakdown_simce(df)
    analisis_punto_4_demanda_cuello_botella(df)
    
    print(f"\n✅ Narrativa generada exitosamente en: {OUTPUT_DIR}")