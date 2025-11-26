import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'figures/analisis_demanda'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores
C_FREE = '#2ca02c'  # Verde
C_PAID = '#d62728'  # Rojo

def cargar_datos():
    print(">>> Cargando datos...")
    df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
    
    # Asegurar que ratio_alumno_curso sea numérico
    df['ratio_alumno_curso'] = pd.to_numeric(df['ratio_alumno_curso'], errors='coerce')
    
    # Clasificar Pago
    def clasificar_pago(row):
        pago = str(row['PAGO_MENSUAL']).upper()
        dep = str(row['categoria_dependencia']).upper()
        if pago == 'GRATUITO': return 'Gratuito'
        if pago == 'SIN INFORMACION' and any(x in dep for x in ['MUNICIPAL', 'SLEP', 'ADMIN']): return 'Gratuito'
        return 'Pagado'

    df['TIPO_PAGO'] = df.apply(clasificar_pago, axis=1)
    return df

def analisis_saturacion_aulas(df):
    """
    Analiza si los colegios pagados tienen aulas más llenas (mayor ratio alumno/curso),
    especialmente en comunas donde son escasos.
    """
    print(">>> Generando gráfico de Saturación de Aulas...")

    # 1. Calcular métricas por comuna
    comuna_stats = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Pct_Oferta_Pagada': (x['TIPO_PAGO'] == 'Pagado').mean(),
        'Cant_Colegios_Pagados': (x['TIPO_PAGO'] == 'Pagado').sum(),
        'Alumnos_Curso_Pagado': x.loc[x['TIPO_PAGO'] == 'Pagado', 'ratio_alumno_curso'].mean(),
        'Alumnos_Curso_Gratuito': x.loc[x['TIPO_PAGO'] == 'Gratuito', 'ratio_alumno_curso'].mean()
    }))

    # 2. Filtrar: Nos interesan comunas donde EXISTE oferta pagada pero no es dominante
    # (Ej: Comunas de clase media/baja donde hay pocos colegios particulares)
    # Filtro: Al menos 1 colegio pagado y menos del 50% de la oferta es pagada
    target_comunas = comuna_stats[
        (comuna_stats['Cant_Colegios_Pagados'] > 0) & 
        (comuna_stats['Pct_Oferta_Pagada'] < 0.50)
    ].sort_values('Pct_Oferta_Pagada')

    # Tomar las 15 comunas más representativas de este fenómeno (menor oferta pagada)
    sample = target_comunas.head(15)

    # 3. Graficar Comparativa
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(sample))
    width = 0.35
    
    # Barras
    plt.bar(x - width/2, sample['Alumnos_Curso_Gratuito'], width, label='Gratuito (Mun/Subv)', color=C_FREE, alpha=0.8)
    plt.bar(x + width/2, sample['Alumnos_Curso_Pagado'], width, label='Pagado (Part/Copago)', color=C_PAID, alpha=0.8)
    
    # Línea de referencia (ej. 35 alumnos es un curso bastante lleno)
    plt.axhline(y=35, color='gray', linestyle='--', alpha=0.5, label='Ref: 35 alumnos/curso')

    plt.xticks(x, sample.index, rotation=45, ha='right')
    plt.ylabel('Promedio de Alumnos por Curso (Aula)', fontsize=12)
    plt.title('Evidencia de Alta Demanda: Saturación de Aulas en Colegios Pagados\n(En comunas con baja oferta privada)', fontsize=14)
    plt.legend()
    
    # Anotaciones de brecha
    for i in x:
        val_paid = sample['Alumnos_Curso_Pagado'].iloc[i]
        val_free = sample['Alumnos_Curso_Gratuito'].iloc[i]
        diff = val_paid - val_free
        if diff > 5: # Si la diferencia es notable (>5 alumnos más)
            plt.text(i + width/2, val_paid + 0.5, f"+{diff:.0f}", ha='center', color='black', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_saturacion_aulas_curso.png'), dpi=150)
    plt.close()
    
    print(f"Gráfico guardado en {OUTPUT_DIR}/04_saturacion_aulas_curso.png")
    
    # Imprimir datos para verificación
    print("\n--- Datos de las Comunas Analizadas ---")
    print(sample[['Pct_Oferta_Pagada', 'Alumnos_Curso_Gratuito', 'Alumnos_Curso_Pagado']])

if __name__ == "__main__":
    df = cargar_datos()
    analisis_saturacion_aulas(df)