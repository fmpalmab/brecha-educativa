import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# =============================================================================
# CONFIGURACIÓN DE ESTILO
# =============================================================================
sns.set_theme(style="whitegrid", context="talk")

def main():
    print("--- Iniciando Análisis de Oferta y Demanda (Hito 3) ---")
    
    # Crear carpeta para guardar las imágenes
    output_dir = 'figures/oferta_demanda'
    os.makedirs(output_dir, exist_ok=True)

    # 1. CARGAR DATOS
    archivo_datos = 'data/processed/base_consolidada_rm_2024.csv'
    if not os.path.exists(archivo_datos):
        print("⚠️ Error: No se encuentra el archivo de datos procesados.")
        return
    
    df = pd.read_csv(archivo_datos)
    
    # 2. PROCESAMIENTO
    print("Clasificando y agrupando datos...")
    
    # Crear etiqueta legible para el gráfico
    # Si dice "GRATUITO" es "Gratuito", si no es "Pagado" (Copago o Particular)
    df['Tipo de Financiamiento'] = df['PAGO_MENSUAL_NORM'].apply(
        lambda x: 'Gratuito (Público/Subv)' if str(x).strip().upper() == 'GRATUITO' else 'Pagado (Copago/Privado)'
    )
    
    # Agrupar por Comuna y Tipo
    # Calculamos: Cuántos colegios hay (count) y Cuántos alumnos tienen en promedio (mean de MAT_TOTAL)
    comunal_pago = df.groupby(['NOM_COM_RBD', 'Tipo de Financiamiento']).agg({
        'RBD': 'count',           # Cantidad de colegios
        'MAT_TOTAL': 'mean'       # Tamaño promedio (Alumnos por colegio)
    }).reset_index()
    
    # Renombrar columnas para que Seaborn las use de etiquetas automáticamente
    comunal_pago.columns = ['Comuna', 'Tipo de Financiamiento', 'Cantidad de Colegios', 'Alumnos Promedio por Colegio']
    
    # Ordenar las comunas para que el gráfico sea legible (ej. por cantidad total de colegios)
    # Esto ayuda a que no salgan en orden alfabético aleatorio
    orden_comunas = df['NOM_COM_RBD'].value_counts().index

    # 3. GENERAR GRÁFICOS
    
    # --- GRÁFICO 1: Disponibilidad de Oferta (Cantidad de Colegios) ---
    print("Generando gráfico de Cantidad de Colegios...")
    plt.figure(figsize=(16, 8))
    
    sns.barplot(
        data=comunal_pago, 
        x='Comuna', 
        y='Cantidad de Colegios', 
        hue='Tipo de Financiamiento', 
        order=orden_comunas, # Mantiene un orden lógico (de más a menos colegios)
        palette='viridis'    # Paleta accesible y profesional
    )
    
    plt.title('Oferta Educativa: Cantidad de Colegios Gratuitos vs. Pagados por Comuna', fontsize=18, pad=20)
    plt.xticks(rotation=90, fontsize=10) # Rotar nombres de comunas para leerlos bien
    plt.ylabel('Número de Establecimientos')
    plt.xlabel('') # Quitamos "Comuna" porque es obvio
    plt.legend(title='Modalidad', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_cantidad_colegios_pago.png', dpi=300)
    plt.close()

    # --- GRÁFICO 2: Intensidad de la Demanda (Tamaño de los Colegios) ---
    print("Generando gráfico de Tamaño de Colegios (Mega-Colegios)...")
    plt.figure(figsize=(16, 8))
    
    sns.barplot(
        data=comunal_pago, 
        x='Comuna', 
        y='Alumnos Promedio por Colegio', 
        hue='Tipo de Financiamiento', 
        order=orden_comunas,
        palette='magma' # Usamos otra paleta para distinguir visualmente del anterior
    )
    
    plt.title('El Fenómeno de los "Mega-Colegios": Tamaño Promedio por Comuna', fontsize=18, pad=20)
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel('Promedio de Alumnos por Establecimiento')
    plt.xlabel('')
    
    # Línea de referencia: Tamaño "manejable" (ej. 500 alumnos) vs "Masivo"
    plt.axhline(y=800, color='red', linestyle='--', alpha=0.5, label='Zona de Alta Masividad (>800)')
    plt.legend(title='Modalidad', bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_tamano_promedio_colegios.png', dpi=300)
    plt.close()

    print(f"¡Listo! Gráficos guardados en la carpeta '{output_dir}'.")

if __name__ == "__main__":
    main()