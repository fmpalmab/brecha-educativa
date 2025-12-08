import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIGURACIÓN ---
OUTPUT_DIR = 'figures/finales'
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# Colores
COLOR_FREE = '#2ca02c'  # Verde
COLOR_PAID = '#d62728'  # Rojo

def main():
    print(">>> GENERANDO ANÁLISIS DE DETALLE (CRITERIO: PEOR DESEMPEÑO) <<<")
    
    # 1. Cargar Datos
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024_final.csv')
    except FileNotFoundError:
        print("❌ Falta el archivo de datos final. Ejecuta el Bloque 1.")
        return

    # Clasificación Binaria
    df['TIPO_PAGO'] = df['PAGO_MENSUAL'].apply(
        lambda x: 'Gratuito' if str(x).strip().upper() == 'GRATUITO' or 'MUNICIPAL' in str(x).upper() else 'Pagado'
    )

    # -------------------------------------------------------------------------
    # PASO 1: SELECCIONAR COMUNAS CRÍTICAS (PEOR SIMCE + N > 5)
    # -------------------------------------------------------------------------
    print("Identificando comunas críticas por bajo desempeño...")
    
    # Calcular métricas por comuna
    # Usamos el promedio de ambos niveles para el ranking general
    df['SIMCE_GLOBAL'] = df[['SIMCE_4B_AVG', 'SIMCE_2M_AVG']].mean(axis=1)
    
    comuna_stats = df.groupby('NOM_COM_RBD').agg({
        'RBD': 'count',             # Cantidad de colegios
        'SIMCE_GLOBAL': 'mean'      # Promedio SIMCE
    }).reset_index()
    
    # FILTRO 1: Más de 5 establecimientos (para evitar comunas rurales con 1 colegio)
    comuna_stats = comuna_stats[comuna_stats['RBD'] > 5]
    
    # FILTRO 2: Las 15 con PEOR promedio (orden ascendente)
    top_comunas_criticas = comuna_stats.sort_values('SIMCE_GLOBAL', ascending=True).head(15)['NOM_COM_RBD'].tolist()
    
    print(f"Comunas seleccionadas (Bajo Desempeño): {top_comunas_criticas}")

    # -------------------------------------------------------------------------
    # PASO 2: GRAFICAR DISTRIBUCIÓN DETALLADA
    # -------------------------------------------------------------------------
    df_zoom = df[df['NOM_COM_RBD'].isin(top_comunas_criticas)].copy()
    
    niveles = [
        {'col': 'SIMCE_4B_AVG', 'nombre': '4° Básico', 'file': '4b'},
        {'col': 'SIMCE_2M_AVG', 'nombre': 'II Medio', 'file': '2m'}
    ]
    
    for nivel in niveles:
        col_simce = nivel['col']
        nombre_nivel = nivel['nombre']
        tag = nivel['file']
        
        # Ordenar el eje Y por el promedio de la comuna en ESTE nivel específico (de peor a mejor)
        order = df_zoom.groupby('NOM_COM_RBD')[col_simce].mean().sort_values(ascending=True).index
        
        plt.figure(figsize=(14, 12))
        
        # A. PUNTITOS INDIVIDUALES (Strip Plot)
        sns.stripplot(
            data=df_zoom, 
            y='NOM_COM_RBD', 
            x=col_simce, 
            hue='TIPO_PAGO', 
            palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID},
            order=order,
            alpha=0.5, 
            size=5, 
            jitter=0.25,
            dodge=False,
            zorder=1
        )
        
        # B. PROMEDIOS (Dumbbell)
        means = df_zoom.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])[col_simce].mean().unstack()
        
        # Líneas conectoras
        for comuna in order:
            if comuna in means.index:
                val_free = means.loc[comuna, 'Gratuito']
                val_paid = means.loc[comuna, 'Pagado']
                if pd.notnull(val_free) and pd.notnull(val_paid):
                    plt.plot([val_free, val_paid], [comuna, comuna], color='gray', linewidth=1.5, alpha=0.6, zorder=2)
        
        # Puntos Grandes (Promedios)
        plt.scatter(means['Gratuito'], means.index, color=COLOR_FREE, s=120, edgecolor='k', lw=1, label='Promedio Gratuito', zorder=3)
        plt.scatter(means['Pagado'], means.index, color=COLOR_PAID, s=120, edgecolor='k', lw=1, label='Promedio Pagado', zorder=3)

        # Estética
        plt.title(f'Zoom a la Precariedad: Distribución SIMCE {nombre_nivel}\n(15 Comunas con menor desempeño promedio)', fontsize=16, pad=15)
        plt.xlabel(f'Puntaje Promedio {nombre_nivel}')
        plt.ylabel('')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Leyenda limpia
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Colegio (Individual)', markerfacecolor='gray', markersize=5, alpha=0.5),
            Line2D([0], [0], marker='o', color='w', label='Promedio Gratuito', markerfacecolor=COLOR_FREE, markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Promedio Pagado', markerfacecolor=COLOR_PAID, markersize=10, markeredgecolor='k'),
        ]
        plt.legend(handles=legend_elements, loc='lower right', title='Leyenda')

        plt.tight_layout()
        save_path = f'{OUTPUT_DIR}/16_{tag}_detalle_distribucion_peor_desempeno.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Gráfico guardado: {save_path}")

if __name__ == "__main__":
    main()