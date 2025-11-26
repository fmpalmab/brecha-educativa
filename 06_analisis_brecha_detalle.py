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
    print(">>> GENERANDO ANÁLISIS DE DETALLE: SOBRERREPRESENTACIÓN Y DISTRIBUCIÓN <<<")
    
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
    # PASO 1: IDENTIFICAR COMUNAS CON MAYOR GAP DE SOBRERREPRESENTACIÓN
    # -------------------------------------------------------------------------
    print("Identificando comunas críticas (Demanda > Oferta en sector Pagado)...")
    
    comuna_agg = df.groupby('NOM_COM_RBD').apply(lambda x: pd.Series({
        'Total_Colegios': len(x),
        'Total_Matricula': x['MAT_TOTAL'].sum(),
        'Col_Pagados': (x['TIPO_PAGO'] == 'Pagado').sum(),
        'Mat_Pagada': x.loc[x['TIPO_PAGO'] == 'Pagado', 'MAT_TOTAL'].sum()
    })).dropna()
    
    # Cálculo de Gap
    comuna_agg['Pct_Oferta_Pagada'] = comuna_agg['Col_Pagados'] / comuna_agg['Total_Colegios']
    comuna_agg['Pct_Demanda_Pagada'] = comuna_agg['Mat_Pagada'] / comuna_agg['Total_Matricula']
    
    # GAP POSITIVO = La demanda está más concentrada en pagados que la oferta física
    # (Ej: 60% colegios son pagados, pero 80% alumnos van a ellos)
    comuna_agg['GAP_SOBRERREPRESENTACION'] = comuna_agg['Pct_Demanda_Pagada'] - comuna_agg['Pct_Oferta_Pagada']
    
    # Seleccionar Top 15 Comunas con mayor Gap Positivo
    top_comunas = comuna_agg.sort_values('GAP_SOBRERREPRESENTACION', ascending=False).head(15).index.tolist()
    
    print(f"Comunas seleccionadas: {top_comunas}")

    # -------------------------------------------------------------------------
    # PASO 2: GRAFICAR DISTRIBUCIÓN DETALLADA (DUMBBELL + STRIP PLOT)
    # -------------------------------------------------------------------------
    # Filtramos la base solo para estas comunas
    df_zoom = df[df['NOM_COM_RBD'].isin(top_comunas)].copy()
    
    # Iteramos por nivel (4to Básico y II Medio)
    niveles = [
        {'col': 'SIMCE_4B_AVG', 'nombre': '4° Básico', 'file': '4b'},
        {'col': 'SIMCE_2M_AVG', 'nombre': 'II Medio', 'file': '2m'}
    ]
    
    for nivel in niveles:
        col_simce = nivel['col']
        nombre_nivel = nivel['nombre']
        tag = nivel['file']
        
        print(f"Generando gráfico para {nombre_nivel}...")
        
        # Preparar datos para el gráfico
        # Ordenar comunas por el Gap calculado antes para mantener consistencia
        df_zoom['NOM_COM_RBD'] = pd.Categorical(df_zoom['NOM_COM_RBD'], categories=top_comunas, ordered=True)
        df_plot = df_zoom.dropna(subset=[col_simce]).sort_values('NOM_COM_RBD')
        
        if df_plot.empty:
            continue

        plt.figure(figsize=(14, 12))
        
        # A. PUNTITOS INDIVIDUALES (Strip Plot)
        # jitter=True dispersa los puntos para que no se solapen
        sns.stripplot(
            data=df_plot, 
            y='NOM_COM_RBD', 
            x=col_simce, 
            hue='TIPO_PAGO', 
            palette={'Gratuito': COLOR_FREE, 'Pagado': COLOR_PAID},
            alpha=0.4,  # Transparencia para ver densidad
            size=4, 
            jitter=0.25,
            dodge=False, # Superponer en la misma línea de la comuna
            zorder=1
        )
        
        # B. PROMEDIOS Y LÍNEA CONECTORA (Dumbbell)
        # Calculamos promedios para dibujar las líneas
        means = df_plot.groupby(['NOM_COM_RBD', 'TIPO_PAGO'])[col_simce].mean().unstack()
        
        # Dibujar líneas grises conectando los promedios
        for comuna in top_comunas:
            if comuna in means.index:
                val_free = means.loc[comuna, 'Gratuito']
                val_paid = means.loc[comuna, 'Pagado']
                if pd.notnull(val_free) and pd.notnull(val_paid):
                    plt.plot([val_free, val_paid], [comuna, comuna], color='gray', linewidth=2, alpha=0.8, zorder=2)
        
        # Dibujar los puntos grandes de los promedios encima
        # Gratuito
        plt.scatter(
            means['Gratuito'], means.index, 
            color=COLOR_FREE, s=150, edgecolor='black', linewidth=1.5, label='Promedio Gratuito', zorder=3
        )
        # Pagado
        plt.scatter(
            means['Pagado'], means.index, 
            color=COLOR_PAID, s=150, edgecolor='black', linewidth=1.5, label='Promedio Pagado', zorder=3
        )

        # Detalles estéticos
        plt.title(f'Realidad detrás del Promedio: Distribución SIMCE {nombre_nivel}\n(En Comunas con alta Sobrerrepresentación de Demanda Privada)', fontsize=16, pad=15)
        plt.xlabel('Puntaje SIMCE')
        plt.ylabel('')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Arreglar leyenda (estaba duplicada por el stripplot y el scatter manual)
        handles, labels = plt.gca().get_legend_handles_labels()
        # Seleccionamos solo los últimos 2 (los scatters grandes) o creamos custom
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Colegio (Individual)', markerfacecolor='gray', markersize=5, alpha=0.5),
            Line2D([0], [0], marker='o', color='w', label='Promedio Gratuito', markerfacecolor=COLOR_FREE, markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='Promedio Pagado', markerfacecolor=COLOR_PAID, markersize=10, markeredgecolor='k'),
            Line2D([0], [0], color='gray', lw=2, label='Brecha Promedio')
        ]
        plt.legend(handles=legend_elements, loc='lower right', title='Leyenda')

        plt.tight_layout()
        save_path = f'{OUTPUT_DIR}/16_{tag}_detalle_distribucion_sobrerrepresentacion.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Gráfico guardado: {save_path}")

if __name__ == "__main__":
    main()