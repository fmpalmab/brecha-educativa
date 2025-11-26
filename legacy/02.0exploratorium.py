import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def main():
    print("--- Iniciando Análisis Estadístico ---")
    
    # 1. Configuración de directorios y estilos
    # Creamos carpetas para organizar las salidas
    os.makedirs('figures', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Estilo profesional para los gráficos
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'figure.max_open_warning': 0}) # Evitar warnings por muchas figuras

    # 2. Cargar datos procesados
    input_file = 'data/processed/base_consolidada_rm_2024.csv'
    if not os.path.exists(input_file):
        sys.exit(f"ERROR: No se encontró {input_file}. Ejecuta 01_limpieza_datos.py primero.")
        
    df = pd.read_csv(input_file)
    print(f"Datos cargados: {len(df)} registros.")

    # Archivo de reporte de texto
    report_path = 'reports/hallazgos_estadisticos.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE ESTADÍSTICAS - BRECHA EDUCATIVA RM\n")
        f.write("===============================================\n\n")

        # ---------------------------------------------------------
        # ANÁLISIS 1: DESIGUALDAD INTRA-COMUNAL (La "Lotería" Educativa)
        # ---------------------------------------------------------
        print("Generando análisis de varianza por comuna...")
        
        stats_comuna = df.groupby('NOM_COM_RBD')['ratio_alumno_docente'].agg(
            media='mean',
            mediana='median',
            desviacion='std',
            count='count'
        ).reset_index()

        # Filtramos comunas con suficientes datos (>10 colegios)
        stats_comuna = stats_comuna[stats_comuna['count'] > 10]
        
        # Top 10 comunas más desiguales (Mayor desviación estándar)
        top_desigualdad = stats_comuna.sort_values('desviacion', ascending=False).head(10)
        
        f.write("1. TOP 10 COMUNAS CON MAYOR VARIANZA INTERNA (DESIGUALDAD):\n")
        f.write(top_desigualdad[['NOM_COM_RBD', 'media', 'desviacion']].to_string(index=False))
        f.write("\n\nInsight: En estas comunas, el promedio es engañoso porque conviven colegios hacinados con otros vacíos.\n\n")

        # GRÁFICO 1: Boxplot Ranking
        plt.figure(figsize=(16, 10)) # Tamaño grande para que se lean las etiquetas
        
        # Ordenar comunas por mediana para el gráfico
        order = df.groupby('NOM_COM_RBD')['ratio_alumno_docente'].median().sort_values().index
        
        sns.boxplot(
            data=df, 
            x='NOM_COM_RBD', 
            y='ratio_alumno_docente', 
            order=order,
            palette="viridis",
            showfliers=False, # Ocultar outliers extremos para limpieza visual
            linewidth=1
        )
        
        plt.xticks(rotation=90, fontsize=8)
        plt.title('Distribución de Personificación por Comuna (Ranking de Medianas)', fontsize=16)
        plt.ylabel('Estudiantes por Docente (Menos es mejor)')
        plt.xlabel('')
        plt.axhline(y=df['ratio_alumno_docente'].mean(), color='r', linestyle='--', label='Promedio Regional')
        plt.legend()
        plt.tight_layout()
        
        # GUARDAR IMAGEN
        plt.savefig('figures/01_boxplot_ranking_comunas.png', dpi=300)
        plt.close() # Cerrar para liberar memoria

        # ---------------------------------------------------------
        # ANÁLISIS 2: DEPENDENCIA (Público vs Privado)
        # ---------------------------------------------------------
        print("Generando análisis por dependencia...")
        
        stats_dep = df.groupby('categoria_dependencia')['ratio_alumno_docente'].describe()
        f.write("2. ESTADÍSTICAS POR TIPO DE DEPENDENCIA:\n")
        f.write(stats_dep.to_string())
        f.write("\n\n")

        # GRÁFICO 2: Densidad (KDE)
        plt.figure(figsize=(12, 7))
        sns.kdeplot(
            data=df, 
            x='ratio_alumno_docente', 
            hue='categoria_dependencia', 
            fill=True, 
            common_norm=False, 
            palette="tab10",
            alpha=0.2,
            linewidth=2
        )
        plt.title('Comparación de Densidad: ¿Qué tipo de colegio está más saturado?', fontsize=14)
        plt.xlim(0, 45)
        plt.xlabel('Ratio Alumnos/Docente')
        plt.tight_layout()
        
        # GUARDAR IMAGEN
        plt.savefig('figures/02_densidad_dependencia.png', dpi=300)
        plt.close()

        # ---------------------------------------------------------
        # ANÁLISIS 3: IMPACTO DEL PRECIO (Murallas Económicas)
        # ---------------------------------------------------------
        print("Generando análisis de precios...")
        
        # Filtramos los que tienen precio informado
        df_precios = df[df['orden_precio'] >= 0].copy()
        
        # GRÁFICO 3: Boxplot Precios
        plt.figure(figsize=(12, 8))
        
        orden_precios = [
            'GRATUITO', 
            '$1.000 A $10.000', 
            '$10.001 A $25.000', 
            '$25.001 A $50.000', 
            '$50.001 A $100.000', 
            'MAS DE $100.000'
        ]
        
        sns.boxplot(
            data=df_precios,
            x='PAGO_MENSUAL',
            y='ratio_alumno_docente',
            order=orden_precios,
            palette="Blues"
        )
        
        plt.xticks(rotation=30)
        plt.title('Relación Precio vs. Calidad (Ratio Alumno/Docente)', fontsize=14)
        plt.tight_layout()
        
        # GUARDAR IMAGEN
        plt.savefig('figures/03_impacto_precio.png', dpi=300)
        plt.close()
        
        f.write("3. ANÁLISIS DE PRECIOS COMPLETADO.\n")
        f.write("Revisar gráfico 'figures/03_impacto_precio.png' para ver el punto de quiebre.\n")

    print("\n--- Proceso finalizado ---")
    print("Gráficos guardados en: figures/")
    print("Resumen estadístico en: reports/hallazgos_estadisticos.txt")

if __name__ == "__main__":
    main()