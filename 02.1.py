import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def main():
    print("--- Iniciando Análisis Profundo (Batch 2) ---")
    
    # 1. Cargar datos
    df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
    
    # Crear carpeta para nuevas figuras
    os.makedirs('figures/deep_dive', exist_ok=True)
    
    # Configuración visual
    sns.set_theme(style="whitegrid")
    
    # -------------------------------------------------------------------------
    # ANÁLISIS 1: HACINAMIENTO (Alumnos por Sala vs Alumnos por Profe)
    # -------------------------------------------------------------------------
    # Este gráfico revela la verdad: ¿Tienen muchos profes o solo cursos chicos?
    
    print("Generando matriz de Calidad (Scatter Plot)...")
    
    plt.figure(figsize=(12, 8))
    
    # Filtramos outliers extremos para visualizar mejor
    df_clean = df[(df['ratio_alumno_docente'] < 50) & (df['ratio_alumno_curso'] < 60)]
    
    sns.scatterplot(
        data=df_clean,
        x='ratio_alumno_docente',
        y='ratio_alumno_curso',
        hue='categoria_dependencia',
        alpha=0.6,
        palette='viridis'
    )
    
    # Líneas promedio para dividir en cuadrantes
    avg_docente = df_clean['ratio_alumno_docente'].mean()
    avg_curso = df_clean['ratio_alumno_curso'].mean()
    
    plt.axvline(avg_docente, color='red', linestyle='--', alpha=0.5)
    plt.axhline(avg_curso, color='red', linestyle='--', alpha=0.5)
    
    plt.title('Matriz de Calidad: Personificación (X) vs. Hacinamiento (Y)', fontsize=14)
    plt.xlabel('Alumnos por Profesor (Personificación)')
    plt.ylabel('Alumnos por Curso (Hacinamiento)')
    
    # Anotaciones de cuadrantes
    plt.text(avg_docente + 5, avg_curso + 10, "ZONA CRÍTICA\n(Pocos profes, salas llenas)", color='darkred', fontweight='bold')
    plt.text(avg_docente - 10, avg_curso - 10, "ZONA IDEAL\n(Muchos profes, salas pequeñas)", color='green', fontweight='bold')
    
    plt.savefig('figures/deep_dive/01_matriz_calidad.png', dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # ANÁLISIS 2: RURALIDAD (El factor olvidado)
    # -------------------------------------------------------------------------
    print("Analizando brecha Urbano vs Rural...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(x='RURAL_RBD', y='ratio_alumno_docente', data=df, ax=axes[0], palette="Set2")
    axes[0].set_title('Personificación: Urbano (0) vs Rural (1)')
    
    sns.boxplot(x='RURAL_RBD', y='ratio_alumno_curso', data=df, ax=axes[1], palette="Set2")
    axes[1].set_title('Hacinamiento: Urbano (0) vs Rural (1)')
    
    plt.tight_layout()
    plt.savefig('figures/deep_dive/02_brecha_rural.png', dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # ANÁLISIS 3: TOP Y BOTTOM 20 (Nombres y Apellidos)
    # -------------------------------------------------------------------------
    # A veces el insight es un caso específico.
    
    print("Identificando colegios extremos...")
    
    cols_interes = ['NOM_RBD', 'NOM_COM_RBD', 'categoria_dependencia', 'ratio_alumno_docente', 'ratio_alumno_curso']
    
    # Los "Peores" en personificación (Ratios más altos)
    peores = df.sort_values('ratio_alumno_docente', ascending=False).head(20)[cols_interes]
    
    # Los "Mejores" en personificación (Ratios más bajos, filtrando errores de datos con > 50 alumnos)
    mejores = df[df['MAT_TOTAL'] > 50].sort_values('ratio_alumno_docente', ascending=True).head(20)[cols_interes]
    
    # Guardar reporte de texto
    with open('reports/analisis_profundo_batch2.txt', 'w', encoding='utf-8') as f:
        f.write("REPORTE DE DEEP DIVE (BATCH 2)\n")
        f.write("================================\n\n")
        
        f.write("1. COMPARACIÓN URBANO VS RURAL:\n")
        rural_stats = df.groupby('RURAL_RBD')[['ratio_alumno_docente', 'ratio_alumno_curso']].mean()
        f.write(rural_stats.to_string())
        f.write("\n\n(Nota: 0 = Urbano, 1 = Rural)\n\n")
        
        f.write("2. TOP 20 COLEGIOS CON MAYOR CARGA POR DOCENTE (CRÍTICO):\n")
        f.write(peores.to_string(index=False))
        f.write("\n\n")
        
        f.write("3. TOP 20 COLEGIOS CON MEJOR PERSONIFICACIÓN (>50 alumnos):\n")
        f.write(mejores.to_string(index=False))
        f.write("\n\n")
        
        # Correlación
        corr = df[['ratio_alumno_docente', 'ratio_alumno_curso', 'MAT_TOTAL', 'orden_precio']].corr()
        f.write("4. MATRIZ DE CORRELACIÓN:\n")
        f.write(corr.to_string())

    print("¡Listo! Revisa la carpeta 'figures/deep_dive' y el reporte en 'reports/'.")

if __name__ == "__main__":
    main()