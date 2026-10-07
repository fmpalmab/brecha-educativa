"""
Compatibility wrapper for 05_analisis_estadistico_final_v2.py.
Forwards to brecha_educativa.visualizer and analysis.
"""

import pandas as pd
from brecha_educativa.config import CONSOLIDATED_FILE, FIGURES_DIR
from brecha_educativa.processor import process_raw_data
from brecha_educativa.visualizer import plot_correlation_heatmap

def main():
    if not CONSOLIDATED_FILE.is_file():
        process_raw_data()
    df = pd.read_csv(CONSOLIDATED_FILE)
    plot_correlation_heatmap(df, output_path=FIGURES_DIR / "matriz_correlacion_general.png")
    print("Matriz de correlación generada exitosamente.")

if __name__ == "__main__":
    main()