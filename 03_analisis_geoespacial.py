"""
Compatibility wrapper for 03_analisis_geoespacial.py.
Forwards to brecha_educativa.spatial.
"""

import pandas as pd
from brecha_educativa.config import CONSOLIDATED_FILE, FIGURES_DIR
from brecha_educativa.processor import process_raw_data
from brecha_educativa.spatial import plot_schools_map

def main():
    if not CONSOLIDATED_FILE.is_file():
        process_raw_data()
    df = pd.read_csv(CONSOLIDATED_FILE)
    plot_schools_map(df, output_path=FIGURES_DIR / "mapa_colegios_santiago.png")

if __name__ == "__main__":
    main()