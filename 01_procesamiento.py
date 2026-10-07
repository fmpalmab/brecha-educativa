"""
Compatibility wrapper for 01_procesamiento.py.
Forwards to brecha_educativa.processor.
"""

from brecha_educativa.processor import process_raw_data

def main():
    df = process_raw_data()
    print(f"Procesamiento completado: {len(df)} establecimientos consolidados.")

if __name__ == "__main__":
    main()