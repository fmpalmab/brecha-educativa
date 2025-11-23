import pandas as pd
import geopandas as gpd
import unicodedata
import os

# Función de normalización (la misma que usamos)
def normalizar(texto):
    if pd.isna(texto): return ""
    texto = str(texto).upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = u"".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    return texto.strip()

def main():
    print("--- DIAGNÓSTICO DE NOMBRES DE COMUNA ---")
    
    # 1. Cargar tus datos (CSV)
    try:
        df = pd.read_csv('data/processed/base_consolidada_rm_2024.csv')
        comunas_csv = set(df['NOM_COM_RBD'].unique())
        print(f"✅ CSV cargado: {len(comunas_csv)} comunas encontradas.")
    except:
        print("❌ Error cargando CSV. Verifica la ruta.")
        return

    # 2. Cargar el Mapa (Shapefile)
    # Ajusta la ruta si tu shapefile tiene otro nombre
    try:
        # Busca el archivo .shp en la carpeta
        shape_path = 'data/external/comunas_rm/comunas_rm.shp' 
        if not os.path.exists(shape_path):
             # Intento de búsqueda automática
             for root, dirs, files in os.walk('data'):
                 for file in files:
                     if file.endswith(".shp"):
                         shape_path = os.path.join(root, file)
                         break
        
        gdf = gpd.read_file(shape_path)
        
        # Buscar la columna que tiene el nombre (a veces es NOM_COM, COMUNA, NAME, etc)
        col_nombre = [c for c in gdf.columns if 'NOM' in c.upper() or 'COM' in c.upper()][0]
        print(f"✅ Mapa cargado desde: {shape_path}")
        print(f"   Columna de nombre detectada: '{col_nombre}'")
        
        # Normalizar nombres del mapa
        gdf['nombre_norm'] = gdf[col_nombre].apply(normalizar)
        comunas_mapa = set(gdf['nombre_norm'].unique())
        
    except Exception as e:
        print(f"❌ Error cargando Mapa: {e}")
        return

    # 3. COMPARACIÓN (Aquí está la verdad)
    print("\n---------------------------------------------------")
    print("🔍 RESULTADOS DEL CRUCE:")
    
    en_ambos = comunas_csv.intersection(comunas_mapa)
    solo_csv = comunas_csv - comunas_mapa
    solo_mapa = comunas_mapa - comunas_csv

    print(solo_mapa)
    print("")
    print(solo_csv)
    
    print(f"Coincidencias perfectas: {len(en_ambos)}")
    print(f"Comunas en CSV sin mapa: {len(solo_csv)}")
    print(f"Comunas en Mapa sin datos: {len(solo_mapa)}")
    
    if len(solo_csv) > 0:
        print("\n⚠️  NOMBRES EN TU CSV QUE NO ENCUENTRAN MAPA (Revisa esto):")
        print(sorted(list(solo_csv)))
        
    if len(solo_mapa) > 0:
        print("\n⚠️  NOMBRES EN EL MAPA QUE NO TIENEN DATOS (Posibles candidatos):")
        # Mostramos solo los de la RM para no llenar la pantalla si el mapa es de todo Chile
        print(sorted(list(solo_mapa))[:20]) 

if __name__ == "__main__":
    main()