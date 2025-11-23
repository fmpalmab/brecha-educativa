import pandas as pd
import numpy as np

# 1. Cargar los datos
# Usamos low_memory=False porque hay columnas mixtas
df_ee = pd.read_csv('data/raw/EE_2024.csv', sep=';', encoding='utf-8', low_memory=False)
df_mat = pd.read_csv('data/raw/Matricula_2024.csv', sep=';', encoding='utf-8', low_memory=False)
df_doc = pd.read_csv('data/raw/Docente_2024.csv', sep=';', encoding='utf-8', low_memory=False)

# --- CORRECCIÓN DE LA LLAVE RBD (CRÍTICO) ---
# Forzamos a número y eliminamos errores para que el merge no falle
df_ee['RBD'] = pd.to_numeric(df_ee['RBD'], errors='coerce')
df_mat['RBD'] = pd.to_numeric(df_mat['RBD'], errors='coerce')
df_doc['RBD'] = pd.to_numeric(df_doc['RBD'], errors='coerce')

df_ee = df_ee.dropna(subset=['RBD'])
df_mat = df_mat.dropna(subset=['RBD'])
df_doc = df_doc.dropna(subset=['RBD'])

df_ee['RBD'] = df_ee['RBD'].astype(int)
df_mat['RBD'] = df_mat['RBD'].astype(int)
df_doc['RBD'] = df_doc['RBD'].astype(int)
# -------------------------------------------

# 2. Filtrar solo Región Metropolitana (Código 13)
df_ee_rm = df_ee[df_ee['COD_REG_RBD'] == 13].copy()

# 3. Seleccionar columnas relevantes (AHORA INCLUYE PRECIO)
cols_ee = [
    'RBD', 'NOM_RBD', 'COD_COM_RBD', 'NOM_COM_RBD', 
    'COD_DEPE', 'COD_DEPE2', 'LATITUD', 'LONGITUD', 'RURAL_RBD',
    'PAGO_MENSUAL'  # <--- ¡NUEVA COLUMNA!
]
# Limpiamos nombres de columnas por si hay espacios extraños
df_ee_rm.columns = df_ee_rm.columns.str.strip() 
# Verificamos que existan antes de filtrar
cols_existentes = [c for c in cols_ee if c in df_ee_rm.columns]
df_ee_rm = df_ee_rm[cols_existentes]

cols_mat = ['RBD', 'MAT_TOTAL', 'CUR_SIM_TOT']
cols_doc = ['RBD', 'DC_TOT']

# 4. Unir los DataFrames
df_master = df_ee_rm.merge(df_mat[cols_mat], on='RBD', how='left')
df_master = df_master.merge(df_doc[cols_doc], on='RBD', how='left')

# 5. Limpieza de datos nulos y ceros técnicos
df_master = df_master.dropna(subset=['LATITUD', 'LONGITUD', 'DC_TOT', 'MAT_TOTAL'])
df_master = df_master[df_master['DC_TOT'] > 0] 
df_master = df_master[df_master['MAT_TOTAL'] > 0]

# 6. Crear Indicador de Personificación
df_master['ratio_alumno_docente'] = df_master['MAT_TOTAL'] / df_master['DC_TOT']
df_master['ratio_alumno_curso'] = df_master['MAT_TOTAL'] / df_master['CUR_SIM_TOT']

# 7. Clasificación de Dependencia (Legible)
def clasificar_dependencia(cod):
    if cod == 1: return 'Municipal (Corporación)'
    elif cod == 2: return 'Municipal (DAEM)'
    elif cod == 3: return 'Particular Subvencionado'
    elif cod == 4: return 'Particular Pagado'
    elif cod == 5: return 'Admin. Delegada'
    elif cod == 6: return 'SLEP (Público)'
    else: return 'Otro'

df_master['categoria_dependencia'] = df_master['COD_DEPE'].apply(clasificar_dependencia)

# 8. Limpieza y Orden del Precio (Para poder graficarlo)
# El campo PAGO_MENSUAL suele venir como texto: "GRATUITO", "$1.000 A $10.000", etc.
# Vamos a crear una columna numérica ordenada para facilitar los mapas de color.
precio_map = {
    'GRATUITO': 0,
    '$1.000 A $10.000': 1,
    '$10.001 A $25.000': 2,
    '$25.001 A $50.000': 3,
    '$50.001 A $100.000': 4,
    'MAS DE $100.000': 5,
    'SIN INFORMACION': -1 # O puedes usar np.nan
}

df_master['orden_precio'] = df_master['PAGO_MENSUAL'].map(precio_map).fillna(-1)

# Vista previa para verificar
print(f"Total establecimientos: {len(df_master)}")
print("\nEjemplo de datos con precio:")
print(df_master[['NOM_RBD', 'categoria_dependencia', 'ratio_alumno_docente', 'PAGO_MENSUAL']].head())

# Guardar
df_master.to_csv('data/processed/base_consolidada_rm_2024.csv', index=False)
print("¡Archivo guardado con precios incluidos!")