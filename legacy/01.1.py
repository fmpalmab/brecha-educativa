import pandas as pd

# Rutas de archivos (ajusta según tu estructura)
path_base = 'data/processed/base_consolidada_rm_2024.csv'
path_simce_4b = 'data/raw/simce4b2024_rbd_preliminar.csv'
path_simce_2m = 'data/raw/simce2m2024_rbd_preliminar.csv'

# 1. Cargar la base consolidada (esta suele ser UTF-8 si la generaste tú, si falla usa encoding='latin-1' también)
df_base = pd.read_csv(path_base, sep=',')

# 2. Cargar bases SIMCE con la codificación correcta
# Agregamos encoding='latin-1' para evitar el UnicodeDecodeError
df_simce_4b = pd.read_csv(path_simce_4b, sep=';', encoding='latin-1')
df_simce_2m = pd.read_csv(path_simce_2m, sep=';', encoding='latin-1')

# --- Resto del código de procesamiento ---

cols_4b = ['rbd', 'prom_lect4b_rbd', 'prom_mate4b_rbd']
cols_2m = ['rbd', 'prom_lect2m_rbd', 'prom_mate2m_rbd']

df_4b_sel = df_simce_4b[cols_4b].copy()
df_2m_sel = df_simce_2m[cols_2m].copy()

df_4b_sel.rename(columns={
    'rbd': 'RBD', 
    'prom_lect4b_rbd': 'simce_4b_lectura_2024', 
    'prom_mate4b_rbd': 'simce_4b_matematica_2024'
}, inplace=True)

df_2m_sel.rename(columns={
    'rbd': 'RBD', 
    'prom_lect2m_rbd': 'simce_2m_lectura_2024', 
    'prom_mate2m_rbd': 'simce_2m_matematica_2024'
}, inplace=True)

df_base['RBD'] = pd.to_numeric(df_base['RBD'], errors='coerce')
df_4b_sel['RBD'] = pd.to_numeric(df_4b_sel['RBD'], errors='coerce')
df_2m_sel['RBD'] = pd.to_numeric(df_2m_sel['RBD'], errors='coerce')

df_final = pd.merge(df_base, df_4b_sel, on='RBD', how='left')
df_final = pd.merge(df_final, df_2m_sel, on='RBD', how='left')

output_path = 'data/processed/base_consolidada_rm_2024_con_simce.csv'
df_final.to_csv(output_path, index=False, sep=',')

print("Proceso finalizado con éxito.")