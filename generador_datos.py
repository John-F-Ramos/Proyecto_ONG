import pandas as pd
import numpy as np
import random

# Configuración inicial
np.random.seed(42)
NUM_REGISTROS = 5000

# 1. Generar IDs (en NoSQL esto sería el _id, pero usamos un int por ahora)
ids = np.arange(1001, 1001 + NUM_REGISTROS)

# 2. Variables Categóricas
canales = ['Redes Sociales', 'Evento', 'Calle', 'Referido', 'Email']
causas = ['Niñez', 'Salud', 'Ambiente', 'Humanitaria', 'Animales']

data_canales = np.random.choice(canales, NUM_REGISTROS, p=[0.3, 0.1, 0.25, 0.15, 0.2])
data_causas = np.random.choice(causas, NUM_REGISTROS)

# 3. Variables Numéricas
antiguedad = np.random.randint(1, 61, NUM_REGISTROS)
contactos = np.random.randint(0, 13, NUM_REGISTROS)

# Monto con lógica de outliers (para limpiar después)
monto = np.random.normal(25, 10, NUM_REGISTROS)
monto = np.abs(monto) + 5
# Crear Outliers (Errores para limpiar)
indices_outliers = np.random.choice(range(NUM_REGISTROS), size=15, replace=False)
monto[indices_outliers] = monto[indices_outliers] * 100 

# 4. Target (Churn)
prob_abandono = 0.2 + np.where(monto < 10, 0.3, 0) + np.where(antiguedad < 3, 0.2, 0)
prob_abandono = np.clip(prob_abandono, 0, 1)
abandono = np.random.binomial(1, prob_abandono)

# 5. Crear DataFrame
df = pd.DataFrame({
    'id_donante': ids,
    'antiguedad_meses': antiguedad,
    'monto_promedio': np.round(monto, 2),
    'canal_captacion': data_canales,
    'interes_causa': data_causas,
    'contactos_anuales': contactos,
    'abandono': abandono
})

# 6. Inyectar Nulos (Para limpiar después)
indices_nulos = np.random.choice(df.index, size=int(NUM_REGISTROS * 0.05), replace=False)
df.loc[indices_nulos, 'canal_captacion'] = np.nan

# 7. Exportar a CSV listo para NoSQL
# index=False es importante para que no cree una columna extra de índice
df.to_csv('donantes_ong_nosql.csv', index=False, encoding='utf-8')

print("Archivo 'donantes_ong_nosql.csv' generado exitosamente.")