import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==========================================
# 1. CARGA DE DATOS
# ==========================================
print("--- 1. Cargando Datos ---")

# RUTA: Esta ruta puede cambiar dependiendo de la ubicación del archivo que deseamos analizar
RUTA_ARCHIVO = r'E:\Academic Content\Ciencias de datos 2\Proyecto\donantes_ong_nosql.csv'

try:
    df = pd.read_csv(RUTA_ARCHIVO)
    print(f"Datos cargados exitosamente: {df.shape[0]} registros.")
except FileNotFoundError:
    print(f"ERROR CRÍTICO: No se encuentra el archivo en:\n{RUTA_ARCHIVO}")
    sys.exit()

# ==========================================
# 2. LIMPIEZA
# ==========================================
print("\n--- 2. Aplicando Reglas de Limpieza y ETL ---")

df['canal_captacion'] = df['canal_captacion'].fillna('Desconocido')
df['interes_causa'] = df['interes_causa'].replace('Niñez', 'Desarrollo Infantil')

limite_monto = df['monto_promedio'].quantile(0.99)
df_clean = df[df['monto_promedio'] <= limite_monto].copy()
print(f"Registros después de limpieza de outliers: {df_clean.shape[0]}")

# ==========================================
# 3. PREPARACIÓN PARA ML
# ==========================================
print("\n--- 3. Preparación para ML (Split Train/Test) ---")

features = ['antiguedad_meses', 'monto_promedio', 'contactos_anuales', 'canal_captacion', 'interes_causa']
target = 'abandono'

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f" > Total registros usados: {len(X)}")
print(f" > Set de Entrenamiento (80%): {X_train.shape[0]} donantes")
print(f" > Set de Prueba (20%): {X_test.shape[0]} donantes")

categorical_features = ['canal_captacion', 'interes_causa']
numerical_features = ['antiguedad_meses', 'monto_promedio', 'contactos_anuales']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# ==========================================
# 4. ENTRENAMIENTO
# ==========================================
print("\n--- 4. Entrenando Modelos ---")

# Usamos class_weight='balanced'
modelo_lr = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))])
modelo_lr.fit(X_train, y_train)

modelo_rf = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])
modelo_rf.fit(X_train, y_train)

print("Modelos entrenados correctamente (con balanceo de clases).")

# ==========================================
# 5. GENERACIÓN DE MÉTRICAS (CSV PARA MARIO)
# ==========================================
print("\n" + "="*40)
print("   GENERANDO REPORTE DE MÉTRICAS (MARIO)")
print("="*40)

metricas_data = []

def registrar_metricas(nombre, modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metricas_data.append({
        'Modelo': nombre,
        'Accuracy_Global': round(acc, 4),
        'Recall_Abandono (Clase 1)': round(report['1']['recall'], 4),
        'Precision_Abandono (Clase 1)': round(report['1']['precision'], 4),
        'F1_Score_Abandono': round(report['1']['f1-score'], 4),
        'TP (Abandono Detectado)': tp,
        'FN (Abandono NO Detectado)': fn,
        'TN (Fiel Correcto)': tn,
        'FP (Falsa Alarma)': fp
    })
    
    print(f"Modelo procesado: {nombre} | Recall Clase 1: {report['1']['recall']:.2f}")

registrar_metricas("Regresion Logistica", modelo_lr, X_test, y_test)
registrar_metricas("Random Forest", modelo_rf, X_test, y_test)

df_mario = pd.DataFrame(metricas_data)

# ==========================================
# 6. EXPORTAR ARCHIVOS (PARA FANY Y MARIO)
# ==========================================
print("\n" + "="*40)
print("   GENERANDO INSIGHTS PARA DASHBOARD (FANY)")
print("="*40)

# CAMBIO ESTRATÉGICO: Usamos Regresión Logística para el CSV de Fany
# Razón: Tuvo mejor Recall (0.47 vs 0.05), detectará más gente en riesgo para el dashboard.
df['probabilidad_abandono'] = modelo_lr.predict_proba(df[features])[:, 1]

# Segmentación de Riesgo
df['riesgo_categoria'] = pd.cut(df['probabilidad_abandono'], 
                                bins=[0, 0.4, 0.7, 1.0], 
                                labels=['Bajo', 'Medio', 'Alto'])

# IMPRIMIR RESUMEN PARA FANY EN CONSOLA 
conteo_riesgo = df['riesgo_categoria'].value_counts().sort_index()
print("Resumen de Segmentación de Riesgo:")
print("-" * 30)
for categoria, cantidad in conteo_riesgo.items():
    print(f" > Riesgo {categoria}: \t{cantidad} donantes")
print("-" * 30)
# ----------------------------------------------------

df_fany = df[['id_donante', 'abandono', 'probabilidad_abandono', 'riesgo_categoria', 'antiguedad_meses', 'monto_promedio']]

archivo_fany = 'predicciones_finales_FANY.csv'
archivo_mario = 'metricas_modelo_MARIO.csv'

try:
    df_fany.to_csv(archivo_fany, index=False)
    print(f"✅ [1/2] Archivo para Fany generado: '{archivo_fany}'")
    
    df_mario.to_csv(archivo_mario, index=False)
    print(f"✅ [2/2] Archivo para Mario generado: '{archivo_mario}'")
    
    print("\n--- RESUMEN DE ENTREGA ---")
    print(f"1. Enviar '{archivo_fany}' a Fany (Dashboard).")
    print(f"2. Enviar '{archivo_mario}' a Mario (Informe de Negocio).")

except PermissionError as e:
    print(f"\n❌ ERROR DE PERMISO: Uno de los archivos CSV está abierto.")
    print(f"Detalle: {e}")
    print("Cierra los archivos en Excel y vuelve a ejecutar.")