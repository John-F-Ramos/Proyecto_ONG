# 🤝 Proyecto ONG - Predicción de Abandono de Donantes

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 📋 Descripción del Proyecto

Este proyecto utiliza **Machine Learning** para predecir el abandono (churn) de donantes en una organización no gubernamental (ONG). El objetivo principal es identificar de manera temprana a los donantes que tienen mayor probabilidad de dejar de contribuir, permitiendo implementar estrategias de retención proactivas.

## 🎯 Objetivos

- **Predecir el abandono de donantes** utilizando modelos de clasificación
- **Segmentar donantes por nivel de riesgo** (Bajo, Medio, Alto)
- **Generar insights accionables** para el equipo de fidelización
- **Optimizar estrategias de retención** basadas en datos

## 📊 Estructura del Proyecto

```
Proyecto_ONG/
│
├── generador_datos.py              # Script para generar datos sintéticos de donantes
├── modelo_churn.py                 # Pipeline completo de ML (limpieza, entrenamiento, predicción)
├── donantes_ong_nosql.csv          # Dataset de 5,000 donantes
├── metricas_modelo_MARIO.csv       # Métricas de rendimiento de los modelos
├── predicciones_finales.csv        # Predicciones del modelo
└── predicciones_finales_FANY.csv   # Dataset enriquecido para dashboard
```

## 🔧 Características Principales

### 1. Generación de Datos (`generador_datos.py`)

Crea un dataset sintético con **5,000 registros** que incluye:

- **Variables categóricas**: Canal de captación, Causa de interés
- **Variables numéricas**: Antigüedad, Monto promedio, Contactos anuales
- **Target**: Variable binaria de abandono (0 = Activo, 1 = Abandono)
- **Casos reales**: Outliers y valores nulos para simular datos del mundo real

### 2. Pipeline de Machine Learning (`modelo_churn.py`)

#### 🧹 Limpieza de Datos
- Imputación de valores nulos en canal de captación
- Eliminación de outliers en montos (percentil 99)
- Normalización de categorías

#### 🤖 Modelos Implementados

| Modelo | Descripción | Configuración |
|--------|-------------|---------------|
| **Regresión Logística** | Modelo lineal balanceado | `class_weight={0:1, 1:3}` |
| **Random Forest** | Ensemble de árboles de decisión | `n_estimators=100, class_weight='balanced'` |

#### 📈 Preprocesamiento Automático
- **StandardScaler** para variables numéricas
- **OneHotEncoder** para variables categóricas
- Pipeline de Scikit-Learn para flujo reproducible

#### 🎯 Segmentación de Riesgo
Los donantes se clasifican en tres categorías según probabilidad de abandono:
- 🟢 **Bajo**: 0% - 40%
- 🟡 **Medio**: 40% - 50%
- 🔴 **Alto**: 50% - 100%

## 📦 Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes)

### Dependencias

```bash
pip install pandas numpy scikit-learn
```

O crea un archivo `requirements.txt`:

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## 🚀 Uso

### 1. Generar Datos Sintéticos

```bash
python generador_datos.py
```

**Salida**: Genera el archivo `donantes_ong_nosql.csv` con 5,000 registros.

### 2. Entrenar Modelos y Generar Predicciones

```bash
python modelo_churn.py
```

**⚠️ Importante**: Actualiza la ruta del archivo en la línea 18 del script:

```python
RUTA_ARCHIVO = r'ruta/a/tu/donantes_ong_nosql.csv'
```

**Salidas**:
- `metricas_modelo_MARIO.csv`: Métricas de evaluación de modelos
- `predicciones_finales_FANY.csv`: Dataset con predicciones y segmentación

## 📊 Variables del Dataset

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `id_donante` | int | Identificador único del donante |
| `antiguedad_meses` | int | Meses desde la primera donación (1-60) |
| `monto_promedio` | float | Promedio de donaciones mensuales |
| `canal_captacion` | str | Canal de adquisición (Redes Sociales, Evento, Calle, Referido, Email) |
| `interes_causa` | str | Causa de interés (Niñez/Desarrollo Infantil, Salud, Ambiente, Humanitaria, Animales) |
| `contactos_anuales` | int | Número de contactos anuales con la ONG (0-12) |
| `abandono` | int | Target: 1 = Abandonó, 0 = Activo |

## 📈 Resultados Esperados

El script muestra en consola:

```
--- RESUMEN DE ENTREGA ---
1. Enviar 'predicciones_finales_FANY.csv' a Fany (Dashboard).
2. Enviar 'metricas_modelo_MARIO.csv' a Mario (Informe de Negocio).

Resumen de Segmentación de Riesgo:
------------------------------
 > Riesgo Bajo:     3,500 donantes
 > Riesgo Medio:   1,000 donantes
 > Riesgo Alto:      500 donantes
------------------------------
```

## 🎓 Metodología

1. **Carga de datos**: Lectura del CSV con validación de existencia
2. **ETL y limpieza**: Manejo de nulos, outliers y normalización
3. **Split estratificado**: 80% entrenamiento, 20% prueba
4. **Entrenamiento**: Dos modelos con balanceo de clases
5. **Evaluación**: Métricas enfocadas en Recall (detectar abandonos)
6. **Exportación**: Archivos para análisis de negocio y dashboard

## 🔍 Métricas de Evaluación

El archivo `metricas_modelo_MARIO.csv` incluye:

- **Accuracy Global**: Precisión general del modelo
- **Recall (Clase 1)**: % de abandonos detectados correctamente
- **Precision (Clase 1)**: % de predicciones de abandono correctas
- **F1-Score**: Media armónica entre precisión y recall
- **Matriz de Confusión**: TP, TN, FP, FN

## 🤝 Contribuciones

Este proyecto es parte de un trabajo académico de **Ciencia de Datos**.

## 👥 Equipo

- **Desarrollo y Modelado**: John F. Ramos
- **Dashboard (Fany)**: Visualización de predicciones
- **Análisis de Negocio (Mario)**: Interpretación de métricas

## 📝 Licencia

Este proyecto es de código abierto y está disponible para fines educativos.

## 📧 Contacto

Para preguntas o sugerencias, contacta al propietario del repositorio: [@John-F-Ramos](https://github.com/John-F-Ramos)

---

⭐ **¿Te resultó útil este proyecto?** ¡Dale una estrella al repositorio!