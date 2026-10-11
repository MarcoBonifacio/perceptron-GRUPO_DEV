# Clasificador de tumores con Perceptrón — versión 1.0.0

Proyecto base de la **Evaluación Parcial** de Construcción de Software. Un Perceptrón implementado desde cero
con NumPy clasifica tumores de mama como **malignos (0)** o **benignos (1)** a partir de medidas de las células.

## Datos

- `datos/pacientes.csv`: 569 pacientes, 10 medidas y la columna `diagnostico`.
  Proviene del dataset *Breast Cancer Wisconsin (Diagnostic)* (UCI). Se borraron algunos valores para practicar la limpieza.
- `datos/hospital_b.csv`: 60 pacientes de otro hospital. Se usa para reproducir el error del Issue de la Tarea 1.

## Estructura

```text
├── datos/                 # archivos CSV
├── src/
│   ├── datos.py           # cargar, limpiar, estandarizar y dividir
│   ├── perceptron.py      # clase Perceptron
│   └── metricas.py        # accuracy y otras métricas
├── main.py                # entrena y evalúa los modelos
├── .github/workflows/ci.yml
├── CHANGELOG.md
└── requirements.txt
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py --datos datos/pacientes.csv --objetivo diagnostico
```

El argumento opcional `--features` selecciona las características del Modelo 3.
Por defecto utiliza `concavidad puntos_concavos area textura`; por ejemplo:

```bash
python main.py --datos datos/pacientes.csv --objetivo diagnostico --features radio textura
```

## Integrantes

| Nombre | Código | Rol |
|---|---|---|
| Baca Medina, Jerson Alejandro | 74162753 | usuario#1 |
| Bonifacio Ruiz, Marco Antonio | 60863027 |usuario#2 |
| Cahuana Mendoza, Bryan Alexander | 72738694 | Líder |
| Llanos Luna, Yury Yohan | 60351789 | usuario#3 |
| Salazar Arenas, Benjamin | 72948599 | usuario#4 |

## Resultados

Resultados con `datos/pacientes.csv`, 30 épocas y la partición de prueba determinista
del programa. La matriz de confusión sigue el orden `[[TN, FP], [FN, TP]]`.

| Modelo | Accuracy | Error | Matriz de confusión |
|---|---:|---:|---|
| Modelo 1: tasa 0.01, features base | 0.867 | 0.133 | `[[34, 4], [11, 64]]` |
| Modelo 2: tasa 0.5, decaimiento 0.1, features base | 0.876 | 0.124 | `[[37, 1], [13, 62]]` |
| Modelo 3: tasa 0.01, features seleccionadas | 0.920 | 0.080 | `[[35, 3], [6, 69]]` |

**¿Por qué los Modelos 1 y 2 daban igual antes de la Tarea 4, si sus tasas eran
distintas?** Una tasa distinta cambia el tamaño de las actualizaciones, pero no
garantiza predicciones distintas. Ambos usaban los mismos datos y la misma partición
de prueba; al convertir las salidas en clases 0/1, las diferencias de pesos no
cambiaron las métricas observadas. Al añadir decaimiento de 0.1 al Modelo 2, su tasa
efectiva varía por época y sus métricas pasan a 0.876 de accuracy y 0.124 de error.

**¿Qué cambió con las features del Modelo 3 y por qué?** El Modelo 3 sustituye las
características base (`radio`, `textura`, `perimetro`, `area`) por `concavidad`,
`puntos_concavos`, `area` y `textura`. Con estas variables obtiene 0.920 de accuracy y
0.080 de error en esta partición, lo que indica que la representación seleccionada
separa mejor las clases evaluadas; no implica que el resultado se generalice a
cualquier conjunto de datos.

**Los errores por época nunca llegan a 0: ¿qué dice eso sobre si los datos son
linealmente separables?** Los errores de entrenamiento no llegan a cero en las 30
épocas (la última época deja 68, 53 y 36 errores en los Modelos 1, 2 y 3,
respectivamente). Esto sugiere que las clases no son perfectamente separables por una
frontera lineal en las características usadas. Sin embargo, por sí solo no demuestra
que sean inseparables: más épocas u otros parámetros podrían cambiar el resultado.

**En este problema, ¿qué error es más grave, un falso positivo o un falso negativo?**
En los datos, 0 representa un tumor maligno y 1 uno benigno; por tanto, con la matriz
`[[TN, FP], [FN, TP]]`, un falso positivo es un tumor maligno clasificado como benigno.
Ese error es clínicamente más grave porque puede retrasar la detección y el tratamiento
del cáncer. Un falso negativo corresponde aquí a clasificar un tumor benigno como
maligno, lo que puede generar estudios o seguimiento innecesarios.
