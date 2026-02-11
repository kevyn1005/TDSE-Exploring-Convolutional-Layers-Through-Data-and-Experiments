**Estudiante:** Kevyn Daniel Forero Gonzalez  
**Curso:** Laboratorio 03  
**Dataset:** CIFAR-10 

### Objetivos Específicos
1. Establecer una **línea base** con un modelo no convolucional (MLP)
2. Diseñar una **CNN desde cero** con justificación arquitectónica
3. Realizar **experimentos controlados** sobre aspectos de las capas convolucionales
4. **Interpretar** por qué las CNNs superan a las arquitecturas densas
5. Identificar **limitaciones** de las convoluciones en ciertos dominios

### Problema de Clasificación
Clasificar imágenes de 6 categorías de animales del dataset CIFAR-10:
-  Pájaro (Bird)
-  Gato (Cat)
-  Ciervo (Deer)
-  Perro (Dog)
-  Rana (Frog)
-  Caballo (Horse)

---

## Descripción del Conjunto de Datos

### CIFAR-10 Animals Subset

**Origen:** CIFAR-10 Dataset (Canadian Institute For Advanced Research)  
**Subconjunto:** 6 de 10 clases (solo animales)

### Características Técnicas

 Característica | Valor 
---------------|-------
 **Imágenes de Entrenamiento** | 30,000 
 **Imágenes de Prueba** | 6,000 
 **Dimensiones** | 32×32 píxeles 
 **Canales** | RGB (3 canales) 
 **Rango Original** | [0, 255] 
 **Rango Normalizado** | [0.0, 1.0] 
 **Clases** | 6 (balanceadas) 

### Distribución de Clases

```
Pájaro:  5,000 train / 1,000 test
Gato:    5,000 train / 1,000 test  
Ciervo:  5,000 train / 1,000 test
Perro:   5,000 train / 1,000 test
Rana:    5,000 train / 1,000 test
Caballo: 5,000 train / 1,000 test
```

### Preprocesamiento Aplicado

1. **Normalización**: División por 255 para escalar a [0, 1]
2. **Data Augmentation** (solo entrenamiento):
   - Flip horizontal aleatorio
   - Rotación aleatoria (±10%)
   - Traslación aleatoria (±10%)

### Justificación del Dataset para CNNs

El dataset de animales es ideal para estudiar CNNs porque:

1. **Invarianza Espacial**: Los animales pueden aparecer en diferentes posiciones de la imagen
2. **Jerarquía de Características**:
   - **Nivel bajo**: Bordes, texturas, colores
   - **Nivel medio**: Partes del cuerpo (patas, alas, cabeza)
   - **Nivel alto**: Forma completa del animal
3. **Patrones Locales**: Características como ojos, patas y alas están espacialmente localizadas
4. **Color Relevante**: Los 3 canales RGB contienen información discriminativa entre especies
5. **Variabilidad Visual**: Diferentes ángulos, posturas y fondos requieren generalización

---

## Arquitecturas Implementadas

### 1. Modelo Baseline (No Convolucional)

**Tipo:** Multi-Layer Perceptron (MLP)  
**Propósito:** Establecer línea base de comparación

#### Arquitectura

```
Input: 32×32×3 → Flatten → 3,072 features
    
Dense(256) + ReLU + Dropout(0.3)
   
Dense(128) + ReLU + Dropout(0.3)
    
Dense(6) + Softmax
    
Output: 6 clases
```

#### Diagrama Detallado

```

 Layer (type)                    │ Output Shape       │    Param    

 flatten                         │ (None, 3072)       │          0   
 dense                           │ (None, 256)        │    786,688   
 dropout                         │ (None, 256)        │          0   
 dense_1                         │ (None, 128)        │     32,896   
 dropout_1                       │ (None, 128)        │          0   
 dense_2                         │ (None, 6)          │        774   

```

**Total de Parámetros:** 820,358 (3.13 MB)

#### Características del Baseline

-  **Fortalezas**: 
  - Simplicidad conceptual
  - Rápido de entrenar
-  **Limitaciones**:
  - Pierde estructura espacial al hacer Flatten
  - No captura patrones locales
  - Alto número de parámetros
  - Propenso a overfitting

---

### 2. Modelo Convolucional (CNN)

**Tipo:** Convolutional Neural Network  
**Propósito:** Explotar sesgo inductivo espacial

#### Arquitectura de 3 Bloques

```
Input: 32×32×3
    ↓
 BLOQUE 1 
│ Conv2D(32, 3×3) + BN + ReLU            │  → 32×32×32
│ Conv2D(32, 3×3) + ReLU                 │  → 32×32×32
│ MaxPooling(2×2)                        │  → 16×16×32
│ Dropout(0.25)                          │

    
 BLOQUE 2 
│ Conv2D(64, 3×3) + BN + ReLU            │  → 16×16×64
│ Conv2D(64, 3×3) + ReLU                 │  → 16×16×64
│ MaxPooling(2×2)                        │  → 8×8×64
│ Dropout(0.25)                          │

    ↓
 BLOQUE 3 
│ Conv2D(128, 3×3) + BN + ReLU           │  → 8×8×128
│ Conv2D(128, 3×3) + ReLU                │  → 8×8×128
│ MaxPooling(2×2)                        │  → 4×4×128
│ Dropout(0.25)                          │

    ↓
 HEAD 
│ Flatten                                │  → 2,048
│ Dense(256) + BN + Dropout(0.5)         │  → 256
│ Dense(6) + Softmax                     │  → 6

```


**Total de Parámetros:** 815,014 (3.11 MB)


---

#### Observaciones Clave

1. **CNN tiene MENOS parámetros** pero MEJOR rendimiento
2. **Diferencia Train-Val en CNN**: ~7% (vs ~6% en Baseline)
   - CNN generaliza mejor gracias a:
     - Data augmentation
     - Dropout
     - Batch normalization
3. **Loss significativamente menor** en CNN indica mejor confianza en predicciones

---

### Experimento Controlado: Número de Filtros

**Variable Manipulada:** Número de filtros por bloque  
**Variables Fijas:** Kernel size (3×3), capas (3 bloques), activación (ReLU), pooling (2×2)

#### Configuraciones Probadas

| Configuración | Filtros B1 | Filtros B2 | Filtros B3 | Parámetros | Test Acc | Test Loss |
|--------------|-----------|-----------|-----------|-----------|----------|-----------|
| **Reducida** | 16 | 32 | 64 | 208,390 | 73.2% | 0.782 |
| **Estándar** | 32 | 64 | 128 | 815,014 | 76.8% | 0.685 |
| **Aumentada** | 64 | 128 | 256 | 3,231,878 | 77.5% | 0.672 |

#### Análisis del Experimento

##### Hallazgos Cuantitativos

1. **Rendimiento**:
   - Reducida → Estándar: **+3.6%** accuracy (↑4× parámetros)
   - Estándar → Aumentada: **+0.7%** accuracy (↑4× parámetros)
   - **Ley de rendimientos decrecientes** clara

2. **Eficiencia**:
   - **Mejor relación accuracy/parámetro**: Configuración Estándar
   - Aumentada mejora poco pero aumenta mucho complejidad
   - Reducida viable para recursos limitados

3. **Generalización**:
   - Train-Val gap similar en las 3 (~7-8%)
   - Data augmentation + regularización efectivos en todas


---

## Interpretación y Análisis

### ¿Por qué las CNNs Superan al Baseline?

#### 1. **Sesgo Inductivo Espacial**

Las CNNs incorporan **suposiciones** sobre la estructura de imágenes que el MLP no tiene:

##### a) Localidad Espacial
```
MLP:  Cada píxel conectado a TODAS las neuronas
      - Trata imagen como vector 1D
      - Pierde estructura 2D

CNN:  Filtros pequeños (3×3) ven solo píxeles cercanos
      - Explota correlación espacial
      - Captura patrones locales (bordes, texturas)
```

**Ejemplo**: Para detectar "ojo de gato":
- **MLP**: Debe aprender pesos para cada posición posible del ojo
- **CNN**: Un solo filtro detecta "forma circular oscura" en CUALQUIER posición


##### c) Invarianza a Traslación
```
Si un "ojo" aparece en esquina superior izquierda:
  MLP: Debe aprender nuevos pesos para detectarlo

Si aparece en centro:
  MLP: Debe aprender OTROS pesos diferentes

CNN: El MISMO filtro responde en ambas posiciones
```

**Resultado**: Generalización automática a diferentes posiciones


#### 2. **Eficiencia Paramétrica**

```
Dataset: 30,000 imágenes de 32×32×3

MLP Baseline:
  • 820,358 parámetros
  • 48.3% test accuracy
  • Ratio: 17,000 parámetros por % de accuracy

CNN:
  • 815,014 parámetros  
  • 76.8% test accuracy
  • Ratio: 10,600 parámetros por % de accuracy

Conclusión: CNN es 60% más eficiente
```

### Problemas Donde Convolución NO es Apropiada

#### 1. **Series Temporales Largas con Dependencias Globales**

```
Ejemplo: Predicción de mercado de valores

Secuencia: [precio_día1, precio_día2, ..., precio_día365]

 Problema CNNs:
   - Kernels pequeños (3-7) solo ven ventanas locales
   - Eventos hace 200 días pueden ser cruciales
   - Necesita muchas capas para receptive field grande

 Mejor opción: RNNs, LSTMs, Transformers
   - Attention puede ver TODA la secuencia
   - Memoria explícita de eventos pasados
```
FOTOS SAGE MAKER

## Autor

**Kevyn Daniel Forero Gonzalez**  
