# Proyecto WikiHow KDTree

Este proyecto es un estudio de caso desarrollado para la actividad curricular **Diseño y Análisis de Algoritmos**. El objetivo principal es comparar el rendimiento de dos métodos para encontrar vecinos más cercanos en un conjunto de datos: un enfoque iterativo y uno basado en un árbol KD (KD-Tree).

## Descripción del Proyecto

El programa utiliza una base de datos en formato JSONL (`data/wikihow.jsonl`) que contiene preguntas y respuestas extraídas del sitio web WikiHow. A partir de esta información:

1. Se procesan las preguntas y respuestas para obtener sus representaciones vectoriales (embeddings) utilizando el proyecto [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
2. Los embeddings se almacenan como conjuntos de puntos en un espacio de dimensión fija.
3. Se comparan dos métodos para encontrar el vecino más cercano:
   - **Iterativo**: Recorre todos los puntos y calcula la distancia para encontrar el vecino más cercano.
   - **KD-Tree**: Construye un árbol KD para realizar búsquedas más eficientes.

## Estructura del Proyecto

- **`src/project_wikihow.cpp`**: Código principal que implementa la comparación entre los métodos iterativo y KD-Tree.
- **`data/wikihow.jsonl`**: Archivo de entrada con preguntas y respuestas en formato JSONL.
- **`results.csv`**: Archivo de salida que contiene los resultados de las pruebas (tiempos de ejecución).
- **`llama_client.h`**: Interfaz para interactuar con el proyecto `llama.cpp` y obtener embeddings.

## Requisitos

`sudo apt install libcurl4-openssl-dev`

`sudo apt install libjsoncpp-dev`

`sudo apt install libeigen3-dev`

> Estas instrucciones de instalación van a variar con la distribución Linux.


## Instalación

`$ mkdir proyecto && cd proyecto`

Dentro de ese directorio, se deja la carpeta del proyecto

### Formato del proyecto

```
proyecto/
|
--> llama.cpp/
--> eigen/
--> proyecto_wikihow_kdtree/ (este proyecto)
```

### Eigen

Por defecto en el `CMakeLists.txt`, la carpeta de la bilbioteca Eigen lo lee desde la carpeta anterior a este. Si quiere edite la dirección de `EIGEN_ROOT_DIR` y asigne la carpeta en la que esté.

Puedes tambien descargar Eigen desde Gitlab:

`git clone https://gitlab.com/libeigen/eigen.git`

### Llama.cpp install

`git clone https://github.com/ggml-org/llama.cpp`

`cd llama.cpp`

`cmake -B build`

 `cmake --build build --config Release`

## Compilación



`cmake .`

`make`

## Ejecución

`./llama_wikihow data/wikihow.jsonl <database_size> <step> results.csv`

- `data/wikihow.jsonl`: archivo de base de datos
- `<database_size>`: cuántos datos se van a extraer del archivo, siendo el caso más grande.
- `<step>`: El paso, del 0 al tamaño máximo, cada cuánto irá incrementando.
- `results.csv`: \[opcional\] nombre del archivo de salida


`./llama_wikihow_with_leaft_sizes data/wikihow.jsonl <database_size> <step> <num_leaf_size_cases> results.csv`

- `data/wikihow.jsonl`: archivo de base de datos
- `<database_size>`: cuántos datos se van a extraer del archivo, siendo el caso más grande.
- `<step>`: El paso, del 0 al tamaño máximo, cada cuánto irá incrementando.
- `<num_leaf_size_cases>`: número de cantidad casos de tamaños caso base: [1, 2, ..., num_leaf_size_case]
- `results.csv`: \[opcional\] nombre del archivo de salida
---