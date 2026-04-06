# Quijote IR TUI

Aplicación de recuperación de información sobre *Don Quijote de la Mancha* para la práctica 4 de PLN.

El proyecto permite consultar el texto del Quijote en tres modos:

- búsqueda clásica
- búsqueda semántica
- RAG

La interfaz principal es una TUI en terminal hecha con `textual`, pero también incluye una CLI auxiliar para reconstruir recursos y lanzar búsquedas desde comandos.

> Asignatura: Procesamiento del Lenguaje Natural, UCM  
> Curso: 2025-2026

## Integrantes

- Héctor García Rincón
- Pablo Manuel Rodríguez Sosa

## Qué hace la práctica

El objetivo de la práctica es construir un buscador inteligente del Quijote.

El sistema:

1. lee el corpus HTML incluido en el repositorio
2. limpia y normaliza el texto
3. lo divide en chunks con solapamiento
4. construye un índice clásico
5. construye embeddings semánticos
6. permite consultar el corpus desde terminal
7. en modo RAG, recupera fragmentos relevantes y genera una respuesta con Ollama citando las fuentes usadas

La idea importante es esta: el sistema no responde “de memoria”. Siempre trabaja sobre fragmentos reales recuperados del texto.

## Modos de uso

### 1. Búsqueda clásica

Usa tokens normalizados, eliminación de stopwords, lematización y un ranking TF-IDF implementado con `numpy`.

Es el modo más fiable cuando la consulta contiene palabras que aparecen realmente en el texto.

Ejemplos típicos:

- `molinos de viento`
- `Dulcinea`
- `Sancho gobernador`
- `caballero andante`

### 2. Búsqueda semántica

Usa vectores de `spaCy` con `es_core_news_lg`.

Sirve para consultas reformuladas o menos literales. Para mejorar la recuperación, el sistema combina el vector del título del capítulo con el del cuerpo del chunk, mezcla consulta original y consulta lematizada, y aplica un pequeño re-ranking léxico sobre los mejores candidatos semánticos.

Ejemplo:

- `caballería andante`

### 3. RAG

El modo RAG:

1. ejecuta búsqueda clásica
2. ejecuta búsqueda semántica
3. fusiona ambos rankings
4. selecciona varias fuentes relevantes
5. construye un contexto etiquetado como `[F1]`, `[F2]`, etc.
6. llama a un LLM local mediante Ollama
7. devuelve una respuesta breve, grounded y con referencias a las fuentes recuperadas

Este modo está pensado para responder preguntas más naturales, pero siempre apoyándose en fragmentos reales del corpus.

## Cómo funciona internamente

### Ingesta

El corpus está en:

- `p4/quijote/quijote.html`

El sistema extrae capítulos y párrafos, descartando front-matter irrelevante.

### Preprocesado

Se aplica el mismo pipeline al corpus y a la consulta:

- limpieza básica
- minúsculas
- stopwords
- lematización
- normalización homogénea

Esto permite que la consulta y el texto se comparen en condiciones similares.

### Chunking

El texto no se busca “capítulo entero contra consulta”, sino en fragmentos de tamaño medio.

Cada chunk:

- pertenece a un capítulo
- conserva referencias a sus párrafos de origen
- guarda texto original y texto normalizado
- tiene overlap configurable con el chunk siguiente

### Índice clásico

Se construye un índice invertido TF-IDF para:

- términos normalizados
- lemas

Luego ambos se combinan con pesos configurables.

### Índice semántico

Se generan embeddings a partir de `spaCy`:

- se obtiene un vector para el título del capítulo y otro para el cuerpo del chunk
- ambos se combinan con pesos configurables
- la consulta mezcla versión original y versión lematizada
- sobre los mejores candidatos se aplica un pequeño bonus léxico
- se usa similitud coseno con `numpy`

### RAG

La capa RAG está separada en:

- `app/rag/retriever.py`
- `app/rag/context_builder.py`
- `app/rag/prompts.py`
- `app/rag/generator.py`
- `app/rag/formatter.py`

Esto permite mantener el diseño modular y no meter la lógica del LLM dentro de la TUI.

## Requisitos

- Linux
- Python 3.12 o superior
- `uv`
- modelo de spaCy `es_core_news_lg`
- Ollama instalado para el modo RAG

## Instalación

Desde la raíz del monorepo:

```bash
uv sync --project p4
uv run --project p4 python -m spacy download es_core_news_lg
ollama pull qwen3:8b
```

Si quieres ajustar parámetros, copia:

```bash
cp p4/.env.example p4/.env
```

y modifica las variables `QUIJOTE_*`.

## Preparación inicial recomendada

Antes de usar la TUI por primera vez:

```bash
uv run --project p4 python -m p4.main build-chunks
uv run --project p4 python -m p4.main build-classical
uv run --project p4 python -m p4.main build-semantic
```

También puedes reconstruir todo de una vez:

```bash
uv run --project p4 python -m p4.main build-all
```

## Uso rápido

### Abrir la TUI

```bash
uv run --project p4 python -m p4.main tui
```

o simplemente:

```bash
uv run --project p4 python -m p4.main
```

### Buscar desde la CLI

Clásica:

```bash
uv run --project p4 python -m p4.main search "molinos de viento" --mode classical
```

Semántica:

```bash
uv run --project p4 python -m p4.main search "caballería andante" --mode semantic
```

RAG:

```bash
uv run --project p4 python -m p4.main search "¿Qué interpreta don Quijote al ver los molinos?" --mode rag --top-k 3
```

Comando RAG dedicado:

```bash
uv run --project p4 python -m p4.main rag "Resume el episodio de los molinos y cita las fuentes" --max-sources 3
```

## Uso de la TUI

La TUI permite:

- elegir el modo `classical`, `semantic` o `rag`
- escribir la consulta
- lanzar la búsqueda
- navegar resultados
- ver detalles del chunk seleccionado
- reconstruir recursos desde la propia interfaz

### Controles útiles

- `Enter`: buscar desde la caja de consulta
- `Ctrl+r`: buscar
- `c`: cambiar a clásica
- `s`: cambiar a semántica
- `g`: cambiar a RAG
- `/`: volver a enfocar la caja de búsqueda
- `↑` y `↓`: moverse por resultados
- `q`: salir

### Qué muestra cada zona

Barra lateral:

- selector de modo
- caja de consulta
- botones de reconstrucción
- estado de artefactos

Panel principal:

- en modo RAG, una respuesta generada arriba
- una tabla de resultados o fuentes recuperadas
- un panel de detalle con el fragmento recuperado
- un panel de estado con mensajes de error o progreso

## Salida esperada del modo RAG

Una respuesta RAG correcta debe:

- responder en español
- basarse en el contexto recuperado
- incluir referencias como `[F1]`, `[F2]`
- mostrar debajo las fuentes correspondientes

Ejemplo conceptual:

```text
Don Quijote interpreta los molinos como gigantes [F1][F2].
```

y después una tabla con:

- `Ref`
- `Usada`
- `Score`
- `Parte`
- `Capítulo`
- `Chunk`
- `Fragmento`

Así se puede comprobar que la cita no es inventada.

## Comandos disponibles

Estado:

```bash
uv run --project p4 python -m p4.main status
```

Reconstrucción:

```bash
uv run --project p4 python -m p4.main build-chunks
uv run --project p4 python -m p4.main build-classical
uv run --project p4 python -m p4.main build-semantic
uv run --project p4 python -m p4.main build-all
uv run --project p4 python -m p4.main build-all --no-semantic
```

Consulta:

```bash
uv run --project p4 python -m p4.main search "molinos de viento" --mode classical
uv run --project p4 python -m p4.main search "caballería andante" --mode semantic
uv run --project p4 python -m p4.main search "¿Qué interpreta don Quijote al ver los molinos?" --mode rag --top-k 3
uv run --project p4 python -m p4.main rag "Resume el episodio de los molinos y cita las fuentes" --max-sources 3
```

## Configuración

Los valores por defecto están en:

- `p4/settings.toml`

Se pueden sobreescribir con:

- `p4/.env`
- variables de entorno `QUIJOTE_*`

### Parámetros más importantes

Corpus y rutas:

- `paths.corpus_path`
- `paths.artifacts_dir`
- `paths.logs_dir`

Preprocesado:

- `preprocessing.spacy_model`

Chunking:

- `chunking.target_words`
- `chunking.overlap_words`

Búsqueda clásica:

- `search.top_k`
- `search.surface_weight`
- `search.lemma_weight`

Semántica:

- `semantic.batch_size`

Ollama:

- `ollama.host`
- `ollama.timeout_seconds`

RAG:

- `rag.enabled`
- `rag.generation_model`
- `rag.temperature`
- `rag.num_predict`
- `rag.hybrid_top_k`
- `rag.max_sources`
- `rag.max_context_chars`
- `rag.max_source_chars`
- `rag.classical_weight`
- `rag.semantic_weight`
- `rag.rrf_k`

### Ejemplo mínimo de `.env`

```bash
cp p4/.env.example p4/.env
```

Si quieres ver el modelo configurado explícitamente:

```bash
echo 'QUIJOTE_RAG__GENERATION_MODEL=qwen3:8b' >> p4/.env
```

## Estructura del proyecto

```text
p4/
├── app/
│   ├── chunking.py
│   ├── classical_search.py
│   ├── cli.py
│   ├── config.py
│   ├── errors.py
│   ├── ingestion.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── rag/
│   ├── semantic_search.py
│   ├── services.py
│   └── tui/
├── artifacts/
├── logs/
├── quijote/
│   └── quijote.html
├── .env.example
├── pyproject.toml
├── settings.toml
└── main.py
```

## Pruebas manuales recomendadas

### Pruebas básicas

1. Comprobar estado:

```bash
uv run --project p4 python -m p4.main status
```

Esperado:

- existen `chunks`
- existe `classical`
- existe `semantic_manifest`
- existe `semantic_embeddings`

2. Búsqueda clásica:

```bash
uv run --project p4 python -m p4.main search "molinos de viento" --mode classical --top-k 3
```

Esperado:

- aparece `Capítulo VIII`
- el fragmento habla de gigantes, aspas y molinos

3. Búsqueda semántica:

```bash
uv run --project p4 python -m p4.main search "caballería andante" --mode semantic --top-k 2
```

Esperado:

- devuelve fragmentos razonables, aunque no necesariamente perfectos

4. RAG:

```bash
uv run --project p4 python -m p4.main rag "Resume el episodio de los molinos y cita las fuentes" --max-sources 3
```

Esperado:

- respuesta breve
- referencias como `[F1]`
- tabla de fuentes debajo

### Consultas útiles para la TUI

Clásica:

- `molinos de viento`
- `Dulcinea`
- `Sancho gobernador`
- `sierra morena`
- `caballero andante`

Semántica:

- `caballería andante`
- `vida de escuderos`

RAG:

- `¿Qué interpreta don Quijote al ver los molinos?`
- `Resume el episodio de los molinos y cita las fuentes`
- `¿Qué le dice Sancho a don Quijote sobre los molinos?`

### Consultas borde

- consulta vacía
- `xyzqwert`
- `de la y que`

Esperado:

- no se rompe la interfaz
- se informa de que no hay resultados o de que falta consulta

## Limitaciones conocidas, explicadas claro

### 1. El RAG depende de Ollama

Si Ollama no está arrancado, el modo RAG no puede generar respuesta.

Eso no rompe:

- la búsqueda clásica
- la búsqueda semántica

Solo afecta al modo RAG.

### 2. La semántica con spaCy funciona, pero no siempre es brillante

La búsqueda semántica usa `es_core_news_lg`, que ofrece vectores útiles, pero no es un sistema moderno especializado en embeddings de alta calidad.

Eso significa:

- técnicamente funciona
- a veces recupera bien
- otras veces devuelve pasajes menos relacionados de lo deseable

Como el RAG mezcla clásica y semántica, muchas veces irá bien si la clásica ya recupera bien, pero no conviene vender la semántica como perfecta.

### 3. El sistema responde solo con lo recuperado

El RAG está diseñado para contestar usando únicamente el contexto recuperado.

Si la evidencia no basta:

- debe decirlo
- no debería inventar hechos

### 4. Los modelos no vienen en el repositorio

El repositorio incluye:

- código
- corpus
- artefactos regenerables

No incluye:

- `es_core_news_lg`
- modelos de Ollama

Por eso hay que instalarlos aparte.

### 5. Las referencias son a fragmentos reales, no a spans milimétricos

Las referencias `[F1]`, `[F2]`, etc. apuntan a chunks reales recuperados del corpus.

Eso da buena trazabilidad, pero no es una cita académica exacta palabra por palabra.

## Problemas frecuentes

### Falta el modelo de spaCy

Síntoma:

- error al usar semántica

Solución:

```bash
uv run --project p4 python -m spacy download es_core_news_lg
```

### Ollama no está activo

Síntoma:

- error claro indicando que no se pudo conectar con Ollama

Solución:

```bash
ollama serve
```

### Falta el modelo de Ollama

Síntoma:

- error indicando que el modelo configurado no está disponible

Solución:

```bash
ollama pull qwen3:8b
```

o cambia `rag.generation_model`.

### Sale un warning de `VIRTUAL_ENV` al lanzar p4 desde otra práctica

Si vienes de otro entorno del monorepo, puedes lanzar `p4` sin ese warning así:

```bash
VIRTUAL_ENV= uv run --project p4 python -m p4.main tui
```

## Resumen corto

Esta práctica implementa:

- preprocesado completo
- chunking con overlap
- búsqueda clásica
- búsqueda semántica
- TUI interactiva
- RAG híbrido con Ollama y referencias verificables

La mejor forma de lucirla en demo suele ser:

- usar clásica para consultas literales
- usar RAG para preguntas formuladas en lenguaje natural
- enseñar siempre las fuentes para demostrar trazabilidad
