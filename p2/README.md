# P2: Art Attack con el sonido

**Asignatura:** Procesamiento del Lenguaje Natural  
**Grupo:** G02  
**Integrantes:**
- Héctor García Rincón (integrante **a**)
- Pablo Manuel Rodríguez Sosa (integrante **b**)

---

## Contenido de la entrega

| Fichero | Descripción |
|---|---|
| `originales/es_a.mp3` | Grabación del pangrama en español por el integrante a |
| `originales/es_b.mp3` | Grabación del pangrama en español por el integrante b |
| `originales/en_a.mp3` | Grabación del pangrama en inglés por el integrante a |
| `originales/en_b.mp3` | Grabación del pangrama en inglés por el integrante b |
| `originales/fr_a.mp3` | Grabación del pangrama en francés por el integrante a |
| `originales/fr_b.mp3` | Grabación del pangrama en francés por el integrante b |
| `sinteticos/es_a.mp3` | Pangrama sintético en español compuesto con voz del integrante a |
| `sinteticos/en_b.mp3` | Pangrama sintético en inglés compuesto principalmente con voz del integrante b |

---

## Proceso seguido

### Fase 1: Aproximación manual con Praat

En un primer momento intentamos trabajar directamente con Praat. Cargamos los audios y nos pusimos a buscar visualmente en la gráfica dónde empezaba y acababa cada fonema. Fuimos seleccionando y extrayendo a mano letras, sílabas o fonemas del pangrama original para luego intentar juntarlos y crear la frase objetivo.

Pronto vimos que era un proceso demasiado lento y que esta forma de trabajar resultaba poco eficiente. Además, los resultados sonaban bastante mal, ya que las uniones quedaban artificiales y los cortes eran evidentes. No obstante, esta primera toma de contacto nos vino muy bien para empezar a relacionar la forma del audio y el espectrograma con los distintos tipos de sonidos (consonantes sordas, momentos de ruido, pausas antes de la liberación de aire en las oclusivas, vocales acentuadas, etc).

La principal conclusión de esta fase fue que recortar letras sueltas da muchos problemas por culpa de la coarticulación: un sonido cambia mucho según lo que tenga antes y después. Si lo aislas de su entorno, pierde naturalidad.

### Fase 2: Automatización con WebMAUS Basic

Dado que hacerlo todo a mano requería demasiado tiempo, buscamos alternativas y dimos con **WebMAUS Basic**. Se trata de una herramienta que realiza la alineación forzada entre el texto y el audio de forma automática.

Le subimos nuestras grabaciones junto con el texto correspondiente y WebMAUS nos generó unos archivos `.TextGrid`. Estos archivos ya nos daban marcado temporalmente el inicio y el fin de cada fonema y sílaba en el audio original. Gracias a estos TextGrids pudimos ahorrarnos el trabajo manual y extraer los fonemas ya segmentados, lo que agilizó el desarrollo de la práctica enormemente.

### Fase 3: Composición del audio final

A la hora de montar el pangrama objetivo a partir de los recortes, vimos que lo ideal era usar las sílabas completas siempre que el pangrama original las contuviera, ya que así evitábamos romper la coarticulación de esa sílaba concreta. Sin embargo, en la mayoría de los casos no estaban disponibles, por lo que tuvimos que recurrir a la concatenación de fonemas sueltos. 

Cuando ocurría esto, procurábamos escoger fonemas del original que viniesen de un contexto fonético similar, para que al unirlos encajasen mejor. Para juntar todo el conjunto fuimos encadenando los fragmentos, ajustando sus amplitudes en Praat y aplicando pequeños fundidos (*overlap-add*, *crossfade*) para que los saltos fueran menos bruscos.

### Mezcla de idiomas (Inglés + Español)

Para sintetizar el pangrama en inglés, utilizamos casi en su totalidad la grabación del integrante *b*. Pero hubo un par de tramos que tuvimos que sustituir empleando material extraído de las grabaciones en español:

- La **vocal /a/**: En el audio original en inglés no teníamos una variante que encajara de forma natural para palabras como *realized*, así que empleamos una extraída del pangrama en español.
- El inicio **"ex-"** de la palabra *expensive*: Aprovechamos el corte del español correspondiente a la secuencia /ks/ (de la x) porque la articulación se apreciaba con mayor claridad.

Esta prueba nos sirvió para comprobar que ciertos inventarios fonéticos se solapan parcialmente y que es posible reutilizar piezas de otro idioma si la sonoridad encaja en el contexto.

---

## Lo que hemos aprendido del espectrograma

Dedicar tiempo a analizar las gráficas nos ha ayudado mucho a reconocer visualmente los distintos grupos de sonidos. Las características más relevantes en las que nos hemos fijado son:

### Vocales

Son con diferencia el tipo de sonido más fácil de aislar. Se caracterizan por:

- **Onda periódica**: En el oscilograma se observa claramente un patrón regular y repetitivo generado por la vibración periódica del aire.
- **Formantes**: En el espectrograma destacan como bandas horizontales oscuras y continuas, marcando la concentración de la energía.
- **Tónicas (con acento)**: Tienen una presencia mayor. En el espectrograma aparecen más oscuras (mayor intensidad de energía) y en el eje temporal suelen ser visiblemente más largas que las átonas.

### Consonantes fricativas sordas

Hablamos de sonidos como la /s/, la /f/ o el sonido de la jota española.

- Consisten en **ruido aperiódico puro**: En el oscilograma, la onda es completamente irregular y no hay ciclos que se repitan.
- En el espectrograma se visualizan como una **zona de energía dispersa** (como ruido estático o mancha difuminada), sin ninguna banda de frecuencias definida.

### Consonantes fricativas sonoras

Por ejemplo, la /d/ y /g/ suaves aproximantes, o la /v/ y /z/ en inglés.

- Combinan el **ruido aperiódico** propio de la fricción con una **frecuencia fundamental** estable. Es decir, en la parte inferior del espectrograma se ve una "barra de sonoridad" debido a la vibración de las cuerdas vocales, pero en las frecuencias más altas sigue habiendo una banda de ruido.
- En el oscilograma se refleja mediante una onda irregular pero que sigue un patrón cíclico de fondo recurrente.

### Consonantes oclusivas

Los típicos sonidos explosivos como /p, t, k/ (sordas) y /b, d, g/ (sonoras):

- Tienen una **fase de compresión o silencio** previa a la detonación. El flujo de aire se detiene, y tanto en el oscilograma como en el espectrograma esto se traduce en un hueco prácticamente en blanco.
- Seguido del silencio, se produce la **explosión**: la liberación del aire genera una línea vertical muy marcada y extremadamente breve en el espectrograma.

### Consonantes nasales

Sonidos como la /m/, /n/ o /ɲ/ (ñ):

- Como son sonoras, en la parte baja del espectrograma se sigue viendo la barra de sonoridad (las cuerdas vocales vibran) y el oscilograma muestra una onda periódica.
- Sin embargo, a diferencia de las vocales, en el espectrograma tienen mucha menos intensidad. Las bandas de frecuencia se ven mucho más claras o tenues, dándole a la gráfica un aspecto general como de estar "lavado" o apagado, debido a que el sonido sale por la nariz y pierde fuerza.


## Problemas encontrados

- **El obstáculo de la coarticulación:** Fue, con diferencia, la mayor dificultad. Al aislar un fonema y sacarlo de su entorno para colocarlo en otra posición el resultado a menudo sonaba artificial, como el habla de un robot. En el lenguaje natural los fonemas se influyen y se fusionan, por eso intentamos aliviar el problema reutilizando sílabas íntegras siempre que nos fue posible.
- **Saltos de tono (pitch):** Al ir combinando partes registradas en distintos momentos de la grabación original, la entonación sufría cambios bruscos. Tratar de suavizar este aspecto de forma manual mediante la opción `Manipulation` de Praat nos resultó bastante tedioso.
- **Diferencias de volumen (amplitud):** Ocurría algo parecido a los saltos de tono. Algunos de los recortes extraídos venían con mucha intensidad y otros muy flojos. Hubo que ir nivelando y normalizando las amplitudes antes de la concatenación final para que el audio mantuviera una coherencia.
- **Falta de fonemas en la misma lengua:** Al montar el audio en este caso del inglés, encontrábamos huecos de ciertos fonemas objetivo que no teníamos en nuestra muestra o cuya dicción era menos nítida. Esta carencia fue lo que nos motivó a "importar" un par de recortes del español, tal como detallamos anteriormente.

---

## Resultados y conclusión

Los audios finales sintéticos no son demasiado buenos. Si conoces la frase del pangrama, entonces puede que consigas entender lo que se está diciendo, pero si te lo ponen sin saber nada no se entiende nada. Algunas palabras sueltas si que han están bien conseguidas como:
- **En inglés:**
  - Jim, quickly, beautiful, that, expensive
- **En castellano:**:
  - Jugoso, kiwi, piña, exquisito, lleva

En general, estas palabras que si se entienden mejor tienen en comñun que sus sílabas estaban ya en el pangrama original.

Como era de esperar, las frases tienen un efecto robótico. Al haber concatenado trozos de diferentes palabras, se pierde la fluidez natural y la línea de entonación general de la frase queda un poco a trompicones.

Ha sido una práctica curiosa que nos ha servido especialmente para darnos cuenta de lo complejo que es el habla humana: pronunciar una misma letra nunca suena igual dos veces porque siempre depende de los sonidos que la rodean.

---
