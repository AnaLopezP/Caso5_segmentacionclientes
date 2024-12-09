# Caso5_segmentacionclientes

## 1. PCA
Previamente hay que escalar los datos.
Para reducir la dimensionalidad. Pasos:
- Estandarización de los datos.
- Cálculo de la matriz de covarianza o matriz de correlación
- Descomposición de la matriz
- Selección de componentes principales
- Transformación de datos

El resultado final del PCA es un conjunto de componentes principales que son combinaciones lineales de las variables originales y que capturan la mayor parte de la variabilidad presente en los datos originales. Estos componentes pueden utilizarse para visualización, análisis exploratorio de datos o como entrada para otros algoritmos de aprendizaje automático. 


-------- VARIANZA EXPLICADA -------- 
A mayor varianza explicada, los componentes capturan más cosas relevantes.
[0.28566613 0.1294993 ]
PCA 1: 28.57%
PCA 2: 12.95%

PC1: Hay que mirar los más cercanos a 1

PC2: igual
Ambos se clasifican en: alta carga positiva, carga positiva moderada y carga positiva baja e igual para los negativos.

El PC1 y el 2 son influidos por las cargas.


                               pca1      pca2
ID                         0.014166 (pos.b) 0.007990 (pos.b)
Genero                    -0.031717 (neg.b) -0.009543 (neg.b)
Casado                     0.714952 (pos.a)  0.034253 (pos.b)
Edad                       0.876659 (pos.a) -0.001048 (neg.b)
Graduado                   0.388151 (pos.m) -0.031597 (neg.b)
Gastoscore                 0.549283 (pos.a)  0.031929 (pos.b)
Family_Size               -0.296808 (neg.m)  0.021934 (pos.b)
Generacion                -0.897507 (neg.a)  0.002209 (pos.b)
Ingresos anuales brutos    0.738120 (pos.a)  0.034110 (pos.b)
Gasto_medio_mensual_cuota  0.522092 (pos.a) 0.040811 (pos.b)
Abandono                   0.026898 (pos.b) -0.469561 (neg.m)
Dias_cliente               0.001005 (pos.b)  0.918050 (pos.aa)
Campanna_anno             -0.062089 (neg.b)  0.934181 (pos.aa)

Cogemos las altas (positivas y negativas). Estas son las variables influyentes:

PC1: casado, edad, gastoscore, generacion, ingresos anuales brutos, gasto medio
PC2: dias cliente, campaña año, abandono

Sin borrar nada y datos escalados

                               pca1      pca2
ID                         0.012527  0.007733
Genero                    -0.030284 -0.010157
Casado                     0.697382  0.034361
Edad                       0.917553 -0.000201
Graduado                   0.365726 -0.031016
Profesion                  0.209722  0.002849
Experiencia laboral        0.882638  0.002593
Gastoscore                 0.532882  0.030865
Family_Size               -0.298049  0.023423
anno_nacimiento           -0.917778  0.000780
Generacion                -0.916719  0.001376
Ingresos anuales brutos    0.706555  0.033764
Provincia                 -0.011909 -0.002722
Digital_encuesta          -0.211405 -0.000277
Gasto_medio_mensual_cuota  0.491693  0.041819
Abandono                   0.025172 -0.468733
Dias_cliente               0.000743  0.918271
Campanna_anno             -0.060587  0.934344

PC1: casado, edad, gastoscore, generacion, ingresos anuales brutos, gasto medio, anno_nacimiento
PC2: dias cliente, campaña año, abandono


### CONLUSI: 
El PC1 está muy influido por variables relacionadas con las carácterísticas del cliente como persona, lo que sugiere que está relacionado con el perfil socioeconómico de los clientes. 
Por otra parte, el PC2 es influido por variables que relacionan la situación del cliente con la empresa. 



## COSAS QUE NOS QUEDAN

### Del modelo:
interpretar los resultados: analizar y etiquetar los clusters
validacion (?)

### De marketing:
• Identificar patrones de comportamiento de los clients
• Personalizar estrategias de marketing
• Mejorar la retencio n de clients
• Optimizar la asignacio n de recursos
