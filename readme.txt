Autores: Manuel Manzano (@manmanher) y Fernando Garrucho (fergarfer1).
Este repositorio contiene los códigos y dataset recopilado para el 3º Trabajo de la asignatura Aplicaciones IoT, del Máster Universitario de Ingeniería de Telecomunicación de la Universidad de Sevilla.
- En la carpeta datos se encuentra el dataset utilizado. El dataset contiene 53 clases de cartas (52 correspondientes a la baraja francesa y un joker) divididos en 7624 imagenes de entrenamiento, 265 imágenes de validación y 265 imágenes de test. Las imágenes son de 224x224x3.
- inferenciaV2 : Contiene el código utilizado para la clasificación de una imagen con el modelo entrenado, el código está listo para ser ejecutado en la coral de la raspberry pi.
- conversor_cuant_notebook.ipynb: Contiene el código utilizado para la conversión del modelo entrenado a un modelo cuantizado apto para la coral de la raspberry pi.
