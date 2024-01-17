# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 12:11:51 2024

@author: neo_a
"""

# Importar los frameworks o librerias necesarios para el script
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os

# Cargo el modelo de segmentación semántica preentrenado en tensorflow_hub
model = hub.load("https://tfhub.dev/tensorflow/deeplabv3/1")

# Obtener el directorio actual del script
script_directory = os.path.dirname(os.path.realpath(__file__))

# Ruta al archivo de video en la carpeta videos
video_path = os.path.join(script_directory, 'videos', 'video_prueba.mp4')

# Abrir el video a analizar
cap = cv2.VideoCapture(video_path)

# Ciclo while para realizar el análisis del video que se ha abierto
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Se realiza el preprocesamiento del fotograma
    input_tensor = tf.image.convert_image_dtype (frame, dtype=tf.unit8)[tf.newaxis, ...]
    
    # Se realiza la segmentación
    result = model(input_tensor)
    
    # Obtener la máscara de segmentación
    mask = result['segmentation_mask']
    
    # Se convierte la máscara a imagen
    segmented_frame = tf.image.convert_image_dtype(mask, dtype=tf.unit8)
    
    # Se Aplica la máscara al fotrograma original
    segmented_frame = cv2.cvtColor(segmented_frame.numpy(), cv2.COLOR_GRAY2BGR)
    output_frame = cv2.addWeighted(frame, 1, segmented_frame, 0.7, 0)
    
    # Se muestra el fotograma segmentado
    cv2.imshow("segmentation", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()