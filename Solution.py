# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:30:44 2024

@author: neo_a
"""

import cv2
import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

# Cargamos el modelo Deevlabv3 de torchvision
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Transformaciones para preprocesar la imagen
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((513, 513)),
    transforms.ToTensor(),
])

# Ahora abrir el video con OpenCV
# Obtenemos la ruta del directorio actual
script_directory = os.path.dirname(os.path.realpath(__file__))

# Ruta al archivo de video en la carpeta videos
video_path = os.path.join(script_directory, 'videos', 'video_prueba.mp4')

# Abrir el video a analizar
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video. Verifica la ruta y el formato del video.")
    
# Ciclo while para realizar el análisis del video que se ha abierto
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocesamiento del frame para que coincida con las expectativas del modelo
    input_tensor = transform(frame).unsqueeze(0)
    
    # Realizamos la segmentación
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    # Obtenemos la máscara de la segmentación
    mask = output.argmax(0).byte().cpu().numpy()
    
    # Imprimir información para depuración
    # print("Input Tensor Shape:", input_tensor.shape)
    # print("Output Shape:", output.shape)
    # print("Mask Shape:", mask.shape)
    
    # Aplicar un umbral para resaltar la segmentación
    threshold = 128
    mask = (mask > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = mask.astype(np.uint8)
    
    # Aplicamos la máscara al frame original
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Imprimir información adicional para depuración
    # print("Segmentacion aplicada")
    
    # Mostramos el frame segmentado
    cv2.imshow('segmentation', segmented_frame)
    
    # Esperar 1 milisegundo y verificar si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cv2.waitKey(30)

cap.release()
cv2.destroyAllWindows()
