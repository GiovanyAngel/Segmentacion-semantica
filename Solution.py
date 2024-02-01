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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

cv2.namedWindow('segmentation', cv2.WINDOW_FULLSCREEN)
    
# Ciclo while para realizar el análisis del video que se ha abierto
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocesamiento del frame p                                                                                                                                                              ara que coincida con las expectativas del modelo
    input_tensor = transform(frame).unsqueeze(0)
    
    # Realizamos la segmentación
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        
        
    probs = torch.nn.functional.softmax(output, dim=0)
    predicted_class = torch.argmax(probs, dim=0).byte().cpu().numpy()
    
    # Obtenemos la máscara de la segmentación
    mask = cv2.resize(predicted_class, (frame.shape[1], frame.shape[0]))
    
    # Aplicar un umbral para resaltar la segmentación
    threshold = 128
    mask = (mask > threshold).astype(np.uint8) * 255
    
    # Aplicamos la máscara al frame original
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (1920, 1080))
    cv2.imshow('original', frame_rgb)
    
    # Esperar 1 milisegundo y verificar si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    cv2.waitKey(15)

cap.release()
cv2.destroyAllWindows()
