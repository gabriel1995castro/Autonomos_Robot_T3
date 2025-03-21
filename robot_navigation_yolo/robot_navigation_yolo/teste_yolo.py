import torch
import cv2

# Carregar o modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')  # Usando o modelo YOLOv5n (versão mais leve)

# Carregar a imagem
image_path = '/home/gabriel/ros2_ws/src/robot_navigation_yolo/image.png'  # Substitua pelo caminho da sua imagem
image = cv2.imread(image_path)
image = cv2.rotate(image, cv2.ROTATE_180)
# Realizar a detecção
results = model(image)

# Exibir a imagem com as detecções
results.show()  # Exibe a imagem com as caixas de detecção

# Se quiser salvar a imagem com as caixas de detecção
results.save()  # Salva a imagem com as caixas em uma pasta 'runs/detect/exp'

# Você também pode acessar os resultados de forma programática
detections = results.xywh[0]  # Coordenadas de caixas, classes e confiança
print(detections)
