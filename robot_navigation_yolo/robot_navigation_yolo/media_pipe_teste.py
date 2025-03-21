import cv2
import mediapipe as mp

# Inicializa o MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Carrega a imagem de entrada
image_path = '/home/gabriel/ros2_ws/src/robot_navigation_yolo/Screenshot from 2025-03-18 22-36-56.png'
image = cv2.imread(image_path)

# Converte a imagem de BGR para RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Processa a imagem
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(rgb_image)

# Desenha os retângulos ao redor dos rostos detectados
if results.detections:
    for detection in results.detections:
        # Desenha a caixa delimitadora do rosto
        mp_draw.draw_detection(image, detection)

# Exibe a imagem com a detecção do rosto
cv2.imshow("Face Detection", image)

# Salva a imagem com a detecção (opcional)
cv2.imwrite('imagem_com_rosto.jpg', image)

# Espera por uma tecla para fechar a janela
cv2.waitKey(0)
cv2.destroyAllWindows()
