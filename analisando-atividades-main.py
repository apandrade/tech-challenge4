import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm
import time 

def detect_activity(video_path, output_path):
    # Inicializar o MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Função para calcular o ângulo entre três pontos (usando o produto escalar)
    def calculate_angle(a, b, c):
        ab = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def is_hand_near_face(hand_landmark, nose_landmark, threshold=0.36):
        # Calcular a distância Euclidiana entre o ponto da mão (pulso) e o ponto do nariz
        distance = np.sqrt((hand_landmark.x - nose_landmark.x) ** 2 + 
                           (hand_landmark.y - nose_landmark.y) ** 2)
        return distance < threshold

    # Função para determinar a atividade com base em heurísticas de ângulo
    def determine_activity(landmarks):
        # Definir ângulos importantes para atividades
        # Braços
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        # Calcular ângulos dos braços
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # Pernas
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # Calcular ângulos das pernas
        left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        
        sum_right_shoulder = right_shoulder.x + right_shoulder.y + right_shoulder.z
        sum_left_shoulder = left_shoulder.x + left_shoulder.y + left_shoulder.z
        distancia_wrist_elbow_left = np.sqrt((left_wrist.x - left_elbow.x) ** 2 + (left_wrist.y - left_elbow.y) ** 2 + (left_wrist.z - left_elbow.z) ** 2)
        distancia_wrist_elbow_right = np.sqrt((right_wrist.x - right_elbow.x) ** 2 + (right_wrist.y - right_elbow.y) ** 2 + (right_wrist.z - right_elbow.z) ** 2)

        # x: 0.485459656
        # y: 0.447289497
        # z: -1.13765788
        # visibility: 0.999392927

        # x: 0.485459656 → Representa a coordenada horizontal do ponto detectado, normalizada em relação à largura da imagem (0 = extremo esquerdo, 1 = extremo direito).
        # y: 0.447289497 → Representa a coordenada vertical do ponto detectado, normalizada em relação à altura da imagem (0 = topo, 1 = base).
        # z: -1.13765788 → Indica a profundidade do ponto em relação à câmera. Valores negativos indicam que o ponto está mais próximo da câmera.
        # visibility: 0.999392927 → Representa a confiança do modelo de que esse ponto foi detectado corretamente. Um valor próximo de 1 significa alta certeza.
        if(left_ankle.visibility < 0.3 and right_ankle.visibility < 0.3): #Tornozelos não visíveis
            if(left_elbow.visibility < 0.4 and right_elbow.visibility < 0.4):
                if((left_wrist.visibility > 0.5 and is_hand_near_face(left_wrist, nose) and right_wrist.y < right_shoulder.y) or
                   (right_wrist.visibility > 0.5 and is_hand_near_face(right_wrist, nose) and left_wrist.y < left_shoulder.y)):
                    return "Mao no rosto"
                return "Atividade desconhecida 1"
            else:
                if((left_wrist.visibility > 0.5 and is_hand_near_face(left_wrist, nose) and right_wrist.y < right_shoulder.y) or
                   (right_wrist.visibility > 0.5 and is_hand_near_face(right_wrist, nose) and left_wrist.y < left_shoulder.y)):
                    return "Mao no rosto"
                elif(abs(left_shoulder.y - right_shoulder.y) > 0.6): #Ombros um acima do outro na horaizontal
                    if((left_wrist.visibility > 0.5 and is_hand_near_face(left_wrist, nose)) or
                        (right_wrist.visibility > 0.5 and is_hand_near_face(right_wrist, nose))):
                        return "Mao no rosto"
                    return "Deitado"
                elif((left_wrist.visibility > 0.5 and 45 <left_arm_angle < 130) or
                       (right_wrist.visibility > 0.5 and 45 < right_arm_angle < 130)): #Braços flexionados
                    if((right_wrist.visibility > 0.3 and abs(right_wrist.y - nose.y) < 0.3 and right_wrist.y < right_shoulder.y) or 
                        (left_wrist.visibility and abs(left_wrist.y - nose.y) < 0.3) and left_wrist.y < left_shoulder.y): # Pulsos acima do nariz
                        return "Acenando"   
                    else:
                        return "Escrevendo ou Teclando"
                elif((right_elbow.visibility > 0.5 and abs(right_elbow.y - right_shoulder.y) < 0.3) or
                      (left_elbow.visibility > 0.5 and abs(left_elbow.y - left_shoulder.y) < 0.3)): #Braços visiveis
                    if(abs(right_elbow.z - right_shoulder.z) > 0.2 or abs(left_elbow.z - left_shoulder.z) > 0.2):
                        return "Braco aberto"
                    else:
                        return "Parado"
                elif(right_elbow.visibility > 0.4 or left_elbow.visibility > 0.4): #Braços visiveis
                    return "Parado"
                else:
                    return "Atividade desconhecida 3"
        else: #Tornozelos possivelmente visíveis
            if(left_leg_angle > 150 and right_leg_angle > 150):
                if(abs(left_leg_angle - right_leg_angle) > 1):
                    return "Caminhando"
                else:
                    return "Em pe"
            else:
                if(abs(left_shoulder.y - right_shoulder.y) > 0.6):
                    return "Deitado"  #Ombros um acima do outro na horaizontal
                return "Sentado"

    # Loop para processar cada frame do vídeo com barra de progresso
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar o frame para detectar a pose
        results = pose.process(rgb_frame)

        # Desenhar as anotações da pose no frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Determinar a atividade com base nos ângulos
            activity = determine_activity(results.pose_landmarks.landmark)

            # Exibir a atividade no frame
            cv2.putText(frame, f'Atividade: {activity}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

        # Exibir o frame processado
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(0.05)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para o vídeo de entrada e saída
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'input_video.mp4')  # Nome do vídeo de entrada
output_video_path = os.path.join(script_dir, 'output_video_atividade.mp4')  # Nome do vídeo de saída

# Processar o vídeo
detect_activity(input_video_path, output_video_path)
