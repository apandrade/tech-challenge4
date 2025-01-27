import cv2
import mediapipe as mp
import os
from tqdm import tqdm
import numpy as np

# Função para calcular ângulos entre três pontos
def calculate_angle(a, b, c):
    a = np.array(a)  # ponto A
    b = np.array(b)  # ponto B (pivô)
    c = np.array(c)  # ponto C
    
    ab = a - b
    bc = c - b
    
    # Calcular o ângulo entre os vetores ab e bc
    angle = np.arctan2(np.linalg.det([ab, bc]), np.dot(ab, bc)) * 180.0 / np.pi
    return abs(angle)

# Função para detectar atividade
def detect_activity(landmarks):
    if landmarks:
        # Obter as coordenadas dos marcos principais (por exemplo, mãos, braços, ombros, etc.)
        left_shoulder = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
                         landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x,
                          landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y]
        left_hip = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].x,
                    landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].y]
        right_hip = [landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x,
                     landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y]
        left_knee = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x,
                     landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y]
        right_knee = [landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x,
                      landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y]
        left_elbow = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x,
                      landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y]
        right_elbow = [landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x,
                       landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y]
        left_wrist = [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x,
                      landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y]
        right_wrist = [landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x,
                       landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y]

        # Calcular ângulos
        left_leg_angle = calculate_angle(left_hip, left_knee, [landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                                              landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].y])
        right_leg_angle = calculate_angle(right_hip, right_knee, [landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].x,
                                                                  landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].y])

        # Calcular ângulos dos braços
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Detectar atividade com base nos ângulos e posições
        if abs(left_shoulder[1] - right_shoulder[1]) < 0.05 and abs(left_hip[1] - right_hip[1]) < 0.05:  # Posição deitada (tronco quase horizontal)
            return "Deitado"
        elif abs(left_knee[1] - right_knee[1]) < 0.05:  # Indica que a pessoa está parada (sem movimentação de pernas)
            return "Parado"
        elif left_leg_angle < 45 and right_leg_angle < 45:  # Pode indicar postura sentada, mas não deitada
            return "Sentado"
        elif left_arm_angle > 120 and right_arm_angle > 120:  # Braços levantados como para dançar
            return "Dançando"
        elif left_arm_angle < 45 and right_arm_angle < 45:  # Braços baixos, indicando possível relaxamento
            return "Relaxando"
        elif left_arm_angle > 160 and right_arm_angle > 160:  # Braços estendidos, como no caso de teclando no celular
            return "Teclando no celular"
        elif abs(left_shoulder[0] - right_shoulder[0]) < 0.1:  # Braços próximos, possivelmente tocando outra pessoa
            return "Tocando em outra pessoa"
        elif abs(left_wrist[0] - left_shoulder[0]) < 0.2 and abs(right_wrist[0] - right_shoulder[0]) < 0.2:  # Mão próxima do rosto
            return "Mão no rosto"
        elif left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:  # Mão levantada para dar tchau
            return "Dando tchau"
        else:
            return "Atividade indefinida"

    return "Sem atividade"


def detect_pose(video_path, output_path):
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

            # Detectar atividade com base nos marcos
            activity = detect_activity(results.pose_landmarks)
            cv2.putText(frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

        # Exibir o frame processado (pressione 'q' para sair)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Garantir que o vídeo seja finalizado corretamente
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para o vídeo de entrada e saída
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'input_video.mp4')  # Nome do vídeo de entrada
output_video_path = os.path.join(script_dir, 'atividades.mp4')  # Nome do vídeo de saída

# Processar o vídeo
detect_pose(input_video_path, output_video_path)
