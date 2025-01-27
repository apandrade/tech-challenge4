import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

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

    # Função para determinar a atividade com base em heurísticas de ângulo
    def determine_activity(landmarks):
        # Definir ângulos importantes para atividades
        # Verificar ângulos entre ombro, cotovelo e punho para atividades
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Calcular ângulos dos braços
        left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Detecção baseada em heurísticas de ângulos
        if left_arm_angle < 45 or right_arm_angle < 45:
            return "Pessoa teclando no celular"  # Braço em ângulo agudo típico de alguém segurando um celular
        elif left_arm_angle > 160 and right_arm_angle > 160:
            return "Pessoa sentada"  # Braços esticados ao lado indicam alguém sentado
        elif left_arm_angle > 90 and right_arm_angle < 90:
            return "Pessoa escrevendo no papel"  # Braço esquerdo levantado e braço direito mais baixo
        elif right_arm_angle > 150:
            return "Pessoa acenando com a mão esquerda"  # Braço esquerdo levantado em ângulo acentuado
        elif left_arm_angle > 160 and right_arm_angle < 80:
            return "Pessoa dançando"  # Movimento amplo de braços típico de dança
        elif left_arm_angle < 30 and right_arm_angle < 30:
            return "Pessoa teclando no computador"  # Braços quase retos indicando digitação no computador
        elif left_arm_angle > 170 and right_arm_angle > 170:
            return "Pessoa parada"  # Braços esticados ao lado e posição estática
        else:
            return "Atividade desconhecida"

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
