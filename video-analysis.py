import cv2
import mediapipe as mp
import os
from deepface import DeepFace
from tqdm import tqdm

# Emotion and pose analysis
def detect_emotions_and_poses(video_path, output_path):
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

    # Inicializar o MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Loop para processar cada frame do vídeo
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Analisar o frame para detectar faces e expressões
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Iterar sobre cada face detectada
        for face in result:
            # Obter a caixa delimitadora da face
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
            # Obter a emoção dominante
            dominant_emotion = face['dominant_emotion']
            
            # Desenhar um retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Escrever a emoção dominante acima da face
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Processar o frame para detectar poses
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Desenhar as anotações da pose no frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'input_video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, 'output/output_video.mp4')  # Nome do vídeo de saída

# Chamar a função para detectar emoções e poses no vídeo e salvar o vídeo processado
detect_emotions_and_poses(input_video_path, output_video_path)
