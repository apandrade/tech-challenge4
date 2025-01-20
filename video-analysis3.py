import cv2
import mediapipe as mp
import os
from deepface import DeepFace
from tqdm import tqdm
import numpy as np

# Função para calcular a diferença média entre landmarks de poses
def calculate_pose_difference(pose1, pose2):
    if not pose1 or not pose2:
        return float('inf')
    diff = np.linalg.norm(np.array(pose1) - np.array(pose2), axis=1)
    return np.mean(diff)

# Função principal
def detect_emotions_and_poses(video_path, output_path, report_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Variáveis de controle para análise
    last_pose_landmarks = None
    anomalies = []
    activities_summary = {"emotions": {}, "total_anomalies": 0}

    for frame_index in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(
                frame, actions=['emotion'], enforce_detection=True, detector_backend='opencv'
            )
        except Exception as e:
            print(f"Erro no DeepFace: {e}")
            continue

        # Analisar emoções
        if isinstance(result, list):
            for face in result:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                dominant_emotion = face['dominant_emotion']
                activities_summary["emotions"][dominant_emotion] = activities_summary["emotions"].get(dominant_emotion, 0) + 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Analisar poses
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Detectar movimento anômalo
            current_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            if last_pose_landmarks:
                pose_diff = calculate_pose_difference(current_landmarks, last_pose_landmarks)
                if pose_diff > 0.05:  # Limite de diferença para detectar anomalias
                    anomalies.append(frame_index)
            last_pose_landmarks = current_landmarks

        # Escrever frame processado no vídeo
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Gerar relatório
    activities_summary["total_anomalies"] = len(anomalies)
    with open(report_path, "w") as report_file:
        report_file.write("Resumo da Análise do Vídeo\n")
        report_file.write(f"Total de frames analisados: {total_frames}\n")
        report_file.write(f"Número de anomalias detectadas: {len(anomalies)}\n")
        report_file.write("Emoções detectadas:\n")
        for emotion, count in activities_summary["emotions"].items():
            report_file.write(f"  {emotion}: {count} vezes\n")
        report_file.write("\nFrames com anomalias:\n")
        report_file.write(", ".join(map(str, anomalies)))

# Caminho para arquivos
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'input_video.mp4')
output_video_path = os.path.join(script_dir, 'output/output_video3.mp4')
report_path = os.path.join(script_dir, 'output/summary_report.txt')

# Chamar a função principal
detect_emotions_and_poses(input_video_path, output_video_path, report_path)
