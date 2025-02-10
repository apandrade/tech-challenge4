import cv2
import mediapipe as mp
import os
from deepface import DeepFace
from tqdm import tqdm
import numpy as np
import time

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Parâmetros
POSE_DIFFERENCE_THRESHOLD = 0.05
HAND_NEAR_FACE_THRESHOLD = 0.36

# Funções auxiliares

def calculate_pose_difference(pose1, pose2):
    if not pose1 or not pose2:
        return float('inf')
    diff = np.linalg.norm(np.array(pose1) - np.array(pose2), axis=1)
    return np.mean(diff)

def calculate_angle(a, b, c):
    ab = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - c.y])
    cos_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def is_hand_near_face(hand_landmark, nose_landmark, threshold=HAND_NEAR_FACE_THRESHOLD):
    distance = np.sqrt((hand_landmark.x - nose_landmark.x) ** 2 + 
                       (hand_landmark.y - nose_landmark.y) ** 2)
    return distance < threshold

def detect_emotions(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
    except Exception as e:
        print(f"Erro no DeepFace: {e}")
        return []
    return result

def draw_emotions(frame, emotions):
    for face in emotions:
        x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
        dominant_emotion = face['dominant_emotion']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def detect_pose_and_anomalies(frame, pose_model, last_pose_landmarks, frame_index, anomalies):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        current_landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

        if last_pose_landmarks:
            pose_diff = calculate_pose_difference(current_landmarks, last_pose_landmarks)
            if pose_diff > POSE_DIFFERENCE_THRESHOLD:
                anomalies.append(frame_index)
        return current_landmarks
    return None

def determine_activity(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
    if(left_ankle.visibility < 0.3 and right_ankle.visibility < 0.3):
        if(left_elbow.visibility < 0.4 and right_elbow.visibility < 0.4):
            if((left_wrist.visibility > 0.5 and is_hand_near_face(left_wrist, nose) and right_wrist.y < right_shoulder.y) or
               (right_wrist.visibility > 0.5 and is_hand_near_face(right_wrist, nose) and left_wrist.y < left_shoulder.y)):
                return "Mao no rosto"
            return "Atividade desconhecida 1"
        else:
            if((left_wrist.visibility > 0.5 and is_hand_near_face(left_wrist, nose) and right_wrist.y < right_shoulder.y) or
               (right_wrist.visibility > 0.5 and is_hand_near_face(right_wrist, nose) and left_wrist.y < left_shoulder.y)):
                return "Mao no rosto"
            elif(abs(left_shoulder.y - right_shoulder.y) > 0.6):
                if((left_wrist.visibility > 0.5 and is_hand_near_face(left_wrist, nose)) or
                   (right_wrist.visibility > 0.5 and is_hand_near_face(right_wrist, nose))):
                    return "Mao no rosto"
                return "Deitado"
            elif((left_wrist.visibility > 0.5 and 45 <left_arm_angle < 130) or
                (right_wrist.visibility > 0.5 and 45 < right_arm_angle < 130)):
                if((right_wrist.visibility > 0.3 and abs(right_wrist.y - nose.y) < 0.3 and right_wrist.y < right_shoulder.y) or 
                    (left_wrist.visibility and abs(left_wrist.y - nose.y) < 0.3) and left_wrist.y < left_shoulder.y):
                    return "Acenando"  
                else:
                    return "Escrevendo ou Teclando"
            elif((right_elbow.visibility > 0.5 and abs(right_elbow.y - right_shoulder.y) < 0.3) or
                (left_elbow.visibility > 0.5 and abs(left_elbow.y - left_shoulder.y) < 0.3)):
                if(abs(right_elbow.z - right_shoulder.z) > 0.2 or abs(left_elbow.z - left_shoulder.z) > 0.2):
                    return "Braco aberto"
                else:
                    return "Parado"
            elif(right_elbow.visibility > 0.4 or left_elbow.visibility > 0.4):
                return "Parado"
            else:
                return "Atividade desconhecida 3"
    else:
        if(left_leg_angle > 150 and right_leg_angle > 150):
            if(abs(left_leg_angle - right_leg_angle) > 1):
                return "Caminhando"
            else:
                return "Em pe"
        else:
            if(abs(left_shoulder.y - right_shoulder.y) > 0.6):
                return "Deitado"
            return "Sentado"


def process_video(video_path, output_path, report_path):
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

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    last_pose_landmarks = None
    anomalies = []
    activities_summary = {"emotions": {}, "total_anomalies": 0}

    for frame_index in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        emotions = detect_emotions(frame)
        draw_emotions(frame, emotions)
        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Processar o frame para detectar a pose
        results = pose.process(rgb_frame)

        current_landmarks = detect_pose_and_anomalies(frame, pose, last_pose_landmarks, frame_index, anomalies)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            activity = determine_activity(results.pose_landmarks.landmark)
            cv2.putText(frame, f'Atividade: {activity}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2, cv2.LINE_AA)
            last_pose_landmarks = current_landmarks

        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

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

 #Execução principal
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_video_path = os.path.join(script_dir, 'input_video.mp4')
    output_video_path = os.path.join(script_dir, 'output_video_atividade.mp4')
    report_path = os.path.join(script_dir, 'output/summary_report.txt')
    process_video(input_video_path, output_video_path, report_path)
