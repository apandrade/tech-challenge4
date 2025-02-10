# POS FIAP ALURA - IA PARA DEVS
## Tech Challenge Fase 4
### Integrantes Grupo 26

- André Philipe Oliveira de Andrade(RM357002) - andrepoandrade@gmail.com
- Joir Neto (RM356391) - joirneto@gmail.com
- Marcos Jen San Hsie(RM357422) - marcosjsh@gmail.com
- Michael dos Santos Silva(RM357009) - michael.shel96@gmail.com
- Sonival dos Santos(RM356905) - sonival.santos@gmail.com

Video(Youtube): 

Github: https://github.com/apandrade/tech-challenge4

# Análise de Vídeo com Detecção de Pose e Emoções

Este projeto realiza **análise de vídeo** utilizando **Visão Computacional e Aprendizado de Máquina** para identificar poses humanas, detectar emoções e relatar anomalias nos movimentos.  

## 🛠️ Funcionalidades

- **Detecção de pose humana** usando **MediaPipe**  
- **Identificação de atividades** baseadas na posição do corpo  
- **Detecção de emoções faciais** utilizando **DeepFace**  
- **Registro de anomalias nos movimentos**  
- **Geração de relatório** com estatísticas da análise  
- **Processamento de vídeo** e salvamento de saída anotada  

---

## 🚀 Como Executar o Projeto

### 1️⃣ Clone do repositório
```bash
git clone https://github.com/apandrade/tech-challenge4.git
cd tech-challenge4
```

### 2️⃣ Instale as dependências
```bash
pip install -r requirements.txt
```

### 3️⃣ Coloque um vídeo chamado `input_video.mp4` na mesma pasta do script e rode:
```bash
python analisando-video.py
```


## 📚 Bibliotecas Utilizadas

O código faz uso das seguintes bibliotecas:

| Biblioteca | Função |
|------------|------------------------------------------------|
| `opencv-python` | Processamento de vídeo e exibição de saída |
| `mediapipe` | Detecção da pose humana |
| `deepface` | Reconhecimento facial e análise de emoções |
| `tqdm` | Barra de progresso durante o processamento |
| `numpy` | Cálculos matemáticos para análise de poses |
| `os` | Manipulação de diretórios e arquivos |
| `time` | Medição de tempo de execução |

---

## 🔍 Estrutura do Código

### 🏃 Detecção de Pose (MediaPipe)

- **`detect_pose_and_anomalies(frame, pose_model, last_pose_landmarks, frame_index, anomalies)`**  
  → Processa cada frame para detectar poses e registra anomalias nos movimentos.
  
  Entrada: Frame, modelo de pose, landmarks do frame anterior, índice do frame e lista de anomalias.  
  Saída: Landmarks atuais.

- **`determine_activity(landmarks)`**  
  → Analisa os pontos do corpo e classifica a atividade como "Em pé", "Sentado", "Deitado", "Acenando", etc.
  
  Entrada: Landmarks detectados.  
  Saída: Atividade classificada (ex: caminhando, sentado, acenando).

- **`calculate_pose_difference(pose1, pose2)`**  
  → Compara poses para identificar mudanças bruscas que podem indicar anomalias.
  
  Entrada: Dois conjuntos de landmarks (pose1 e pose2).  
  Saída: Diferença média entre os landmarks.

- **`calculate_angle(a, b, c)`**
  → Calcula o ângulo entre três landmarks. Classifica movimentos com base nos ângulos dos membros (braços, pernas).

  Entrada: Três landmarks (a, b, c).  
  Saída: Ângulo em graus.

- **`is_hand_near_face(hand_landmark, nose_landmark, threshold)`**
  → Verifica se a mão está próxima ao rosto. Detecta gestos como acenar ou coçar o rosto.

  Entrada: Landmark da mão, landmark do nariz e limiar de distância.  
  Saída: True se a mão estiver próxima ao rosto, False caso contrário.

### 😀 Detecção de Emoções (DeepFace)

- **`detect_emotions(frame)`**  
  → Analisa expressões faciais e retorna as emoções detectadas usando o DeepFace. Identifica emoções como felicidade, tristeza, raiva, etc.

  Entrada: Frame do vídeo.  
  Saída: Lista de emoções detectadas.

- **`draw_emotions(frame, emotions)`**  
  → Desenha caixas ao redor do rosto e exibe a emoção detectada no frame.

  Entrada: Frame do vídeo e lista de emoções.  
  Saída: Frame com as emoções desenhadas.


### 🎥 Processamento de Vídeo

- **`process_video(video_path, output_path, report_path)`**  
  → Função principal que executa todo o pipeline de processamento. Lê o vídeo de entrada, analisa cada frame, salva um novo vídeo com as anotações e gera um relatório.
  
  Entrada: Caminho do vídeo de entrada, caminho do vídeo de saída e caminho do relatório.  
  Saída: Vídeo processado e relatório de análise.

### 📊 Relatório Gerado

Após a execução, um **relatório de análise** será salvo em:

```
output/summary_report.txt
```

## 📌 Exemplo de Saída

🔹 **Vídeo anotado:**  
- Detecção da pose com linhas e conexões desenhadas  
- Emoções exibidas sobre os rostos detectados  
- Atividade classificada exibida no canto esquerdo do video  

🔹 **Relatório gerado:**  

```
Resumo da Análise do Vídeo
Total de frames analisados: 3326
Número de anomalias detectadas: 1043
Emoções detectadas:
Atividades detectadas
    -Escrevendo ou teclando: 386 vezes
    -Acenando: 23 vezes
    -Braco aberto: 119 vezes
    -Caminhando: 82 vezes
    -Deitado: 96 vezes
    -Em pe: 9 vezes
    -Mao no rosto: 235 vezes
    -Parado: 621 vezes
    -Sentado: 649 vezes
    -Atividade desconhecida: 493 vezes

Frames com anomalias:
5, 8, 9, 11, 12, 15, 16, 17, 21, 23, 30, 34, 38, 39, 40...
```

---

## 📝 Observações

- O vídeo de entrada deve estar na pasta do script e ser nomeado `input_video.mp4`.  
- O modelo pode ter dificuldades em detectar poses corretamente se a iluminação estiver ruim ou houver muitas pessoas no vídeo.  

---

## 🤖 Melhorias Futuras

- Melhorar detecção e avaliação. 
- Melhorar a precisão da **detecção de anomalias** ajustando os **limiares de diferença**.  
- Implementar suporte para **detecção de múltiplas pessoas** no mesmo vídeo.  
