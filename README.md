# POS FIAP ALURA - IA PARA DEVS
## Tech Challenge Fase 4
### Integrantes Grupo 26

- André Philipe Oliveira de Andrade(RM357002) - andrepoandrade@gmail.com
- Joir Neto (RM356391) - joirneto@gmail.com
- Marcos Jen San Hsie(RM357422) - marcosjsh@gmail.com
- Michael dos Santos Silva(RM357009) - michael.shel96@gmail.com
- Sonival dos Santos(RM356905) - sonival.santos@gmail.com

Video(Youtube): https://youtu.be/fFAfOxIVIys?si=jWSriJLiemTKLL_B

Github: https://github.com/apandrade/tech-challenge4

# Análise de Vídeo com Detecção de Pose e Emoções

Este projeto foi desenvolvido em Python e utiliza duas bibliotecas principais: MediaPipe, para detecção de pose e landmarks corporais, e DeepFace, para análise de emoções. O objetivo é analisar um vídeo de entrada, detectar movimentos e emoções, e gerar um relatório com os resultados.  

O projeto processa vídeos frame a frame, detectando landmarks corporais (como ombros, cotovelos, pulsos, quadris, joelhos e tornozelos) e emoções faciais. Com base nesses dados, o sistema classifica atividades e identifica anomalias, gerando um relatório final com os resultados.

## 🛠️ Funcionalidades

- **Detecção de Pose:** Utiliza o MediaPipe para identificar landmarks corporais.  
- **Identificação de atividades:** Classifica movimentos com base na posição dos landmarks.  
- **Detecção de emoções faciais** utilizando **DeepFace**  
- **Registro de anomalias nos movimentos**  
- **Geração de relatório** Gera um relatório detalhado com anomalias detectadas, emoções predominantes, atividades e anômalias.  
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
  → Função principal que executa todo o pipeline de processamento. O vídeo é processado frame a frame. Para cada frame, os landmarks corporais são detectados, as diferenças de pose são calculadas, e a atividade é classificada. As emoções detectadas são desenhadas no frame, e as anomalias são registradas para gerar um relatório final.
  
  Entrada: Caminho do vídeo de entrada, caminho do vídeo de saída e caminho do relatório.  
  Saída: Vídeo processado e relatório de análise.


## Destaques do Algoritmo
### Visibilidade dos Landmarks
Se um landmark não estiver visível, ele não é considerado na análise. Por exemplo, se os tornozelos não estiverem visíveis, a pessoa provavelmente está sentada ou deitada.

### Distância entre os Landmarks
A distância entre os landmarks ajuda a inferir movimentos. Por exemplo, se o pulso está longe da cintura, é provável que o braço esteja aberto.

### Observação das Posições
A posição relativa dos landmarks é crucial. Se o pulso está acima do ombro, a pessoa pode estar acenando ou levantando a mão.

### Heurística dos Ângulos
Os ângulos entre os landmarks permitem classificar movimentos mais complexos. Por exemplo, um ângulo grande entre o quadril, joelho e tornozelo indica que a pessoa está em pé ou caminhando.


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
	-sad: 874 vezes
	-fear: 417 vezes
	-happy: 1055 vezes
	-angry: 120 vezes
	-neutral: 834 vezes
	-surprise: 116 vezes
	-disgust: 1 vezes
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

## 🌟 Considerações Finais
Este projeto demonstra como técnicas de visão computacional podem ser usadas para analisar movimentos e emoções humanas de forma automatizada.  
Futuramente, podemos expandir o projeto para incluir mais atividades, melhorar a precisão da detecção e fazer integrações com APIs externa para disponibilizar o serviço na internet.

## 🤖 Melhorias Futuras

- Melhorar detecção e implementar novas atividades no algorítimo. 
- Melhorar a precisão da **detecção de anomalias** ajustando os **limiares de diferença**.  
- Implementar suporte para **detecção de múltiplas pessoas** no mesmo vídeo.
