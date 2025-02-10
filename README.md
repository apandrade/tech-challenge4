# 📌  Tech Challenge 4 - Análise de Vídeo com Detecção de Pose e Emoções

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

### 1️⃣ Instale as dependências

Antes de rodar o código, instale as bibliotecas necessárias, contidas no arquivo `requirements.txt`:

### 2️⃣ Execute o script principal

Coloque um vídeo chamado `input_video.mp4` na mesma pasta do script e rode:

```bash
python analisando-video.py
```

---

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

- **`determine_activity(landmarks)`**  
  → Analisa os pontos do corpo e classifica a atividade como "Em pé", "Sentado", "Deitado", "Acenando", etc.

- **`calculate_pose_difference(pose1, pose2)`**  
  → Compara poses para identificar mudanças bruscas que podem indicar anomalias.

### 😀 Detecção de Emoções (DeepFace)

- **`detect_emotions(frame)`**  
  → Analisa expressões faciais e retorna as emoções detectadas.

- **`draw_emotions(frame, emotions)`**  
  → Desenha caixas ao redor do rosto e exibe a emoção detectada no frame.

### 🎥 Processamento de Vídeo

- **`process_video(video_path, output_path, report_path)`**  
  → Lê o vídeo de entrada, analisa cada frame, salva um novo vídeo com as anotações e gera um relatório.

### 📊 Relatório Gerado

Após a execução, um **relatório de análise** será salvo em:

```
output/summary_report.txt
```

Ele conterá:
- Número total de frames analisados
- Quantidade de anomalias detectadas
- Emoções mais comuns
- Frames onde anomalias ocorreram

---

## 📌 Exemplo de Saída

🔹 **Vídeo anotado:**  
- Detecção da pose com linhas e conexões desenhadas  
- Emoções exibidas sobre os rostos detectados  
- Atividade classificada exibida no canto da tela  

🔹 **Relatório gerado:**  

```
Resumo da Análise do Vídeo
Total de frames analisados: 3000
Número de anomalias detectadas: 45
Emoções detectadas:
  feliz: 10 vezes
  neutro: 25 vezes
  surpreso: 5 vezes

Frames com anomalias:
123, 456, 789, ...
```

---

## 📝 Observações

- O vídeo de entrada deve estar na pasta do script e ser nomeado `input_video.mp4`.  
- O modelo pode ter dificuldades em detectar poses corretamente se a iluminação estiver ruim ou houver muitas pessoas no vídeo.  
- O script suporta **interrupção manual** (pressionando `Q` durante a execução).

---

## 🤖 Melhorias Futuras

- Melhorar detecção e avaliação. 
- Melhorar a precisão da **detecção de anomalias** ajustando os **limiares de diferença**.  
- Implementar suporte para **detecção de múltiplas pessoas** no mesmo vídeo.  
