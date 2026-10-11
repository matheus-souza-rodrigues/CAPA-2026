# Comparação de detectores de estágio único para alvos marítimos em VSNT

Trabalho final da disciplina de **Inteligência Artificial (IIA)** do C-ApA-Of-SA — Centro de Instrução Almirante Alexandrino, 2026.

**Autores:** 1T Vinicius Jorge Pinheiro Maia, 1T Victor Valle de Moraes Vieira e 1T Matheus de Souza Rodrigues.

## O que o trabalho faz

Compara três detectores de estágio único — **YOLOv8, YOLO11 e YOLO26** — na detecção de embarcações e obstáculos em imagens de um veículo de superfície não tripulado (VSNT). Os modelos, pré-treinados no conjunto COCO, são avaliados **antes e depois** da adaptação ao domínio marítimo por *transfer learning*, sob um protocolo único: mesmos dados, mesma partição, mesmas anotações e mesmas métricas.

- **Dados:** MODD2 (*Marine Obstacle Detection Dataset 2*), imagens retificadas da câmera esquerda, com as anotações oficiais do conjunto, feitas à mão e verificadas por especialista. Nenhum rótulo é gerado automaticamente.
- **Partição:** por sequência (20 de treino, 4 de validação e 4 de teste). As cenas de teste nunca são vistas no treino.
- **Métricas:** precisão, revocação, F1, mAP@50, mAP@50-95, tempo de inferência e revocação por faixa de tamanho do alvo.
- **Dificuldade do problema:** a mediana da caixa ocupa cerca de 0,02% da imagem, e cerca de 92% dos alvos ocupam menos de 1% dela. É um problema de alvos pequenos.

## Resultados

Depois da adaptação ao domínio (conjunto de teste, IoU de 0,50):

| Modelo | Parâmetros (M) | mAP@50 | mAP@50-95 | F1 |
|---|---|---|---|---|
| YOLOv8 | 3,01 | 0,667 | **0,352** | 0,682 |
| YOLO11 | 2,58 | **0,698** | 0,343 | **0,745** |
| YOLO26 | 2,38 | 0,621 | 0,316 | 0,641 |

- A adaptação eleva o F1 entre 0,382 e 0,493, puxada por um aumento de três a quase cinco vezes na revocação.
- O YOLO11 obtém o maior mAP@50 e o maior F1. O YOLO26, apesar de ser o mais recente, não o supera em nenhuma métrica de detecção.
- A diferença entre os modelos é decidida pelos alvos muito pequenos (abaixo de 0,05% da imagem), que são 72% do conjunto de teste.
- O tempo de inferência variou bastante entre sessões de GPU na nuvem, e a comparação de latência para uso embarcado fica para avaliação em hardware específico.

## Arquivo

| Arquivo | O que é |
|---|---|
| `deteccao_vsnt_modd2.ipynb` | Notebook completo: baixa o MODD2, monta o dataset, avalia os modelos pré-treinados, faz o ajuste fino e gera as tabelas e figuras do trabalho |

O notebook também avalia o SSD300, que foi rodado durante o experimento mas não entrou na versão final deste trabalho.

## Como executar

1. Abra `deteccao_vsnt_modd2.ipynb` no **Google Colab**.
2. Em *Ambiente de execução → Alterar o tipo de ambiente*, escolha **GPU (T4)**.
3. Execute as células **na ordem**. O MODD2 é baixado pelo próprio notebook a partir dos arquivos oficiais do conjunto, e os pesos dos modelos são baixados pela biblioteca Ultralytics. Nada disso está guardado no repositório.

## Parâmetros principais

Ficam na célula **"Configuração central do projeto"** (Parte 1):

| Parâmetro | Valor | Para que serve |
|---|---|---|
| `EPOCAS` | 20 | Épocas de treino |
| `IMGSZ` | 768 | Resolução de entrada dos YOLO (o SSD300 usa 300) |
| `PASSO_QUADROS` | 5 | Usa 1 a cada 5 quadros (cerca de 2.300 imagens) |
| `CONF_MIN` | 0,25 | Confiança mínima para aceitar uma detecção |
| `IOU_NMS` | 0,45 | Supressão de detecções repetidas |
| `SEQUENCIAS_USADAS` | todas | Lista vazia usa as 28 sequências do MODD2 |

## Dados

BOVCON, B.; MANDELJC, R.; PERŠ, J.; KRISTAN, M. Stereo obstacle detection for unmanned surface vehicles by IMU-assisted semantic segmentation. *Robotics and Autonomous Systems*, v. 104, p. 1–13, 2018.
