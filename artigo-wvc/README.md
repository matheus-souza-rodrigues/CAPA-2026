# Comparing One-Stage Detectors for Maritime Target Detection on USVs

Código do artigo **"Comparing One-Stage Detectors for Maritime Target Detection on USVs"**, submetido à *International Conference on Artificial Intelligence, Computer, Data Sciences and Applications* (ACDSA).

**Autores:** Matheus de Souza Rodrigues, Rigel Procópio Fernandes e Ighor de Souza Ribeiro — Centro de Instrução Almirante Alexandrino (CIAA), Marinha do Brasil.

> Esta pasta se chama `artigo-wvc` porque o artigo foi preparado inicialmente para o Workshop de Visão Computacional (WVC) 2026. A versão atual foi submetida à ACDSA.

## O que o artigo faz

Compara quatro detectores de estágio único — **YOLOv8, YOLO11, YOLO26 e SSD300** — na detecção de alvos marítimos a partir de um veículo de superfície não tripulado (VSNT). Os modelos, pré-treinados no conjunto COCO, são avaliados **antes e depois** da adaptação ao domínio por *transfer learning*, com os mesmos dados, a mesma partição e as mesmas métricas.

- **Dados:** MODD2 (*Marine Obstacle Detection Dataset 2*), adquirido por um VSNT real, imagens retificadas da câmera esquerda, com as anotações oficiais feitas à mão e verificadas por especialista.
- **Partição:** por sequência. As cenas de teste nunca são vistas no treino.
- **Métricas:** precisão, revocação, F1, mAP@50, mAP@50-95, tempo de inferência e revocação por faixa de tamanho do alvo.

## Resultados

Depois da adaptação ao domínio (conjunto de teste, IoU de 0,50):

| Modelo | Parâmetros (M) | mAP@50 | mAP@50-95 | F1 |
|---|---|---|---|---|
| YOLOv8 | 3,01 | 0,667 | **0,352** | 0,682 |
| YOLO11 | 2,58 | **0,698** | 0,343 | **0,745** |
| YOLO26 | 2,38 | 0,621 | 0,316 | 0,641 |
| SSD300 | 23,75 | 0,318 | 0,159 | 0,395 |

- A adaptação eleva o F1 entre 0,265 e 0,493, puxada por um aumento de três a cinco vezes na revocação.
- O YOLO11 obtém o maior mAP@50 e o maior F1. O YOLO26, apesar de ser o mais recente, não o supera em nenhuma métrica de detecção.
- A ordenação entre os modelos é decidida pelos alvos abaixo de 0,05% da imagem, que são 72% do conjunto de teste.
- O SSD300 é o mais prejudicado, porque reduz a imagem para 300 × 300 pixels e um alvo de poucos pixels praticamente desaparece.
- O tempo de inferência variou bastante entre sessões de GPU na nuvem, e o compromisso entre precisão e latência para uso embarcado fica para avaliação em hardware específico.

## Arquivo

| Arquivo | O que é |
|---|---|
| `artigo_WVC2026_epocas20.ipynb` | Notebook completo do artigo: obtenção do MODD2, montagem do dataset, avaliação antes do treino, ajuste fino dos quatro modelos e avaliação final |

## Como executar

1. Abra o notebook no **Google Colab**.
2. Em *Ambiente de execução → Alterar o tipo de ambiente*, escolha **GPU (T4)**.
3. Execute as células **na ordem**. O notebook tenta obter o MODD2 nesta ordem: uma pasta local já extraída (`PASTA_LOCAL`), uma cópia no Google Drive (`DRIVE_FILE_ID`) ou os arquivos oficiais do conjunto. Os pesos dos modelos são baixados automaticamente.

## Parâmetros principais

Ficam na célula **"Configuração central do projeto"** (Parte 1):

| Parâmetro | Valor | Para que serve |
|---|---|---|
| `EPOCAS` | 20 | Épocas de treino |
| `IMGSZ` | 768 | Resolução de entrada dos YOLO (o SSD300 usa 300) |
| `PASSO_QUADROS` | 5 | Usa 1 a cada 5 quadros |
| `CONF_MIN` | 0,25 | Confiança mínima para aceitar uma detecção |
| `IOU_NMS` | 0,45 | Supressão de detecções repetidas |
| `SEMENTE` | 42 | Reprodutibilidade |

## Relação com os outros trabalhos

Este estudo deu origem ao trabalho de Inteligência Artificial (pasta `artigo-iia`) e serviu de linha de base para o TCC (pasta `tcc`), que passa da detecção genérica de embarcações para a identificação de um alvo específico e a estimativa de sua marcação e distância.

## Dados

BOVCON, B.; MANDELJC, R.; PERŠ, J.; KRISTAN, M. Stereo obstacle detection for unmanned surface vehicles by IMU-assisted semantic segmentation. *Robotics and Autonomous Systems*, v. 104, p. 1–13, 2018.
