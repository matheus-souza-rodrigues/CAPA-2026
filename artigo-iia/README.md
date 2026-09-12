# Comparação de detectores de estágio único — MODD2

Código do artigo *Comparação de Detectores de Estágio Único para Detecção de Alvos Marítimos em VSNT* (disciplina de Inteligência Artificial, 2026).

## O experimento

YOLOv8, YOLO11 e YOLO26 são comparados sob um protocolo único — mesmos dados, mesma partição, mesmas anotações e mesmas métricas — antes e depois da adaptação ao domínio marítimo por *transfer learning*.

- **Dataset:** MODD2, imagens retificadas da câmera esquerda, com as anotações oficiais do conjunto (nenhum rótulo gerado automaticamente).
- **Partição:** por sequência, não por quadro — as cenas de teste nunca foram vistas no treino.
- **Métricas:** mAP@50, mAP@50-95, tempo de inferência e revocação por faixa de tamanho do alvo.
- **Característica do problema:** a mediana da área das caixas é de 0,02% da imagem e 92% dos alvos ocupam menos de 1% dela — detecção predominantemente de alvos pequenos.

## Arquivo

`deteccao_vsnt_modd2.ipynb` — notebook completo, feito para rodar no Google Colab com GPU. Baixa o MODD2, monta o dataset, avalia os modelos pré-treinados no COCO, faz o *fine-tuning* e gera as tabelas e figuras do artigo.

O notebook avalia também o SSD300, que foi rodado durante o experimento e não entrou na versão final do artigo.

## Como reproduzir

1. Abra o notebook no Google Colab e ative a GPU (*Ambiente de execução → Alterar o tipo de ambiente*).
2. Execute as células em ordem. Os pesos e o dataset são baixados pelo próprio notebook e não estão versionados aqui.
3. Os parâmetros do experimento (`EPOCAS`, `PASSO_QUADROS`, `CONF_MIN`) ficam todos na célula de configuração central.
