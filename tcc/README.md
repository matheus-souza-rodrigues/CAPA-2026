# TCC — Identificação de alvos por visão computacional embarcada em VSNT

Trabalho de Conclusão de Curso do C-ApA-Of-SA (CIAA, 2026)
**1T Matheus de Souza Rodrigues** — orientador: Doutor Rigel Procópio Fernandes

O código identifica uma embarcação designada como **alvo** entre as demais embarcações da cena e estima sua **marcação** e sua **distância**, usando imagens do conjunto público MODD2 (*Marine Obstacle Detection Dataset 2*), adquirido por um veículo de superfície não tripulado real.

## Resumo do método

1. **Rótulo do alvo.** O MODD2 anota os obstáculos quadro a quadro, mas não diz qual caixa é qual embarcação. O código liga as caixas de quadros seguidos (trilha) e marca a lancha escolhida como `alvo`; as demais caixas viram `outros`.
2. **Partição temporal.** A lancha aparece em uma única sequência. O trecho com ela **distante** vai para o teste, o intermediário para a validação e o trecho com ela **próxima** para o treino (simula receber imagens do alvo antes da missão e reconhecê-lo depois, mais longe).
3. **Aumento de dados.** Cada quadro de treino com o alvo gera 5 variantes: espelhamento, filtro da média 3×3 e 5×5, e redução da cena para 60% e 40%.
4. **Treino.** YOLO11 *nano* (Ultralytics), partindo dos pesos do COCO, ajustado em duas classes (`alvo` e `outros`).
5. **Avaliação.** Precisão, revocação, F1 e matriz de confusão, comparando o modelo pré-treinado com o especializado.
6. **Marcação e distância.** Marcação pela coluna do centro da caixa; distância por triangulação estéreo (casamento de gabarito entre as câmeras esquerda e direita), com a incerteza correspondente e verificação pelo comprimento implícito da embarcação.

## Arquivos

| Arquivo | O que é |
|---|---|
| `codigo_identificacao_alvo_vsnt.ipynb` | Notebook completo, do download dos dados às estimativas de marcação e distância |
| `codigo_gerar_graficos.py` | Refaz os gráficos do TCC a partir das saídas do notebook |

## Como executar

1. Abra `codigo_identificacao_alvo_vsnt.ipynb` no **Google Colab**.
2. Em *Ambiente de execução → Alterar o tipo de ambiente*, escolha **GPU (T4)**. Sem GPU o treino não termina.
3. Execute as células **na ordem**. O notebook:
   - monta o Google Drive (para guardar pesos e saídas entre sessões);
   - baixa e descompacta o MODD2 (se o download falhar por cota do Drive, coloque o zip do conjunto na raiz do seu Drive ou na pasta `tcc_vsnt`, e rode a célula de novo);
   - monta a trilha do alvo, a partição, o aumento de dados e o dataset no formato YOLO;
   - avalia o modelo pré-treinado, treina o especializado e gera as métricas e figuras;
   - calcula marcação e distância ao longo da aproximação;
   - ao final, compacta as saídas em um zip para download.
4. Tempo aproximado: cerca de **1 hora** de treino em GPU T4.

## Organização do notebook

| Parte | O que faz |
|---|---|
| 1 | Configuração e dependências |
| 2 | Dados do MODD2 e trilha do alvo |
| 3 | Partição, aumento de dados e dataset YOLO de duas classes |
| 4 | Linha de base: YOLO11 do COCO, sem treino |
| 5 | Ajuste fino em `alvo` e `outros` |
| 6 | Avaliação: precisão, revocação, F1, mAP, matriz de confusão e desempenho por tamanho |
| 7 | Calibração estéreo da sequência |
| 8 | Marcação e distância ao longo da aproximação |
| 9 | Resumo, zip e download |

## Parâmetros principais

Todos ficam na célula **"Configuração central"** (Parte 1). Os mais úteis para adaptar o trabalho:

| Parâmetro | Valor | Para que serve |
|---|---|---|
| `SEQUENCIA_ALVO` | `kope75-00-00013780-00014195` | Sequência do MODD2 onde está o alvo |
| `IOU_MINIMO`, `DIST_CENTRO_MAX`, `RAZAO_LARGURA_MAX` | 0,30 / 1,2 / 1,5 | Critério de ligação das caixas na trilha |
| `FRACAO_TESTE`, `FRACAO_VALID`, `BUFFER` | 0,30 / 0,15 / 8 | Partição temporal da trilha do alvo |
| `PASSO_OUTRAS` | 3 | Usa 1 quadro a cada 3 das sequências sem o alvo |
| `PESOS`, `EPOCAS`, `IMGSZ`, `BATCH` | `yolo11n.pt` / 40 / 960 / 8 | Treino |
| `SEMENTE` | 42 | Semente de aleatoriedade (reprodutibilidade) |
| `AUG_FLIP`, `AUG_BLUR`, `AUG_AFASTAR` | sim / (3, 5) / (0,60, 0,40) | Aumento de dados |
| `CONF_MIN`, `IOU_AVALIACAO_2` | 0,25 / 0,30 | Avaliação |

### Para usar outro alvo

1. Rode a Parte 2 e veja as trilhas mais longas que o código lista e desenha.
2. Escolha a embarcação desejada e ajuste `SEQUENCIA_ALVO` (e `N_ALVO_ESPERADO`, que é só uma conferência).
3. Rode o restante normalmente.

## Resultados obtidos (execução do TCC)

- Identificação do alvo nos 91 quadros de teste: **85 identificados como alvo, 0 confundidos** com outra embarcação, 6 não detectados (todos durante uma oclusão parcial).
- Detecção: F1 de **0,925** (YOLO11 do COCO) para **0,941** (especializado).
- Marcação: diferença média de **0,09°** em relação à anotação.
- Distância: de **72 a 98 m**, com incerteza de 7 a 13 m; comprimento implícito da lancha em torno de **6,5 m**.

## Limitações

- O alvo aparece em uma única sequência: treino e teste compartilham a mesma cena.
- Quadros consecutivos são quase idênticos, então a amostra efetiva é menor que o número de quadros.
- O MODD2 não fornece distância de referência; a distância é verificada pela incerteza e pela coerência do comprimento implícito.

## Como citar

RODRIGUES, M. S. **Identificação de alvos por visão computacional embarcada em veículos de superfície não tripulados**: discriminação entre contatos e estimativa de marcação e distância. Monografia (Curso de Aperfeiçoamento Avançado em Sistemas de Armas) — Centro de Instrução Almirante Alexandrino, Rio de Janeiro, 2026.

Conjunto de dados: BOVCON, B. et al. Stereo obstacle detection for unmanned surface vehicles by IMU-assisted semantic segmentation. *Robotics and Autonomous Systems*, v. 104, p. 1–13, 2018.
