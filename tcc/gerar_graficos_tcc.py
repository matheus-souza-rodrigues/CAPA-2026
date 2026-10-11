import json, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]
AZUL, LARANJA, AGUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, CINZA, GRADE = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
    "axes.edgecolor": CINZA, "axes.labelcolor": TINTA, "xtick.color": CINZA, "ytick.color": CINZA,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "figure.facecolor": "white",
})
def br(v, casas=3):
    return f"{v:.{casas}f}".replace(".", ",")
def grade(ax, eixo="y"):
    ax.grid(axis=eixo, color=GRADE, linewidth=0.6); ax.set_axisbelow(True)
def salvar(fig, nome):
    fig.savefig(f"{OUT}/{nome}"); plt.close(fig); print("ok", nome)

# ---------------------------------------------------------------- dados
D = json.load(open(f"{SRC}/alvo_definicao.json"))
Q = D["quadros"]; IDX = [q["indice"] for q in Q]; W = [q["caixa_xyxy"][2] - q["caixa_xyxy"][0] for q in Q]
n = len(Q); nt, nv, B = int(n * .30), int(n * .15), 8
F_TE, F_VA, F_TR = (0, nt), (nt + B, nt + B + nv), (nt + B + nv + B, n)
assert (nt, nv, n - F_TR[0]) == (91, 45, 153)

# ---------------------------------------------------------------- 1. perfil de aproximacao
fig, ax = plt.subplots(figsize=(6.3, 2.7))
for (a, b), cor, nome in ((F_TE, AGUA, "teste"), (F_VA, LARANJA, "validação"), (F_TR, AZUL, "treino")):
    ax.axvspan(IDX[a], IDX[b - 1], color=cor, alpha=0.14, lw=0, zorder=0)
    ax.text((IDX[a] + IDX[b - 1]) / 2, 0.96, f"{nome}\n{b - a} quadros", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=TINTA)
ax.plot(IDX, W, color=TINTA, lw=1.4, zorder=3)
ax.set_xlabel("quadro da sequência"); ax.set_ylabel("largura da caixa do alvo (px)")
ax.set_ylim(40, 300); grade(ax)
salvar(fig, "perfil_alvo.png")

# ---------------------------------------------------------------- 2. distribuicao de tamanho
tr = np.array(W[F_TR[0]:]); te = np.array(W[F_TE[0]:F_TE[1]])
aug = np.concatenate([tr, tr, tr, tr * 0.60, tr * 0.40])       # flip, media 3x3, media 5x5, reducoes
lo, hi = te.min(), te.max()
dentro = lambda v: float(np.mean((v >= lo) & (v <= hi)))
print("dentro da faixa do teste: original %.3f | aumentado %.3f | tudo %.3f" % (dentro(tr), dentro(aug), dentro(np.concatenate([tr, aug]))))
fig, ax = plt.subplots(figsize=(6.3, 3.0))
bins = np.linspace(0, 290, 30)
ax.axvspan(lo, hi, color=AGUA, alpha=0.10, lw=0, zorder=0)
ax.text((lo + hi) / 2, 0.97, f"faixa do teste\n{lo:.0f} a {hi:.0f} px", transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=9, color=CINZA)
for nome, v, cor, ls in (("treino, variantes aumentadas", aug, LARANJA, "-"), ("treino, quadros originais", tr, AZUL, "-"),
                         ("teste", te, AGUA, "-")):
    ax.hist(v, bins=bins, histtype="stepfilled", color=cor, alpha=0.18, zorder=2)
    ax.hist(v, bins=bins, histtype="step", color=cor, lw=1.6, label=f"{nome} ({len(v)} caixas)", zorder=3)
ax.set_xlabel("largura da caixa do alvo (px)"); ax.set_ylabel("número de caixas")
ax.set_xlim(0, 300); ax.margins(y=0.18); ax.legend(loc="upper right", fontsize=9); grade(ax)
salvar(fig, "distribuicao_tamanho.png")

# ---------------------------------------------------------------- 3. curvas de treino
R = list(csv.DictReader(open(f"{SRC}/saidas_v2/treino_results.csv")))
ep = [int(r["epoch"]) for r in R]
col = lambda k: [float(r[k]) for r in R]
fit = [0.1 * a + 0.9 * b for a, b in zip(col("metrics/mAP50(B)"), col("metrics/mAP50-95(B)"))]
melhor = ep[int(np.argmax(fit))]; print("melhor epoca", melhor)
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5))
ax = axes[0]
ax.plot(ep, col("train/box_loss"), color=AZUL, lw=1.6, label="caixa")
ax.plot(ep, col("train/cls_loss"), color=LARANJA, lw=1.6, ls="--", label="classe")
ax.set_ylim(0, 4.2); ax.set_title("perda no treino", fontsize=10); ax.legend(fontsize=8.5)
ax = axes[1]
ax.plot(ep, col("metrics/precision(B)"), color=AZUL, lw=1.6, label="precisão")
ax.plot(ep, col("metrics/recall(B)"), color=LARANJA, lw=1.6, ls="--", label="revocação")
ax.set_ylim(0.4, 1.0); ax.set_title("validação", fontsize=10); ax.legend(fontsize=8.5, loc="lower right")
ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0,4", "0,6", "0,8", "1,0"])
ax = axes[2]
ax.plot(ep, col("metrics/mAP50(B)"), color=AZUL, lw=1.6, label="mAP@50")
ax.plot(ep, col("metrics/mAP50-95(B)"), color=LARANJA, lw=1.6, ls="--", label="mAP@50-95")
ax.set_ylim(0.4, 1.0); ax.set_title("validação", fontsize=10); ax.legend(fontsize=8.5, loc="center", bbox_to_anchor=(0.5, 0.5))
ax.set_yticks([0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0,4", "0,6", "0,8", "1,0"])
for ax in axes:
    ax.axvline(melhor, color=CINZA, lw=0.8, ls=":"); ax.set_xlabel("época"); ax.set_xlim(1, 29); ax.set_xticks([1, 10, 19, 29]); grade(ax)
fig.tight_layout(w_pad=1.2)
salvar(fig, "curvas_treino.png")

# ---------------------------------------------------------------- 4. comparativo de metricas
M = np.loadtxt(f"{SRC}/saidas_v2/matriz_confusao.csv", delimiter=",", dtype=int)
TP, FP, FN = int(M[0, 0] + M[0, 1] + M[1, 0] + M[1, 1]), int(M[2].sum()), int(M[0, 2] + M[1, 2])
prf = lambda tp, fp, fn: (tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn))
depois = prf(TP, FP, FN); antes = prf(178, 25, 4)
print("depois", TP, FP, FN, [round(x, 3) for x in depois], "| antes", [round(x, 3) for x in antes])
assert [round(x, 3) for x in depois] == [0.917, 0.967, 0.941] and [round(x, 3) for x in antes] == [0.877, 0.978, 0.925]
fig, ax = plt.subplots(figsize=(5.2, 2.8))
x = np.arange(3); w = 0.34
for desl, vals, cor, nome in ((-w / 2 - 0.012, antes, AZUL, "YOLO11 pré-treinado (COCO)"), (w / 2 + 0.012, depois, LARANJA, "YOLO11 especializado")):
    ax.bar(x + desl, vals, w, color=cor, label=nome)
    for xi, v in zip(x + desl, vals):
        ax.text(xi, v + 0.012, br(v), ha="center", va="bottom", fontsize=9, color=TINTA)
ax.set_xticks(x); ax.set_xticklabels(["precisão", "revocação", "F1"]); ax.set_ylim(0, 1.22); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([br(v, 1) for v in np.arange(0, 1.01, 0.2)])
ax.legend(loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.03)); grade(ax); ax.tick_params(axis="x", length=0)
salvar(fig, "comparativo_metricas.png")

# ---------------------------------------------------------------- 5. matriz de confusao
fig, ax = plt.subplots(figsize=(4.3, 3.3))
ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max() * 1.15)
ax.set_xticks(range(3)); ax.set_xticklabels(["alvo", "outros", "não detectado"])
ax.set_yticks(range(3)); ax.set_yticklabels(["alvo", "outros", "sem anotação"])
ax.set_xlabel("saída do modelo"); ax.set_ylabel("anotação")
for i in range(3):
    for j in range(3):
        if i == 2 and j == 2:
            ax.text(j, i, "–", ha="center", va="center", fontsize=13, color=CINZA); continue
        ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=13, color="white" if M[i, j] > M.max() / 2 else TINTA)
for k in (0.5, 1.5):
    ax.axhline(k, color="white", lw=2); ax.axvline(k, color="white", lw=2)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
salvar(fig, "matriz_confusao.png")

# ---------------------------------------------------------------- 6. revocacao por faixa de tamanho
faixas = ["menor que 70 px", "70 a 100 px", "100 a 150 px"]; tot = [11, 77, 3]
ac_antes, ac_depois = [10, 76, 1], [9, 76, 0]
assert sum(ac_depois) == M[0, 0] and sum(tot) == M[0].sum() and sum(ac_antes) == 178 - 91
fig, ax = plt.subplots(figsize=(5.6, 2.9))
x = np.arange(3)
for desl, ac, cor, nome in ((-w / 2 - 0.012, ac_antes, AZUL, "YOLO11 pré-treinado (COCO)"), (w / 2 + 0.012, ac_depois, LARANJA, "YOLO11 especializado")):
    vals = [a / t for a, t in zip(ac, tot)]
    ax.bar(x + desl, vals, w, color=cor, label=nome)
    for xi, v, a, t in zip(x + desl, vals, ac, tot):
        ax.text(xi, v + 0.012, f"{a}/{t}", ha="center", va="bottom", fontsize=9, color=TINTA)
ax.set_xticks(x); ax.set_xticklabels([f"{f}\n({t} quadros)" for f, t in zip(faixas, tot)])
ax.set_ylabel("revocação do alvo"); ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([br(v, 1) for v in np.arange(0, 1.01, 0.2)])
ax.legend(loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.03)); grade(ax); ax.tick_params(axis="x", length=0)
salvar(fig, "revocacao_por_faixa.png")

# ---------------------------------------------------------------- 7. parametros do alvo
P = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(f"{SRC}/saidas_v2/parametros_alvo.csv"))]
q = np.array([r["quadro"] for r in P]); g = lambda k: np.array([r[k] for r in P])
fig, axes = plt.subplots(3, 1, figsize=(6.3, 6.6), sharex=True)
ax = axes[0]
ax.plot(q, g("marcacao_gt_deg"), color=CINZA, lw=2.6, alpha=0.45, label="a partir da caixa anotada")
ax.plot(q, g("marcacao_deg"), color=AZUL, lw=1.3, label="a partir da caixa detectada")
ax.set_ylabel("marcação (graus)"); ax.set_ylim(0, 34); ax.legend(fontsize=9, loc="upper center")
ax = axes[1]
z, e = g("distancia_m"), g("incerteza_m")
ax.fill_between(q, z - e, z + e, color=LARANJA, alpha=0.20, lw=0, label="faixa de incerteza (0,5 px de disparidade)")
ax.plot(q, z, color=LARANJA, lw=1.4, label="distância estimada")
ax.set_ylabel("distância (m)"); ax.set_ylim(55, 135); ax.legend(fontsize=9, loc="upper center", ncol=2)
ax = axes[2]
c = g("comprimento_implicito_m"); nucleo = (q != 13847) & (q != 13931); med = float(np.median(c[nucleo]))
ax.plot(q, c, color=AGUA, lw=1.4)
ax.axhline(med, color=CINZA, lw=0.9, ls="--")
ax.text(q[-1], med - 0.12, f"mediana {br(med, 2)} m", ha="right", va="top", fontsize=9, color=CINZA)
for qi in (13847, 13931):
    ci = float(c[q == qi][0]); ax.plot([qi], [ci], "o", ms=5, mfc="white", mec=TINTA, mew=1.0, zorder=5)
    ax.annotate(f"quadro {qi}", (qi, ci), xytext=(8 if qi == 13847 else -8, -2), textcoords="offset points",
                ha="left" if qi == 13847 else "right", va="center", fontsize=9, color=TINTA)
ax.set_ylabel("comprimento implícito (m)"); ax.set_xlabel("quadro da sequência"); ax.set_ylim(5, 10.6)
for ax in axes: grade(ax)
fig.tight_layout(h_pad=0.8)
salvar(fig, "parametros_alvo.png")
print("mediana nucleo", med)
