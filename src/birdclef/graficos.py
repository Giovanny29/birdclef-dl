"""Figuras dos experimentos (PNG em `resultados/`)."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from birdclef import treino
from birdclef.experimentos import PASTA_RESULTADOS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TEXTO, TEXTO_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
AZUIS = LinearSegmentedColormap.from_list("azuis", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": TEXTO_2,
    "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titlecolor": TEXTO, "figure.dpi": 150,
})

ROTULOS = {"sem": "sem aumento", "specaug": "SpecAugment", "mixup": "mixup", "ruido": "ruído", "todas": "todas"}


def _salvar(fig, caminho: Path):
    fig.tight_layout()
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)


def e2_barras(df: pd.DataFrame, metrica: str = "teste_f1_macro"):
    """F1-macro no teste por configuração: média e desvio das sementes, pontos individuais."""
    ordem = list(ROTULOS)
    g = df.groupby("aumento")[metrica]
    media, desvio = g.mean().reindex(ordem), g.std().reindex(ordem)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(ordem))
    ax.bar(x, media, width=0.6, color=SERIES[0], yerr=desvio, capsize=4, error_kw={"ecolor": TEXTO_2, "lw": 1})
    for i, a in enumerate(ordem):
        v = df.loc[df["aumento"] == a, metrica]
        ax.scatter(np.full(len(v), i), v, s=14, color=TEXTO, zorder=3)
        ax.text(i, media[a] + desvio[a] + 0.01, f"{media[a]:.3f}", ha="center", va="bottom", color=TEXTO, fontsize=9)
    ax.set_xticks(x, [ROTULOS[a] for a in ordem])
    ax.set_ylabel("F1-macro (teste)")
    ax.set_ylim(max(0, media.min() - 0.15), min(1, (media + desvio).max() + 0.06))
    ax.grid(axis="x", visible=False)
    ax.set_title("E2: aumento de dados no M1 (3 sementes)")
    _salvar(fig, PASTA_RESULTADOS / "e2" / "e2_f1_teste.png")


def e2_curvas(semente: int = 42):
    """F1-macro de validação por época, uma linha por configuração."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for cor, a in zip(SERIES, ROTULOS):
        h = pd.read_csv(PASTA_RESULTADOS / "e2" / f"historico_M1_{a}_s{semente}_f0.csv")
        ax.plot(h["epoca"], h["f1_macro"], color=cor, lw=2, label=ROTULOS[a])
    ax.set_xlabel("época")
    ax.set_ylabel("F1-macro (validação)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"E2: curvas de validação (semente {semente})")
    _salvar(fig, PASTA_RESULTADOS / "e2" / "e2_curvas.png")


def e3_linhas(df: pd.DataFrame, metrica: str = "f1_macro"):
    """F1-macro em função da SNR do ruído, com e sem aumento."""
    ordem = ["limpo"] + [f"{s} dB" for s in (20, 10, 5, 0)]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(ordem))
    for cor, a in zip(SERIES, ["sem", "todas"]):
        g = df[df["aumento"] == a].groupby("teste")[metrica]
        m, s = g.mean().reindex(ordem), g.std().reindex(ordem)
        ax.fill_between(x, m - s, m + s, color=cor, alpha=0.15, lw=0)
        ax.plot(x, m, color=cor, lw=2, marker="o", ms=6, label=ROTULOS[a])
        ax.text(x[-1] + 0.08, m.iloc[-1], f"{ROTULOS[a]}  {m.iloc[-1]:.2f}", va="center", color=TEXTO, fontsize=9)
    ax.set_xticks(x, ordem)
    ax.set_xlim(-0.2, len(ordem) - 0.2 + 0.9)
    ax.set_xlabel("teste (SNR do ruído ESC-50, fold 5)")
    ax.set_ylabel("F1-macro (teste)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("E3: robustez a ruído de fundo (M1)")
    _salvar(fig, PASTA_RESULTADOS / "e3" / "e3_snr.png")


def matriz(probs: np.ndarray, y: np.ndarray, classes: list[str], nomes: dict, titulo: str, caminho: Path):
    cm = treino.matriz_confusao(y, probs, len(classes))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap=AZUIS, vmin=0, vmax=1)
    rot = [nomes.get(c, c) for c in classes]
    ax.set_xticks(range(len(classes)), rot, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), rot, fontsize=7)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cm_norm[i, j] > 0.5 else TEXTO)
    ax.set_xlabel("predita")
    ax.set_ylabel("verdadeira")
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, fraction=0.04, label="fração da classe verdadeira")
    _salvar(fig, caminho)


def todas():
    from birdclef import dados
    e2 = pd.read_csv(PASTA_RESULTADOS / "e2" / "e2_execucoes.csv")
    e2_barras(e2)
    e2_curvas()
    pasta_e2 = PASTA_RESULTADOS / "e2"
    classes = json.loads((pasta_e2 / "classes.json").read_text())
    nomes = dados.carregar_metadados().drop_duplicates("primary_label").set_index("primary_label")["common_name"].to_dict()
    melhor = e2.groupby("aumento")["teste_f1_macro"].mean().idxmax()
    y = np.load(pasta_e2 / "y_teste.npy")
    probs = np.load(pasta_e2 / f"probs_teste_M1_{melhor}_s42_f0.npy")
    matriz(probs, y, classes, nomes, f"M1 ({ROTULOS[melhor]}), teste limpo, semente 42", pasta_e2 / "matriz_confusao.png")
    e3_csv = PASTA_RESULTADOS / "e3" / "e3_execucoes.csv"
    if e3_csv.exists():
        e3_linhas(pd.read_csv(e3_csv))


# ============================================================================= Parte 2

NOMES_MODELOS = {"M1": "M1 CNN", "M2": "M2 CRNN", "M3": "M3 PANNs CNN14", "M4": "M4 AST (linear)"}
COR_MODELO = dict(zip(NOMES_MODELOS, SERIES))


def e1_barras(df: pd.DataFrame, ens: pd.DataFrame | None = None, metrica: str = "teste_f1_macro"):
    """E1: F1-macro no teste por modelo (média ± desvio dos 5 folds, pontos = folds, losango =
    ensemble dos 5)."""
    ordem = [m for m in NOMES_MODELOS if m in set(df["modelo"])]
    g = df.groupby("modelo")[metrica]
    media, desvio = g.mean().reindex(ordem), g.std().reindex(ordem)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(ordem))
    ax.bar(x, media, width=0.6, color=[COR_MODELO[m] for m in ordem], yerr=desvio, capsize=4,
           error_kw={"ecolor": TEXTO_2, "lw": 1})
    for i, m in enumerate(ordem):
        v = df.loc[df["modelo"] == m, metrica]
        ax.scatter(np.full(len(v), i), v, s=14, color=TEXTO, zorder=3)
        ax.text(i, media[m] + desvio[m] + 0.012, f"{media[m]:.3f}", ha="center", va="bottom", color=TEXTO, fontsize=9)
        if ens is not None and m in set(ens["modelo"]):
            ax.scatter([i + 0.22], ens.loc[ens["modelo"] == m, "f1_macro"], marker="D", s=26, color=TEXTO_2, zorder=3,
                       label="ensemble dos 5 folds" if i == 0 else None)
    ax.set_xticks(x, [NOMES_MODELOS[m] for m in ordem])
    ax.set_ylabel("F1-macro (teste)")
    ax.set_ylim(max(0, (media - desvio).min() - 0.1), min(1.0, (media + desvio).max() + 0.07))
    ax.grid(axis="x", visible=False)
    if ens is not None:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("E1: arquiteturas (5 folds, mesmo teste de 600 gravações)")
    _salvar(fig, PASTA_RESULTADOS / "e1" / "e1_f1_teste.png")


def e1_curvas(fold: int = 0):
    """F1-macro de validação por época, um modelo por linha (fold `fold`)."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    pasta = PASTA_RESULTADOS / "e1" / "historicos"
    for m in NOMES_MODELOS:
        aum = "sem" if m == "M4" else "todas"
        caminho = pasta / f"historico_{m}_{aum}_s42_f{fold}.csv"
        if m == "M1" and not caminho.exists():
            caminho = PASTA_RESULTADOS / "e2" / f"historico_M1_todas_s42_f{fold}.csv"
        if caminho.exists():
            h = pd.read_csv(caminho)
            ax.plot(h["epoca"], h["f1_macro"], color=COR_MODELO[m], lw=2, label=NOMES_MODELOS[m])
    ax.set_xlabel("época")
    ax.set_ylabel("F1-macro (validação)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"E1: curvas de validação (fold {fold})")
    _salvar(fig, PASTA_RESULTADOS / "e1" / "e1_curvas.png")


def e3b_linhas(df: pd.DataFrame, metrica: str = "f1_macro"):
    """E3b: F1-macro x SNR, uma cor por modelo; linha cheia = com aumento, tracejada = sem."""
    ordem = ["limpo"] + [f"{s} dB" for s in (20, 10, 5, 0)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(ordem))
    for m in NOMES_MODELOS:
        for aum, estilo in (("todas", "-"), ("sem", "--")):
            g = df[(df["modelo"] == m) & (df["aumento"] == aum)]
            if g.empty:
                continue
            v = g.groupby("teste")[metrica].mean().reindex(ordem)
            ax.plot(x, v, color=COR_MODELO[m], lw=2, ls=estilo, marker="o", ms=4,
                    label=f"{NOMES_MODELOS[m]} ({'com' if aum == 'todas' else 'sem'} aumento)")
    ax.set_xticks(x, ordem)
    ax.set_xlabel("teste (SNR do ruído ESC-50, fold 5)")
    ax.set_ylabel("F1-macro (teste)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    ax.set_title("E3: robustez a ruído de fundo, todos os modelos (fold 0)")
    _salvar(fig, PASTA_RESULTADOS / "e3b" / "e3b_snr.png")


def e4_curva(df: pd.DataFrame, metrica: str = "teste_f1_macro"):
    """E4: F1-macro no teste x fração das gravações de treino."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for m in NOMES_MODELOS:
        g = df[df["modelo"] == m].sort_values("fracao")
        if g.empty:
            continue
        ax.plot(g["fracao"] * 100, g[metrica], color=COR_MODELO[m], lw=2, marker="o", label=NOMES_MODELOS[m])
        ax.text(g["fracao"].iloc[-1] * 100 + 2, g[metrica].iloc[-1], f"{g[metrica].iloc[-1]:.2f}", va="center", fontsize=8)
    ax.set_xticks([25, 50, 100], ["25%", "50%", "100%"])
    ax.set_xlabel("gravações de treino usadas (≈ 450, 900 e 1800)")
    ax.set_ylabel("F1-macro (teste)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("E4: eficiência de dados (fold 0)")
    _salvar(fig, PASTA_RESULTADOS / "e4" / "e4_fracao.png")


def todas_parte2():
    from birdclef import dados
    pasta = PASTA_RESULTADOS / "e1"
    if (pasta / "e1_execucoes.csv").exists():
        df = pd.read_csv(pasta / "e1_execucoes.csv")
        ens = pd.read_csv(pasta / "e1_ensemble.csv") if (pasta / "e1_ensemble.csv").exists() else None
        e1_barras(df, ens)
        e1_curvas()
        classes = json.loads((pasta / "classes.json").read_text())
        nomes = dados.carregar_metadados().drop_duplicates("primary_label").set_index("primary_label")["common_name"].to_dict()
        y = np.load(pasta / "y_teste.npy")
        for m in df["modelo"].unique():
            aum = "sem" if m == "M4" else "todas"
            probs = np.mean([np.load(pasta / f"probs_teste_{m}_{aum}_s42_f{f}.npy") for f in sorted(df.loc[df.modelo == m, "fold"])], axis=0)
            matriz(probs, y, classes, nomes, f"{NOMES_MODELOS[m]}, ensemble dos folds, teste limpo", pasta / f"matriz_{m}.png")
    if (PASTA_RESULTADOS / "e3b" / "e3b_execucoes.csv").exists():
        e3b_linhas(pd.read_csv(PASTA_RESULTADOS / "e3b" / "e3b_execucoes.csv"))
    if (PASTA_RESULTADOS / "e4" / "e4_execucoes.csv").exists():
        e4_curva(pd.read_csv(PASTA_RESULTADOS / "e4" / "e4_execucoes.csv"))


if __name__ == "__main__":
    todas()
    todas_parte2()
