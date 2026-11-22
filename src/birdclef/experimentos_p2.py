"""Experimentos da Parte 2: M2, M3 e M4 ao lado do M1.

| Exp. | O que roda                                                                 | Hipótese |
|------|----------------------------------------------------------------------------|----------|
| E1   | M1, M2, M3, M4 nos 5 folds (semente 42), aumento "todas" (M4: sem aumento)  | H1, H2   |
| E2b  | M2 e M3 sem aumento no fold 0 (o "todas" do fold 0 vem do E1)               | H3       |
| E3b  | M1-M4 no teste com ruído da ESC-50 (fold 5) a 20, 10, 5 e 0 dB              | H3       |
| E4   | M1, M3 e M4 com 25%, 50% e 100% das gravações de treino, fold 0             | H1       |

Protocolo comum (o mesmo da Parte 1, `birdclef.treino`): divisões por gravação, teste fixo de 600
gravações usado só na avaliação final, parada antecipada pelo F1-macro de validação por gravação,
predição da gravação = média das janelas. Todo modelo é avaliado nas mesmas janelas.

Cada execução vira um checkpoint em `checkpoints/<modelo>_<aumento>_s<semente>_f<fold>[_p<pct>].pt`
e é pulada se já existir, então dá para interromper e retomar. O `M1_todas_s42_f0` do E2 da Parte 1
é reaproveitado como fold 0 do M1 no E1.

Uso pela linha de comando (a partir da raiz do repositório):

    python -m birdclef.experimentos_p2 e1 M2          # E1 de um modelo
    python -m birdclef.experimentos_p2 e1             # E1 de todos (M1, M2, M3, M4)
    python -m birdclef.experimentos_p2 e2b
    python -m birdclef.experimentos_p2 e3b
    python -m birdclef.experimentos_p2 e4
    python -m birdclef.experimentos_p2 tudo
"""

from dataclasses import asdict, replace
import gc
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from birdclef import aumento, modelos, preproc, treino
from birdclef.experimentos import (PASTA_CHECKPOINTS, PASTA_RESULTADOS, carregar_checkpoint,
                                   carregar_dados, dados_teste)
from birdclef.treino import ConfigTreino

MODELOS = ("M1", "M2", "M3", "M4")
FOLDS = (0, 1, 2, 3, 4)
SEMENTE = 42
FRACOES = (0.25, 0.5, 1.0)

# Hiperparâmetros por modelo. M1 e M2 usam exatamente os do M1 na Parte 1. O M3 (80 M de
# parâmetros) converge em poucas épocas e precisa de lote menor para caber em 4 GB de VRAM; o corpo
# pré-treinado usa taxa 10x menor que a cabeça nova. O M4 só treina uma camada linear: é barato, então
# ganha mais épocas e paciência.
CFG_TREINO = {
    "M1": ConfigTreino(modelo="M1"),
    "M2": ConfigTreino(modelo="M2"),
    "M3": ConfigTreino(modelo="M3", epocas=15, paciencia=4, lote=16, lr=1e-3, lr_backbone=1e-4),
    "M4": ConfigTreino(modelo="M4", epocas=60, paciencia=10, lote=128, lr=1e-3),
}


def aumento_de(modelo: str, nome: str) -> aumento.ConfigAumento:
    """Configuração de aumento `nome` (de `aumento.CONFIGS_E2`) ajustada à entrada do modelo.

    As máscaras do SpecAugment são definidas em bandas e quadros. O M3 usa 64 bandas (em vez de
    128) e quadros de 10 ms (em vez de 16 ms), então as larguras são reescaladas para cobrir a mesma
    fração do espectro e o mesmo tempo que no M1: 16 → 8 bandas, 40 → 64 quadros (~0,64 s).
    O M4 recebe embeddings fixos e não tem aumento.
    """
    if modelo.startswith("M4"):  # M4 e M4FT (E5.6): entrada sem eixo (bandas, quadros) do log-Mel
        return aumento.SEM
    cfg = aumento.CONFIGS_E2[nome]
    if modelo == "M3":
        cfg = replace(cfg, largura_freq=8, largura_tempo=64)
    return cfg


def nome_aumento_e1(modelo: str) -> str:
    return "sem" if modelo.startswith("M4") else "todas"


def nome_execucao(modelo: str, aum: str, semente: int, fold: int, fracao: float = 1.0) -> str:
    base = f"{modelo}_{aum}_s{semente}_f{fold}"
    return base if fracao >= 1.0 else f"{base}_p{int(round(fracao * 100))}"


# ----------------------------------------------------------------------------- dados

_CACHE: dict = {}


def dados_modelo(modelo: str):
    """`X`, `janelas` e `R` da entrada do modelo, com cache (só uma configuração por vez em memória)."""
    nome = modelos.ENTRADA[modelo]
    if nome not in _CACHE:
        _CACHE.clear()
        gc.collect()
        X, janelas, R = carregar_dados(preproc.CONFIGS[nome])
        _CACHE[nome] = (X, janelas, R)
    return _CACHE[nome]


def checar_mesmas_janelas() -> pd.DataFrame:
    """Confere que todas as entradas já geradas têm as mesmas janelas, na mesma ordem."""
    ref = pd.read_csv(preproc.pasta_config(preproc.MEL128_32K) / "janelas.csv")
    linhas = []
    for nome, cfg in preproc.CONFIGS.items():
        caminho = preproc.pasta_config(cfg) / "janelas.csv"
        if caminho.exists():
            j = pd.read_csv(caminho)
            iguais = len(j) == len(ref) and (j[["filename", "inicio_s", "particao", "fold"]].to_numpy()
                                             == ref[["filename", "inicio_s", "particao", "fold"]].to_numpy()).all()
            linhas.append({"entrada": nome, "janelas": len(j), "iguais_ao_mel128_32k": bool(iguais)})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------- uma execução

def executar(modelo: str, aum: str, fold: int, semente: int = SEMENTE, fracao: float = 1.0,
             pasta_hist: str = "historicos") -> tuple[treino.Resultado, str]:
    """Treina (ou recarrega) uma execução e salva checkpoint e histórico."""
    nome = nome_execucao(modelo, aum, semente, fold, fracao)
    ck = PASTA_CHECKPOINTS / f"{nome}.pt"
    if ck.exists():
        return carregar_checkpoint(ck, modelo), nome
    X, janelas, R = dados_modelo(modelo)
    cfg_aum = aumento_de(modelo, aum)
    cfg = replace(CFG_TREINO[modelo], semente=semente, fracao_treino=fracao)
    print(f"--- {nome}  (aumento {cfg_aum.nome}, {modelos.n_parametros(modelos.criar(modelo, 20, pretreinado=False)):,} parâmetros)", flush=True)
    res = treino.treinar(X, janelas, fold, cfg_aum, cfg, R=R)
    treino.salvar(res, ck, historico=res.historico.to_dict("list"), metricas_val=res.metricas_val,
                  segundos=res.segundos, aumento=asdict(cfg_aum), treino=asdict(cfg))
    pasta = PASTA_RESULTADOS / pasta_hist
    pasta.mkdir(parents=True, exist_ok=True)
    res.historico.to_csv(pasta / f"historico_{nome}.csv", index=False)
    return res, nome


def avaliar(res: treino.Resultado, modelo: str, nome: str, pasta) -> tuple[dict, np.ndarray, np.ndarray]:
    """Métricas no teste limpo; guarda as probabilidades por gravação."""
    X, janelas, _ = dados_modelo(modelo)
    X_te, jan_te = dados_teste(X, janelas)
    m, y, probs = treino.avaliar_teste(res, X_te, jan_te)
    pasta.mkdir(parents=True, exist_ok=True)
    np.save(pasta / f"probs_teste_{nome}.npy", probs)
    return m, y, probs


def linha(modelo, aum, fold, fracao, res, m_te) -> dict:
    return {
        "modelo": modelo, "aumento": aum, "fold": fold, "fracao": fracao,
        "epocas": len(res.historico), "melhor_epoca": int(res.metricas_val["epoca"]), "minutos": res.segundos / 60,
        **{f"val_{k}": res.metricas_val[k] for k in ("f1_macro", "acc_top1", "acc_top3", "map")},
        **{f"teste_{k}": v for k, v in m_te.items()},
    }


# ----------------------------------------------------------------------------- E1

def e1(modelos_=MODELOS, folds=FOLDS) -> pd.DataFrame:
    """Cada modelo nos 5 folds. Teste: média ± desvio dos 5 modelos de fold e o *ensemble*
    (média das probabilidades dos 5)."""
    saida = PASTA_RESULTADOS / "e1"
    csv = saida / "e1_execucoes.csv"
    anteriores = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    linhas = []
    for modelo in modelos_:
        aum = nome_aumento_e1(modelo)
        for fold in folds:
            ini = time.time()
            res, nome = executar(modelo, aum, fold, pasta_hist="e1/historicos")
            m_te, y, _ = avaliar(res, modelo, nome, saida)
            linhas.append(linha(modelo, aum, fold, 1.0, res, m_te))
            print(f"{nome}: F1 val {res.metricas_val['f1_macro']:.3f}  F1 teste {m_te['f1_macro']:.3f}  ({(time.time() - ini) / 60:.1f} min)", flush=True)
        np.save(saida / "y_teste.npy", y)
        (saida / "classes.json").write_text(json.dumps(res.classes))
    df = pd.DataFrame(linhas)
    if len(anteriores):
        df = pd.concat([anteriores[~anteriores["modelo"].isin(modelos_)], df], ignore_index=True)
    df.to_csv(csv, index=False)
    resumir_e1()
    return df


def probs_e1(modelo: str, folds=FOLDS) -> np.ndarray:
    aum = nome_aumento_e1(modelo)
    return np.stack([np.load(PASTA_RESULTADOS / "e1" / f"probs_teste_{nome_execucao(modelo, aum, SEMENTE, f)}.npy") for f in folds])


def resumir_e1() -> pd.DataFrame:
    saida = PASTA_RESULTADOS / "e1"
    df = pd.read_csv(saida / "e1_execucoes.csv")
    resumo = df.drop(columns=["aumento", "fold", "fracao"]).groupby("modelo").agg(["mean", "std"])
    resumo.to_csv(saida / "e1_resumo.csv")
    y = np.load(saida / "y_teste.npy")
    ens = []
    for modelo in df["modelo"].unique():
        folds = sorted(df.loc[df["modelo"] == modelo, "fold"])
        p = probs_e1(modelo, folds).mean(0)
        ens.append({"modelo": modelo, "folds": len(folds), **treino.metricas(y, p)})
    pd.DataFrame(ens).to_csv(saida / "e1_ensemble.csv", index=False)
    return resumo


# ----------------------------------------------------------------------------- comparações (H1, H2)

def bootstrap_diferenca(y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, n: int = 5000,
                        semente: int = 0) -> dict:
    """Diferença de F1-macro (A − B) no teste com intervalo de 95% por *bootstrap* pareado:
    as mesmas gravações de teste são reamostradas para os dois modelos."""
    rng = np.random.default_rng(semente)
    pa, pb = probs_a.argmax(1), probs_b.argmax(1)
    obs = f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")
    difs = np.empty(n)
    for i in range(n):
        k = rng.integers(len(y), size=len(y))
        difs[i] = f1_score(y[k], pa[k], average="macro") - f1_score(y[k], pb[k], average="macro")
    lo, hi = np.percentile(difs, [2.5, 97.5])
    return {"diferenca": obs, "ic95_inf": lo, "ic95_sup": hi, "p_bilateral": float(min(1.0, 2 * min((difs <= 0).mean(), (difs >= 0).mean())))}


def comparacoes_e1(pares=(("M2", "M1"), ("M3", "M1"), ("M3", "M2"), ("M4", "M1"), ("M3", "M4"))) -> pd.DataFrame:
    """H1/H2: diferenças por fold (pareadas) e *bootstrap* do *ensemble* no teste."""
    saida = PASTA_RESULTADOS / "e1"
    df = pd.read_csv(saida / "e1_execucoes.csv")
    y = np.load(saida / "y_teste.npy")
    linhas = []
    for a, b in pares:
        if not {a, b} <= set(df["modelo"]):
            continue
        fa = df[df.modelo == a].set_index("fold")["teste_f1_macro"]
        fb = df[df.modelo == b].set_index("fold")["teste_f1_macro"]
        comum = fa.index.intersection(fb.index)
        d = (fa.loc[comum] - fb.loc[comum])
        boot = bootstrap_diferenca(y, probs_e1(a, comum).mean(0), probs_e1(b, comum).mean(0))
        linhas.append({"comparacao": f"{a} − {b}", "folds": len(comum), "dif_media_folds": d.mean(),
                       "dif_desvio_folds": d.std(), "folds_A_melhor": int((d > 0).sum()),
                       **{f"ensemble_{k}": v for k, v in boot.items()}})
    out = pd.DataFrame(linhas)
    out.to_csv(saida / "e1_comparacoes.csv", index=False)
    return out


# ----------------------------------------------------------------------------- E2b e E3b

def e2b(modelos_=("M2", "M3")) -> pd.DataFrame:
    """Sem aumento no fold 0 para M2 e M3; com o "todas" do fold 0 do E1 forma o par do E3."""
    saida = PASTA_RESULTADOS / "e2b"
    linhas = []
    for modelo in modelos_:
        for aum in ("sem", "todas"):
            res, nome = executar(modelo, aum, 0, pasta_hist="e2b/historicos" if aum == "sem" else "e1/historicos")
            m_te, _, _ = avaliar(res, modelo, nome, saida)
            linhas.append(linha(modelo, aum, 0, 1.0, res, m_te))
    df = pd.DataFrame(linhas)
    df.to_csv(saida / "e2b_execucoes.csv", index=False)
    return df


def versoes_teste(modelo: str) -> dict:
    """Teste limpo e com ruído (mesmas janelas e mesmos ruídos para todos os modelos)."""
    X, janelas, _ = dados_modelo(modelo)
    pasta = preproc.pasta_config(preproc.CONFIGS[modelos.ENTRADA[modelo]])
    jan_te = pd.read_csv(pasta / "janelas_teste.csv")
    X_limpo, jan_limpo = dados_teste(X, janelas)
    assert (jan_limpo[["filename", "inicio_s"]].to_numpy() == jan_te[["filename", "inicio_s"]].to_numpy()).all()
    v = {"limpo": np.asarray(X_limpo)}
    for snr in aumento.SNRS_TESTE:
        v[f"{snr} dB"] = np.load(pasta / f"X_teste_snr{snr}.npy")
    return v, jan_te


def e3b(modelos_=MODELOS) -> pd.DataFrame:
    """Robustez a ruído de todos os modelos, fold 0, semente 42. M1 usa os checkpoints da
    Parte 1 (mesma semente e fold), M4 só tem a versão sem aumento."""
    linhas = []
    for modelo in modelos_:
        versoes, jan_te = versoes_teste(modelo)
        configs = ("sem",) if modelo == "M4" else ("sem", "todas")
        for aum in configs:
            ck = PASTA_CHECKPOINTS / f"{nome_execucao(modelo, aum, SEMENTE, 0)}.pt"
            if not ck.exists():
                print(f"(pulando {ck.name}: rode e1/e2b antes)")
                continue
            res = carregar_checkpoint(ck, modelo)
            for versao, Xv in versoes.items():
                m, _, _ = treino.avaliar_teste(res, Xv, jan_te)
                linhas.append({"modelo": modelo, "aumento": aum, "teste": versao, **m})
    df = pd.DataFrame(linhas)
    saida = PASTA_RESULTADOS / "e3b"
    saida.mkdir(parents=True, exist_ok=True)
    df.to_csv(saida / "e3b_execucoes.csv", index=False)
    return df


# ----------------------------------------------------------------------------- E4

def e4(modelos_=("M1", "M3", "M4"), fracoes=FRACOES) -> pd.DataFrame:
    """Eficiência de dados: fold 0, semente 42; 100% é a execução do fold 0 do E1."""
    saida = PASTA_RESULTADOS / "e4"
    linhas = []
    for modelo in modelos_:
        aum = nome_aumento_e1(modelo)
        for fr in fracoes:
            res, nome = executar(modelo, aum, 0, fracao=fr, pasta_hist="e4/historicos" if fr < 1 else "e1/historicos")
            m_te, _, _ = avaliar(res, modelo, nome, saida)
            linhas.append(linha(modelo, aum, 0, fr, res, m_te))
            print(f"{nome}: F1 teste {m_te['f1_macro']:.3f}", flush=True)
    df = pd.DataFrame(linhas)
    df.to_csv(saida / "e4_execucoes.csv", index=False)
    return df


if __name__ == "__main__":
    qual = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    quais_modelos = tuple(sys.argv[2:]) or MODELOS
    if qual in ("e1", "tudo"):
        e1(quais_modelos)
        comparacoes_e1()
    if qual in ("e2b", "tudo"):
        e2b()
    if qual in ("e3b", "tudo"):
        e3b()
    if qual in ("e4", "tudo"):
        e4()
