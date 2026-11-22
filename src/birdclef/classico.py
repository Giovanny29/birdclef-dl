"""M0: linha de base clássica (características de processamento de sinais + classificador
tradicional), sem rede neural. Responde: quanto o *deep learning* ganha sobre a abordagem clássica?

Características por janela de 5 s, calculadas a partir do log-Mel 128 × 313 já salvo (mel128_32k),
sem reler os áudios (o MFCC é a DCT do log-Mel):

- MFCC: 20 primeiros coeficientes da DCT-II (ortonormal) do log-Mel em dB, por quadro;
  média e desvio no tempo (40);
- deltas dos MFCC (diferença entre quadros vizinhos): média e desvio (40);
- descritores espectrais por quadro, a partir da potência Mel: centroide e largura de banda
  (em Hz, pelas frequências centrais das bandas), *rolloff* de 85% (Hz) e *flatness*
  (média geométrica / aritmética); média e desvio no tempo (8);
- perfil de energia: média em dB de 16 faixas de 8 bandas Mel (16).

Total: 104 características. Elas são padronizadas com média e desvio do treino.

Classificadores candidatos (escolha pela validação do fold 0, como no E5): regressão logística,
SVM com kernel RBF e *random forest*. Cada janela é um exemplo, com peso 1/(janelas da gravação),
como no sorteio das redes; a probabilidade da gravação é a média das janelas.

Limitação: sem forma de onda, ficam de fora características temporais finas como a taxa de
cruzamentos por zero; o log-Mel já descarta a fase e comprime a resolução em frequência.
"""

from __future__ import annotations

import time

import librosa
import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from birdclef import preproc, treino
from birdclef.experimentos import PASTA_RESULTADOS

PASTA = PASTA_RESULTADOS / "e5"
N_MFCC = 20


def nomes_caracteristicas() -> list[str]:
    n = [f"mfcc{i}_{s}" for s in ("media", "desvio") for i in range(N_MFCC)]
    n += [f"delta{i}_{s}" for s in ("media", "desvio") for i in range(N_MFCC)]
    n += [f"{d}_{s}" for d in ("centroide", "largura", "rolloff", "flatness") for s in ("media", "desvio")]
    n += [f"energia_faixa{k}" for k in range(16)]
    return n


def caracteristicas(x_db: np.ndarray, cfg=preproc.MEL128_32K) -> np.ndarray:
    """(n, 128, quadros) em dB -> (n, 104)."""
    x_db = x_db.astype(np.float32)
    n, m, t = x_db.shape
    mfcc = dct(x_db, type=2, axis=1, norm="ortho")[:, :N_MFCC]           # (n, 20, t)
    delta = np.diff(mfcc, axis=2)
    f_centro = librosa.mel_frequencies(n_mels=m + 2, fmin=cfg.fmin, fmax=cfg.fmax)[1:-1].astype(np.float32)
    pot = np.power(10.0, x_db / 10) + 1e-10                               # (n, m, t)
    tot = pot.sum(1)                                                      # (n, t)
    centroide = (pot * f_centro[None, :, None]).sum(1) / tot
    largura = np.sqrt((pot * (f_centro[None, :, None] - centroide[:, None, :]) ** 2).sum(1) / tot)
    acum = np.cumsum(pot, axis=1) / tot[:, None, :]
    rolloff = f_centro[np.argmax(acum >= 0.85, axis=1)]
    flatness = np.exp(np.log(pot).mean(1)) / pot.mean(1)
    faixas = x_db.reshape(n, 16, m // 16, t).mean(axis=(2, 3))
    partes = [mfcc.mean(2), mfcc.std(2), delta.mean(2), delta.std(2)]
    for d in (centroide, largura, rolloff, flatness):
        partes += [d.mean(1, keepdims=True), d.std(1, keepdims=True)]
    partes.append(faixas)
    return np.concatenate(partes, axis=1)


def extrair(bloco: int = 512) -> tuple[np.ndarray, pd.DataFrame]:
    """Características de todas as janelas (com cache em resultados/e5/caracteristicas_m0.npy)."""
    cache = PASTA / "caracteristicas_m0.npy"
    X, janelas = preproc.carregar(preproc.MEL128_32K)
    if cache.exists():
        return np.load(cache), janelas
    F = np.concatenate([caracteristicas(np.asarray(X[i:i + bloco])) for i in range(0, len(X), bloco)])
    PASTA.mkdir(parents=True, exist_ok=True)
    np.save(cache, F.astype(np.float32))
    return F, janelas


CANDIDATOS = {
    "regressão logística C=0,1": lambda: LogisticRegression(C=0.1, max_iter=3000),
    "regressão logística C=1": lambda: LogisticRegression(C=1.0, max_iter=3000),
    "regressão logística C=10": lambda: LogisticRegression(C=10.0, max_iter=3000),
    "SVM RBF C=1": lambda: SVC(C=1.0, gamma="scale", probability=True, random_state=42),
    "SVM RBF C=10": lambda: SVC(C=10.0, gamma="scale", probability=True, random_state=42),
    "random forest 500 árvores": lambda: RandomForestClassifier(n_estimators=500, n_jobs=2, random_state=42),
}


def _por_gravacao(probs, janelas, idx, classes):
    df = pd.DataFrame(probs, index=janelas["filename"].to_numpy()[idx])
    g = df.groupby(level=0, sort=False).mean()
    rot = janelas.iloc[idx].groupby("filename", sort=False)["primary_label"].first().loc[g.index]
    return g.to_numpy(), rot.map({c: i for i, c in enumerate(classes)}).to_numpy()


def rodar(fold: int = 0, refazer: bool = False) -> tuple[pd.DataFrame, dict]:
    """Escolhe o classificador na validação do fold 0 e avalia só o escolhido no teste.

    O resultado fica em resultados/e5/ (m0_validacao.csv, m0_resultado.json) e é reaproveitado,
    para o notebook rodar do início ao fim sem retreinar (`refazer=True` força)."""
    import json
    arq = PASTA / "m0_resultado.json"
    if arq.exists() and (PASTA / "m0_validacao.csv").exists() and not refazer:
        return pd.read_csv(PASTA / "m0_validacao.csv"), json.loads(arq.read_text(encoding="utf-8"))
    F, janelas = extrair()
    classes = sorted(janelas["primary_label"].unique())
    alvo = janelas["primary_label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    idx_tr, idx_val = treino.indices(janelas, fold)
    idx_te = np.flatnonzero((janelas["particao"] == "teste").to_numpy())
    pesos = 1.0 / janelas.iloc[idx_tr].groupby("filename")["filename"].transform("size").to_numpy()
    esc = StandardScaler().fit(F[idx_tr])
    Ftr, Fval, Fte = esc.transform(F[idx_tr]), esc.transform(F[idx_val]), esc.transform(F[idx_te])

    linhas, modelos_ = [], {}
    for nome, fabrica in CANDIDATOS.items():
        ini = time.time()
        clf = fabrica()
        clf.fit(Ftr, alvo[idx_tr], sample_weight=pesos)
        p, y = _por_gravacao(clf.predict_proba(Fval), janelas, idx_val, classes)
        linhas.append({"classificador": nome, **{f"val_{k}": v for k, v in treino.metricas(y, p).items()},
                       "minutos": (time.time() - ini) / 60})
        modelos_[nome] = clf
        print(f"{nome}: F1 val {linhas[-1]['val_f1_macro']:.3f} ({linhas[-1]['minutos']:.1f} min)", flush=True)
    tab = pd.DataFrame(linhas).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    escolhido = tab.loc[0, "classificador"]
    p_te, y_te = _por_gravacao(modelos_[escolhido].predict_proba(Fte), janelas, idx_te, classes)
    m_te = treino.metricas(y_te, p_te)
    np.save(PASTA / "probs_teste_M0.npy", p_te)
    tab.to_csv(PASTA / "m0_validacao.csv", index=False)
    res = {"escolhido": escolhido, "minutos": float(tab.loc[0, "minutos"]), **{f"teste_{k}": float(v) for k, v in m_te.items()}}
    arq.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return tab, res


if __name__ == "__main__":
    tab, res = rodar()
    print(tab.round(3).to_string())
    print(res)
