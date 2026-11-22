"""E5: melhorias testadas depois do E1-E4, com o mesmo protocolo.

Regras (as mesmas do relatório):

- toda escolha (manter ou descartar, hiperparâmetros, pesos do ensemble, temperatura) é feita pela
  **validação do fold 0** (480 gravações, as mesmas para todos os modelos);
- cada técnica é comparada com a execução equivalente do E1 (mesmo modelo, fold 0, semente 42),
  mudando uma coisa por vez;
- o **teste** (600 gravações) só é olhado uma vez por técnica *já escolhida* na validação; as
  descartadas ficam sem número de teste na tabela (de propósito);
- tudo vai para `resultados/e5/e5_execucoes.csv`, inclusive o que não funcionou.

Exceção natural: a divisão por gravador (E5.3) não é uma escolha, é um teste de robustez a
vazamento; ela muda a própria divisão, então é avaliada no seu próprio teste.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from birdclef import aumento, dados, treino
from birdclef import experimentos_p2 as e
from birdclef.experimentos import PASTA_CHECKPOINTS, PASTA_RESULTADOS, carregar_checkpoint, dados_teste
from birdclef.treino import ConfigTreino

PASTA = PASTA_RESULTADOS / "e5"
CSV = PASTA / "e5_execucoes.csv"
FOLD = 0
COLUNAS = ["tecnica", "modelo", "o_que_mudou", "base", "val_f1_base", "val_f1", "dif_val",
           "teste_f1_base", "teste_f1", "minutos", "veredito", "obs"]


# ----------------------------------------------------------------------------- registro

def registrar(**linha) -> pd.DataFrame:
    """Acrescenta (ou substitui, pela chave `tecnica`) uma linha na tabela do E5."""
    PASTA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=COLUNAS)
    if linha.get("val_f1") is not None and linha.get("val_f1_base") is not None:
        linha["dif_val"] = linha["val_f1"] - linha["val_f1_base"]
    df = df[df["tecnica"] != linha["tecnica"]]
    df = pd.concat([df, pd.DataFrame([{c: linha.get(c) for c in COLUNAS}])], ignore_index=True)
    df.to_csv(CSV, index=False)
    return df


def tabela() -> pd.DataFrame:
    return pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=COLUNAS)


# ----------------------------------------------------------------------------- predições

def _cache(nome: str):
    return PASTA / "probs" / f"{nome}.npy"


def probs_val(modelo: str, nome_ck: str, janelas_=None, deslocamentos=(0,)) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades por gravação na validação do fold 0 (com cache) e os rótulos.

    `deslocamentos`: TTA, em segundos (ver `prever_tta`)."""
    sufixo = "" if tuple(deslocamentos) == (0,) else "_tta" + "_".join(f"{d:g}" for d in deslocamentos)
    cache = _cache(f"val_{nome_ck}{sufixo}")
    X, janelas, _ = e.dados_modelo(modelo)
    janelas = janelas if janelas_ is None else janelas_
    classes = sorted(janelas["primary_label"].unique())
    _, idx_val = treino.indices(janelas, FOLD)
    rot = janelas.iloc[idx_val].groupby("filename", sort=False)["primary_label"].first()
    y = rot.map({c: i for i, c in enumerate(classes)}).to_numpy()
    if cache.exists():
        return np.load(cache), y
    res = carregar_checkpoint(PASTA_CHECKPOINTS / f"{nome_ck}.pt", modelo)
    probs = prever_tta(res, modelo, X, janelas, idx_val, deslocamentos)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, probs)
    return probs, y


def probs_teste(modelo: str, nome_ck: str, deslocamentos=(0,)) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades por gravação no teste limpo (com cache) e os rótulos."""
    sufixo = "" if tuple(deslocamentos) == (0,) else "_tta" + "_".join(f"{d:g}" for d in deslocamentos)
    cache = _cache(f"teste_{nome_ck}{sufixo}")
    X, janelas, _ = e.dados_modelo(modelo)
    classes = sorted(janelas["primary_label"].unique())
    idx = np.flatnonzero((janelas["particao"] == "teste").to_numpy())
    rot = janelas.iloc[idx].groupby("filename", sort=False)["primary_label"].first()
    y = rot.map({c: i for i, c in enumerate(classes)}).to_numpy()
    if cache.exists():
        return np.load(cache), y
    res = carregar_checkpoint(PASTA_CHECKPOINTS / f"{nome_ck}.pt", modelo)
    probs = prever_tta(res, modelo, X, janelas, idx, deslocamentos)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, probs)
    return probs, y


def prever_tta(res, modelo: str, X, janelas, idx, deslocamentos=(0,)) -> np.ndarray:
    """TTA por deslocamento no tempo: cada janela é deslocada circularmente por `d` segundos
    (eixo dos quadros do log-Mel) e as probabilidades das versões são promediadas. Com (0,) é a
    predição normal. O M4 (embedding fixo) não tem eixo de tempo: só (0,)."""
    if modelo in ("M6", "M6bA"):  # M6 com atenção: a gravação sai da atenção sobre as janelas, não da média
        from birdclef import m6
        return m6.probs_gravacoes(res.modelo, X, janelas, idx, res.norm, res.classes)[0]
    if modelo == "M4" or tuple(deslocamentos) == (0,):
        return treino.prever_gravacoes(res.modelo, X, janelas, idx, res.norm, res.classes)[0]
    quadros_por_s = X.shape[-1] / 5.0
    total = 0
    for d in deslocamentos:
        k = int(round(d * quadros_por_s))
        Xd = _Deslocado(X, k)
        total = total + treino.prever_gravacoes(res.modelo, Xd, janelas, idx, res.norm, res.classes)[0]
    return total / len(deslocamentos)


class _Deslocado:
    """Vista de `X` que aplica `np.roll` no eixo do tempo ao indexar (sem copiar `X` inteiro)."""

    def __init__(self, X, k):
        self.X, self.k = X, k

    def __getitem__(self, idx):
        return np.roll(np.asarray(self.X[idx]), self.k, axis=-1)

    @property
    def shape(self):
        return self.X.shape


def f1(y, p) -> float:
    return float(f1_score(y, p.argmax(1), average="macro"))


# ----------------------------------------------------------------------------- E5.1 ensemble e TTA

def nome_e1(modelo: str, fold: int = FOLD) -> str:
    return e.nome_execucao(modelo, e.nome_aumento_e1(modelo), e.SEMENTE, fold)


def grade_simplex(n: int, passo: float = 0.05):
    k = int(round(1 / passo))
    for c in itertools.product(range(k + 1), repeat=n - 1):
        if sum(c) <= k:
            yield np.array([*c, k - sum(c)]) / k


def ensemble(membros=("M1", "M2", "M3"), passo: float = 0.05) -> dict:
    """Média simples e média ponderada (pesos escolhidos na validação do fold 0) dos modelos do
    fold 0 do E1. Devolve as métricas de validação de tudo e os pesos escolhidos."""
    val = {m: probs_val(m, nome_e1(m)) for m in membros}
    y = next(iter(val.values()))[1]
    individuais = {m: f1(y, p) for m, (p, _) in val.items()}
    P = np.stack([val[m][0] for m in membros])
    simples = f1(y, P.mean(0))
    melhor, pesos = -1, None
    for w in grade_simplex(len(membros), passo):
        s = f1(y, np.tensordot(w, P, 1))
        # desempate: prefere pesos mais uniformes (menos sobreajuste à validação)
        if s > melhor + 1e-9 or (abs(s - melhor) < 1e-9 and np.std(w) < np.std(pesos)):
            melhor, pesos = s, w
    return {"membros": list(membros), "val_individual": individuais, "val_media_simples": simples,
            "val_ponderada": melhor, "pesos": dict(zip(membros, map(float, pesos)))}


def teste_ensemble(membros, pesos) -> float:
    P = [probs_teste(m, nome_e1(m))[0] for m in membros]
    y = probs_teste(membros[0], nome_e1(membros[0]))[1]
    w = np.array([pesos[m] for m in membros])
    return f1(y, np.tensordot(w / w.sum(), np.stack(P), 1))


# ----------------------------------------------------------------------------- variantes de treino

def treinar_variante(nome: str, modelo: str, cfg: ConfigTreino, cfg_aum=None, janelas_=None,
                     fold: int = FOLD, alvos_extra=None) -> tuple[treino.Resultado, float]:
    """Treina (ou recarrega) uma variante do E5; checkpoint em `checkpoints/<nome>.pt`."""
    ck = PASTA_CHECKPOINTS / f"{nome}.pt"
    if ck.exists():
        r = carregar_checkpoint(ck, modelo)
        return r, r.segundos / 60
    X, janelas, R = e.dados_modelo(modelo)
    janelas = janelas if janelas_ is None else janelas_
    cfg_aum = cfg_aum or e.aumento_de(modelo, e.nome_aumento_e1(modelo))
    print(f"--- {nome}", flush=True)
    res = treino.treinar(X, janelas, fold, cfg_aum, cfg, R=R, alvos_extra=alvos_extra)
    treino.salvar(res, ck, historico=res.historico.to_dict("list"), metricas_val=res.metricas_val,
                  segundos=res.segundos, aumento=asdict(cfg_aum), treino=asdict(cfg))
    (PASTA / "historicos").mkdir(parents=True, exist_ok=True)
    res.historico.to_csv(PASTA / "historicos" / f"historico_{nome}.csv", index=False)
    return res, res.segundos / 60


def val_f1_checkpoint(nome: str) -> float:
    ck = torch.load(PASTA_CHECKPOINTS / f"{nome}.pt", map_location="cpu", weights_only=False)
    return float(ck["metricas_val"]["f1_macro"])


# ----------------------------------------------------------------------------- E5.3 divisão por gravador

def janelas_por_autor(semente: int = dados.SEMENTE) -> pd.DataFrame:
    """Mesmas janelas, nova divisão agrupada por gravador (`author`).

    Teste: um dos 5 grupos de `StratifiedGroupKFold` (≈ 20% das gravações), sem nenhum autor em
    comum com o treino. Folds: `StratifiedGroupKFold` de novo nos 80% restantes. Estratificado por
    espécie na medida do possível (um autor com centenas de gravações de uma espécie limita isso).
    """
    div = pd.read_csv(dados.pasta_dados() / "subconjunto" / "divisoes.csv")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semente)
    idx_tr, idx_te = next(sgkf.split(div, div["primary_label"], div["author"]))
    div["particao_autor"] = "treino"
    div.loc[idx_te, "particao_autor"] = "teste"
    div["fold_autor"] = -1
    tr = div.loc[idx_tr]
    for k, (_, iv) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semente)
                                .split(tr, tr["primary_label"], tr["author"])):
        div.loc[tr.index[iv], "fold_autor"] = k
    _, janelas, _ = e.dados_modelo("M1")
    mapa = div.set_index("filename")
    j = janelas.copy()
    j["particao"] = mapa.loc[j["filename"], "particao_autor"].to_numpy()
    j["fold"] = mapa.loc[j["filename"], "fold_autor"].to_numpy()
    return j


def resumo_divisao_autor(j: pd.DataFrame) -> pd.DataFrame:
    div = pd.read_csv(dados.pasta_dados() / "subconjunto" / "divisoes.csv").set_index("filename")
    g = j.drop_duplicates("filename").assign(author=lambda d: div.loc[d["filename"], "author"].to_numpy())
    te, tr = g[g.particao == "teste"], g[g.particao == "treino"]
    return pd.DataFrame([{
        "gravacoes_teste": len(te), "gravacoes_treino": len(tr),
        "autores_teste": te.author.nunique(), "autores_treino": tr.author.nunique(),
        "autores_em_comum": len(set(te.author) & set(tr.author)),
        "especies_no_teste": te.primary_label.nunique(),
        "min_gravacoes_teste_por_especie": int(te.primary_label.value_counts().min()),
    }])


def avaliar_teste_janelas(res, modelo: str, janelas_) -> tuple[dict, np.ndarray, np.ndarray]:
    X, _, _ = e.dados_modelo(modelo)
    X_te, jan_te = dados_teste(X, janelas_)
    return treino.avaliar_teste(res, X_te, jan_te)


# ----------------------------------------------------------------------------- E5.7 calibração

def _log(p):
    return np.log(np.clip(p, 1e-12, 1.0))


def com_temperatura(p: np.ndarray, T: float) -> np.ndarray:
    z = _log(p) / T
    z -= z.max(1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def nll(y, p) -> float:
    return float(-_log(p[np.arange(len(y)), y]).mean())


def ece(y, p, bins: int = 15) -> float:
    """Erro de calibração esperado (confiança do top-1 contra acurácia, em `bins` faixas)."""
    conf, pred = p.max(1), p.argmax(1)
    acerto = (pred == y).astype(float)
    bordas = np.linspace(0, 1, bins + 1)
    total = 0.0
    for a, b in zip(bordas[:-1], bordas[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(acerto[m].mean() - conf[m].mean())
    return float(total)


def ajustar_temperatura(y, p, grade=np.round(np.arange(0.3, 3.01, 0.01), 2)) -> float:
    """Temperatura que minimiza a NLL na validação (busca em grade; F1 não muda, só a confiança)."""
    return float(grade[np.argmin([nll(y, com_temperatura(p, T)) for T in grade])])


def curva_confiabilidade(y, p, bins: int = 10) -> pd.DataFrame:
    conf, pred = p.max(1), p.argmax(1)
    acerto = (pred == y).astype(float)
    bordas = np.linspace(0, 1, bins + 1)
    linhas = []
    for a, b in zip(bordas[:-1], bordas[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            linhas.append({"faixa": (a + b) / 2, "confianca": conf[m].mean(), "acuracia": acerto[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------- destilação

def alvos_professor(professor: str = "M3_todas_s42_f0", modelo_prof: str = "M3", temperatura: float = 2.0) -> np.ndarray:
    """Probabilidades do professor (M3 do fold 0) em cada janela de treino do fold 0, suavizadas
    pela temperatura (p^(1/T), renormalizado). Linhas fora do treino ficam em zero e nunca são
    usadas. O professor viu exatamente as mesmas janelas de treino, então nada da validação ou do
    teste vaza para o aluno. Cache em resultados/e5/probs/professor_<nome>.npy."""
    cache = _cache(f"professor_{professor}_T{temperatura:g}")
    if cache.exists():
        return np.load(cache)
    X, janelas, _ = e.dados_modelo(modelo_prof)
    idx_tr, _ = treino.indices(janelas, FOLD)
    res = carregar_checkpoint(PASTA_CHECKPOINTS / f"{professor}.pt", modelo_prof)
    p = treino.prever_janelas(res.modelo, X, idx_tr, res.norm)
    p = np.power(p, 1.0 / temperatura)
    p /= p.sum(1, keepdims=True)
    alvos = np.zeros((len(janelas), p.shape[1]), dtype=np.float32)
    alvos[idx_tr] = p
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, alvos)
    del res
    torch.cuda.empty_cache()
    return alvos


# ----------------------------------------------------------------------------- agregação das janelas

def probs_janelas(modelo: str, nome_ck: str, particao: str = "val"):
    """Probabilidades por janela (sem agregar), o arquivo de cada janela e o rótulo de cada
    gravação na ordem de `agregar`. `particao`: "val" (fold 0) ou "teste"."""
    cache = _cache(f"janelas_{particao}_{nome_ck}")
    X, janelas, _ = e.dados_modelo(modelo)
    if particao == "val":
        _, idx = treino.indices(janelas, FOLD)
    else:
        idx = np.flatnonzero((janelas["particao"] == "teste").to_numpy())
    arquivos = janelas["filename"].to_numpy()[idx]
    classes = sorted(janelas["primary_label"].unique())
    rot = janelas.iloc[idx].groupby("filename", sort=False)["primary_label"].first()
    y = rot.map({c: i for i, c in enumerate(classes)}).to_numpy()
    if cache.exists():
        return np.load(cache), arquivos, y
    res = carregar_checkpoint(PASTA_CHECKPOINTS / f"{nome_ck}.pt", modelo)
    p = treino.prever_janelas(res.modelo, X, idx, res.norm)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, p)
    return p, arquivos, y


def agregar(pj: np.ndarray, arquivos: np.ndarray, modo: str = "media", k: int = 3) -> np.ndarray:
    """Probabilidade da gravação a partir das janelas: "media" (o padrão do projeto), "max"
    (máximo por classe, renormalizado), "topk" (média das k janelas mais confiantes) ou
    "geometrica" (média dos logaritmos, renormalizada). Ordem das gravações: a de aparição."""
    ordem = pd.unique(arquivos)
    if modo == "media":
        out = pd.DataFrame(pj, index=arquivos).groupby(level=0, sort=False).mean().loc[ordem].to_numpy()
    elif modo == "max":
        out = pd.DataFrame(pj, index=arquivos).groupby(level=0, sort=False).max().loc[ordem].to_numpy()
    elif modo == "geometrica":
        lg = pd.DataFrame(np.log(np.clip(pj, 1e-12, 1)), index=arquivos).groupby(level=0, sort=False).mean()
        out = np.exp(lg.loc[ordem].to_numpy())
    elif modo == "topk":
        conf = pj.max(1)
        pos = pd.Series(np.arange(len(arquivos))).groupby(arquivos, sort=False).apply(list)
        out = np.stack([pj[np.array(ix)[np.argsort(-conf[ix])[:k]]].mean(0) for ix in pos.loc[ordem]])
    else:
        raise ValueError(modo)
    return out / out.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- variantes treinadas

def _cfg(modelo: str, **mudancas) -> ConfigTreino:
    return replace(e.CFG_TREINO[modelo], **mudancas) if modelo in e.CFG_TREINO else ConfigTreino(modelo=modelo, **mudancas)


# Nome do checkpoint -> (modelo, configuração, o que mudou em relação ao E1). Fold 0, semente 42,
# aumento "todas" (o mesmo do E1). Uma mudança por vez.
VARIANTES = {
    "M1_todas_e50_s42_f0": ("M1", _cfg("M1", epocas=50, paciencia=10), "50 épocas (E1: 30) e paciência 10 (E1: 6)"),
    "M2_todas_e50_s42_f0": ("M2", _cfg("M2", epocas=50, paciencia=10), "50 épocas (E1: 30) e paciência 10 (E1: 6)"),
    "M1_todas_ls01_s42_f0": ("M1", _cfg("M1", suavizacao=0.1), "label smoothing 0,1"),
    "M1_todas_destil_s42_f0": ("M1", _cfg("M1", destilacao=0.5),
                               "destilação: alvo = 0,5·rótulo + 0,5·probabilidades do M3 do fold 0 (temperatura 2)"),
    "M1PCEN_todas_s42_f0": ("M1PCEN", replace(e.CFG_TREINO["M1"], modelo="M1PCEN"), "PCEN no lugar do log na entrada da mesma CNN"),
    "M3_todas_2fases_s42_f0": ("M3", _cfg("M3", epocas_cabeca=2), "2 primeiras épocas só a cabeça treina (corpo congelado)"),
    "M3_todas_ls01_s42_f0": ("M3", _cfg("M3", suavizacao=0.1), "label smoothing 0,1"),
    "M3_todas_lrb3e-5_s42_f0": ("M3", _cfg("M3", lr_backbone=3e-5), "lr do corpo 3e-5 (E1: 1e-4)"),
    "M3_todas_lrb3e-4_s42_f0": ("M3", _cfg("M3", lr_backbone=3e-4), "lr do corpo 3e-4 (E1: 1e-4)"),
    "M5_todas_s42_f0": ("M5", ConfigTreino(modelo="M5", epocas=30, paciencia=6, lote=64, lr=1e-3, lr_backbone=3e-4),
                        "EfficientNet-B0 do ImageNet no lugar do M1 (mesma entrada 128 × 313)"),
    "M4FT_sem_s42_f0": ("M4FT", ConfigTreino(modelo="M4FT", epocas=6, paciencia=3, lote=8, lr=1e-3, lr_backbone=3e-5),
                        "AST com os 2 últimos blocos + LayerNorm treináveis (M4: só a cabeça linear)"),
    # E5.9: M6, a CNN do projeto, em ablação (uma peça por vez; ver birdclef.m6). Mesmos
    # hiperparâmetros e aumento do M1; lr_backbone só ativa os grupos de parâmetros (PCEN sem
    # weight decay), com a mesma taxa 1e-3.
    "M6a_todas_s42_f0": ("M6a", replace(e.CFG_TREINO["M1"], modelo="M6a", lr_backbone=1e-3),
                         "PCEN treinável por banda (M1PCEN: PCEN fixa)"),
    "M6b_todas_s42_f0": ("M6b", replace(e.CFG_TREINO["M1"], modelo="M6b", lr_backbone=1e-3),
                         "M6a + blocos residuais com Squeeze-and-Excitation"),
    "M6c_todas_s42_f0": ("M6c", replace(e.CFG_TREINO["M1"], modelo="M6c", lr_backbone=1e-3),
                         "M6b + canal com a posição da banda (CoordConv)"),
}
# Divisão por gravador (E5.3): mesmo modelo e configuração do E1, só a divisão muda.
VARIANTES_AUTOR = {"M1_todas_autor_s42_f0": "M1", "M3_todas_autor_s42_f0": "M3"}


def rodar_treinos(nomes=None) -> None:
    """Treina (ou pula, se já existir) as variantes do E5, na ordem dada."""
    nomes = nomes or list(VARIANTES) + list(VARIANTES_AUTOR)
    janelas_autor = None
    for nome in nomes:
        try:
            janelas_autor = _rodar_um(nome, janelas_autor)
        except Exception as erro:  # uma variante com problema não impede as outras
            import traceback
            traceback.print_exc()
            print(f"{nome}: FALHOU ({type(erro).__name__}: {erro})", flush=True)
            torch.cuda.empty_cache()


def _rodar_um(nome: str, janelas_autor):
    ini = time.time()
    if nome in VARIANTES:
        modelo, cfg, _ = VARIANTES[nome]
        alvos = alvos_professor() if cfg.destilacao else None
        res, _ = treinar_variante(nome, modelo, cfg, alvos_extra=alvos)
    else:
        modelo = VARIANTES_AUTOR[nome]
        if janelas_autor is None or modelos_entrada(modelo) != janelas_autor[0]:
            janelas_autor = (modelos_entrada(modelo), janelas_por_autor_para(modelo))
        res, _ = treinar_variante(nome, modelo, e.CFG_TREINO[modelo], janelas_=janelas_autor[1])
    print(f"{nome}: F1 val {res.metricas_val['f1_macro']:.3f} ({(time.time() - ini) / 60:.1f} min)", flush=True)
    return janelas_autor


def modelos_entrada(modelo: str) -> str:
    from birdclef import modelos
    return modelos.ENTRADA[modelo]


def janelas_por_autor_para(modelo: str) -> pd.DataFrame:
    """A divisão por gravador aplicada às janelas da entrada do modelo (mesmas janelas, mesma ordem)."""
    j = janelas_por_autor()
    _, jm, _ = e.dados_modelo(modelo)
    assert (jm["filename"].to_numpy() == j["filename"].to_numpy()).all()
    return j


if __name__ == "__main__":
    import sys
    rodar_treinos(sys.argv[1:] or None)
