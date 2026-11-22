"""Treino e avaliação, comuns a todos os modelos.

- Amostragem: cada época sorteia janelas com peso 1/(janelas da gravação), para que toda gravação
  pese igual, por mais longa que seja.
- Perda: entropia cruzada com rótulos suaves (necessária para o mixup); AdamW; cosseno na taxa de
  aprendizado; precisão mista na GPU.
- Parada antecipada pelo F1-macro de validação, medido por gravação.
- Predição de uma gravação: média das probabilidades das suas janelas.

Parte 2 (sem mudar o comportamento do M1):

- `ConfigTreino.lr_backbone`: taxa de aprendizado menor para o corpo pré-treinado (M3);
- `indices(..., fracao=)`: usa só uma fração das gravações de treino, estratificada por espécie e
  aninhada (as 25% estão dentro das 50%), para o E4 (eficiência de dados).

Melhorias testadas no E5 (desligadas por padrão, então E1-E4 não mudam):

- `ConfigTreino.suavizacao`: *label smoothing* (rótulo-alvo vira (1 - s) no correto e s/C no resto);
- `ConfigTreino.epocas_cabeca`: ajuste fino em duas fases; nas primeiras épocas só a camada nova
  (`saida`) treina, com o corpo pré-treinado congelado, e depois o modelo todo;
- `ConfigTreino.destilacao` + `treinar(..., alvos_extra=)`: destilação de conhecimento (o alvo
  mistura o rótulo com as probabilidades de um modelo professor em cada janela).
"""

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score

from birdclef import aumento, modelos
from birdclef.aumento import ConfigAumento


@dataclass
class ConfigTreino:
    modelo: str = "M1"
    epocas: int = 30
    paciencia: int = 6
    lote: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    semente: int = 42
    lr_backbone: float | None = None  # só modelos com `grupos_parametros` (M3)
    fracao_treino: float = 1.0        # E4: fração das gravações de treino usadas
    suavizacao: float = 0.0           # E5: label smoothing
    epocas_cabeca: int = 0            # E5: épocas iniciais com o corpo congelado (só a `saida` treina)
    destilacao: float = 0.0           # E5: peso das probabilidades do professor no alvo (precisa de `alvos_extra`)


@dataclass
class Resultado:
    modelo: torch.nn.Module
    classes: list[str]
    norm: tuple[float, float]
    historico: pd.DataFrame
    metricas_val: dict = field(default_factory=dict)
    segundos: float = 0.0


def dispositivo() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixar_semente(semente: int) -> None:
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


# ----------------------------------------------------------------------------- métricas

def metricas(y: np.ndarray, probs: np.ndarray) -> dict:
    """Métricas por gravação. `y`: índices das classes; `probs`: (n, classes)."""
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    onehot = np.eye(probs.shape[1])[y]
    return {
        "f1_macro": f1_score(y, pred, average="macro"),
        "acc_top1": float((pred == y).mean()),
        "acc_top3": float((top3 == y[:, None]).any(1).mean()),
        "map": average_precision_score(onehot, probs, average="macro"),
    }


def matriz_confusao(y: np.ndarray, probs: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y, probs.argmax(1), labels=range(n_classes))


# ----------------------------------------------------------------------------- dados na GPU

GB = 2 ** 30


class XNaGPU:
    """Linhas de `X` (log-Mel em float16) copiadas para a VRAM, indexadas pelo índice global.

    Otimização da Parte 2 (sem efeito nos números: os mesmos valores float16 viram float32, só que
    na GPU). Na máquina de 7,7 GB de RAM, manter `X` inteiro em memória (1,9 GB) junto com o
    contexto CUDA levava o Windows a comprimir páginas, e o sorteio de lotes na CPU deixava a GPU
    ociosa: ~65 s por época do M1 contra ~20 s na Parte 1. Com as linhas de treino e validação na
    VRAM, cada lote é indexado direto na GPU. A cópia é feita em blocos a partir do `X` mapeado do
    disco, sem duplicar `X` na RAM.
    """

    def __init__(self, X, linhas: np.ndarray, disp: torch.device, bloco: int = 1024):
        linhas = np.sort(np.asarray(linhas))
        self.pos = np.full(len(X), -1, dtype=np.int64)
        self.pos[linhas] = np.arange(len(linhas))
        self.dados = torch.empty((len(linhas), *X.shape[1:]), dtype=torch.float16, device=disp)
        for i in range(0, len(linhas), bloco):
            self.dados[i:i + bloco] = torch.from_numpy(np.asarray(X[linhas[i:i + bloco]], dtype=np.float16)).to(disp)
        self.shape = X.shape
        self.disp = disp

    def lote(self, idx: np.ndarray) -> torch.Tensor:
        p = self.pos[idx]
        assert (p >= 0).all(), "índice fora das linhas copiadas para a GPU"
        return self.dados[torch.from_numpy(p).to(self.disp)].float()


def lote_tensor(X, idx: np.ndarray, disp: torch.device) -> torch.Tensor:
    """Lote de `X` como float32 em `disp`, venha `X` de um `XNaGPU` ou de um array/memmap."""
    if isinstance(X, XNaGPU):
        return X.lote(idx)
    return torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(disp, non_blocking=True)


def cabe_na_gpu(X, linhas: np.ndarray, modelo, disp: torch.device) -> bool:
    """Se as `linhas` de `X` cabem na VRAM livre, deixando a folga que o treino do modelo pede
    (`modelo.vram_treino_gb`, padrão 1,3 GB: M1/M2 com lote 64)."""
    if disp.type != "cuda":
        return False
    livre, _ = torch.cuda.mem_get_info()
    precisa = len(linhas) * int(np.prod(X.shape[1:])) * 2
    return precisa + getattr(modelo, "vram_treino_gb", 1.3) * GB < livre


# ----------------------------------------------------------------------------- predição

@torch.no_grad()
def prever_janelas(modelo, X, idx: np.ndarray, norm, lote: int = 256) -> np.ndarray:
    # Correção (Parte 2, GPU de 4 GB): o lote de inferência era fixo em 256. Para o CNN14 (M3)
    # isso pede ~4,2 GiB de VRAM só na passagem direta, mais que a GPU tem: no Windows o driver
    # transborda para a memória compartilhada (fica dezenas de vezes mais lento) ou dá OOM.
    # Modelos grandes declaram `lote_inferencia` (M3: 32, pico ~1,5 GiB); os demais seguem com 256.
    lote = min(lote, getattr(modelo, "lote_inferencia", lote))
    modelo.eval()
    disp = next(modelo.parameters()).device
    media, desvio = norm
    saidas = []
    for i in range(0, len(idx), lote):
        x = lote_tensor(X, idx[i:i + lote], disp)
        with torch.autocast(disp.type, enabled=disp.type == "cuda"):
            saidas.append(torch.softmax(modelo((x - media) / desvio).float(), 1).cpu().numpy())
    return np.concatenate(saidas) if saidas else np.empty((0, 0))


def prever_gravacoes(modelo, X, janelas: pd.DataFrame, idx: np.ndarray, norm, classes) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Probabilidades por gravação (média das janelas), rótulos e nomes dos arquivos."""
    probs = prever_janelas(modelo, X, idx, norm)
    df = pd.DataFrame(probs, index=janelas["filename"].to_numpy()[idx])
    por_grav = df.groupby(level=0, sort=False).mean()
    rotulo = janelas.iloc[idx].groupby("filename", sort=False)["primary_label"].first().loc[por_grav.index]
    y = rotulo.map({c: i for i, c in enumerate(classes)}).to_numpy()
    return por_grav.to_numpy(), y, por_grav.index


# ----------------------------------------------------------------------------- treino

def gravacoes_fracao(janelas: pd.DataFrame, gravacoes: np.ndarray, fracao: float, semente: int = 42) -> set:
    """Fração `fracao` das `gravacoes`, por espécie, em ordem aleatória fixa.

    Como a ordem não depende da fração, os subconjuntos são aninhados (25% ⊂ 50% ⊂ 100%), e são os
    mesmos para todos os modelos: a comparação do E4 muda só a quantidade de dados.
    """
    rotulo = janelas.drop_duplicates("filename").set_index("filename")["primary_label"]
    df = pd.DataFrame({"filename": gravacoes, "especie": rotulo.loc[gravacoes].to_numpy()})
    df["ordem"] = np.random.default_rng(semente).permutation(len(df))
    df = df.sort_values(["especie", "ordem"])
    escolhidas = df.groupby("especie", group_keys=False).apply(
        lambda g: g.head(max(1, int(round(fracao * len(g))))))
    return set(escolhidas["filename"])


def indices(janelas: pd.DataFrame, fold_val: int, fracao: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    treino = (janelas["particao"] == "treino") & (janelas["fold"] != fold_val)
    val = (janelas["particao"] == "treino") & (janelas["fold"] == fold_val)
    if fracao < 1.0:
        manter = gravacoes_fracao(janelas, janelas.loc[treino, "filename"].unique(), fracao)
        treino &= janelas["filename"].isin(manter)
    return np.flatnonzero(treino.to_numpy()), np.flatnonzero(val.to_numpy())


def estatisticas(X, idx: np.ndarray, n: int = 3000, semente: int = 0) -> tuple[float, float]:
    """Média e desvio globais (em dB) de uma amostra das janelas de treino."""
    amostra = np.random.default_rng(semente).choice(idx, size=min(n, len(idx)), replace=False)
    x = np.asarray(X[np.sort(amostra)], dtype=np.float32)
    return float(x.mean()), float(x.std())


def treinar(
    X: np.ndarray,
    janelas: pd.DataFrame,
    fold_val: int,
    cfg_aum: ConfigAumento = aumento.SEM,
    cfg: ConfigTreino = ConfigTreino(),
    R: np.ndarray | None = None,
    verbose: bool = True,
    alvos_extra: np.ndarray | None = None,
) -> Resultado:
    """Treina um modelo com o fold `fold_val` como validação e os demais folds de treino.

    `X` pode ser um array ou um memmap; se as linhas de treino e validação couberem na VRAM, são
    copiadas para lá (`XNaGPU`) e os lotes saem direto da GPU. `R` são os log-Mel dos ruídos de
    treino, obrigatórios se `cfg_aum.ruido`.
    """
    fixar_semente(cfg.semente)
    disp = dispositivo()
    classes = sorted(janelas["primary_label"].unique())
    alvo = janelas["primary_label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    idx_tr, idx_val = indices(janelas, fold_val, cfg.fracao_treino)
    norm = estatisticas(X, idx_tr)
    media, desvio = norm

    janelas_por_grav = janelas.iloc[idx_tr].groupby("filename")["filename"].transform("size").to_numpy()
    pesos = torch.tensor(1.0 / janelas_por_grav, dtype=torch.double)
    R_t = torch.from_numpy(np.asarray(R, dtype=np.float32)).to(disp) if cfg_aum.ruido else None

    torch.backends.cudnn.benchmark = True
    modelo = modelos.criar(cfg.modelo, len(classes)).to(disp)
    if hasattr(modelo, "definir_norm"):  # M1PCEN (E5): a camada PCEN desfaz a normalização por dentro
        modelo.definir_norm(media, desvio)
    if getattr(modelo, "entrada_normalizada", False):
        # entrada já normalizada na origem (fbank do extrator do AST, E5.6): não renormalizar,
        # senão o modelo pré-treinado recebe uma distribuição diferente da do pré-treino
        norm = (0.0, 1.0)
        media, desvio = norm
    if cfg.lr_backbone is not None and hasattr(modelo, "grupos_parametros"):
        params = modelo.grupos_parametros(cfg.lr_backbone, cfg.lr)
    else:
        params = modelo.parameters()
    otim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    agenda = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=cfg.epocas)
    escala = torch.amp.GradScaler(enabled=disp.type == "cuda")

    linhas = np.concatenate([idx_tr, idx_val])
    Xt = XNaGPU(X, linhas, disp) if cabe_na_gpu(X, linhas, modelo, disp) else X
    if verbose:
        print(f"dados de treino/validação {'na GPU' if isinstance(Xt, XNaGPU) else 'na CPU (memmap)'}", flush=True)

    melhor, melhor_estado, sem_melhora, hist = -1.0, None, 0, []
    inicio = time.time()
    def congelar_corpo(congelado: bool) -> None:
        cabeca = {id(p) for p in modelo.saida.parameters()}
        for p in modelo.parameters():
            if id(p) not in cabeca:
                p.requires_grad_(not congelado)

    for epoca in range(1, cfg.epocas + 1):
        if cfg.epocas_cabeca:
            congelar_corpo(epoca <= cfg.epocas_cabeca)
        modelo.train()
        sorteio = idx_tr[torch.multinomial(pesos, len(idx_tr), replacement=True).numpy()]
        perda_total = 0.0
        for i in range(0, len(sorteio), cfg.lote):
            lote = sorteio[i:i + cfg.lote]
            x = lote_tensor(Xt, lote, disp)
            y = F.one_hot(torch.from_numpy(alvo[lote]), len(classes)).float().to(disp)
            if cfg.destilacao:
                # destilação (E5): alvo = (1 - λ)·rótulo + λ·probabilidades do professor na mesma janela
                q = torch.from_numpy(alvos_extra[lote]).float().to(disp)
                y = (1 - cfg.destilacao) * y + cfg.destilacao * q
            if cfg.suavizacao:
                y = y * (1 - cfg.suavizacao) + cfg.suavizacao / len(classes)

            if cfg_aum.ruido:
                com_ruido = torch.rand(len(lote), device=disp) < cfg_aum.prob_ruido
                if com_ruido.any():
                    r = R_t[torch.randint(len(R_t), (int(com_ruido.sum()),), device=disp)]
                    snr = torch.empty(len(r), device=disp).uniform_(cfg_aum.snr_min, cfg_aum.snr_max)
                    x[com_ruido] = aumento.misturar_ruido_mel(x[com_ruido], r, snr)
            x = (x - media) / desvio
            if cfg_aum.specaugment:
                x = aumento.spec_augment(x, cfg_aum)
            if cfg_aum.mixup:
                x, y = aumento.mixup(x, y, cfg_aum.alfa_mixup)

            with torch.autocast(disp.type, enabled=disp.type == "cuda"):
                logits = modelo(x)
            perda = -(y * F.log_softmax(logits.float(), 1)).sum(1).mean()
            otim.zero_grad(set_to_none=True)
            escala.scale(perda).backward()
            escala.step(otim)
            escala.update()
            perda_total += perda.item() * len(lote)
        agenda.step()

        probs, y_val, _ = prever_gravacoes(modelo, Xt, janelas, idx_val, norm, classes)
        m = metricas(y_val, probs)
        hist.append({"epoca": epoca, "perda": perda_total / len(sorteio), **m})
        if verbose:
            print(f"época {epoca:2d}  perda {hist[-1]['perda']:.3f}  F1 val {m['f1_macro']:.3f}  top1 {m['acc_top1']:.3f}  ({time.time() - inicio:.0f} s)", flush=True)
        if m["f1_macro"] > melhor:
            melhor, sem_melhora = m["f1_macro"], 0
            melhor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= cfg.paciencia:
                break

    modelo.load_state_dict(melhor_estado)
    del Xt
    torch.cuda.empty_cache()
    historico = pd.DataFrame(hist)
    metricas_val = historico.loc[historico["f1_macro"].idxmax()].to_dict()
    return Resultado(modelo, classes, norm, historico, metricas_val, time.time() - inicio)


def salvar(res: Resultado, caminho: Path, **extra) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"estado": res.modelo.state_dict(), "classes": res.classes, "norm": res.norm, **extra}, caminho)


def avaliar_teste(res: Resultado, X_teste, janelas_teste: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    """Métricas no teste. `X_teste` e `janelas_teste` contêm só as janelas de teste."""
    idx = np.arange(len(janelas_teste))
    probs, y, _ = prever_gravacoes(res.modelo, X_teste, janelas_teste, idx, res.norm, res.classes)
    return metricas(y, probs), y, probs
