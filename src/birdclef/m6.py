"""M6: CNN do projeto, desenhada a partir dos resultados do E1 ao E5.

Cada peça é uma estrutura conhecida, escolhida por causa de algo que medimos:

| peça | estrutura conhecida | motivo (resultado nosso) |
|---|---|---|
| 1. PCEN treinável | PCEN aprendível por banda (Wang et al., 2017; LEAF, Zeghidour et al., 2021) | a PCEN fixa foi a melhor técnica do E5 (+2,3 pontos no teste, +2,6 a 0 dB) |
| 2. blocos residuais com SE | ResNet (He et al., 2016) + Squeeze-and-Excitation (Hu et al., 2018) | o M1 tem só 4 blocos simples; atalhos e atenção entre canais quase não custam parâmetros |
| 3. posição na frequência | canal de coordenada, como no CoordConv (Liu et al., 2018) | cada espécie canta numa faixa própria (Grad-CAM, confusão Saltator × Wood-Wren), e uma CNN comum é invariante à posição |
| 4. atenção nas janelas | *multiple-instance learning* com atenção com portas (Ilse et al., 2018) | o rótulo é da gravação (rótulo fraco); regras fixas de agregação (máximo, top-3) não melhoraram o teste no E5 |

Ablação (uma peça por vez, fold 0, mesmo aumento e hiperparâmetros do M1):

- `M6a`: M1 + PCEN treinável (peça 1). Comparação direta com o M1PCEN do E5 (PCEN fixa).
- `M6b`: M6a com blocos residuais e SE (peças 1 e 2). Cada bloco residual tem 2 convoluções, então
  as larguras caem pela metade (16-32-64-128) para manter o porte do M1 (~0,3 contra 0,39 milhão de
  parâmetros e custo por lote parecido), o mesmo critério usado no M2: a comparação mede a
  arquitetura, não a capacidade. Com as larguras do M1, o bloco residual custaria 3,3 vezes mais.
- `M6c`: M6b com o canal de frequência (peças 1, 2 e 3).
- `M6`: M6c + atenção nas janelas (peça 4). Treinada numa segunda etapa: o corpo do M6c fica
  congelado e só a atenção é aprendida, reaproveitando a camada de saída do M6c. Assim, a única
  coisa que muda em relação ao M6c é *como* as janelas viram a predição da gravação: média das
  probabilidades (M6c) ou média dos embeddings ponderada por uma atenção aprendida (M6).

As três primeiras variantes são treinadas pelo mesmo `treino.treinar` de todos os modelos (janela a
janela, com o rótulo da gravação). A peça 4 precisa da gravação inteira e é treinada aqui
(`treinar_atencao`).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

QUADROS_POR_S = 32000 / 512  # log-Mel do M1: hop de 512 amostras a 32 kHz (16 ms por quadro)


# ----------------------------------------------------------------------------- peça 1

def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


class PCENTreinavel(nn.Module):
    """PCEN com os 4 parâmetros aprendidos **por banda Mel** (128 × 4 = 512 parâmetros).

    PCEN(t, f) = (E / (eps + M)^alfa + delta)^r - delta^r, com M[t] = (1 - s)·M[t-1] + s·E[t].

    - s (suavização no tempo) controla a constante de tempo: cada banda aprende quanto do passado
      recente conta como "fundo" a ser removido;
    - alfa controla quanto da energia de fundo é dividida (ganho automático);
    - delta e r controlam a compressão (como o log, mas aprendida).

    Inicializada nos valores da PCEN fixa do E5 (constante de tempo 0,4 s, alfa 0,98, delta 2,
    r 0,5), então no início é exatamente a PCEN que já funcionou. Restrições por reparametrização:
    s, alfa e r em (0, 1) via sigmoide; delta > 0 via exponencial. Mesmo estado inicial do filtro
    da PCEN fixa (M[-1] = E[0], solução exata sem cancelamento). Contas em float32 e no domínio
    logarítmico onde possível, fora do autocast.
    """

    def __init__(self, n_bandas: int = 128, sr: int = 32000, hop: int = 512, tempo: float = 0.4,
                 alfa: float = 0.98, delta: float = 2.0, r: float = 0.5, eps: float = 1e-6):
        super().__init__()
        t_q = tempo * sr / hop
        s = float((np.sqrt(1 + 4 * t_q ** 2) - 1) / (2 * t_q ** 2))
        um = torch.ones(n_bandas)
        self.p_s = nn.Parameter(um * _logit(s))
        self.p_alfa = nn.Parameter(um * _logit(alfa))
        self.p_delta = nn.Parameter(um * float(np.log(delta)))
        self.p_r = nn.Parameter(um * _logit(r))
        self.eps = eps

    def valores(self) -> dict:
        """Parâmetros atuais por banda (para inspecionar o que foi aprendido)."""
        return {"s": torch.sigmoid(self.p_s), "alfa": torch.sigmoid(self.p_alfa),
                "delta": torch.exp(self.p_delta), "r": torch.sigmoid(self.p_r)}

    def forward(self, x_db: torch.Tensor) -> torch.Tensor:
        import torchaudio.functional as AF
        with torch.autocast(x_db.device.type, enabled=False):
            v = self.valores()
            log_e = x_db.float() * (np.log(10.0) / 10) + 31 * np.log(2.0)   # potência × 2^31, em log
            e = torch.exp(log_e)                                               # (B, M, T)
            s = v["s"]
            a = torch.stack([torch.ones_like(s), -(1 - s)], 1)                 # (M, 2), um filtro por banda
            b = torch.stack([s, torch.zeros_like(s)], 1)
            t = torch.arange(1, e.shape[-1] + 1, device=e.device, dtype=e.dtype)
            m = AF.lfilter(e, a, b, clamp=False, batching=True) + (1 - s)[:, None] ** t * e[..., :1]
            g = torch.exp(log_e - v["alfa"][:, None] * torch.log(self.eps + m.clamp_min(0)))
            d, r = v["delta"][:, None], v["r"][:, None]
            return (g + d) ** r - d ** r


# ----------------------------------------------------------------------------- peça 2

class BlocoResSE(nn.Module):
    """Bloco residual (2 × conv 3x3 com BN) com Squeeze-and-Excitation, depois MaxPool 2x2."""

    def __init__(self, entrada: int, saida: int, reducao: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(saida)
        self.conv2 = nn.Conv2d(saida, saida, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(saida)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(saida, saida // reducao, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(saida // reducao, saida, 1), nn.Sigmoid())
        self.atalho = (nn.Sequential(nn.Conv2d(entrada, saida, 1, bias=False), nn.BatchNorm2d(saida))
                       if entrada != saida else nn.Identity())
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = h * self.se(h)
        return self.pool(F.relu(h + self.atalho(x)))


# ----------------------------------------------------------------------------- peça 4

class AtencaoJanelas(nn.Module):
    """Atenção com portas sobre as janelas de uma gravação (Ilse et al., 2018).

    peso_i ∝ exp(w · (tanh(V h_i) ⊙ sigmoide(U h_i))), normalizado entre as janelas da gravação;
    embedding da gravação = Σ peso_i · h_i. `w` começa em zero: no início todos os pesos são iguais
    e o embedding da gravação é a média dos embeddings das janelas.
    """

    def __init__(self, dim: int, oculta: int = 64):
        super().__init__()
        self.V = nn.Linear(dim, oculta)
        self.U = nn.Linear(dim, oculta)
        self.w = nn.Linear(oculta, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def pesos(self, h: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        """h: (G, J, D) embeddings das janelas; mascara: (G, J) True nas janelas válidas."""
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        return torch.softmax(a.masked_fill(~mascara, float("-inf")), dim=1)

    def forward(self, h: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        return (self.pesos(h, mascara)[..., None] * h).sum(1)


# ----------------------------------------------------------------------------- modelo

class M6(nn.Module):
    """CNN do projeto. As flags ligam as peças (ver a docstring do módulo)."""

    def __init__(self, n_classes: int, residual: bool = True, frequencia: bool = True, atencao: bool = False,
                 canais=(32, 64, 128, 256), dropout: float = 0.3, n_bandas: int = 128):
        super().__init__()
        from birdclef.modelos import BlocoConv
        self.register_buffer("media", torch.tensor(0.0))
        self.register_buffer("desvio", torch.tensor(1.0))
        self.pcen = PCENTreinavel(n_bandas)
        self.bn = nn.BatchNorm2d(1)
        self.frequencia = frequencia
        if frequencia:  # canal constante com a posição da banda, de -1 (grave) a +1 (agudo)
            self.register_buffer("coord", torch.linspace(-1, 1, n_bandas)[:, None])
        blocos, entrada = [], 2 if frequencia else 1
        for c in canais:
            blocos.append(BlocoResSE(entrada, c) if residual else BlocoConv(entrada, c))
            entrada = c
        self.extrator = nn.Sequential(*blocos)
        self.dropout = nn.Dropout(dropout)
        self.saida = nn.Linear(entrada, n_classes)
        self.dim = entrada
        self.atencao = AtencaoJanelas(entrada) if atencao else None

    def definir_norm(self, media: float, desvio: float) -> None:
        self.media.fill_(media)
        self.desvio.fill_(desvio)

    def grupos_parametros(self, lr_backbone: float, lr_cabeca: float) -> list[dict]:
        # os parâmetros da PCEN ficam sem weight decay: o decaimento puxaria os parâmetros
        # reparametrizados para zero (s = 0,5, alfa = 0,5...), longe da PCEN que funcionou
        pcen = list(self.pcen.parameters())
        ids = {id(p) for p in pcen}
        return [{"params": pcen, "lr": lr_cabeca, "weight_decay": 0.0},
                {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr_cabeca}]

    def _frente(self, x: torch.Tensor) -> torch.Tensor:
        p = self.bn(self.pcen(x * self.desvio + self.media).unsqueeze(1))   # (B, 1, M, T)
        if self.frequencia:
            p = torch.cat([p, self.coord.expand(p.shape[0], 1, -1, p.shape[-1]).to(p.dtype)], 1)
        return p

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        h = self.extrator(self._frente(x)).mean(dim=2)    # média na frequência -> (B, C, T')
        return h.mean(dim=2) + h.amax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.saida(self.dropout(self.embedding(x)))

    def logits_gravacao(self, h: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        """Logits por gravação a partir dos embeddings das janelas (G, J, D) (peça 4)."""
        return self.saida(self.atencao(h, mascara))


LARGURAS_RES = (16, 32, 64, 128)  # mesmo porte do M1 (ver M6b na docstring do módulo)
VARIANTES_M6 = {"M6a": dict(residual=False, frequencia=False),
                "M6b": dict(residual=True, frequencia=False, canais=LARGURAS_RES),
                "M6c": dict(residual=True, frequencia=True, canais=LARGURAS_RES),
                "M6": dict(residual=True, frequencia=True, atencao=True, canais=LARGURAS_RES),
                # atenção sobre o M6b: o canal de frequência piorou a validação (M6c < M6b) e foi
                # descartado, então a peça 4 também é testada sobre a melhor base, o M6b
                "M6bA": dict(residual=True, frequencia=False, atencao=True, canais=LARGURAS_RES)}
# variantes com a atenção nas janelas (a gravação sai de `logits_gravacao`, não da média)
COM_ATENCAO = ("M6", "M6bA")
# base (sem atenção) de cada variante com atenção
BASE_ATENCAO = {"M6": "M6c", "M6bA": "M6b"}


def criar(nome: str, n_classes: int) -> M6:
    return M6(n_classes, **VARIANTES_M6[nome])


# ----------------------------------------------------------------------------- peça 4: treino e predição

def embeddings_janelas(modelo: M6, X, idx: np.ndarray, norm, lote: int = 256) -> np.ndarray:
    disp = next(modelo.parameters()).device
    modelo.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), lote):
            x = torch.from_numpy(np.asarray(X[np.sort(idx[i:i + lote])], dtype=np.float32)).to(disp)
            ordem = np.argsort(np.argsort(idx[i:i + lote]))
            with torch.autocast(disp.type, enabled=disp.type == "cuda"):
                h = modelo.embedding((x - norm[0]) / norm[1])
            out.append(h.float().cpu().numpy()[ordem])
    return np.concatenate(out)


def agrupar(H: np.ndarray, arquivos: np.ndarray, max_janelas: int = 12):
    """(n_janelas, D) -> (G, J, D) com máscara, na ordem de aparição das gravações."""
    ordem = pd.unique(arquivos)
    pos = pd.Series(np.arange(len(arquivos))).groupby(arquivos, sort=False).apply(list).loc[ordem]
    G = np.zeros((len(ordem), max_janelas, H.shape[1]), dtype=np.float32)
    M = np.zeros((len(ordem), max_janelas), dtype=bool)
    for g, ix in enumerate(pos):
        ix = ix[:max_janelas]
        G[g, :len(ix)] = H[ix]
        M[g, :len(ix)] = True
    return G, M, ordem


def _rotulos(janelas, idx, ordem, classes):
    rot = janelas.iloc[idx].groupby("filename", sort=False)["primary_label"].first().loc[ordem]
    return rot.map({c: i for i, c in enumerate(classes)}).to_numpy()


def probs_gravacoes(modelo: M6, X, janelas, idx, norm, classes):
    """Probabilidades por gravação do M6 completo (embeddings -> atenção -> saída)."""
    H = embeddings_janelas(modelo, X, idx, norm)
    G, M, ordem = agrupar(H, janelas["filename"].to_numpy()[idx])
    disp = next(modelo.parameters()).device
    with torch.no_grad():
        lg = modelo.logits_gravacao(torch.from_numpy(G).to(disp), torch.from_numpy(M).to(disp))
    return torch.softmax(lg.float(), 1).cpu().numpy(), _rotulos(janelas, idx, ordem, classes)


def treinar_atencao(base: str = "M6c_todas_s42_f0", nome: str = "M6_todas_s42_f0", modelo_final: str = "M6", epocas: int = 300,
                    paciencia: int = 30, lote: int = 64, lr: float = 1e-3, descarte: float = 0.3,
                    semente: int = 42) -> dict:
    """Etapa 2 do M6: congela o corpo e a saída do `base` (M6c) e aprende só a atenção nas janelas
    das gravações de treino do fold 0. Escolha da época pela validação do fold 0. Durante o treino,
    cada janela é descartada com probabilidade `descarte` (mantendo ao menos uma), para a atenção
    não depender de uma janela específica. Salva `checkpoints/<nome>.pt` no formato dos outros."""
    from birdclef import treino
    from birdclef import experimentos_p2 as e
    from birdclef.experimentos import PASTA_CHECKPOINTS, carregar_checkpoint

    ck_path = PASTA_CHECKPOINTS / f"{nome}.pt"
    if ck_path.exists():
        return torch.load(ck_path, map_location="cpu", weights_only=False)
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    modelo_base = BASE_ATENCAO[modelo_final]
    X, janelas, _ = e.dados_modelo(modelo_base)
    classes = sorted(janelas["primary_label"].unique())
    rb = carregar_checkpoint(PASTA_CHECKPOINTS / f"{base}.pt", modelo_base)
    disp = treino.dispositivo()
    modelo = criar(modelo_final, len(classes)).to(disp)
    faltando = modelo.load_state_dict(rb.modelo.state_dict(), strict=False).missing_keys
    assert all(k.startswith("atencao.") for k in faltando), faltando
    idx_tr, idx_val = treino.indices(janelas, 0)
    arq = janelas["filename"].to_numpy()
    ini = time.time()
    Gtr, Mtr, otr = agrupar(embeddings_janelas(modelo, X, idx_tr, rb.norm), arq[idx_tr])
    Gva, Mva, ova = agrupar(embeddings_janelas(modelo, X, idx_val, rb.norm), arq[idx_val])
    ytr, yva = _rotulos(janelas, idx_tr, otr, classes), _rotulos(janelas, idx_val, ova, classes)
    Gtr_t, Mtr_t, ytr_t = (torch.from_numpy(np.array(a)).to(disp) for a in (Gtr, Mtr, ytr))
    Gva_t, Mva_t = torch.from_numpy(Gva).to(disp), torch.from_numpy(Mva).to(disp)

    for p in modelo.parameters():
        p.requires_grad_(False)
    for p in modelo.atencao.parameters():
        p.requires_grad_(True)
    otim = torch.optim.AdamW(modelo.atencao.parameters(), lr=lr, weight_decay=1e-4)
    modelo.eval()  # dropout desligado; só a atenção treina

    def avaliar():
        with torch.no_grad():
            p = torch.softmax(modelo.logits_gravacao(Gva_t, Mva_t), 1).cpu().numpy()
        return treino.metricas(yva, p)

    m0 = avaliar()
    melhor, melhor_estado, sem, hist = m0["f1_macro"], {k: v.clone() for k, v in modelo.atencao.state_dict().items()}, 0, []
    hist.append({"epoca": 0, "perda": float("nan"), **m0})
    for ep in range(1, epocas + 1):
        perm = rng.permutation(len(Gtr))
        tot = 0.0
        for i in range(0, len(perm), lote):
            b = torch.from_numpy(perm[i:i + lote]).to(disp)
            masc = Mtr_t[b] & (torch.rand(Mtr_t[b].shape, device=disp) > descarte)
            vazias = ~masc.any(1)
            masc[vazias] = Mtr_t[b][vazias]  # gravação sem nenhuma janela sorteada: mantém todas
            perda = F.cross_entropy(modelo.logits_gravacao(Gtr_t[b], masc), ytr_t[b])
            otim.zero_grad()
            perda.backward()
            otim.step()
            tot += perda.item() * len(b)
        m = avaliar()
        hist.append({"epoca": ep, "perda": tot / len(perm), **m})
        if m["f1_macro"] > melhor:
            melhor, sem = m["f1_macro"], 0
            melhor_estado = {k: v.clone() for k, v in modelo.atencao.state_dict().items()}
        else:
            sem += 1
            if sem >= paciencia:
                break
    modelo.atencao.load_state_dict(melhor_estado)
    h = pd.DataFrame(hist)
    m_best = h.loc[h["f1_macro"].idxmax()].to_dict()
    ck = {"estado": modelo.state_dict(), "classes": classes, "norm": rb.norm,
          "historico": h.to_dict("list"), "metricas_val": m_best, "segundos": time.time() - ini,
          "treino": {"modelo": modelo_final, "base": base, "epocas": epocas, "paciencia": paciencia, "lote": lote,
                     "lr": lr, "descarte_janelas": descarte, "o_que_treina": f"só a atenção (corpo e saída do {modelo_base} congelados)"},
          "val_f1_inicial_media_embeddings": m0["f1_macro"]}
    torch.save(ck, ck_path)
    return ck


def carregar(nome: str = "M6_todas_s42_f0"):
    """Modelo com atenção (M6 ou M6bA) e o checkpoint; a arquitetura vem do checkpoint."""
    from birdclef import treino
    from birdclef.experimentos import PASTA_CHECKPOINTS
    ck = torch.load(PASTA_CHECKPOINTS / f"{nome}.pt", map_location=treino.dispositivo(), weights_only=False)
    m = criar(ck["treino"]["modelo"], len(ck["classes"])).to(treino.dispositivo())
    m.load_state_dict(ck["estado"])
    return m.eval(), ck


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["M6bA"]:
        ck = treinar_atencao(base="M6b_todas_s42_f0", nome="M6bA_todas_s42_f0", modelo_final="M6bA")
    else:
        ck = treinar_atencao()
    print(f"M6 (atenção): F1 val {ck['metricas_val']['f1_macro']:.3f} "
          f"(média dos embeddings, antes de treinar: {ck['val_f1_inicial_media_embeddings']:.3f})")
