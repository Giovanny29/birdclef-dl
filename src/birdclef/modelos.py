"""Modelos do projeto. Todos recebem um lote (B, ...) e devolvem logits (B, classes).

| ID | Classe        | Entrada                         | Inicialização |
|----|---------------|---------------------------------|---------------|
| M1 | `CNN`         | log-Mel 128 × 313 (mel128_32k)  | aleatória     |
| M2 | `CRNN`        | log-Mel 128 × 313 (mel128_32k)  | aleatória     |
| M3 | `Cnn14`       | log-Mel 64 × 501 (mel64_32k)    | PANNs/AudioSet|
| M4 | `CabecaLinear`| embedding 768 do AST congelado  | AST/AudioSet  |

M1 é da Parte 1 e não mudou. M2, M3 e M4 são da Parte 2.

O M4 não recebe espectrograma: o AST (87 M de parâmetros, 1214 tokens por janela) não cabe para
ajuste fino numa GPU de 4 GB, então ele é usado congelado como extrator de embeddings
(`birdclef.ast_emb`) e só a cabeça linear é treinada (*linear probe*). Ver o notebook 04.
"""

from pathlib import Path
import urllib.request

import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------------------------------------------------------- M1 e M2

class BlocoConv(nn.Module):
    """Conv 3x3 - BN - ReLU - MaxPool (2x2 por padrão)."""

    def __init__(self, entrada: int, saida: int, pool=(2, 2)):
        super().__init__()
        self.camadas = nn.Sequential(
            nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(saida),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x):
        return self.camadas(x)


class CNN(nn.Module):
    """M1: 4 blocos Conv-BN-ReLU-Pool, pooling global e uma camada linear.

    O pooling global faz a média na frequência e combina média e máximo no tempo: o canto pode
    ocupar só um trecho da janela, e o máximo preserva esse trecho.
    """

    def __init__(self, n_classes: int, canais=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocos, entrada = [], 1
        for c in canais:
            blocos.append(BlocoConv(entrada, c))
            entrada = c
        self.extrator = nn.Sequential(*blocos)
        self.dropout = nn.Dropout(dropout)
        self.saida = nn.Linear(entrada, n_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        h = self.extrator(x.unsqueeze(1))  # (B, C, M', T')
        h = h.mean(dim=2)                  # média na frequência -> (B, C, T')
        return h.mean(dim=2) + h.amax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.saida(self.dropout(self.embedding(x)))


class CRNN(nn.Module):
    """M2: blocos convolucionais + GRU bidirecional no eixo do tempo.

    Os 3 primeiros blocos reduzem frequência e tempo (2x2); o 4º reduz só a frequência (2x1),
    para a GRU receber uma sequência mais longa (39 passos de ~128 ms numa janela de 5 s).
    Depois da média na frequência, a GRU lê a sequência e a saída é resumida por média + máximo
    no tempo, o mesmo pooling do M1.

    Tamanho escolhido para ficar do mesmo porte do M1 (~375 mil contra ~393 mil parâmetros),
    como pede a hipótese H2: a diferença entre os dois é a modelagem temporal, não a capacidade.
    """

    def __init__(self, n_classes: int, canais=(32, 64, 128, 128), oculto: int = 96, dropout: float = 0.3):
        super().__init__()
        pools = [(2, 2), (2, 2), (2, 2), (2, 1)]
        blocos, entrada = [], 1
        for c, p in zip(canais, pools):
            blocos.append(BlocoConv(entrada, c, pool=p))
            entrada = c
        self.extrator = nn.Sequential(*blocos)
        self.gru = nn.GRU(entrada, oculto, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.saida = nn.Linear(2 * oculto, n_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        h = self.extrator(x.unsqueeze(1))      # (B, C, M', T')
        h = h.mean(dim=2).transpose(1, 2)      # (B, T', C)
        with torch.autocast(h.device.type, enabled=False):
            h, _ = self.gru(h.float())         # GRU em float32 (estável com precisão mista)
        return h.mean(dim=1) + h.amax(dim=1)   # (B, 2*oculto)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.saida(self.dropout(self.embedding(x)))


# ----------------------------------------------------------------------------- M3: PANNs CNN14

URL_CNN14 = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
ARQUIVO_CNN14 = "Cnn14_mAP=0.431.pth"


def pasta_pretreinados() -> Path:
    return Path(__file__).resolve().parents[2] / "checkpoints" / "pretreinados"


def baixar_cnn14() -> Path:
    """Baixa (uma vez) os pesos oficiais do PANNs CNN14 (~330 MB, Zenodo, licença MIT)."""
    destino = pasta_pretreinados() / ARQUIVO_CNN14
    if not destino.exists():
        destino.parent.mkdir(parents=True, exist_ok=True)
        print(f"baixando {ARQUIVO_CNN14} (~330 MB) de {URL_CNN14}", flush=True)
        urllib.request.urlretrieve(URL_CNN14, destino)
    return destino


class BlocoConvPANNs(nn.Module):
    """Bloco do CNN14 oficial: (Conv3x3-BN-ReLU) x2 + average pooling. Mesmos nomes de camada do
    código original (`conv1`, `bn1`, ...) para que os pesos carreguem direto."""

    def __init__(self, entrada: int, saida: int):
        super().__init__()
        self.conv1 = nn.Conv2d(entrada, saida, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(saida, saida, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(saida)
        self.bn2 = nn.BatchNorm2d(saida)

    def forward(self, x, pool=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool)


class Cnn14(nn.Module):
    """M3: PANNs CNN14 (Kong et al., 2020), reimplementado com os nomes de camada do repositório
    oficial (github.com/qiuqiangkong/audioset_tagging_cnn, MIT) para carregar os pesos do AudioSet.

    Diferenças em relação ao original, todas intencionais:

    - o espectrograma não é calculado aqui: chega pré-computado (`mel64_32k`, mesma configuração
      do PANNs: 32 kHz, n_fft 1024, hop 320, 64 bandas, 50-14000 Hz);
    - o SpecAugment e o mixup internos do PANNs foram retirados: o aumento é o do projeto
      (`birdclef.aumento`), igual para todos os modelos;
    - a saída `fc_audioset` (527 classes, sigmoide) é trocada por `saida` (n_classes, logits).

    A entrada chega padronizada pela média e desvio globais do treino (como nos outros modelos); a
    `bn0` (BatchNorm por banda Mel) reestima suas estatísticas no ajuste fino.
    """

    def __init__(self, n_classes: int, pretreinado: bool = True, n_mels: int = 64):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(n_mels)
        canais = (64, 128, 256, 512, 1024, 2048)
        entrada = 1
        for i, c in enumerate(canais, start=1):
            setattr(self, f"conv_block{i}", BlocoConvPANNs(entrada, c))
            entrada = c
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.saida = nn.Linear(2048, n_classes, bias=True)
        if pretreinado:
            self.carregar_audioset()

    def carregar_audioset(self) -> None:
        ck = torch.load(baixar_cnn14(), map_location="cpu", weights_only=False)
        estado = ck.get("model", ck)
        proprio = self.state_dict()
        usados = {k: v for k, v in estado.items() if k in proprio and v.shape == proprio[k].shape}
        faltando = [k for k in proprio if k not in usados and not k.startswith("saida.")]
        if faltando:
            raise RuntimeError(f"pesos do CNN14 não casaram: {faltando[:5]}...")
        self.load_state_dict(usados, strict=False)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(1)  # (B, M, T) -> (B, 1, T, M), layout do PANNs
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for i in range(1, 7):
            pool = (2, 2) if i < 6 else (1, 1)
            x = getattr(self, f"conv_block{i}")(x, pool=pool)
            x = F.dropout(x, p=0.2, training=self.training)
        x = x.mean(dim=3)                       # média na frequência -> (B, C, T')
        x = x.amax(dim=2) + x.mean(dim=2)       # máximo + média no tempo
        x = F.dropout(x, p=0.5, training=self.training)
        return F.relu_(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.saida(F.dropout(self.embedding(x), p=0.5, training=self.training))

    def grupos_parametros(self, lr_backbone: float, lr_cabeca: float) -> list[dict]:
        """Taxa menor no corpo pré-treinado e maior na camada nova (ajuste fino discriminativo)."""
        cabeca = list(self.saida.parameters())
        ids = {id(p) for p in cabeca}
        corpo = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": corpo, "lr": lr_backbone}, {"params": cabeca, "lr": lr_cabeca}]


# ----------------------------------------------------------------------------- M4: cabeça sobre o AST

class CabecaLinear(nn.Module):
    """M4: classificador linear sobre o embedding (768) do AST congelado (*linear probe*)."""

    def __init__(self, n_classes: int, dim: int = 768, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.saida = nn.Linear(dim, n_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.saida(self.dropout(x))


# ----------------------------------------------------------------------------- fábrica

# Configuração de espectrograma de cada modelo (nome da pasta em dados/espectrogramas/).
ENTRADA = {"M1": "mel128_32k", "M2": "mel128_32k", "M3": "mel64_32k", "M4": "ast_emb"}


def criar(nome: str, n_classes: int, pretreinado: bool = True) -> nn.Module:
    """`pretreinado=False` cria o M3 sem baixar os pesos do AudioSet (usado ao recarregar um
    checkpoint já treinado, que traz todos os pesos)."""
    if nome == "M1":
        return CNN(n_classes)
    if nome == "M2":
        return CRNN(n_classes)
    if nome == "M3":
        return Cnn14(n_classes, pretreinado=pretreinado)
    if nome == "M4":
        return CabecaLinear(n_classes)
    raise ValueError(f"modelo desconhecido: {nome}")


def n_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)
