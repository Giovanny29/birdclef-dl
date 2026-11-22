"""E5.6: ajuste fino parcial do AST (os últimos blocos), comparado ao M4 (AST congelado + linear).

O M4 usa o AST inteiro congelado porque o ajuste fino completo não cabe em 4 GB de VRAM. Aqui
testamos o meio-termo: congelar os primeiros blocos e treinar só os `n_blocos` últimos, o
LayerNorm final e uma cabeça nova. Os blocos congelados não guardam ativações para o *backward*
(nenhum tensor deles exige gradiente), então a memória fica perto da de uma passagem direta.

Entrada: os fbanks de 128 bandas que o próprio `ASTFeatureExtractor` calcula (16 kHz, quadros de
10 ms, normalização do AudioSet), nos mesmos inícios de janela do `mel128_32k`. Uma janela de 5 s
tem ~500 quadros; o extrator completa até 1024 com um valor constante. Para economizar disco,
guardamos só os primeiros `QUADROS` quadros e o modelo recompõe o enchimento (mesmo valor que o
extrator usaria). Saída em `dados/espectrogramas/ast_fbank/` (`X.npy` float16 ~3 GB, fora do git).

Uma mudança por vez em relação ao M4: mesma entrada do AST, sem aumento de dados; muda só o que
treina (cabeça → cabeça + últimos blocos).
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from birdclef import dados, preproc
from birdclef.ast_emb import MODELO_AST, janelas_nos_inicios, pasta_cache_ast

QUADROS = 512           # quadros guardados por janela (5 s ≈ 498)
QUADROS_AST = 1024      # comprimento que o AST espera (10,24 s)
AST_FBANK = preproc.AST_FBANK


def extrator():
    from transformers import ASTFeatureExtractor
    return ASTFeatureExtractor.from_pretrained(MODELO_AST, cache_dir=pasta_cache_ast())


def valor_enchimento(ext=None) -> float:
    """Valor dos quadros de enchimento após a normalização do extrator (fbank 0 normalizado)."""
    ext = ext or extrator()
    return float((0.0 - ext.mean) / (ext.std * 2))


def preparar_fbank() -> None:
    """Fbanks de todas as janelas (mesma ordem do janelas.csv do mel128_32k), em memmap."""
    from numpy.lib.format import open_memmap
    cfg_ref = preproc.MEL128_32K
    saida = preproc.pasta_config(AST_FBANK)
    if (saida / "X.npy").exists() and (saida / "janelas.csv").exists():
        return
    saida.mkdir(parents=True, exist_ok=True)
    janelas = pd.read_csv(preproc.pasta_config(cfg_ref) / "janelas.csv")
    ext = extrator()
    X = open_memmap(saida / "X_parcial.npy", mode="w+", dtype=np.float16, shape=(len(janelas), QUADROS, 128))
    pasta_audio = dados.pasta_dados() / "bruto" / "train_audio"
    i = 0
    for arquivo, grupo in tqdm(janelas.groupby("filename", sort=False), desc="fbank do AST"):
        sinal = preproc.carregar_audio(pasta_audio / arquivo, AST_FBANK)
        ondas = janelas_nos_inicios(sinal, grupo["inicio_s"].to_numpy(), AST_FBANK.sr)
        fb = ext(list(ondas), sampling_rate=16000, return_tensors="np")["input_values"]  # (n, 1024, 128)
        X[i:i + len(fb)] = fb[:, :QUADROS].astype(np.float16)
        i += len(fb)
    assert i == len(janelas)
    X.flush(); del X
    (saida / "X_parcial.npy").rename(saida / "X.npy")
    shutil.copy(preproc.pasta_config(cfg_ref) / "janelas.csv", saida / "janelas.csv")


class ASTParcial(nn.Module):
    """AST pré-treinado com só os `n_blocos` últimos blocos (e o LayerNorm final) treináveis."""

    entrada_normalizada = True  # o fbank já vem normalizado pelo extrator: treino não renormaliza
    lote_inferencia = 16
    vram_treino_gb = 99.0       # X (3 GB) fica no memmap, nunca na VRAM

    def __init__(self, n_classes: int, pretreinado: bool = True, n_blocos: int = 2):
        super().__init__()
        from transformers import ASTConfig, ASTModel
        if pretreinado:
            self.ast = ASTModel.from_pretrained(MODELO_AST, cache_dir=pasta_cache_ast())
        else:
            self.ast = ASTModel(ASTConfig.from_pretrained(MODELO_AST, cache_dir=pasta_cache_ast()))
        for p in self.ast.parameters():
            p.requires_grad_(False)
        # transformers 5.x: blocos em `ast.layers` (no 4.x era `ast.encoder.layer`)
        blocos = self.ast.layers if hasattr(self.ast, "layers") else self.ast.encoder.layer
        for bloco in blocos[-n_blocos:]:
            for p in bloco.parameters():
                p.requires_grad_(True)
        for p in self.ast.layernorm.parameters():
            p.requires_grad_(True)
        self.enchimento = valor_enchimento() if pretreinado else 0.4670
        self.dropout = nn.Dropout(0.2)
        self.saida = nn.Linear(self.ast.config.hidden_size, n_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        if t < QUADROS_AST:
            x = torch.cat([x, x.new_full((b, QUADROS_AST - t, f), self.enchimento)], dim=1)
        return self.ast(input_values=x).pooler_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.saida(self.dropout(self.embedding(x)))

    def grupos_parametros(self, lr_backbone: float, lr_cabeca: float) -> list[dict]:
        corpo = [p for p in self.ast.parameters() if p.requires_grad]
        return [{"params": corpo, "lr": lr_backbone}, {"params": self.saida.parameters(), "lr": lr_cabeca}]


if __name__ == "__main__":
    preparar_fbank()
