"""Embeddings do AST congelado para o M4 (*linear probe*).

Por que congelado: o Audio Spectrogram Transformer (Gong et al., 2021) tem ~87 M de parâmetros e
processa 1214 tokens por janela (o extrator completa 5 s até 10,24 s, como no treino original).
O ajuste fino completo não cabe numa GPU de 4 GB, mas a passagem direta sem gradiente cabe. Então
cada janela passa uma única vez pelo AST pré-treinado no AudioSet, o embedding de 768 dimensões
(`pooler_output`, média dos tokens [CLS] e de destilação) é salvo, e o M4 treina só uma cabeça
linear em cima dele. É a avaliação padrão de representações pré-treinadas (*linear probe*).

Mesmas amostras dos outros modelos: as janelas não são recalculadas a 16 kHz (o filtro de energia
poderia mudar na borda); cada gravação é recortada nos mesmos `inicio_s` do `janelas.csv` do
`mel128_32k`. O teste ruidoso (E3) usa os mesmos ruídos do fold 5 da ESC-50, sorteados com a mesma
semente e na mesma ordem de `aumento.preparar_teste_ruidoso`, então cada janela recebe o mesmo ruído
que recebeu nos outros modelos.

Limitação: como o embedding é fixo, o M4 não usa aumento de dados (SpecAugment, mixup e ruído agem
no espectrograma durante o treino). Isso é registrado no E1.

Saída em `dados/espectrogramas/ast_emb/`: `X.npy` (N, 768) float16, `janelas.csv`,
`X_teste_snr<k>.npy` e `janelas_teste.csv`.
"""

import shutil

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from birdclef import aumento, dados, preproc

MODELO_AST = "MIT/ast-finetuned-audioset-10-10-0.4593"


def carregar_ast(disp: torch.device):
    from transformers import ASTFeatureExtractor, ASTModel
    extrator = ASTFeatureExtractor.from_pretrained(MODELO_AST)
    modelo = ASTModel.from_pretrained(MODELO_AST).to(disp).eval()
    if disp.type == "cuda":
        modelo = modelo.half()
    return extrator, modelo


def janelas_nos_inicios(sinal: np.ndarray, inicios_s: np.ndarray, sr: int, duracao: float = 5.0) -> np.ndarray:
    """Recorta janelas de `duracao` s nos inícios dados (mesma regra de completar do preproc)."""
    n = int(sr * duracao)
    if len(sinal) < n:
        sinal = np.tile(sinal, int(np.ceil(n / max(len(sinal), 1))))[:n]
    return np.stack([sinal[int(round(i * sr)):int(round(i * sr)) + n] for i in inicios_s])


@torch.no_grad()
def embeddings(ondas: np.ndarray, extrator, modelo, lote: int = 16) -> np.ndarray:
    disp = next(modelo.parameters()).device
    saidas = []
    for i in range(0, len(ondas), lote):
        ent = extrator(list(ondas[i:i + lote]), sampling_rate=16000, return_tensors="pt")
        x = ent["input_values"].to(disp, dtype=next(modelo.parameters()).dtype)
        saidas.append(modelo(input_values=x).pooler_output.float().cpu().numpy())
    return np.concatenate(saidas)


def gerar(lote: int = 16, refazer: bool = False) -> None:
    """Gera os embeddings de todas as janelas e do teste ruidoso (~15-30 min numa GPU de notebook)."""
    cfg_ref, cfg = preproc.MEL128_32K, preproc.AST_EMB
    saida = preproc.pasta_config(cfg)
    saida.mkdir(parents=True, exist_ok=True)
    janelas = pd.read_csv(preproc.pasta_config(cfg_ref) / "janelas.csv")
    pasta_audio = dados.pasta_dados() / "bruto" / "train_audio"
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extrator, modelo = carregar_ast(disp)

    if refazer or not (saida / "X.npy").exists():
        blocos = []
        for arquivo, grupo in tqdm(janelas.groupby("filename", sort=False), desc="AST (todas as janelas)"):
            sinal = preproc.carregar_audio(pasta_audio / arquivo, cfg)
            blocos.append(embeddings(janelas_nos_inicios(sinal, grupo["inicio_s"].to_numpy(), cfg.sr), extrator, modelo, lote))
        X = np.concatenate(blocos).astype(np.float16)
        # groupby(sort=False) percorre as gravações na ordem em que aparecem; janelas de uma
        # gravação são contíguas no janelas.csv, então a ordem das linhas se mantém.
        assert len(X) == len(janelas)
        np.save(saida / "X.npy", X)
        shutil.copy(preproc.pasta_config(cfg_ref) / "janelas.csv", saida / "janelas.csv")

    if refazer or not (saida / f"X_teste_snr{aumento.SNRS_TESTE[-1]}.npy").exists():
        teste = janelas[janelas["particao"] == "teste"]
        ruidos, _ = aumento.carregar_ruidos([aumento.FOLD_RUIDO_TESTE], cfg)
        rng = np.random.default_rng(dados.SEMENTE)  # mesma semente e ordem de preparar_teste_ruidoso
        por_snr = {snr: [] for snr in aumento.SNRS_TESTE}
        ordem = []
        for arquivo, grupo in tqdm(teste.groupby("filename", sort=False), desc="AST (teste com ruído)"):
            jan = janelas_nos_inicios(preproc.carregar_audio(pasta_audio / arquivo, cfg), grupo["inicio_s"].to_numpy(), cfg.sr)
            escolhidos = ruidos[rng.integers(len(ruidos), size=len(jan))]
            for snr in aumento.SNRS_TESTE:
                mist = np.stack([aumento.misturar_onda(s, r, snr) for s, r in zip(jan, escolhidos)])
                por_snr[snr].append(embeddings(mist, extrator, modelo, lote).astype(np.float16))
            ordem.append(grupo)
        pd.concat(ordem).to_csv(saida / "janelas_teste.csv", index=False)
        for snr in aumento.SNRS_TESTE:
            np.save(saida / f"X_teste_snr{snr}.npy", np.concatenate(por_snr[snr]))


if __name__ == "__main__":
    gerar()
