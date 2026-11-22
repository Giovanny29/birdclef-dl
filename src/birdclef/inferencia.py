"""Inferência com os pesos finais: uma gravação (.ogg, .wav, ...) -> 3 espécies mais prováveis.

Uso (a partir da raiz do repositório, com o ambiente do projeto ativo):

    python -m birdclef.inferencia caminho.ogg                  # modelo padrão: ensemble
    python -m birdclef.inferencia caminho.ogg --modelo M3
    python -m birdclef.inferencia caminho.ogg --modelo M1 --top 5

Os pesos ficam em `checkpoints/finais/<nome>.pt`, cada um com um `<nome>.json` ao lado (classes,
normalização, configuração do log-Mel, hiperparâmetros, métricas e data). Ver `checkpoints/finais/`
e a seção "Como usar os pesos" do README.

A inferência repete exatamente o caminho do treino e da avaliação:

1. o áudio é lido em mono a 32 kHz e recortado com `preproc.janelas_da_gravacao` (janelas de 5 s,
   passo de 2,5 s, descarte das janelas 20 dB abaixo da mais forte, no máximo 12);
2. cada janela vira a entrada do modelo: log-Mel 128 × 313 (M1, M2), log-Mel 64 × 501 (M3) ou o
   embedding de 768 dimensões do AST congelado (M4, mesmos inícios, a 16 kHz);
3. a entrada é padronizada com a média e o desvio do treino gravados no JSON;
4. a probabilidade da gravação é a média das probabilidades das janelas; no ensemble, a média
   ponderada das probabilidades dos modelos (pesos escolhidos na validação, no JSON do ensemble).
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np
import torch

from birdclef import modelos, preproc
from birdclef.preproc import ConfigMel

PASTA_FINAIS = Path(__file__).resolve().parents[2] / "checkpoints" / "finais"


# ----------------------------------------------------------------------------- carregamento

def ler_json(nome: str) -> dict:
    caminho = PASTA_FINAIS / f"{nome}.json"
    if not caminho.exists():
        disponiveis = sorted(p.stem for p in PASTA_FINAIS.glob("*.json"))
        raise FileNotFoundError(f"{caminho} não existe. Disponíveis: {disponiveis}. "
                                "Veja no README como baixar ou recriar os pesos finais.")
    return json.loads(caminho.read_text(encoding="utf-8"))


def config_mel(d: dict) -> ConfigMel:
    nomes = {f.name for f in fields(ConfigMel)}
    return ConfigMel(**{k: v for k, v in d.items() if k in nomes})


class ModeloFinal:
    """Um modelo final (M1-M4) pronto para prever gravações."""

    def __init__(self, nome: str, disp: torch.device | None = None):
        self.info = ler_json(nome)
        self.nome = nome
        self.tipo = self.info["modelo"]
        self.classes = self.info["classes"]
        self.disp = disp or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(PASTA_FINAIS / self.info["arquivo"], map_location=self.disp, weights_only=False)
        self.rede = modelos.criar(self.tipo, len(self.classes), pretreinado=False).to(self.disp).eval()
        self.rede.load_state_dict(ck["estado"])
        self.media = float(self.info["normalizacao"]["media"])
        self.desvio = float(self.info["normalizacao"]["desvio"])
        self.cfg = config_mel(self.info["config_entrada"])
        self._ast = None

    # entradas ---------------------------------------------------------------
    def entradas(self, caminho: Path, janelas32: np.ndarray, inicios: np.ndarray) -> np.ndarray:
        if self.tipo == "M4FT":  # fbank do extrator do AST nos mesmos inícios (E5.6)
            from birdclef import ast_emb, ast_ft
            sinal16 = preproc.carregar_audio(caminho, self.cfg)
            ondas = ast_emb.janelas_nos_inicios(sinal16, inicios, self.cfg.sr)
            fb = ast_ft.extrator()(list(ondas), sampling_rate=16000, return_tensors="np")["input_values"]
            return fb[:, :ast_ft.QUADROS]
        if self.tipo == "M4":
            from birdclef import ast_emb
            if self._ast is None:
                self._ast = ast_emb.carregar_ast(self.disp)
            sinal16 = preproc.carregar_audio(caminho, self.cfg)  # cfg do AST: 16 kHz
            ondas = ast_emb.janelas_nos_inicios(sinal16, inicios, self.cfg.sr)
            return ast_emb.embeddings(ondas, *self._ast)
        return preproc.log_mel(janelas32, self.cfg)

    @torch.no_grad()
    def prever(self, caminho: Path, janelas32: np.ndarray, inicios: np.ndarray) -> np.ndarray:
        """Probabilidades por janela (n_janelas, classes)."""
        x = torch.from_numpy(self.entradas(caminho, janelas32, inicios).astype(np.float32)).to(self.disp)
        x = (x - self.media) / self.desvio
        with torch.autocast(self.disp.type, enabled=self.disp.type == "cuda"):
            logits = self.rede(x)
        return torch.softmax(logits.float(), 1).cpu().numpy()


def recortar(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    """Janelas de 5 s a 32 kHz e seus inícios (s), com o mesmo recorte do treino."""
    cfg = preproc.MEL128_32K
    return preproc.janelas_da_gravacao(preproc.carregar_audio(caminho, cfg), cfg)


def prever_arquivo(caminho: str | Path, modelo: str = "ensemble", top: int = 3) -> list[dict]:
    """As `top` espécies mais prováveis da gravação, com probabilidade.

    `modelo`: "M1", "M2", "M3", "M4", "M4FT", "M5" (os que existirem em checkpoints/finais/) ou "ensemble".
    """
    caminho = Path(caminho)
    janelas32, inicios = recortar(caminho)
    if modelo == "ensemble":
        info = ler_json("ensemble")
        classes, nomes_comuns = info["classes"], info.get("nomes_comuns", {})
        probs = np.zeros(len(classes))
        for membro in info["membros"]:
            m = ModeloFinal(membro["nome"])
            assert m.classes == classes, f"classes de {membro['nome']} diferem das do ensemble"
            probs += membro["peso"] * m.prever(caminho, janelas32, inicios).mean(0)
        probs /= sum(mb["peso"] for mb in info["membros"])
    else:
        m = ModeloFinal(modelo)
        classes, nomes_comuns = m.classes, m.info.get("nomes_comuns", {})
        probs = m.prever(caminho, janelas32, inicios).mean(0)
    ordem = np.argsort(-probs)[:top]
    return [{"especie": classes[i], "nome": nomes_comuns.get(classes[i], ""), "probabilidade": float(probs[i])}
            for i in ordem]


def main(argv=None):
    p = argparse.ArgumentParser(description="Espécies de ave mais prováveis numa gravação (BirdCLEF 2025, 20 espécies).")
    p.add_argument("audio", help="arquivo de áudio (.ogg, .wav, .mp3, ...)")
    p.add_argument("--modelo", default="ensemble", choices=["M1", "M2", "M3", "M4", "M4FT", "M5", "ensemble"])
    p.add_argument("--top", type=int, default=3)
    a = p.parse_args(argv)
    janelas, _ = recortar(Path(a.audio))
    print(f"{a.audio}: {len(janelas)} janela(s) de 5 s | modelo {a.modelo}")
    for k, r in enumerate(prever_arquivo(a.audio, a.modelo, a.top), start=1):
        print(f"  {k}. {r['especie']:<8} {r['nome']:<32} {100 * r['probabilidade']:5.1f}%")


if __name__ == "__main__":
    main()
