"""Pesos finais reutilizáveis em `checkpoints/finais/` (usados por `birdclef.inferencia`).

Para cada tipo de modelo (M1-M5) fica o candidato com **maior F1-macro na validação do fold 0**,
a única validação comum a todos os modelos e a mesma usada nas escolhas do E5. Os candidatos são a
execução do fold 0 do E1 e as variantes do E5 daquele tipo que já tiverem checkpoint. O teste é só
reportado, nunca usado na escolha.

O ensemble final é a média ponderada dos finais, com pesos escolhidos na mesma validação
(`melhorias.grade_simplex`); o JSON dele guarda os membros e os pesos.

Cada `<nome>.pt` guarda só o necessário para inferência (`estado`, `classes`, `norm`), e o
`<nome>.json` ao lado guarda classes, normalização, configuração da entrada (log-Mel ou AST),
janelas, hiperparâmetros, métricas de validação e de teste e a data.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch

from birdclef import dados, modelos, preproc, treino
from birdclef import melhorias as mh
from birdclef.experimentos import PASTA_CHECKPOINTS
from birdclef.inferencia import PASTA_FINAIS

# Variantes do E5 por tipo (nome do checkpoint, descrição). Só entram se o checkpoint existir.
VARIANTES_E5 = {
    "M1": [("M1_todas_e50_s42_f0", "50 épocas, paciência 10")],
    "M2": [("M2_todas_e50_s42_f0", "50 épocas, paciência 10")],
    "M3": [("M3_todas_2fases_s42_f0", "duas fases (cabeça congelada 2 épocas)"),
           ("M3_todas_ls01_s42_f0", "label smoothing 0,1"),
           ("M3_todas_lrb3e-5_s42_f0", "lr do corpo 3e-5"),
           ("M3_todas_lrb3e-4_s42_f0", "lr do corpo 3e-4")],
    "M4": [],
    "M4FT": [("M4FT_sem_s42_f0", "AST com os 2 últimos blocos ajustados")],
    "M5": [("M5_todas_s42_f0", "EfficientNet-B0 ImageNet")],
}


def candidatos(tipo: str) -> list[tuple[str, str]]:
    lista = [] if tipo in ("M5", "M4FT") else [(mh.nome_e1(tipo), "E1, fold 0")]
    lista += VARIANTES_E5.get(tipo, [])
    return [(n, d) for n, d in lista if (PASTA_CHECKPOINTS / f"{n}.pt").exists()]


def escolher(tipo: str) -> pd.DataFrame:
    """F1 de validação (fold 0) de cada candidato do tipo, ordenado do melhor para o pior."""
    linhas = [{"tipo": tipo, "checkpoint": n, "descricao": d, "val_f1": mh.val_f1_checkpoint(n)} for n, d in candidatos(tipo)]
    return pd.DataFrame(linhas).sort_values("val_f1", ascending=False).reset_index(drop=True)


def _config_entrada(tipo: str) -> dict:
    cfg = preproc.CONFIGS[modelos.ENTRADA[tipo]]
    d = asdict(cfg)
    if tipo in ("M4", "M4FT"):
        from birdclef.ast_emb import MODELO_AST
        d["observacao"] = (f"embedding de 768 dimensões (pooler_output) do AST congelado {MODELO_AST}, "
                           "calculado a 16 kHz nos mesmos inícios de janela do mel128_32k")
        d["modelo_ast"] = MODELO_AST
    return d


def _nomes_comuns(classes) -> dict:
    meta = dados.carregar_metadados().drop_duplicates("primary_label").set_index("primary_label")
    return {c: str(meta.loc[c, "common_name"]) for c in classes}


def salvar_final(tipo: str, checkpoint: str, descricao: str) -> dict:
    """Copia o checkpoint para `checkpoints/finais/<tipo>.pt` e escreve o JSON ao lado."""
    PASTA_FINAIS.mkdir(parents=True, exist_ok=True)
    ck = torch.load(PASTA_CHECKPOINTS / f"{checkpoint}.pt", map_location="cpu", weights_only=False)
    torch.save({"estado": ck["estado"], "classes": ck["classes"], "norm": ck["norm"]}, PASTA_FINAIS / f"{tipo}.pt")
    p_val, y_val = mh.probs_val(tipo, checkpoint)
    p_te, y_te = mh.probs_teste(tipo, checkpoint)
    rede = modelos.criar(tipo, len(ck["classes"]), pretreinado=False)
    rede.load_state_dict(ck["estado"])
    info = {
        "nome": tipo,
        "modelo": tipo,
        "descricao": descricao,
        "arquivo": f"{tipo}.pt",
        "origem": f"checkpoints/{checkpoint}.pt",
        "parametros": modelos.n_parametros(rede),
        "classes": ck["classes"],
        "nomes_comuns": _nomes_comuns(ck["classes"]),
        "normalizacao": {"media": float(ck["norm"][0]), "desvio": float(ck["norm"][1]),
                         "como": "x_norm = (entrada - media) / desvio, estatísticas globais do treino"},
        "config_entrada": _config_entrada(tipo),
        "janelas": {k: v for k, v in asdict(preproc.MEL128_32K).items()
                    if k in ("sr", "duracao_janela", "passo_janela", "max_janelas", "limiar_db")},
        "agregacao": "média das probabilidades das janelas da gravação",
        "hiperparametros": {"treino": ck.get("treino"), "aumento": ck.get("aumento")},
        "metricas_validacao": {"fold": 0, **treino.metricas(y_val, p_val)},
        "metricas_teste": treino.metricas(y_te, p_te),
        "data": datetime.now().isoformat(timespec="seconds"),
        "uso": f"python -m birdclef.inferencia caminho.ogg --modelo {tipo}",
    }
    (PASTA_FINAIS / f"{tipo}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return info


def salvar_ensemble(membros: list[str], passo: float = 0.05) -> dict:
    """Pesos escolhidos na validação do fold 0 sobre os finais `membros`; métricas e JSON."""
    val = {m: mh.probs_val(m, json.loads((PASTA_FINAIS / f"{m}.json").read_text(encoding="utf-8"))["origem"].split("/")[-1][:-3]) for m in membros}
    te = {m: mh.probs_teste(m, json.loads((PASTA_FINAIS / f"{m}.json").read_text(encoding="utf-8"))["origem"].split("/")[-1][:-3]) for m in membros}
    y_val, y_te = next(iter(val.values()))[1], next(iter(te.values()))[1]
    Pv, Pt = np.stack([val[m][0] for m in membros]), np.stack([te[m][0] for m in membros])
    melhor, pesos = -1, None
    for w in mh.grade_simplex(len(membros), passo):
        s = mh.f1(y_val, np.tensordot(w, Pv, 1))
        if s > melhor + 1e-9 or (abs(s - melhor) < 1e-9 and np.std(w) < np.std(pesos)):
            melhor, pesos = s, w
    classes = json.loads((PASTA_FINAIS / f"{membros[0]}.json").read_text(encoding="utf-8"))["classes"]
    info = {
        "nome": "ensemble",
        "modelo": "ensemble",
        "descricao": "média ponderada das probabilidades por gravação dos modelos finais",
        "membros": [{"nome": m, "peso": float(w)} for m, w in zip(membros, pesos) if w > 0],
        "pesos_escolhidos_em": f"validação do fold 0 (grade no simplex, passo {passo})",
        "classes": classes,
        "nomes_comuns": _nomes_comuns(classes),
        "metricas_validacao": {"fold": 0, **treino.metricas(y_val, np.tensordot(pesos, Pv, 1))},
        "metricas_teste": treino.metricas(y_te, np.tensordot(pesos, Pt, 1)),
        "data": datetime.now().isoformat(timespec="seconds"),
        "uso": "python -m birdclef.inferencia caminho.ogg --modelo ensemble",
    }
    (PASTA_FINAIS / "ensemble.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return info


def criar(tipos=("M1", "M2", "M3", "M4", "M4FT", "M5"), membros_ensemble=("M1", "M2", "M3")) -> pd.DataFrame:
    """Escolhe e salva o final de cada tipo disponível e o ensemble. Devolve o resumo."""
    linhas = []
    for tipo in tipos:
        esc = escolher(tipo) if candidatos(tipo) else pd.DataFrame()
        if esc.empty:
            continue
        info = salvar_final(tipo, esc.loc[0, "checkpoint"], esc.loc[0, "descricao"])
        linhas.append({"final": tipo, "origem": info["origem"], "descricao": info["descricao"],
                       "candidatos": len(esc), "val_f1": info["metricas_validacao"]["f1_macro"],
                       "teste_f1": info["metricas_teste"]["f1_macro"], "parametros": info["parametros"]})
    membros = [m for m in membros_ensemble if (PASTA_FINAIS / f"{m}.json").exists()]
    if len(membros) >= 2:
        ens = salvar_ensemble(membros)
        linhas.append({"final": "ensemble", "origem": " + ".join(f"{m['nome']}×{m['peso']:.2f}" for m in ens["membros"]),
                       "descricao": ens["descricao"], "candidatos": None,
                       "val_f1": ens["metricas_validacao"]["f1_macro"], "teste_f1": ens["metricas_teste"]["f1_macro"],
                       "parametros": None})
    return pd.DataFrame(linhas)
