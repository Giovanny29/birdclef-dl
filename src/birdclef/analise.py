"""Análises da Parte 2: F1 por espécie, checagem de vazamento por gravador, t-SNE e Grad-CAM.

- `f1_por_especie`: onde cada modelo erra.
- `vazamento_autor`: as divisões são por gravação, mas um mesmo gravador (`author` no xeno-canto)
  costuma subir várias gravações da mesma espécie, no mesmo lugar e com o mesmo equipamento. Se o
  teste for bem mais fácil nas gravações cujo autor também aparece no treino, parte do desempenho
  vem de reconhecer o gravador/ambiente, não a ave. A função separa o teste nos dois grupos.
- `embeddings` + t-SNE: como as espécies se separam no espaço aprendido.
- `grad_cam`: que regiões do espectrograma pesam na decisão (canto ou ruído de fundo?).
"""

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

from birdclef import dados


def f1_por_especie(y: np.ndarray, probs_por_modelo: dict, classes: list[str]) -> pd.DataFrame:
    linhas = {m: f1_score(y, p.argmax(1), average=None, labels=range(len(classes))) for m, p in probs_por_modelo.items()}
    return pd.DataFrame(linhas, index=classes)


def vazamento_autor(arquivos_teste: pd.Index, y: np.ndarray, probs_por_modelo: dict) -> pd.DataFrame:
    """F1-macro e acurácia no teste, separando gravações cujo autor também está no treino."""
    div = pd.read_csv(dados.pasta_dados() / "subconjunto" / "divisoes.csv")
    if "author" not in div.columns:
        raise ValueError("divisoes.csv sem a coluna 'author'")
    autores_treino = set(div.loc[div["particao"] == "treino", "author"])
    autor = div.set_index("filename").loc[arquivos_teste, "author"]
    visto = autor.isin(autores_treino).to_numpy()
    linhas = []
    for m, p in probs_por_modelo.items():
        pred = p.argmax(1)
        for grupo, mask in (("autor também no treino", visto), ("autor só no teste", ~visto)):
            linhas.append({"modelo": m, "grupo": grupo, "gravacoes": int(mask.sum()),
                           "acuracia": float((pred[mask] == y[mask]).mean()),
                           "f1_macro": f1_score(y[mask], pred[mask], average="macro")})
    return pd.DataFrame(linhas)


@torch.no_grad()
def embeddings(modelo, X, idx: np.ndarray, norm, lote: int = 128) -> np.ndarray:
    modelo.eval()
    disp = next(modelo.parameters()).device
    media, desvio = norm
    out = []
    for i in range(0, len(idx), lote):
        x = torch.from_numpy(np.asarray(X[idx[i:i + lote]], dtype=np.float32)).to(disp)
        out.append(modelo.embedding((x - media) / desvio).float().cpu().numpy())
    return np.concatenate(out)


def tsne(E: np.ndarray, semente: int = 42) -> np.ndarray:
    return TSNE(n_components=2, init="pca", perplexity=30, random_state=semente).fit_transform(E)


def grad_cam(modelo, camada, x_norm: torch.Tensor, classe: int | None = None) -> np.ndarray:
    """Grad-CAM (Selvaraju et al., 2017) de uma janela já normalizada, na resolução da entrada.

    `camada`: o módulo convolucional cuja saída (B, C, F, T) é explicada, p. ex.
    `modelo.extrator[-1]` no M1/M2 ou `modelo.conv_block5` no M3.
    """
    ativ, grad = {}, {}
    h1 = camada.register_forward_hook(lambda m, i, o: ativ.__setitem__("a", o))
    h2 = camada.register_full_backward_hook(lambda m, gi, go: grad.__setitem__("g", go[0]))
    modelo.eval()
    x = x_norm[None].requires_grad_(True)
    logits = modelo(x)
    classe = int(logits.argmax(1)) if classe is None else classe
    modelo.zero_grad()
    logits[0, classe].backward()
    h1.remove(); h2.remove()
    a, g = ativ["a"][0], grad["g"][0]                  # (C, F', T')
    cam = torch.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = torch.nn.functional.interpolate(cam[None, None], size=_tamanho_entrada(modelo, x_norm), mode="bilinear")[0, 0]
    cam = cam / (cam.max() + 1e-8)
    from birdclef.modelos import Cnn14
    if isinstance(modelo, Cnn14):
        cam = cam.T  # volta para (bandas, quadros), o layout da entrada
    return cam.detach().cpu().numpy()


def _tamanho_entrada(modelo, x_norm):
    # M3 trabalha com (T, M) internamente; devolvemos sempre no layout (M, T) da entrada
    from birdclef.modelos import Cnn14
    m, t = x_norm.shape
    return (t, m) if isinstance(modelo, Cnn14) else (m, t)
