# app.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import streamlit as st
import re, colorsys

# ============================ Feature Flags ============================
SHOW_SUGGESTED_EPS_VISUAL = False   # mostra/oculta a linha de ε sugerido (só visual)
EXPOSE_FLAGS_IN_UI = True           # exibir flags na UI
# ======================================================================

st.set_page_config(page_title="Explorador DBSCAN – ε", layout="wide")

# ───────────────── Aparência global ─────────────────
def style_plot(fig, height=600, width=None):
    fig.update_layout(
        font=dict(size=18, color="black"),
        legend=dict(
            font=dict(size=20, color="black"),                 # itens
            title=dict(font=dict(size=22, color="black"))      # título
        ),
        xaxis=dict(
            title_font=dict(size=20, color="black"),
            tickfont=dict(size=16, color="black"),
            showline=True, linewidth=1.5, linecolor="black", gridcolor="#E6E6E6"
        ),
        yaxis=dict(
            title_font=dict(size=20, color="black"),
            tickfont=dict(size=16, color="black"),
            showline=True, linewidth=1.5, linecolor="black", gridcolor="#E6E6E6"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
    )
    if width is not None:
        fig.update_layout(width=width)
    return fig

# ───────────────── Helpers de DBSCAN/joelho ─────────────────
def kth_neighbor_distances(X, k: int):
    """Distâncias até o k-ésimo vizinho para cada ponto (ordenadas e originais)."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)  # +1 inclui o próprio ponto
    dists, _ = nbrs.kneighbors(X)
    kth = dists[:, -1]
    return np.sort(kth), kth

def run_dbscan(X, eps, min_samples):
    """Executa DBSCAN e retorna rótulos e contagens úteis."""
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_                 # -1 = ruído
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    outliers = int(np.sum(labels == -1))
    clustered = int(len(labels) - outliers)
    return labels, n_clusters, clustered, outliers

def suggest_eps_from_knee(sorted_kdist):
    """Heurística simples: pico de curvatura discreta (pode superestimar em dados bem separados)."""
    y = sorted_kdist
    if len(y) < 5:
        return float(np.median(y))
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    curv = np.zeros_like(y_n)
    curv[1:-1] = np.abs(y_n[:-2] - 2 * y_n[1:-1] + y_n[2:])
    knee_idx = int(np.argmax(curv))
    return float(y[knee_idx])

def compute_eps_range(X, min_samples):
    """Define um intervalo razoável para o slider de ε a partir de percentis da k-distância."""
    sorted_kdist, _ = kth_neighbor_distances(X, min_samples)
    p10, p95 = np.percentile(sorted_kdist, [10, 95])
    if p95 <= 0:
        p95 = float(sorted_kdist.max()) if sorted_kdist.size else 1.0
    lo = max(1e-6, 0.25 * p10)
    hi = max(lo * 2, p95 * 1.25)
    return float(lo), float(hi), sorted_kdist

# ───────────────── Utilitários de cor ─────────────────
def _to_rgb_tuple(color_str):
    if not isinstance(color_str, str):
        return None
    s = color_str.strip()
    if s.startswith("#"):
        hx = s[1:]
        if len(hx) == 3:
            hx = "".join(ch * 2 for ch in hx)
        if len(hx) != 6:
            return None
        try:
            r = int(hx[0:2], 16) / 255.0
            g = int(hx[2:4], 16) / 255.0
            b = int(hx[4:6], 16) / 255.0
            return (r, g, b)
        except ValueError:
            return None
    m = re.match(r"rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)", s, re.IGNORECASE)
    if m:
        r, g, b = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        if max(r, g, b) > 1.0:
            r, g, b = r/255.0, g/255.0, b/255.0
        return (r, g, b)
    return None

def _is_reddish(color_str: str) -> bool:
    rgb = _to_rgb_tuple(color_str)
    if rgb is None:
        return False
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return (h < 15/360 or h > 345/360) and s >= 0.45

def fixed_color_map(unique_labels):
    """
    Mapa de cores estável:
    - 'Ruído' -> vermelho puro.
    - Clusters usam paleta determinística SEM tons de vermelho.
    """
    palette_raw = (
        pc.qualitative.Set1
        + pc.qualitative.Set2
        + pc.qualitative.Set3
        + pc.qualitative.Plotly
        + pc.qualitative.Dark24
        + pc.qualitative.Light24
    )
    seen, palette = set(), []
    for c in palette_raw:
        if c not in seen:
            seen.add(c); palette.append(c)

    safe_palette = [c for c in palette if not _is_reddish(c)]
    if not safe_palette:
        safe_palette = ["#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#bcbd22", "#8c564b"]

    cmap = {"Ruído": "red"}
    ordered = [lab for lab in sorted(unique_labels) if lab != "Ruído"]
    for i, lab in enumerate(ordered):
        cmap[lab] = safe_palette[i % len(safe_palette)]
    return cmap

# ───────────────── Desenho de círculos de raio ε ─────────────────
def epsilon_circles(df_xy, eps, max_circles=800, seed=42):
    """
    Gera shapes de círculos (em unidades do dado) com raio = ε ao redor de pontos (x1, x2).
    Limita a quantidade para manter performance.
    """
    idx = np.arange(len(df_xy))
    if len(idx) > max_circles:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_circles, replace=False)

    shapes = []
    for i in idx:
        xi = float(df_xy.iloc[i]["x1"]); yi = float(df_xy.iloc[i]["x2"])
        shapes.append(dict(
            type="circle", xref="x", yref="y", layer="below",
            x0=xi - eps, x1=xi + eps, y0=yi - eps, y1=yi + eps,
            line=dict(color="rgba(0,0,0,0.25)", width=1),
            fillcolor="rgba(0,0,0,0.0)"
        ))
    return shapes, len(idx)

# ───────────────── Sidebar ─────────────────
st.sidebar.title("Dados e Parâmetros")

if EXPOSE_FLAGS_IN_UI:
    st.sidebar.markdown("**Dev / Flags**")
    SHOW_SUGGESTED_EPS_VISUAL = st.sidebar.toggle(
        "Mostrar ε sugerido no gráfico", value=SHOW_SUGGESTED_EPS_VISUAL
    )
    st.sidebar.markdown("---")

dataset = st.sidebar.selectbox(
    "Conjunto de dados",
    [
        "Blobs (separáveis)",
        "Duas Luas (não linear)",
        "Círculos Concêntricos",
        "Espiral",
        "Uniforme Aleatório",
    ],
    index=0
)

n_samples = st.sidebar.slider("Tamanho da amostra", 200, 5000, 800, step=100)
random_state = st.sidebar.number_input("Semente (aleatória)", 0, 10000, 42, step=1)

if dataset.startswith("Blobs"):
    centers = st.sidebar.slider("Blobs: número de centros", 2, 8, 4, step=1)
    cluster_std = st.sidebar.slider("Blobs: dispersão dos grupos", 0.2, 3.0, 0.80, step=0.05)
    X, y_true = make_blobs(n_samples=n_samples, centers=centers,
                           cluster_std=cluster_std, random_state=random_state)

elif dataset.startswith("Duas"):
    noise = st.sidebar.slider("Duas Luas: ruído", 0.0, 0.5, 0.08, step=0.01)
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

elif dataset.startswith("Círculos"):
    noise = st.sidebar.slider("Círculos: ruído", 0.0, 0.5, 0.05, step=0.01)
    factor = st.sidebar.slider("Círculos: espaçamento", 0.1, 0.9, 0.5, step=0.05)
    X, y_true = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)

elif dataset.startswith("Espiral"):
    turns = st.sidebar.slider("Espiral: nº de voltas", 1, 6, 3, step=1)
    noise = st.sidebar.slider("Espiral: ruído", 0.0, 0.5, 0.05, step=0.01)
    t = np.linspace(0, turns * 2 * np.pi, n_samples)
    x = t * np.cos(t) + np.random.normal(scale=noise, size=n_samples)
    y = t * np.sin(t) + np.random.normal(scale=noise, size=n_samples)
    X = np.c_[x, y]
    y_true = None

elif dataset.startswith("Uniforme"):
    rng = np.random.RandomState(random_state)
    X = rng.rand(n_samples, 2) * 10 - 5
    y_true = None

min_samples = st.sidebar.slider("min_samples (k)", 2, 50, 10, step=1)

# Intervalo de ε e (internamente) ε sugerido
eps_lo, eps_hi, sorted_kdist = compute_eps_range(X, min_samples)
eps_suggested = suggest_eps_from_knee(sorted_kdist)   # calculado, mas pode não ser mostrado

st.sidebar.markdown("---")

# Slider híbrido para ε
eps_default = (eps_lo + eps_hi) / 2
try:
    eps = st.sidebar.slider(
        "ε (epsilon)",
        float(eps_lo), float(eps_hi),
        float(eps_default),
        step=(eps_hi - eps_lo) / 500  # passo fino
    )
except Exception:
    eps = st.sidebar.select_slider(
        "ε (epsilon)",
        options=np.linspace(eps_lo, eps_hi, 500),
        value=eps_default
    )

# Círculos de raio ε
st.sidebar.markdown("---")
show_eps_radius = st.sidebar.checkbox("Mostrar raio ε em cada ponto", value=False)
max_circles = st.sidebar.slider("Limite de círculos (perf.)", 100, 3000, 800, step=100)

# Sweep
st.sidebar.markdown("---")
do_sweep = st.sidebar.checkbox("Mostrar variação com ε", value=False)
sweep_points = st.sidebar.slider("Resolução da variação (# de ε)", 10, 200, 50, step=5)

# ───────────────── Layout principal ─────────────────
st.title("Explorador DBSCAN – ε (epsilon)")
st.caption("Visual em tema claro, fontes grandes e cores estáveis.")

colA, colB = st.columns([1.2, 1.0], gap="large")

# — Esquerda: dispersão + métricas
with colA:
    labels, n_clusters, clustered, outliers = run_dbscan(X, eps, min_samples)

    cols = ["x1", "x2"] + ([f"x{i}" for i in range(3, X.shape[1] + 1)] if X.shape[1] > 2 else [])
    df = pd.DataFrame(X, columns=cols)
    df["label"] = labels.astype(int)

    # nomes (clusters como números; ruído nomeado)
    df["label_name"] = df["label"].astype(str)
    df.loc[df["label"] == -1, "label_name"] = "Ruído"

    # Ordena legenda: clusters numéricos, depois Ruído
    labels_presentes = list(df["label_name"].unique())
    def _ord_key(lab):
        return (lab == "Ruído", int(lab) if lab != "Ruído" and lab.lstrip("-").isdigit() else 10**9)
    ordered_labels = sorted(labels_presentes, key=_ord_key)
    df["label_name"] = pd.Categorical(df["label_name"], categories=ordered_labels, ordered=True)

    # mapa de cores/símbolos só para categorias presentes
    cmap = fixed_color_map(ordered_labels)
    symbol_map = {**{lab: "circle" for lab in ordered_labels if lab != "Ruído"},
                  "Ruído": "x"}

    st.subheader("Agrupamento atual")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ε (epsilon)", f"{eps:.4f}")
    m2.metric("min_samples (k)", f"{min_samples}")
    m3.metric("Grupos", f"{n_clusters}")
    m4.metric("Ruído", f"{outliers}  ({outliers/len(df):.1%})")

    fig_scatter = px.scatter(
        df, x="x1", y="x2",
        color="label_name",
        symbol="label_name",                 # clusters=circle, Ruído=x
        symbol_map=symbol_map,
        opacity=0.95,
        title="DBSCAN – pontos agrupados e ruído",
        color_discrete_map=cmap,
        category_orders={"label_name": ordered_labels}
    )
    fig_scatter.update_traces(marker=dict(size=10, line=dict(width=0)))

    # Círculos de raio ε (coordenadas reais)
    if show_eps_radius:
        shapes, used = epsilon_circles(df[["x1", "x2"]], eps, max_circles=max_circles)
        fig_scatter.update_yaxes(scaleanchor="x", scaleratio=1.0)  # círculos de verdade
        fig_scatter.update_layout(shapes=shapes)
        st.caption(f"Raio ε desenhado em {used} ponto(s) (limite: {max_circles}).")

    fig_scatter = style_plot(fig_scatter, height=650)
    st.plotly_chart(fig_scatter, use_container_width=True)

# — Direita: curva k-distância (joelho) + variação
with colB:
    st.subheader(f"Curva de k-distância (k = min_samples = {min_samples})")

    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        y=sorted_kdist,
        x=np.arange(1, len(sorted_kdist) + 1),
        mode="lines",
        name="k-distâncias (ordenadas)"
    ))
    fig_k.add_hline(y=eps, line_dash="dash",
                    annotation_text=f"ε = {eps:.4f}", annotation_position="top left")

    # Flag visual do ε sugerido
    if SHOW_SUGGESTED_EPS_VISUAL:
        fig_k.add_hline(y=eps_suggested, line_dash="dot",
                        annotation_text=f"ε sugerido ≈ {eps_suggested:.4f}")

    fig_k.update_layout(
        legend_title_text="Legenda",
        xaxis_title="Índice do ponto (ordenado pela k-distância)",
        yaxis_title=f"Distância até o {min_samples}º vizinho",
        title="Gráfico do joelho (k-distância)"
    )
    fig_k = style_plot(fig_k, height=740)
    st.plotly_chart(fig_k, use_container_width=True)

    if do_sweep:
        st.subheader("Variação com ε")
        eps_grid = np.linspace(eps_lo, eps_hi, sweep_points)
        metrics = []
        for e in eps_grid:
            _, nc, cl, out = run_dbscan(X, e, min_samples)
            metrics.append((e, nc, cl, out))
        sweep_df = pd.DataFrame(metrics, columns=["eps", "n_clusters", "points_in_clusters", "outliers"])

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_df["eps"], y=sweep_df["points_in_clusters"],
                                       mode="lines", name="Pontos em grupos"))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["eps"], y=sweep_df["outliers"],
                                       mode="lines", name="Pontos em ruído"))
        fig_sweep.add_vline(x=eps, line_dash="dash", annotation_text=f"ε = {eps:.4f}")
        fig_sweep.update_layout(xaxis_title="ε (epsilon)", yaxis_title="Quantidade", title="Contagens vs ε")
        fig_sweep = style_plot(fig_sweep, height=320)
        st.plotly_chart(fig_sweep, use_container_width=True)

        fig_clusters = go.Figure()
        fig_clusters.add_trace(go.Scatter(x=sweep_df["eps"], y=sweep_df["n_clusters"],
                                          mode="lines", name="Nº de grupos"))
        fig_clusters.add_vline(x=eps, line_dash="dash", annotation_text=f"ε = {eps:.4f}")
        fig_clusters.update_layout(xaxis_title="ε (epsilon)", yaxis_title="Nº de grupos", title="Grupos vs ε")
        fig_clusters = style_plot(fig_clusters, height=300)
        st.plotly_chart(fig_clusters, use_container_width=True)

# ───────────────── Rodapé ─────────────────
with st.expander("Notas"):
    st.markdown(
        """
- **Cores fixas**: apenas *Ruído* é vermelho; grupos usam paleta estável sem vermelho.
- **Símbolos**: grupos → círculo; *Ruído* → “x”.
- **Raio ε**: círculos são desenhados em coordenadas reais e com limite para manter a performance; ao ativar, a proporção dos eixos é 1:1.
- **Joelho**: o cálculo de **ε sugerido** permanece ativo; a exibição pode ser ligada/desligada pela *feature flag*.
- **Datasets**: Blobs, Duas Luas, Círculos Concêntricos, Espiral e Uniforme Aleatório.
        """
    )
