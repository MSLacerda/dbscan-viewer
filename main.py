# app.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DBSCAN ε Explorer", layout="wide")

# Global style helper (white background + large fonts)
def style_plot(fig, height=600, width=None):
    fig.update_layout(
        font=dict(size=18, color="black"),        # fonte padrão
        legend=dict(                              # legenda maior
            font=dict(size=20, color="black")
        ),
        xaxis=dict(                               # eixo X
            title_font=dict(size=20, color="black"),
            tickfont=dict(size=16, color="black"),
            showline=True, linewidth=1.5, linecolor="black",
            gridcolor="#E6E6E6"
        ),
        yaxis=dict(                               # eixo Y
            title_font=dict(size=20, color="black"),
            tickfont=dict(size=16, color="black"),
            showline=True, linewidth=1.5, linecolor="black",
            gridcolor="#E6E6E6"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
    )
    if width is not None:
        fig.update_layout(width=width)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def kth_neighbor_distances(X, k: int):
    """Return sorted distances to k-th nearest neighbor for each point."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)  # +1 to include the point itself
    dists, _ = nbrs.kneighbors(X)
    kth = dists[:, -1]  # distance to k-th neighbor (excluding self)
    return np.sort(kth), kth

def run_dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_  # -1 are outliers
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    outliers = int(np.sum(labels == -1))
    clustered = int(len(labels) - outliers)
    return labels, n_clusters, clustered, outliers

def suggest_eps_from_knee(sorted_kdist):
    """Simple knee via discrete 2nd derivative on normalized curve."""
    y = sorted_kdist
    if len(y) < 5:
        return float(np.median(y))
    x = np.linspace(0, 1, len(y))
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    curv = np.zeros_like(y_n)
    curv[1:-1] = np.abs(y_n[:-2] - 2 * y_n[1:-1] + y_n[2:])
    knee_idx = int(np.argmax(curv))
    return float(y[knee_idx])

def compute_eps_range(X, min_samples):
    """Derive a slider range for eps from k-distance percentiles."""
    sorted_kdist, _ = kth_neighbor_distances(X, min_samples)
    p10, p95 = np.percentile(sorted_kdist, [10, 95])
    if p95 <= 0:
        p95 = float(sorted_kdist.max()) if sorted_kdist.size else 1.0
    lo = max(1e-6, 0.25 * p10)
    hi = max(lo * 2, p95 * 1.25)
    return float(lo), float(hi), sorted_kdist

def fixed_color_map(unique_labels):
    """
    Return a stable color map so colors don't shuffle when params change.
    - 'Outlier' -> black
    - clusters '0','1','2',... -> deterministic palette
    """
    # deterministic palette with many distinct colors
    palette = (
        pc.qualitative.Set1
        + pc.qualitative.Set2
        + pc.qualitative.Set3
        + pc.qualitative.Plotly
    )
    cmap = {"Outlier": "black"}
    # keep numeric cluster labels sorted for stability
    ordered = [lab for lab in sorted(unique_labels) if lab != "Outlier"]
    for i, lab in enumerate(ordered):
        cmap[lab] = palette[i % len(palette)]
    return cmap

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — data & params
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("Data & Parameters")

dataset = st.sidebar.selectbox(
    "Dataset",
    ["Blobs (separable)", "Two Moons (nonlinear)"],
    index=0
)
n_samples = st.sidebar.slider("Number of samples", 200, 5000, 800, step=100)
random_state = st.sidebar.number_input("Random state", 0, 10000, 42, step=1)

if dataset.startswith("Blobs"):
    centers = st.sidebar.slider("Blobs: number of centers", 2, 8, 4, step=1)
    cluster_std = st.sidebar.slider("Blobs: cluster std", 0.2, 3.0, 0.80, step=0.05)
    X, y_true = make_blobs(
        n_samples=n_samples, centers=centers,
        cluster_std=cluster_std, random_state=random_state
    )
else:
    noise = st.sidebar.slider("Moons: noise", 0.0, 0.5, 0.08, step=0.01)
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

min_samples = st.sidebar.slider("min_samples (k)", 2, 50, 10, step=1)

# eps range + suggested knee
eps_lo, eps_hi, sorted_kdist = compute_eps_range(X, min_samples)
eps_suggested = suggest_eps_from_knee(sorted_kdist)

st.sidebar.markdown("---")
use_suggested = st.sidebar.checkbox("Use suggested ε (knee heuristic)", value=True)
eps_default = np.clip(eps_suggested, eps_lo, eps_hi) if use_suggested else (eps_lo + eps_hi) / 2
eps = st.sidebar.slider("ε (epsilon)", float(eps_lo), float(eps_hi), float(eps_default))

# Optional sweep
st.sidebar.markdown("---")
do_sweep = st.sidebar.checkbox("Show ε sweep (metrics vs ε)", value=False)
sweep_points = st.sidebar.slider("Sweep resolution (# ε values)", 10, 200, 50, step=5)

# ──────────────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────────────
st.title("DBSCAN ε (epsilon) Explorer")
st.caption("White background, large fonts, and fixed legend colors for stable visuals.")

colA, colB = st.columns([1.2, 1.0], gap="large")

# Left: clustering scatter + metrics
with colA:
    labels, n_clusters, clustered, outliers = run_dbscan(X, eps, min_samples)

    # DataFrame
    cols = ["x1", "x2"] + ([f"x{i}" for i in range(3, X.shape[1] + 1)] if X.shape[1] > 2 else [])
    df = pd.DataFrame(X, columns=cols)
    df["label"] = labels.astype(int)

    # Stable label names
    df["label_name"] = df["label"].astype(str)
    df.loc[df["label"] == -1, "label_name"] = "Outlier"

    # Fixed color map
    cmap = fixed_color_map(df["label_name"].unique())

    st.subheader("Current clustering")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ε (epsilon)", f"{eps:.4f}")
    m2.metric("min_samples (k)", f"{min_samples}")
    m3.metric("Clusters", f"{n_clusters}")
    m4.metric("Outliers", f"{outliers}  ({outliers/len(df):.1%})")

    # Scatter (2D)
    fig_scatter = px.scatter(
        df, x="x1", y="x2", color="label_name", opacity=0.95,
        title="DBSCAN clusters (2D)",
        color_discrete_map=cmap
    )
    fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig_scatter = style_plot(fig_scatter, height=650)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Right: k-distance (knee) + sweep
with colB:
    st.subheader(f"k-distance curve (k = min_samples = {min_samples})")

    # Knee plot
    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        y=sorted_kdist,
        x=np.arange(1, len(sorted_kdist) + 1),
        mode="lines",
        name="sorted k-distances"
    ))
    fig_k.add_hline(y=eps, line_dash="dash",
                    annotation_text=f"ε = {eps:.4f}", annotation_position="top left")
    if use_suggested:
        fig_k.add_hline(y=eps_suggested, line_dash="dot",
                        annotation_text=f"suggested ε ≈ {eps_suggested:.4f}")
    fig_k.update_layout(
        xaxis_title="Point index (sorted by k-distance)",
        yaxis_title=f"Distance to {min_samples}° neighbor",
        title="k-distance plot"
    )
    fig_k = style_plot(fig_k, height=740)
    st.plotly_chart(fig_k, use_container_width=True)

    if do_sweep:
        st.subheader("ε sweep (how metrics change with ε)")
        eps_grid = np.linspace(eps_lo, eps_hi, sweep_points)
        metrics = []
        for e in eps_grid:
            _, nc, cl, out = run_dbscan(X, e, min_samples)
            metrics.append((e, nc, cl, out))
        sweep_df = pd.DataFrame(metrics, columns=["eps", "n_clusters", "points_in_clusters", "outliers"])

        # Points in clusters vs outliers
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sweep_df["eps"], y=sweep_df["points_in_clusters"],
                                       mode="lines", name="# points in clusters"))
        fig_sweep.add_trace(go.Scatter(x=sweep_df["eps"], y=sweep_df["outliers"],
                                       mode="lines", name="# outliers"))
        fig_sweep.add_vline(x=eps, line_dash="dash", annotation_text=f"ε = {eps:.4f}")
        fig_sweep.update_layout(xaxis_title="ε (epsilon)", yaxis_title="Count", title="Counts vs ε")
        fig_sweep = style_plot(fig_sweep, height=320)
        st.plotly_chart(fig_sweep, use_container_width=True)

        # Number of clusters
        fig_clusters = go.Figure()
        fig_clusters.add_trace(go.Scatter(x=sweep_df["eps"], y=sweep_df["n_clusters"],
                                          mode="lines", name="# clusters"))
        fig_clusters.add_vline(x=eps, line_dash="dash", annotation_text=f"ε = {eps:.4f}")
        fig_clusters.update_layout(xaxis_title="ε (epsilon)", yaxis_title="# clusters", title="Clusters vs ε")
        fig_clusters = style_plot(fig_clusters, height=300)
        st.plotly_chart(fig_clusters, use_container_width=True)

# Footer notes
with st.expander("Notes"):
    st.markdown(
        """
- **Fixed colors**: Outliers are always black; cluster IDs keep stable, deterministic colors.
- **Styling**: White background and large fonts for readability in light theme.
- **Knee**: Suggested ε is found via a simple curvature heuristic on the k-distance curve.
        """
    )
