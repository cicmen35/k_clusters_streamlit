import random
import numpy as np
import streamlit as st

from k_clust import (
    blobs,
    feature_scale,
    init_centroids,
    load_tsv,
    plot_center_with_data,
    pred_cluster,
    repeat_until_stabilized,
    plot_predicted_centers,
)

@st.cache_data      # cache full k-means iteration history so stepping through steps doesn't recompute clusters
def run_kmeans_history(X: np.ndarray, k: int, seed: int, max_iters: int) -> list[dict]:
    clusters = init_centroids(X, k, seed)
    return repeat_until_stabilized(X, clusters, max_iters, seed)

st.title("K-means Clustering")

# optional tsv upload to use custom data instead of blobs
uploader = st.file_uploader("Upload a TSV file (x, y columns for coordinates)", type=["tsv"], accept_multiple_files=False)

# disable blob sliders if a file is uploaded
has_upload = uploader is not None
n_samples = st.slider("Samples", 100, 1000, 500, step=50, disabled=has_upload)
centers = st.slider("True blobs (centers)", 2, 8, 3, disabled=has_upload)
k = st.slider("Clusters (k)", 2, 8, 3)

# pick a new random seed when button is clicked
rerun = st.button("Random seed")
if rerun:
    st.session_state.seed = random.randint(0, 10000)

random_state = st.number_input("Random seed", key="seed", step=1)

# load uploaded tsv data or generate synthetic blobs
if uploader is not None:
    try:
        X = load_tsv(uploader)
    except ValueError as err:
        st.error(f"Could not load TSV file: {err}")
        st.stop()
else:
    X, _ = blobs(n_samples=n_samples, centers=centers, random_state=int(random_state))

# scale data and initialize centroids
X = feature_scale(X)
#clusters = init_centroids(X, k, int(random_state))
iterations_count = st.number_input("Number of iterations (default = 100, min. value = 1)", min_value=1, value=100, key="iterations", step=1)

# run k-means and retrieve the list of cluster states across all iterations
history = run_kmeans_history(X, k, int(random_state), int(iterations_count))
# final iteration index
max_step = len(history) - 1
# keep current step in bounds if a new run converged in fewer iterations
step = min(st.session_state.get("step", 0), max_step)

# buttons to scrub through every iteration from 0 to max_step
b1, b2 = st.columns(2)
if b1.button("Previous step") and step > 0:
    step -= 1
if b2.button("Next step") and step < max_step:
    step += 1
st.session_state.step = step

col1, col2 = st.columns(2)

# show data with initial random centroids
with col1:
    st.subheader("Initial Centroids")
    st.pyplot(plot_center_with_data(X, history[0]), clear_figure=True)


# show final clustered data and centroids
with col2:
    st.subheader(f"Clustered Data (Step {step}/{max_step})")
    step_clusters = history[step]
    step_pred = pred_cluster(X, step_clusters)
    st.pyplot(plot_predicted_centers(X, step_clusters, step_pred), clear_figure=True)
