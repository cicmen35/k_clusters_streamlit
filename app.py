import random
import streamlit as st

from k_clust import (
    blobs,
    feature_scale,
    init_centroids,
    load_tsv,
    plot_center_with_data,
    repeat_until_stabilized,
    plot_predicted_centers,
)

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
clusters = init_centroids(X, k, int(random_state))

col1, col2 = st.columns(2)

# show data with initial random centroids
with col1:
    st.subheader("Initial Centroids")
    st.pyplot(plot_center_with_data(X, clusters), clear_figure=True)

# run k-means until convergence
clusters, pred = repeat_until_stabilized(X, clusters)

# show final clustered data and centroids
with col2:
    st.subheader("Clustered Data")
    st.pyplot(plot_predicted_centers(X, clusters, pred), clear_figure=True)
