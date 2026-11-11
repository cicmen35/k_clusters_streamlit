import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def load_tsv(file) -> np.ndarray:
    # reset stream position to the start
    file.seek(0)
    try:
        # try reading the tsv directly as numbers
        data = np.loadtxt(file, delimiter="\t")
    except ValueError:
        # if the first line is a text header, rewind and skip it
        file.seek(0)
        data = np.loadtxt(file, delimiter="\t", skiprows=1)

    # make sure we actually got at least 2 columns for x and y
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("TSV file must contain at least 2 numeric columns.")

    # take the first two columns
    return data[:, :2]


def blobs(n_samples: int = 500, centers: int = 3, random_state: int = 23) -> tuple[np.ndarray, np.ndarray]:
    # generate synthetic 2d blob data for clustering
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=centers,
        random_state=random_state,
    )
    return X, y


def feature_scale(X: np.ndarray) -> np.ndarray:
    # standardize features to zero mean and unit variance
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def init_centroids(X: np.ndarray, k: int, random_seed: int = 23) -> dict:
    # randomly initialize k cluster centroids within [-2, 2]
    clusters = {}
    rng = np.random.default_rng(random_seed)

    for index in range(k):
        # pick a random starting point for each centroid
        center = rng.uniform(-2.0, 2.0, size=(X.shape[1],))
        cluster = {
            'center': center,
            'points': []
        }
        clusters[index] = cluster
    return clusters


def plot_center_with_data(X: np.ndarray, clusters: dict) -> plt.Figure:
    # plot raw data points along with initial cluster centroids
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1])
    ax.grid(True)
    # draw initial centroids as red stars
    for i in clusters:
        center = clusters[i]['center']
        ax.scatter(center[0], center[1], marker='*', c='red', s=150)
    return fig


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    # compute euclidean distance between two points
    return float(np.sqrt(np.sum((p1 - p2) ** 2)))


def assign_clusters(X: np.ndarray, clusters: dict) -> dict:
    # assign each data point to the nearest cluster centroid
    k = len(clusters)
    # reset assigned points from the previous step
    for i in range(k):
        clusters[i]['points'] = []

    for idx in range(X.shape[0]):
        dist = []
        curr_x = X[idx]
        # calculate distance to every centroid
        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)
        # pick the closest cluster and add the point to it
        curr_cluster = int(np.argmin(dist))
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters


def update_clusters(X: np.ndarray, clusters: dict, rng: np.random.Generator) -> dict:
    # recalculate each centroid as the mean of its assigned points
    k = len(clusters)
    for i in range(k):
        points = np.array(clusters[i]['points'])
        # only move the centroid if the cluster is not empty
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
        else:
            # if a centroid has no cluster in its area, respawn dead centroid at a random data point 
            clusters[i]['center'] = X[rng.integers(0, X.shape[0])].copy()
    return clusters


def pred_cluster(X: np.ndarray, clusters: dict) -> list[int]:
    # get the closest cluster index for each data point
    k = len(clusters)
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(int(np.argmin(dist)))
    return pred

def snapshot_centers(clusters: dict) -> dict:
    # store only centroids coordinates
    return {
        i: {'center': clusters[i]['center'].copy()} for i in clusters
    }

def repeat_until_stabilized(X: np.ndarray, clusters: dict, max_iters: int = 100, random_seed: int = 23) -> list[dict]:
    # iteratively assign and update clusters until centroids stop moving
    k = len(clusters)
    history = [snapshot_centers(clusters)] # initial random centroids
    rng = np.random.default_rng(random_seed)

    for _ in range(max_iters):
        # save old centroids so we can check if they changed
        old_centers = [clusters[i]['center'].copy() for i in range(k)]

        clusters = assign_clusters(X, clusters)
        clusters = update_clusters(X, clusters, rng)

        new_centers = [clusters[i]['center'] for i in range(k)]
        history.append(snapshot_centers(clusters))

        # stop if centroids have converged
        if np.allclose(old_centers, new_centers):
            break

    return history


def plot_predicted_centers(X: np.ndarray, clusters: dict, pred: list[int]) -> plt.Figure:
    # plot data points colored by cluster assignment with final centroids
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=pred)
    ax.grid(True)
    # draw final centroids as red triangles
    for i in clusters:
        center = clusters[i]['center']
        ax.scatter(center[0], center[1], marker='^', c='red', s=150)
    return fig
