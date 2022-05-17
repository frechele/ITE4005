from collections import deque
import sys
import numpy as np
import os


def load_data(filename: str):
    data_index = []
    data = []
    with open(filename, 'rt') as f:
        for line in f.readlines():
            idx, x_coord, y_coord = line.split()
            x_coord, y_coord = float(x_coord), float(y_coord)

            data_index.append(idx)
            data.append(np.array([x_coord, y_coord]))

    return data_index, np.array(data)


def dbscan(data: np.ndarray, eps: float, min_pts: int):
    status = np.zeros(data.shape[0])

    def _get_neighbor_idx(p_idx):
        distances = np.sqrt(np.sum(np.square(data - data[p_idx]), axis=-1))
        return np.where(distances <= eps)[0].tolist()

    clusters = []
    for idx in range(data.shape[0]):
        if status[idx] != 0:
            continue

        neighbors_idx = _get_neighbor_idx(idx)
        if len(neighbors_idx) < min_pts:
            status[idx] = -1 # mark as outlier
        else:
            new_cluster_id = len(clusters) + 1 # to make 0 as not processed
            new_cluster = [idx]
            status[idx] = new_cluster_id

            neighbors = deque(neighbors_idx)
            while len(neighbors) > 0:
                nb = neighbors.popleft()
                if status[nb] <= 0:
                    if status[nb] == 0:
                        nb_neighbors_idx = _get_neighbor_idx(nb)
                        if len(nb_neighbors_idx) >= min_pts:
                            neighbors += nb_neighbors_idx
                    
                    status[nb] = new_cluster_id
                    new_cluster.append(nb)

            clusters.append(new_cluster)

    return clusters


def main(filename: str, n: int, eps: float, min_pts: int):
    data_index, data = load_data(filename)
    clusters = dbscan(data, eps, min_pts)

    clusters = sorted(clusters, key=lambda x: len(x), reverse=True)[:n]
    
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]

    for cluster_id, cluster in enumerate(clusters):
        with open(f'{basename}_cluster_{cluster_id}.txt', 'wt') as f:
            f.write('\n'.join([data_index[idx] for idx in cluster]))


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('usage: python3 {} <filename> <n> <eps> <min_pts>'.format(sys.argv[0]))

    filename = sys.argv[1]
    n = int(sys.argv[2])
    eps = float(sys.argv[3])
    min_pts = int(sys.argv[4])

    main(filename, n, eps, min_pts)
