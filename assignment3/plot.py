import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

from sklearn.cluster import DBSCAN


basename = sys.argv[1]

data = {}
with open(f'{basename}.txt', 'rt') as f:
    for line in f.readlines():
        idx, x, y = line.split()
        x, y = float(x), float(y)
        data[int(idx)] = np.array([x, y])

points = []
colors = []
for cluster_id, cluster_name in enumerate(glob.glob(f'{basename}_cluster_*')):
    with open(cluster_name, 'rt') as f:
        cluster = []
        for line in f.readlines():
            idx = int(line)
            points.append(data[idx])
            colors.append(cluster_id)

points = np.array(points)
xs = points[:, 0]
ys = points[:, 1]
plt.scatter(xs, ys, c=colors, cmap='rainbow')

model = DBSCAN(eps=15, min_samples=22)

points = np.array(list(data.values()))
colors = model.fit_predict(points)
xs = points[:, 0]
ys = points[:, 1]
plt.scatter(xs, ys, c=colors, s=1, marker='*', cmap='rainbow')

plt.savefig('test.png')
