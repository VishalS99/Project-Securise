from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def palette_perc(k_cluster):
    width = 300
    palette = np.zeros((50, width, 3), np.uint8)

    n_pixels = len(k_cluster.labels_)
    counter = Counter(k_cluster.labels_)  # count how many pixels per cluster
    perc = {}
    for i in counter:
        perc[i] = np.round(counter[i] / n_pixels, 2)
    perc = dict(sorted(perc.items()))

    # for logging purposes
    print(perc)
    print(k_cluster.cluster_centers_)
    print(k_cluster.cluster_centers_[counter.most_common(1)[0][0]])  # Answer

    step = 0

    for idx, centers in enumerate(k_cluster.cluster_centers_):
        palette[:, step:int(step + perc[idx] * width + 1), :] = centers
        step += int(perc[idx] * width + 1)

    return palette


def show_img_compare(img_1, img_2):
    f, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off')
    ax[1].axis('off')
    f.tight_layout()
    plt.show()


if __name__ == '__main__':
    img1 = cv2.imread("target.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # resize image
    # img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

    clt = KMeans(n_clusters=5)

    clt1 = clt.fit(img1.reshape(-1, 3))
    show_img_compare(img1, palette_perc(clt1))
    cv2.imshow('colors',palette_perc(clt1))
