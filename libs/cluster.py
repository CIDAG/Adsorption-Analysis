# Class to store clustering information

from __future__ import annotations
import numpy as np
import numpy.typing as npt


class Cluster:
    """
    Data type to store clustering information
    """

    def __init__(self, max_size: int) -> Cluster:
        self.best_k = 0
        self.labels = np.repeat(None, max_size)
        self.centroids = np.repeat(None, max_size)
        self.scores = np.repeat(None, max_size)
        self.wcss = np.repeat(None, max_size)

    def __getitem__(
        self, k: npt.NDArray[int] | int
    ) -> tuple[
        int,
        npt.NDArray[int],
        npt.NDArray[float],
        npt.NDArray[float],
        npt.NDArray[float],
    ]:
        """
        Return the k value queried, and (labels, centroids, scores) for this k
        """
        return (
            k,
            self.labels[k],
            self.centroids[k],
            self.scores[k],
            self.wcss[k],
        )

    def __len__(self) -> int:
        """
        Return amount of non null data
        """
        return sum(1 if l is not None else 0 for l in self.labels)

    def get_best(
        self,
    ) -> tuple[
        int,
        npt.NDArray[int],
        npt.NDArray[float],
        npt.NDArray[float],
        npt.NDArray[float],
    ]:
        """
        Return the data for the best value of K found by Silhouette
        """
        return (
            self.best_k,
            self.labels[self.best_k],
            self.centroids[self.best_k],
            self.scores[self.best_k],
            self.wcss[self.best_k],
        )

    def insert(
        self,
        k: int,
        labels: npt.NDArray[int],
        centroid: npt.NDArray[float],
        score: float,
        wcss: float,
    ) -> None:
        """
        Insert cluster
        """
        # Only a single element
        if k > len(self.labels):
            self.labels[0] = labels
            self.centroids[0] = centroid
            self.scores[0] = score
            self.wcss[0] = wcss
        # Multiple elements
        else:
            self.labels[k] = labels
            self.centroids[k] = centroid
            self.scores[k] = score
            self.wcss[k] = wcss

    def __str__(self):
        """
        Formatted print
        """
        return (
            "Cluster\n"
            f"best_k = {self.best_k}\n"
            f"labels = {self.labels}\n"
            f"centroids = {self.centroids}\n"
            f"scores = {self.scores}\n"
            f"wcss = {self.wcss}\n"
        )

    def __repr__(self):
        return self.__str__()
