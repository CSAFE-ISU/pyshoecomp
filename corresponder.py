# -*- coding: utf-8 -*-
import warnings
import time
from typing import Optional, Any
import numpy as np
from numpy.typing import ArrayLike
from collections import UserDict
import rtree

import cliquematch

warnings.filterwarnings(action="ignore", message=".*Euclidean.*", module="cliquematch")

class Correspondence(UserDict):
    @classmethod
    def success(cls, Q_corr, K_corr, **params):
        if len(Q_corr) != len(K_corr):
            warnings.warn("mapping may not be one-to-one", RuntimeWarning)
        answer = Correspondence(success=True, size=len(Q_corr), Q=Q_corr, K=K_corr)
        answer.update(**params)
        return answer

    @classmethod
    def failure(cls, **params):
        answer = Correspondence(
            success=False, size=0, Q=np.zeros((1, 2)), K=np.zeros((1, 2))
        )
        answer.update(**params)
        return answer


class Corresponder:
    _extname_ = "<none>"

    def __init__(self, *args, **params):
        pass

    def __call__(self, Q, K, *args, **params) -> Correspondence:
        start = time.time()
        res = self._call_impl(Q, K, **params)
        res["time"] = time.time() - start
        # print("time taken is ", time.time() - start)
        return res

    def _call_impl(self, Q, K, *args, **params) -> Correspondence:
        """
        receive interest points from Q and K
        return a Correspondence object
        """
        raise NotImplementedError("abstract base class")


class DummyMatcher(Corresponder):
    _extname_ = "dummy"

    def _call_impl(self, Q, K, *args, **params) -> Correspondence:
        return Correspondence.success(
            Q_corr=Q.points,
            K_corr=K.points,
        )


class CliqueMatcher(Corresponder):
    _extname_ = "clique1"

    def __init__(self, epsilon: float = 0.05, use_dfs: bool = False, *args, **params):
        super().__init__(epsilon=epsilon, use_dfs=use_dfs, *args, **params)
        self.epsilon = max(0.05, epsilon)
        self.use_dfs = use_dfs

    @staticmethod
    def get_cfunc_for_graph_build(theta_range):
        q_diff = np.zeros(2, dtype=np.float32)
        k_diff = np.zeros(2, dtype=np.float32)
        invmat = np.zeros((2, 2), dtype=np.float32)

        def cfunc(S1, i1, i2, S2, j1, j2) -> bool:
            q_diff = S1[i2] - S1[i1]
            k_diff = S2[j2] - S2[j1]
            invmat[0, 0] = -q_diff[0]  # d
            invmat[0, 1] = -q_diff[1]  # -b
            invmat[1, 0] = -q_diff[1]  # -c
            invmat[1, 1] = q_diff[0]  # a
            res = np.matmul(k_diff, invmat)
            theta = np.pi - np.arctan2(res[1], res[0])
            return theta >= theta_range[0] and theta <= theta_range[1]

        return cfunc

    def _call_impl(self, Q, K, *args, **params) -> Correspondence:
        if len(Q.points) <= 2 or len(K.points) <= 2:
            warnings.warn("not enough interest points")
            return Correspondence.failure()
        # ADD A DECENT CONDITION FUNCTION
        # TO HAVE A SPARSER GRAPH
        # THE RECTANGLE OVERLAP CHECK
        # OR A ROTATION LIMIT
        # OTHERWISE when building edges, give a large epsilon and
        # set use_dfs = False, in the clique search
        try:
            G = cliquematch.A2AGraph(Q.points, K.points)
            G.epsilon = self.epsilon
            if not G.build_edges():
                warnings.warn(
                    "unable to construct correspondence graph", RuntimeWarning
                )
                return Correspondence.failure(graph_V=0, graph_E=0)
        except Exception as e:
            print(e, "construction")
            return Correspondence.failure(graph_V=0, graph_E=0)

        V = G.n_vertices
        E = G.n_edges
        dens = (2.0 * E) / (V * (V - 1))
        ub = min(len(K.points), len(Q.points))
        try:
            clq = np.array(
                G.get_max_clique(upper_bound=ub, use_dfs=False), dtype=np.uint64
            )
            corr = (
                Q.points[(clq - 1) // len(K.points)],
                K.points[(clq - 1) % len(K.points)],
            )
        except Exception:
            print(e, "correspondence")
            warnings.warn("unable to find maximum clique", RuntimeWarning)
            return Correspondence.failure(graph_V=0, graph_E=0)

        answer = Correspondence.success(
            Q_corr=corr[0],
            K_corr=corr[1],
            ub=ub,
            ratio=100 * len(corr[0]) / ub,
            graph_V=G.n_vertices,
            graph_E=G.n_edges,
        )
        del G
        # print("clique size is", answer["size"])
        return answer


class CliqueMatcherWithTuning(Corresponder):
    _extname_ = "clique2"
    def __init__(
        self,
        epsilon: float = 0.05,
        use_dfs: bool = False,
        alpha: float = 0.01,
        *args,
        **params
    ):
        super().__init__(epsilon=epsilon, use_dfs=use_dfs, alpha=alpha, *args, **params)
        self.epsilon = max(0.05, epsilon)
        self.use_dfs = use_dfs
        self.alpha = max(0.0, alpha)

    def _split(self, pts):
        if self.alpha <= 0.0:
            return pts
        is_pt = np.ones(len(pts), dtype=np.bool_)
        ind = rtree.index.Index()
        r = self.alpha
        for i, p in enumerate(pts):
            px, py = p
            nearby = ind.intersection((px - r, py - r, px + r, py + r))
            if any(np.linalg.norm(p - pts[j]) <= r for j in nearby):
                is_pt[i] = False
            else:
                ind.insert(i, (px, py, px, py))
        return pts[is_pt]

    def _call_impl(self, Q, K, *args, **params) -> Correspondence:
        Q_sep_points = self._split(Q.points)
        K_sep_points = self._split(K.points)
        if len(Q_sep_points) <= 2 or len(K_sep_points) <= 2:
            warnings.warn("not enough interest points")
            return Correspondence.failure()
        # ADD A DECENT CONDITION FUNCTION
        # TO HAVE A SPARSER GRAPH
        # THE RECTANGLE OVERLAP CHECK
        # OR A ROTATION LIMIT
        # OTHERWISE when building edges, give a large epsilon and
        # set use_dfs = False, in the clique search
        try:
            G = cliquematch.A2AGraph(Q_sep_points, K_sep_points)
            G.epsilon = self.epsilon
            if not G.build_edges():
                warnings.warn(
                    "unable to construct correspondence graph", RuntimeWarning
                )
                return Correspondence.failure(graph_V=0, graph_E=0)
        except Exception as e:
            print(e, "construction")
            return Correspondence.failure(graph_V=0, graph_E=0)

        V = G.n_vertices
        E = G.n_edges
        dens = (2.0 * E) / (V * (V - 1))
        ub = min(len(K_sep_points), len(Q_sep_points))
        try:
            clq = np.array(
                G.get_max_clique(upper_bound=ub, use_dfs=False), dtype=np.uint64
            )
            corr = (
                Q_sep_points[(clq - 1) // len(K_sep_points)],
                K_sep_points[(clq - 1) % len(K_sep_points)],
            )
        except Exception:
            print(e, "correspondence")
            warnings.warn("unable to find maximum clique", RuntimeWarning)
            return Correspondence.failure(graph_V=0, graph_E=0)

        answer = Correspondence.success(
            Q_corr=corr[0],
            K_corr=corr[1],
            ub=ub,
            ratio=100 * len(corr[0]) / ub,
            graph_V=G.n_vertices,
            graph_E=G.n_edges,
        )
        del G
        # print("clique size is", answer["size"])
        return answer


CORRESPONDER_MAP = {
    x._extname_: x for x in Corresponder.__subclasses__() if x._extname_ != "dummy"
}
