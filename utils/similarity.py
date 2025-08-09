from typing import Sequence, List, Tuple
import numpy as np

def _to_vector(v) -> np.ndarray:
    return np.asarray(v, dtype=float).ravel()

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cosine similarity between two 1-D vectors.
    Returns 0.0 if either vector is zero-length to avoid dividing by zero.
    """
    a = _to_vector(a)
    b = _to_vector(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

def cosine_similarities(query: Sequence[float], matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and each row in matrix.
    matrix can be a list of lists or a 2D numpy array (shape: N x D).
    Returns a 1D numpy array of length N with similarity scores.
    """
    q = _to_vector(query)
    M = np.asarray(matrix, dtype=float)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    if q.shape[0] != M.shape[1]:
        raise ValueError(f"Dim mismatch: query dim {q.shape[0]} vs matrix dim {M.shape[1]}")
    qnorm = np.linalg.norm(q)
    Mnorms = np.linalg.norm(M, axis=1)
    dots = M.dot(q)
    denom = qnorm * Mnorms
    # Avoid divide-by-zero: where denom==0 set similarity to 0.0
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    sims = dots / denom_safe
    sims = np.where(denom == 0.0, 0.0, sims)
    return sims

def top_n_similar(query: Sequence[float], matrix: Sequence[Sequence[float]], n: int = 3
                 ) -> List[Tuple[int, float]]:
    """
    Return top-n (index, score) pairs sorted by score desc.
    Uses argpartition for efficiency when n << len(matrix).
    """
    sims = cosine_similarities(query, matrix)
    N = sims.size
    if N == 0:
        return []
    n = min(n, N)
    if n == N:
        idx_sorted = np.argsort(-sims)
    else:
        # get n largest indices, then sort them descending
        part = np.argpartition(-sims, n-1)[:n]
        idx_sorted = part[np.argsort(-sims[part])]
    return [(int(i), float(sims[i])) for i in idx_sorted]
