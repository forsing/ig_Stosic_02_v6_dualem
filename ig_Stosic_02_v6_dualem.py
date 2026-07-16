from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
dual (e/m) veza — Amari: 
dve veze, histerezis između njih.

Na kategoričkoj porodici (dijagonala):
  m-veza:  Γm ≈ 0  → Euler: p' = p + η v
  e-veza:  Γe_ii = −1/p_i  → v' = v − η Γe ⊙ v²
  (Fisher Levi-Civita je „sredina“; ovde razdvajamo e vs m)

Dva predikciona stanja p_m, p_e; skor blend:
  score = 0.5·(p_m − p_glob) + 0.5·(p_e − p_glob)
Ban last; jedna next. CSV ceo, seed=39.
"""



import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
WINDOW = 100
ETA = 1.0
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def window_p(draws: np.ndarray, end: int, w: int = WINDOW) -> np.ndarray:
    start = max(0, end - w)
    chunk = draws[start:end]
    cnt = Counter(chunk.reshape(-1).tolist())
    n_slots = max(len(chunk) * FRONT_SELECT, 1)
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def project_simplex(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def step_m(p: np.ndarray, v: np.ndarray, eta: float = ETA) -> np.ndarray:
    """m-geodezija / flat u p: samo p + ηv."""
    return project_simplex(p + eta * v)


def step_e(p: np.ndarray, v: np.ndarray, eta: float = ETA) -> tuple[np.ndarray, np.ndarray]:
    """e-veza: Γe_ii = −1/p_i."""
    gamma_e = -1.0 / np.clip(p, 1e-18, None)
    p_new = project_simplex(p + eta * v)
    v_new = v - eta * gamma_e * (v ** 2)
    v_new = v_new - v_new.mean()
    return p_new, v_new


def number_scores(
    p_m: np.ndarray,
    p_e: np.ndarray,
    p_glob: np.ndarray,
    ban: set[int],
) -> dict[int, float]:
    out = {}
    for i in range(FRONT_N):
        n = i + 1
        if n in ban:
            out[n] = -1e18
        else:
            out[n] = float(0.5 * (p_m[i] - p_glob[i]) + 0.5 * (p_e[i] - p_glob[i]))
    return out


def _combo_fit(
    combo: list[int],
    score: dict[int, float],
    target_sum: float,
    pos_means: list[float],
    target_odd: float,
    ban: set[int],
) -> float:
    nums = sorted(combo)
    if any(x in ban for x in nums):
        return -1e18
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(draws: np.ndarray, score: dict[int, float], ban: set[int]) -> list[int]:
    ranked = sorted((n for n in score if n not in ban), key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))

    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, len(ranked) - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))

    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd, ban)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd, ban)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_02_v6(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    ban = set(int(x) for x in last.tolist())
    n = len(draws)
    p0 = window_p(draws, n - 1, WINDOW)
    p1 = window_p(draws, n, WINDOW)
    v = p1 - p0
    v = v - v.mean()

    p_m = step_m(p1, v, ETA)
    p_e, v_e = step_e(p1, v, ETA)
    p_glob = global_p(draws)
    score = number_scores(p_m, p_e, p_glob, ban)

    diverg = float(np.linalg.norm(p_m - p_e))

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {n} | seed={SEED} | WINDOW={WINDOW} | ETA={ETA} | ig_02_v6 dual e/m")
    print(f"last: {last.tolist()}")
    print()

    print("=== e vs m ===")
    print(
        {
            "v_l2": round(float(np.linalg.norm(v)), 6),
            "||p_m − p_e||": round(diverg, 6),
            "||p_m − p1||": round(float(np.linalg.norm(p_m - p1)), 6),
            "||p_e − p1||": round(float(np.linalg.norm(p_e - p1)), 6),
            "v_e_l2": round(float(np.linalg.norm(v_e)), 6),
        }
    )
    print()

    ranked = sorted(
        ((n_, float(score[n_])) for n_ in range(1, FRONT_N + 1) if n_ not in ban),
        key=lambda t: (-t[1], t[0]),
    )
    print("=== top12 skor (0.5 m + 0.5 e excess, ban last) ===")
    print([(n_, round(sc, 6)) for n_, sc in ranked[:12]])
    print()

    combo = predict_next(draws, score, ban)
    print("=== next (ig_02_v6 dual e/m) ===")
    print("next:", combo)
    print("overlap last:", sorted(set(combo) & ban))


if __name__ == "__main__":
    run_ig_02_v6()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | WINDOW=100 | ETA=1.0 | ig_02_v6 dual e/m
last: [4, 5, 6, 11, 12, 18, 28]

=== e vs m ===
{'v_l2': 0.004518, '||p_m − p_e||': 0.0, '||p_m − p1||': 0.004518, '||p_e − p1||': 0.004518, 'v_e_l2': 0.004512}

=== top12 skor (0.5 m + 0.5 e excess, ban last) ===
[(1, 0.014147), (29, 0.009478), (14, 0.007911), (27, 0.007097), (16, 0.00702), (24, 0.006959), (8, 0.006175), (38, 0.00404), (20, 0.001935), (34, 0.001659), (31, 0.001521), (9, 0.001152)]

=== next (ig_02_v6 dual e/m) ===
next: [8, x, 15, y, 24, z, 34]
overlap last: []
"""
