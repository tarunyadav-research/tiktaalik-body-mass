"""
Body mass of Tiktaalik roseae: Monte Carlo estimates with explicit uncertainty.

Method B (primary, Tiktaalik-specific):
    M = C * L * w * d = C * (w/L)^2 * (d/w) * L^3
    w, d  : maximum body width and depth, from measurements of the holotype NUFV 108
            (Hohn-Schulte et al. 2013, Table 1).
    C     : empirical mass coefficient M / (L w d), measured on 157 extant
            fishes of 7 species with known mass, width and depth
            (Brofeldt 1917 "fishcatch" data; data/fishcatch.dat).
    Tested by leave-one-species-out prediction on the same data. None of these
    fishes has Tiktaalik's flattened shape, so C is applied outside the tested range.

Method A (reference, generic fish):
    W = a L^b from extant analogue fishes (FishBase Bayesian LWR), with
    correlated uncertainty in (log a, b). It answers "what would a typical
    fish of this length weigh?", which is the comparison against which
    Tiktaalik's broad body is judged.

If the factors of Method B are independent, as the primary analysis assumes,
Var(ln M) splits exactly into per-factor terms (first-order Sobol indices).

Run:  python model.py > results/report.txt        (writes results/ and figures/)
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 20261102
N = 200_000
HERE = Path(__file__).parent
DATA = HERE / "data" / "fishcatch.dat"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

# Body measurements from Hohn-Schulte et al. (2013, Historical Biology 25:2,
# Table 1), in metres. Total length = skull (240 mm) + trunk (560 mm); the
# trunk is modelled to a point, i.e. without a caudal fin. The table is headed
# "NUFV 109", but its footnote states that skull dimensions and body length
# "were taken from the original publication (Daeschler et al. 2006)", whose
# skeletal reconstruction (Fig. 6) is based on the holotype NUFV 108, the only
# specimen with a preserved trunk and pelvis. Their humerus (30 mm) also fits
# NUFV 108, half the size of NUFV 109 (68 mm; Stewart et al. 2020). We
# therefore attribute these proportions to NUFV 108.
PROPORTIONS = dict(L=0.800, skull_L=0.240, trunk_w=0.180, trunk_d=0.120, skull_w=0.190)

# The 0.80 m reconstruction assumed ~45 presacral vertebrae (Daeschler et al.
# 2006); muCT places the pelvis at vertebrae 31-32 (Stewart et al. 2024).
# Shortening the trunk in proportion gives an alternative holotype length.
PRESACRAL_OLD, PRESACRAL_NEW = 45, 31
L_REVISED = PROPORTIONS["skull_L"] + (PROPORTIONS["L"] - PROPORTIONS["skull_L"]) * PRESACRAL_NEW / PRESACRAL_OLD

# Specimen sizes relative to the holotype: NUFV 108 has the smallest and NUFV
# 109 the largest pectoral fin, "a twofold difference in size" (Shubin et al.
# 2006); NUFV 109 is 1.34 x NUFV 110 by humerus length (Stewart et al. 2020).
SIZE_RATIO = {"NUFV 110": 2.0 / 1.34, "NUFV 109": 2.0}
SIZE_RATIO_CV = 0.10
HOLOTYPE_DEPTH_CRUSH = 1.25  # sensitivity: NUFV 108 is dorsoventrally crushed (Lemberg 2018)
HOHN_SCHULTE_MASS_KG = 6.94  # their volumetric estimate, density 1.0

# Uncertainty of the proportions (single specimen, possible taphonomic
# distortion; dorsoventral compaction would reduce depth).
W_OVER_L_CV = 0.10
D_OVER_W_CV = 0.15

# Fishcatch species used for C. The two bream species are deep-bodied and
# laterally compressed (depth/width ~2.8), the opposite of Tiktaalik, and are
# excluded from the primary analysis (reported as a sensitivity case).
FISHCATCH_SPECIES = {1: "bream", 2: "whitefish", 3: "roach", 4: "silver bream",
                     5: "smelt", 6: "pike", 7: "perch"}
C_SPECIES = (2, 3, 5, 6, 7)

# Extant analogues for Method A. FishBase Bayesian length-weight estimates
# (Froese, Thorson & Reyes 2014). W in g, L in cm total length; 95% intervals.
# "basis" records whether the estimate comes from studies of that species or
# from a body-shape group prior (less informative about the species itself).
ANALOGUES = {
    "Neoceratodus forsteri": dict(a=0.00389, a_lo=0.00180, a_hi=0.00842, b=3.12, b_lo=2.94, b_hi=3.30,
                                  basis="body-shape group", max_L_cm=170, max_W_kg=40.0),
    "Protopterus aethiopicus": dict(a=0.00380, a_lo=0.00153, a_hi=0.00946, b=3.10, b_lo=2.88, b_hi=3.32,
                                    basis="(sub)family-body shape group", max_L_cm=None, max_W_kg=None),
    "Latimeria chalumnae": dict(a=0.01202, a_lo=0.00823, a_hi=0.01756, b=3.03, b_lo=2.93, b_hi=3.13,
                                basis="species", max_L_cm=168, max_W_kg=95.0),
    "Atractosteus spatula": dict(a=0.00229, a_lo=0.00092, a_hi=0.00573, b=3.19, b_lo=2.98, b_hi=3.40,
                                 basis="(sub)family-body shape group", max_L_cm=260, max_W_kg=137.0),
}
RHO_AB = -0.95  # assumed correlation of log a and b; sensitivity -0.80, -0.99

# Reference sizes: (median length in m, length CV, median w/L, basis).
# Two holotype scenarios; larger specimens scale isometrically from each.
W_OVER_L_PUB = PROPORTIONS["trunk_w"] / PROPORTIONS["L"]
W_OVER_L_REV = PROPORTIONS["trunk_w"] / L_REVISED
SCALED_CV = float(np.hypot(0.05, SIZE_RATIO_CV))
SIZES = {
    "NUFV 108, published reconstruction": (
        PROPORTIONS["L"], 0.05, W_OVER_L_PUB,
        "skull + trunk of the Daeschler et al. (2006) reconstruction (Hohn-Schulte et al. 2013); "
        "matches 0.78 m from the pelvis (Shubin et al. 2014)"),
    "NUFV 108, revised trunk": (
        L_REVISED, 0.08, W_OVER_L_REV,
        "trunk shortened by 31/45 presacral vertebrae (Stewart et al. 2024)"),
}
for spec, ratio in SIZE_RATIO.items():
    for scen in ("published reconstruction", "revised trunk"):
        L0, _, wl, _ = SIZES[f"NUFV 108, {scen}"]
        SIZES[f"{spec}, {scen}"] = (L0 * ratio, SCALED_CV, wl,
                                    f"{ratio:.2f} x NUFV 108 (Shubin et al. 2006; Stewart et al. 2020)")
SIZES["largest, as reported"] = (2.75, 0.15, W_OVER_L_PUB,
                                 "upper end of the 1.25-2.75 m range (Stewart et al. 2020); basis unstated")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_fishcatch():
    """Return array of (species, mass_kg, L1_m, L3_m, depth_m, width_m)."""
    rows = []
    for line in DATA.read_text().splitlines():
        p = line.split()
        if len(p) < 8 or "NA" in p[2:8]:
            continue
        sp = int(p[1])
        W, L1, L2, L3, Hpct, Wpct = map(float, p[2:8])
        if W <= 0:
            continue
        rows.append((sp, W / 1000, L1 / 100, L3 / 100, Hpct * L3 / 1e4, Wpct * L3 / 1e4))
    return np.array(rows)


def mass_coefficient(fish):
    """C = M / (L1 w d) in kg/m^3. L1 (nose to start of tail) matches the L of the Tiktaalik proportions."""
    return fish[:, 1] / (fish[:, 2] * fish[:, 4] * fish[:, 5])


def sample_C(rng, fish, n, species=C_SPECIES):
    """Species-balanced bootstrap of C: pick a species, then a fish of that species."""
    C = mass_coefficient(fish)
    pools = [C[fish[:, 0] == s] for s in species]
    pick = rng.integers(len(pools), size=n)
    out = np.empty(n)
    for i, pool in enumerate(pools):
        m = pick == i
        out[m] = rng.choice(pool, size=m.sum())
    return out


def sample_tail_ratio(rng, fish, n, species=C_SPECIES):
    """L1 / L3 (body without caudal fin / total length), species-balanced bootstrap."""
    r = fish[:, 2] / fish[:, 3]
    pools = [r[fish[:, 0] == s] for s in species]
    pick = rng.integers(len(pools), size=n)
    out = np.empty(n)
    for i, pool in enumerate(pools):
        m = pick == i
        out[m] = rng.choice(pool, size=m.sum())
    return out


def validate_method_b(fish):
    """Leave-one-species-out: predict each fish from C pooled over the other species."""
    C = mass_coefficient(fish)
    out = {}
    pred = np.empty(len(fish))
    for s in FISHCATCH_SPECIES:
        test = fish[:, 0] == s
        train = np.isin(fish[:, 0], [t for t in C_SPECIES if t != s])
        species_medians = [np.median(C[fish[:, 0] == t]) for t in C_SPECIES if t != s]
        C_hat = np.median(species_medians)
        pred[test] = C_hat * fish[test, 2] * fish[test, 4] * fish[test, 5]
        err = pred[test] / fish[test, 1] - 1
        out[FISHCATCH_SPECIES[s]] = dict(n=int(test.sum()), median_pct_error=100 * float(np.median(err)),
                                         median_abs_pct_error=100 * float(np.median(np.abs(err))),
                                         used_in_C=s in C_SPECIES)
    return out, pred


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def lognormal(rng, median, cv, n):
    sigma = np.sqrt(np.log(1 + cv**2))
    return median * np.exp(sigma * rng.standard_normal(n))


def sample_lwr(rng, p, n, rho=RHO_AB):
    """Correlated draws of (log10 a, b) from 95% intervals."""
    b_sd = (p["b_hi"] - p["b_lo"]) / (2 * 1.96)
    la_sd = (np.log10(p["a_hi"]) - np.log10(p["a_lo"])) / (2 * 1.96)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    b = p["b"] + b_sd * z1
    log10a = np.log10(p["a"]) + la_sd * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    return log10a, b


def method_a(rng, L_m, rho=RHO_AB, species=None):
    """Mass (kg) of a generic analogue fish of length L. Pools analogues equally unless one is named."""
    n = L_m.size
    if species is not None:
        log10a, b = sample_lwr(rng, ANALOGUES[species], n, rho)
    else:
        names = list(ANALOGUES)
        pick = rng.integers(len(names), size=n)
        log10a, b = np.empty(n), np.empty(n)
        for i, name in enumerate(names):
            m = pick == i
            log10a[m], b[m] = sample_lwr(rng, ANALOGUES[name], m.sum(), rho)
    return 10 ** (log10a + b * np.log10(100 * L_m)) / 1000.0


def sample_shape(rng, fish, n, species=C_SPECIES, w_over_L=W_OVER_L_PUB):
    return {
        "C": sample_C(rng, fish, n, species),
        "w_over_L": lognormal(rng, w_over_L, W_OVER_L_CV, n),
        "d_over_w": lognormal(rng, PROPORTIONS["trunk_d"] / PROPORTIONS["trunk_w"], D_OVER_W_CV, n),
    }


def method_b(L_m, f):
    return f["C"] * f["w_over_L"] ** 2 * f["d_over_w"] * L_m**3


def log_variance_shares(L_m, f):
    """Exact decomposition of Var(ln M) for Method B (independent factors)."""
    parts = {
        "length (cubed)": 9 * np.var(np.log(L_m)),
        "width / length (squared)": 4 * np.var(np.log(f["w_over_L"])),
        "depth / width": np.var(np.log(f["d_over_w"])),
        "mass coefficient C": np.var(np.log(f["C"])),
    }
    total = sum(parts.values())
    return {k: v / total for k, v in parts.items()}, total


def summarize(x):
    q = np.percentile(x, [2.5, 25, 50, 75, 97.5])
    return dict(median=q[2], ci95=(q[0], q[4]), iqr=(q[1], q[3]), mean=float(np.mean(x)))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    fish = load_fishcatch()
    C_all = mass_coefficient(fish)

    out = {"seed": SEED, "n_draws": N, "rho_ab": RHO_AB, "proportions": PROPORTIONS,
           "sizes": {k: dict(L_m=v[0], cv=v[1], w_over_L=v[2], basis=v[3]) for k, v in SIZES.items()},
           "C_by_species": {FISHCATCH_SPECIES[s]: dict(n=int((fish[:, 0] == s).sum()),
                                                       median=float(np.median(C_all[fish[:, 0] == s])))
                            for s in FISHCATCH_SPECIES},
           "estimates": {}}

    validation, pred = validate_method_b(fish)
    out["validation_loso"] = validation
    r = fish[:, 2] / fish[:, 3]
    out["tail_ratio_L1_over_L3"] = {FISHCATCH_SPECIES[s]: float(np.median(r[fish[:, 0] == s])) for s in C_SPECIES}

    # Check against Hohn-Schulte et al. (2013) at their own geometry.
    fixed = sample_C(rng, fish, N)
    hs = fixed * PROPORTIONS["L"] * PROPORTIONS["trunk_w"] * PROPORTIONS["trunk_d"]
    out["fixed_geometry"] = summarize(hs)

    for label, (L0, cv, wl, _) in SIZES.items():
        L = lognormal(rng, L0, cv, N)
        mA = method_a(rng, L)
        # Method A expects total length; our lengths exclude a caudal fin.
        mA_tail = method_a(rng, L / sample_tail_ratio(rng, fish, N))
        f = sample_shape(rng, fish, N, w_over_L=wl)
        mB = method_b(L, f)
        shares, var_total = log_variance_shares(L, f)
        out["estimates"][label] = {
            "L_median_m": L0,
            "length_cv": cv,
            "w_over_L_median": wl,
            "method_B_tiktaalik": summarize(mB),
            "method_A_generic_fish": summarize(mA),
            "method_A_tail_adjusted": summarize(mA_tail),
            "method_A_by_species": {s: summarize(method_a(rng, L, species=s)) for s in ANALOGUES},
            "ratio_B_over_A_median": float(np.median(mB) / np.median(mA)),
            "ratio_B_over_A_tail_median": float(np.median(mB) / np.median(mA_tail)),
            "method_B_var_ln_mass": var_total,
            "method_B_variance_shares": shares,
        }

    # Sensitivities (NUFV 108).
    L = lognormal(rng, *SIZES["NUFV 108, published reconstruction"][:2], N)
    out["sensitivity_nufv108"] = {
        "C from pike only": summarize(method_b(L, sample_shape(rng, fish, N, species=(6,)))),
        "C including breams": summarize(method_b(L, sample_shape(rng, fish, N, species=tuple(FISHCATCH_SPECIES)))),
        # Swap the width only: depth stays 120 mm, so d/w must change with w.
        "skull width instead of trunk width": summarize(
            method_b(L, {**sample_shape(rng, fish, N),
                         "w_over_L": lognormal(rng, PROPORTIONS["skull_w"] / PROPORTIONS["L"], W_OVER_L_CV, N),
                         "d_over_w": lognormal(rng, PROPORTIONS["trunk_d"] / PROPORTIONS["skull_w"],
                                               D_OVER_W_CV, N)})),
        # NUFV 108 is dorsoventrally crushed (Lemberg 2018): depth 25% greater.
        "depth +25% (crushing)": summarize(
            method_b(L, {**sample_shape(rng, fish, N),
                         "d_over_w": HOLOTYPE_DEPTH_CRUSH * lognormal(
                             rng, PROPORTIONS["trunk_d"] / PROPORTIONS["trunk_w"], D_OVER_W_CV, N)})),
    }
    # Same caudal-fin conversion as the main Method A results.
    L = lognormal(rng, *SIZES["largest, as reported"][:2], N)
    out["rho_sensitivity_method_A_largest_reported"] = {
        str(r): summarize(method_a(rng, L / sample_tail_ratio(rng, fish, N), rho=r))
        for r in (-0.80, -0.95, -0.99)}

    # Method A calibration: analogues at their own maximum length vs record weight.
    out["method_A_analogue_checks"] = {
        s: dict(max_L_cm=p["max_L_cm"], published_max_kg=p["max_W_kg"],
                predicted=summarize(method_a(rng, np.full(N, p["max_L_cm"] / 100), species=s)))
        for s, p in ANALOGUES.items() if p["max_L_cm"]}

    out["robustness"] = robustness_checks(fish, validation)

    with open(RESULTS / "estimates.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)

    make_figures(rng, fish, pred, out)
    print_report(out)


def robustness_checks(fish, validation):
    """Checks added in revision. A separate random stream keeps all other results unchanged."""
    rng = np.random.default_rng(SEED + 1)
    res = {}
    d_w = PROPORTIONS["trunk_d"] / PROPORTIONS["trunk_w"]
    L = lognormal(rng, *SIZES["NUFV 108, published reconstruction"][:2], N)

    # 2.75 m read as total length: remove a caudal fin at Brofeldt proportions.
    ratio = sample_tail_ratio(rng, fish, N)
    L_tot = lognormal(rng, 2.75, 0.15, N)
    res["2.75 m as total length"] = summarize(method_b(L_tot * ratio, sample_shape(rng, fish, N)))
    res["tail_ratio_median"] = float(np.median(ratio))

    # Crushing that conserves cross-section area: depth x k, width / k leaves w*d unchanged.
    k = HOLOTYPE_DEPTH_CRUSH
    f = sample_shape(rng, fish, N)
    res["crushing, area conserved"] = summarize(
        method_b(L, {**f, "w_over_L": f["w_over_L"] / k, "d_over_w": f["d_over_w"] * k**2}))

    # Correlated errors in width and depth. Crushing implies a negative correlation.
    res["width-depth correlation"] = {}
    s1, s2 = np.sqrt(np.log(1 + W_OVER_L_CV**2)), np.sqrt(np.log(1 + D_OVER_W_CV**2))
    for rho in (-0.5, 0.5):
        z1, z2 = rng.standard_normal(N), rng.standard_normal(N)
        f = {"C": sample_C(rng, fish, N),
             "w_over_L": W_OVER_L_PUB * np.exp(s1 * z1),
             "d_over_w": d_w * np.exp(s2 * (rho * z1 + np.sqrt(1 - rho**2) * z2))}
        res["width-depth correlation"][str(rho)] = summarize(method_b(L, f))

    # How far Tiktaalik lies outside the calibration shapes.
    C = mass_coefficient(fish)
    rows = []
    for s, name in FISHCATCH_SPECIES.items():
        m = fish[:, 0] == s
        rows.append(dict(species=name, C=float(np.median(C[m])),
                         d_over_w=float(np.median(fish[m, 4] / fish[m, 5])),
                         w_over_L1=float(np.median(fish[m, 5] / fish[m, 2]))))
    res["fish_shapes"] = rows

    # Rough extrapolation of C across species to Tiktaalik's depth/width (7 points).
    x, y = np.log([r["d_over_w"] for r in rows]), np.log([r["C"] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    C_trend = float(np.exp(intercept + slope * np.log(d_w)))
    C_med = float(np.median(sample_C(rng, fish, N)))
    res["C_trend"] = dict(slope=float(slope), r=float(np.corrcoef(x, y)[0, 1]), C_at_tiktaalik=C_trend,
                          C_primary_median=C_med, mass_ratio=C_trend / C_med)
    # The same fit without the breams, which cluster at high depth/width.
    keep = [i for i, s in enumerate(FISHCATCH_SPECIES) if s in C_SPECIES]
    slope5, intercept5 = np.polyfit(x[keep], y[keep], 1)
    C_trend5 = float(np.exp(intercept5 + slope5 * np.log(d_w)))
    res["C_trend_five_species"] = dict(slope=float(slope5), r=float(np.corrcoef(x[keep], y[keep])[0, 1]),
                                       C_at_tiktaalik=C_trend5, mass_ratio=C_trend5 / C_med)

    # Validation across all seven species, without the shape-based exclusion.
    errs = [v["median_abs_pct_error"] for v in validation.values()]
    res["validation_all_species"] = dict(min=min(errs), max=max(errs), median=float(np.median(errs)))
    return res


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(rng, fish, pred, out):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colA, colB = "#2b6cb0", "#c05621"

    # Fig 1: validation of Method B on extant fishes (leave one species out).
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    markers = "osD^vPX"
    for i, (s, name) in enumerate(FISHCATCH_SPECIES.items()):
        m = fish[:, 0] == s
        hollow = s not in C_SPECIES
        ax.scatter(fish[m, 1], pred[m], s=22, marker=markers[i], label=name + (" (excluded from C)" if hollow else ""),
                   facecolors="none" if hollow else None, edgecolors="#4a5568" if hollow else None, alpha=0.85)
    lim = [0.005, 3]
    ax.plot(lim, lim, color="#1a202c", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Measured mass (kg)")
    ax.set_ylabel("Predicted mass, Method B (kg)")
    ax.set_title("Method B on 157 extant fishes (leave one species out)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_validation.png", dpi=200)
    plt.close(fig)

    # Fig 2: mass vs length with 95% bands.
    Ls = np.linspace(0.5, 3.0, 51)
    bands = {"A": [], "B": []}
    for L0 in Ls:
        L = np.full(20_000, L0)
        bands["A"].append(np.percentile(method_a(rng, L / sample_tail_ratio(rng, fish, 20_000)),
                                        [2.5, 50, 97.5]))
        bands["B"].append(np.percentile(method_b(L, sample_shape(rng, fish, 20_000)), [2.5, 50, 97.5]))
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for key, col, name in (("B", colB, "B: Tiktaalik proportions (NUFV 108 reconstruction)"),
                           ("A", colA, "A: generic fish, caudal fin added")):
        q = np.array(bands[key])
        ax.fill_between(Ls, q[:, 0], q[:, 2], color=col, alpha=0.18, lw=0)
        ax.plot(Ls, q[:, 1], color=col, lw=2, label=name)
    ax.set_yscale("log")
    short = {"NUFV 108, published reconstruction": "NUFV 108", "NUFV 110, published reconstruction": "NUFV 110",
             "NUFV 109, published reconstruction": "NUFV 109", "largest, as reported": "2.75 m, no caudal fin"}
    marks = [(SIZES[label][0], text) for label, text in short.items()]
    marks.append((2.75 * out["robustness"]["tail_ratio_median"], "2.75 m total length"))
    for L0, text in marks:
        ax.axvline(L0, color="#a0aec0", lw=0.8, ls="--")
        ax.text(L0 + 0.03, 0.03, text, rotation=90, va="bottom", ha="left", fontsize=8,
                color="#4a5568", transform=ax.get_xaxis_transform())
    ax.set_xlabel("Body length without caudal fin (m)")
    ax.set_ylabel("Body mass (kg)")
    ax.set_title("Mass as a function of length (median, 95% band)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_mass_vs_length.png", dpi=200)
    plt.close(fig)

    # Fig 4: variance decomposition of ln M (Method B) for two sizes.
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    names = list(out["estimates"]["NUFV 108, published reconstruction"]["method_B_variance_shares"])
    y = np.arange(len(names))
    for j, (label, col) in enumerate((("NUFV 108, published reconstruction", colB), ("largest, as reported", "#7b341e"))):
        vals = [100 * out["estimates"][label]["method_B_variance_shares"][k] for k in names]
        ax.barh(y + (0.2 if j == 0 else -0.2), vals, height=0.38, color=col,
                label=f"{label.split(',')[0]} (length CV {round(100 * SIZES[label][1])}%)")
        for yy, v in zip(y, vals):
            ax.text(v + 1, yy + (0.2 if j == 0 else -0.2), f"{v:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Share of Var(ln mass), %")
    ax.set_xlim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("What drives the uncertainty (Method B)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_variance_shares.png", dpi=200)
    plt.close(fig)

    # Fig 5: Tiktaalik lies outside the calibration fishes in shape, and C varies with shape.
    rb = out["robustness"]
    d_w = PROPORTIONS["trunk_d"] / PROPORTIONS["trunk_w"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.9))
    for i, (s, name) in enumerate(FISHCATCH_SPECIES.items()):
        m = fish[:, 0] == s
        hollow = s not in C_SPECIES
        a1.scatter(fish[m, 4] / fish[m, 5], fish[m, 5] / fish[m, 2], s=14, marker=markers[i], label=name,
                   facecolors="none" if hollow else None, edgecolors="#4a5568" if hollow else None, alpha=0.7)
    a1.scatter([d_w, d_w], [W_OVER_L_PUB, W_OVER_L_REV], s=120, marker="*", color="#1a202c", zorder=3)
    a1.annotate("Tiktaalik, published", (d_w, W_OVER_L_PUB), xytext=(8, -4), textcoords="offset points", fontsize=8)
    a1.annotate("Tiktaalik, revised trunk", (d_w, W_OVER_L_REV), xytext=(8, -4), textcoords="offset points", fontsize=8)
    a1.set_xscale("log")
    a1.set_xticks([0.5, 1, 2, 3], ["0.5", "1", "2", "3"])
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlim(0.45, 4)
    a1.set_xlabel("Depth / width")
    a1.set_ylabel("Width / length (no caudal fin)")
    a1.set_title("A. Body shape", loc="left", fontsize=10)
    a1.legend(frameon=False, fontsize=7, loc="upper right", markerscale=1.2)

    rows = rb["fish_shapes"]
    xs = np.array([r["d_over_w"] for r in rows])
    ys = np.array([r["C"] for r in rows])
    for i, r in enumerate(rows):
        hollow = list(FISHCATCH_SPECIES)[i] not in C_SPECIES
        a2.scatter(r["d_over_w"], r["C"], s=36, marker=markers[i], facecolors="none" if hollow else None,
                   edgecolors="#4a5568" if hollow else None)
    grid = np.linspace(np.log(0.6), np.log(3.0), 50)
    keep = [i for i, s in enumerate(FISHCATCH_SPECIES) if s in C_SPECIES]
    for idx, ls, lab in ((list(range(len(rows))), "-", "fit, 7 species"), (keep, "--", "fit, 5 species")):
        slope, icpt = np.polyfit(np.log(xs[idx]), np.log(ys[idx]), 1)
        a2.plot(np.exp(grid), np.exp(icpt + slope * grid), color="#718096", lw=1.2, ls=ls, label=lab)
    a2.axvspan(xs.min(), xs.max(), color="#edf2f7", lw=0, zorder=0)
    C_hs = HOHN_SCHULTE_MASS_KG / (PROPORTIONS["L"] * PROPORTIONS["trunk_w"] * PROPORTIONS["trunk_d"])
    marks = ((rb["C_trend"]["C_primary_median"], "C used here (317)", "#c05621"),
             (C_hs, f"implied by Hohn-Schulte et al. ({C_hs:.0f})", "#2b6cb0"),
             (rb["C_trend"]["C_at_tiktaalik"], f"trend, 7 species ({rb['C_trend']['C_at_tiktaalik']:.0f})", "#1a202c"),
             (rb["C_trend_five_species"]["C_at_tiktaalik"],
              f"trend, 5 species ({rb['C_trend_five_species']['C_at_tiktaalik']:.0f})", "#1a202c"))
    for val, lab, col in marks:
        a2.scatter(d_w, val, s=40, marker="D", color=col, zorder=3)
        a2.annotate(lab, (d_w, val), xytext=(8, -3), textcoords="offset points", fontsize=7.5)
    a2.axvline(d_w, color="#a0aec0", lw=0.8, ls=":")
    a2.set_xscale("log")
    a2.set_xticks([0.5, 1, 2, 3], ["0.5", "1", "2", "3"])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlim(0.45, 4)
    a2.set_ylim(180, 720)
    a2.set_xlabel("Depth / width")
    a2.set_ylabel("Mass coefficient C (kg m$^{-3}$)")
    a2.set_title("B. C against body shape", loc="left", fontsize=10)
    a2.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_shape_and_C.png", dpi=200)
    plt.close(fig)

    # Fig 3: holotype estimates under each assumption.
    est = out["estimates"]
    sens = out["sensitivity_nufv108"]
    main_b = est["NUFV 108, published reconstruction"]["method_B_tiktaalik"]

    def scaled(d, k):
        return dict(median=d["median"] * k, ci95=(d["ci95"][0] * k, d["ci95"][1] * k))

    groups = {"primary": ("#2b6cb0", "o"), "sensitivity": ("#2f855a", "s"), "model form": ("#c05621", "D")}
    items = [
        ("Published reconstruction", main_b, "primary"),
        ("Revised trunk (31 vertebrae)", est["NUFV 108, revised trunk"]["method_B_tiktaalik"], "primary"),
        ("C from pike only", sens["C from pike only"], "sensitivity"),
        ("C including breams", sens["C including breams"], "sensitivity"),
        ("Skull width for trunk width", sens["skull width instead of trunk width"], "sensitivity"),
        ("Depth +25%", sens["depth +25% (crushing)"], "sensitivity"),
        ("Crushing, area conserved", rb["crushing, area conserved"], "sensitivity"),
        ("Width-depth correlation -0.5", rb["width-depth correlation"]["-0.5"], "sensitivity"),
        ("Width-depth correlation +0.5", rb["width-depth correlation"]["0.5"], "sensitivity"),
        ("C from trend, 7 species", scaled(main_b, rb["C_trend"]["mass_ratio"]), "model form"),
        ("C from trend, 5 species", scaled(main_b, rb["C_trend_five_species"]["mass_ratio"]), "model form"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    seen = set()
    for j, (lab, d, g) in enumerate(items):
        col, mk = groups[g]
        ax.plot(d["ci95"], [j, j], color=col, lw=2, solid_capstyle="round")
        ax.scatter(d["median"], j, s=44, marker=mk, color=col, zorder=3, edgecolors="white", linewidths=1.5,
                   label=None if g in seen else g)
        seen.add(g)
        ax.text(d["ci95"][1] * 1.06, j, f"{d['median']:.1f}", va="center", fontsize=8, color="#4a5568")
    ax.axvline(HOHN_SCHULTE_MASS_KG, color="#718096", lw=1, ls="--")
    ax.text(HOHN_SCHULTE_MASS_KG * 1.02, -0.9, "Hohn-Schulte et al. (6.94 kg)", fontsize=7.5, color="#4a5568")
    ax.set_yticks(range(len(items)), [it[0] for it in items], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_ylim(len(items) - 0.4, -1.4)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1.5, 45)
    ax.set_xlabel("Holotype body mass (kg), median and 95% interval")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_holotype_estimates.png", dpi=200)
    plt.close(fig)


def print_report(out):
    def fmt(d):
        return f"{d['median']:.2f} kg (95% {d['ci95'][0]:.2f}-{d['ci95'][1]:.2f})"

    print("C by species:", {k: round(v["median"]) for k, v in out["C_by_species"].items()})
    print("\nValidation (leave one species out):")
    for s, v in out["validation_loso"].items():
        print(f"  {s:12s} n={v['n']:3d} median err {v['median_pct_error']:+.1f}%  "
              f"median |err| {v['median_abs_pct_error']:.1f}%  in C: {v['used_in_C']}")
    print("\nNUFV 108 at fixed published geometry:", fmt(out["fixed_geometry"]),
          f"(Hohn-Schulte et al. 2013: {HOHN_SCHULTE_MASS_KG} kg)")
    for label, e in out["estimates"].items():
        print(f"\n== {label}: L = {e['L_median_m']:.2f} m (CV {e['length_cv']})")
        print("  Method B (Tiktaalik):", fmt(e["method_B_tiktaalik"]))
        print("  Method A (generic)  :", fmt(e["method_A_generic_fish"]))
        print("  Method A (+tail)    :", fmt(e["method_A_tail_adjusted"]))
        print(f"  B/A median ratio    : {e['ratio_B_over_A_median']:.2f} (tail-adjusted {e['ratio_B_over_A_tail_median']:.2f})")
        print("  Var shares (B)      :", {k: round(v, 3) for k, v in e["method_B_variance_shares"].items()})
    print("\nSensitivity (NUFV 108):")
    for k, d in out["sensitivity_nufv108"].items():
        print(f"  {k}: {fmt(d)}")
    print("\nrho sensitivity (Method A, largest reported):")
    for r, d in out["rho_sensitivity_method_A_largest_reported"].items():
        print(f"  rho={r}: {fmt(d)}")
    print("\nMethod A analogue checks:")
    for s, c in out["method_A_analogue_checks"].items():
        print(f"  {s}: L={c['max_L_cm']} cm, record {c['published_max_kg']} kg, predicted {fmt(c['predicted'])}")
    rb = out["robustness"]
    print("\nRobustness checks:")
    print("  2.75 m as total length:", fmt(rb["2.75 m as total length"]), f"(L1/L3 median {rb['tail_ratio_median']:.3f})")
    print("  crushing, area conserved:", fmt(rb["crushing, area conserved"]))
    for r, d in rb["width-depth correlation"].items():
        print(f"  width-depth rho={r}: {fmt(d)}")
    for r in rb["fish_shapes"]:
        print(f"  {r['species']:12s} C={r['C']:.0f} d/w={r['d_over_w']:.2f} w/L1={r['w_over_L1']:.3f}")
    t = rb["C_trend"]
    print(f"  C trend across species: slope {t['slope']:.2f}, r {t['r']:.2f}, C at Tiktaalik d/w {t['C_at_tiktaalik']:.0f}, "
          f"primary C {t['C_primary_median']:.0f}, mass ratio {t['mass_ratio']:.2f}")
    t = rb["C_trend_five_species"]
    print(f"  C trend, five species: slope {t['slope']:.2f}, r {t['r']:.2f}, C at Tiktaalik d/w {t['C_at_tiktaalik']:.0f}, "
          f"mass ratio {t['mass_ratio']:.2f}")
    v = rb["validation_all_species"]
    print(f"  validation, all seven species: median |err| {v['min']:.1f}-{v['max']:.1f}%, median of species {v['median']:.1f}%")


if __name__ == "__main__":
    main()
