"""
Who bears the cost of slow AI adoption.

Two scenarios over a ten year horizon. In the fast scenario adoption starts
now and comes with reskilling, so exposed workers share in the productivity
gain and displacement is brief. In the slow scenario adoption is delayed,
early adopters in high wage exposed occupations capture gains anyway, and when
adoption finally arrives it comes as a shock with less reskilling and a long
spell of wage stagnation for exposed lower wage workers.

The cost of waiting is the income lost under the slow scenario relative to the
fast one, and the question is who pays it.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "outputs")

HORIZON = 10

FAST = dict(
    start_year=0,
    ramp_years=4,
    productivity_gain=0.25,   # output gain on exposed tasks once adopted
    worker_share=0.6,         # share of the gain that reaches wages with reskilling
    displacement_rate=0.08,   # share of exposed workers displaced at peak
    recovery_years=2,
    stagnation=0.0,
)

SLOW = dict(
    start_year=5,
    ramp_years=2,
    productivity_gain=0.25,
    worker_share=0.3,
    displacement_rate=0.18,
    recovery_years=4,
    stagnation=0.02,          # yearly real wage loss for fully exposed workers before adoption
)

EARLY_ADOPTER_WAGE = 75000    # occupations above this wage adopt tools on their own
REPLACEMENT_INCOME = 0.45     # share of wage a displaced worker keeps
TREND_GROWTH = 0.01


def adoption_level(year, p):
    if year < p["start_year"]:
        return 0.0
    return min(1.0, (year - p["start_year"] + 1) / p["ramp_years"])


def simulate(occ, p, early_adopters=False):
    """Return a DataFrame of yearly income per worker and employed share per occupation."""
    rows = []
    displaced = np.zeros(len(occ))
    for year in range(HORIZON + 1):
        a = adoption_level(year, p)
        exposure = occ["ai_exposure"].values
        wage = occ["median_wage"].values * (1 + TREND_GROWTH) ** year
        gain = p["productivity_gain"] * exposure * a * p["worker_share"]
        if early_adopters and a == 0:
            # high wage exposed occupations pick up tools before formal adoption
            early = (occ["median_wage"].values > EARLY_ADOPTER_WAGE) * exposure
            gain = gain + p["productivity_gain"] * early * 0.8 * min(1.0, year / 3)
        if p["stagnation"] > 0 and a == 0:
            wage = wage * (1 - p["stagnation"] * exposure) ** year
        # displacement rises with adoption and fades over the recovery period
        new_disp = p["displacement_rate"] * exposure * a
        displaced = np.maximum(displaced * (1 - 1.0 / p["recovery_years"]), new_disp)
        income = wage * (1 + gain) * (1 - displaced) + wage * REPLACEMENT_INCOME * displaced
        for i, name in enumerate(occ["occupation"]):
            rows.append(dict(year=year, occupation=name, income=income[i],
                             displaced=displaced[i]))
    return pd.DataFrame(rows)


def weighted_gini(values, weights):
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cum_w = np.cumsum(w)
    cum_vw = np.cumsum(v * w)
    cum_w = cum_w / cum_w[-1]
    cum_vw = cum_vw / cum_vw[-1]
    area = np.sum((cum_w[1:] - cum_w[:-1]) * (cum_vw[1:] + cum_vw[:-1])) / 2
    area += cum_w[0] * cum_vw[0] / 2
    return 1 - 2 * area


def gini_path(sim, occ):
    out = []
    for year, g in sim.groupby("year"):
        g = g.set_index("occupation").loc[occ["occupation"]]
        out.append(weighted_gini(g["income"].values, occ["employment"].values))
    return np.array(out)


def quintiles(occ):
    """Assign each occupation to a wage quintile weighted by employment."""
    o = occ.sort_values("median_wage").copy()
    cum = o["employment"].cumsum() / o["employment"].sum()
    o["quintile"] = np.minimum(5, np.ceil(cum * 5).astype(int))
    return o.set_index("occupation")["quintile"]


def cost_table(fast, slow, occ):
    f = fast.groupby("occupation")["income"].sum()
    s = slow.groupby("occupation")["income"].sum()
    cost = (f - s).rename("cost_per_worker")
    disp = slow.groupby("occupation")["displaced"].sum() - fast.groupby("occupation")["displaced"].sum()
    df = occ.set_index("occupation").join(cost).join(disp.rename("extra_displaced_years"))
    df["total_cost"] = df["cost_per_worker"] * df["employment"]
    df["cost_share_of_income"] = df["cost_per_worker"] / (df["median_wage"] * (HORIZON + 1))
    df["quintile"] = quintiles(occ)
    return df


def region_costs(df, regions):
    group_cost = df.groupby("group").apply(
        lambda g: g["total_cost"].sum() / g["employment"].sum())
    out = []
    for _, r in regions.iterrows():
        cost = sum(r[g] * group_cost.get(g, 0.0) for g in group_cost.index)
        out.append(dict(province=r["province"], cost_per_worker=cost))
    return pd.DataFrame(out).sort_values("cost_per_worker", ascending=False)


def main():
    os.makedirs(OUT, exist_ok=True)
    occ = pd.read_csv(os.path.join(DATA, "occupations.csv"))
    regions = pd.read_csv(os.path.join(DATA, "regions.csv"))

    fast = simulate(occ, FAST)
    slow = simulate(occ, SLOW, early_adopters=True)
    df = cost_table(fast, slow, occ)

    df["baseline_income"] = df["median_wage"] * (HORIZON + 1) * df["employment"]
    df["displaced_person_years"] = df["extra_displaced_years"] * df["employment"]
    byq = df.groupby("quintile").agg(total_cost=("total_cost", "sum"),
                                     employment=("employment", "sum"),
                                     baseline_income=("baseline_income", "sum"),
                                     displaced_person_years=("displaced_person_years", "sum"))
    byq["cost_per_worker"] = byq["total_cost"] / byq["employment"]
    byq["share_of_cost"] = byq["total_cost"] / byq["total_cost"].sum()
    byq["cost_pct_of_income"] = 100 * byq["total_cost"] / byq["baseline_income"]
    byq["share_of_displacement"] = byq["displaced_person_years"] / byq["displaced_person_years"].sum()

    g_fast = gini_path(fast, occ)
    g_slow = gini_path(slow, occ)
    reg = region_costs(df, regions)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(byq.index.astype(str), byq["cost_pct_of_income"])
    axes[0].set_xlabel("wage quintile (1 = lowest)")
    axes[0].set_ylabel("income lost, percent of ten year baseline income")
    axes[0].set_title("Cost of slow adoption by wage quintile")
    axes[1].bar(byq.index.astype(str), byq["share_of_displacement"])
    axes[1].set_xlabel("wage quintile (1 = lowest)")
    axes[1].set_ylabel("share of extra displacement person years")
    axes[1].set_title("Who absorbs the extra displacement")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cost_by_quintile.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(HORIZON + 1), g_fast, label="fast adoption with reskilling")
    ax.plot(range(HORIZON + 1), g_slow, label="slow adoption")
    ax.set_xlabel("year")
    ax.set_ylabel("employment weighted wage Gini")
    ax.set_title("Inequality path under each scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gini_paths.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(reg["province"], reg["cost_per_worker"] / 1000)
    ax.invert_yaxis()
    ax.set_xlabel("ten year income lost per worker, thousand dollars")
    ax.set_title("Cost of slow adoption by province")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "region_map_bar.png"), dpi=120)
    plt.close(fig)

    summary = pd.DataFrame([
        dict(metric="total ten year cost of waiting, billion dollars", value=round(df["total_cost"].sum() / 1e9, 2)),
        dict(metric="share of cost borne by bottom quintile", value=round(byq.loc[1, "share_of_cost"], 3)),
        dict(metric="share of cost borne by top quintile", value=round(byq.loc[5, "share_of_cost"], 3)),
        dict(metric="share of extra displacement, bottom two quintiles", value=round(byq.loc[1:2, "share_of_displacement"].sum(), 3)),
        dict(metric="cost as percent of income, quintile 2", value=round(byq.loc[2, "cost_pct_of_income"], 2)),
        dict(metric="cost as percent of income, quintile 5", value=round(byq.loc[5, "cost_pct_of_income"], 2)),
        dict(metric="bottom quintile employment share", value=round(byq.loc[1, "employment"] / byq["employment"].sum(), 3)),
        dict(metric="gini year 10, fast", value=round(g_fast[-1], 4)),
        dict(metric="gini year 10, slow", value=round(g_slow[-1], 4)),
        dict(metric="gini peak, slow", value=round(g_slow.max(), 4)),
        dict(metric="highest cost province", value=reg.iloc[0]["province"]),
    ])
    summary.to_csv(os.path.join(OUT, "scenario_summary.csv"), index=False)
    df.reset_index().round(1).to_csv(os.path.join(OUT, "occupation_costs.csv"), index=False)
    byq.round(3).to_csv(os.path.join(OUT, "quintile_costs.csv"))
    print(summary.to_string(index=False))
    print()
    print(byq.round(3).to_string())


if __name__ == "__main__":
    main()
