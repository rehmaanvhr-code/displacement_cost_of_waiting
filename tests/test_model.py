import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model

DATA = os.path.join(os.path.dirname(model.__file__), "data")


def occ():
    return pd.read_csv(os.path.join(DATA, "occupations.csv"))


def test_adoption_level_ramps():
    p = dict(start_year=2, ramp_years=4)
    assert model.adoption_level(0, p) == 0
    assert 0 < model.adoption_level(3, p) < 1
    assert model.adoption_level(9, p) == 1


def test_weighted_gini_equal_incomes_is_zero():
    g = model.weighted_gini(np.array([5.0, 5.0, 5.0]), np.array([1, 2, 3]))
    assert abs(g) < 1e-9


def test_fast_scenario_income_exceeds_slow_in_total():
    o = occ()
    fast = model.simulate(o, model.FAST)
    slow = model.simulate(o, model.SLOW, early_adopters=True)
    df = model.cost_table(fast, slow, o)
    assert df["total_cost"].sum() > 0


def test_unexposed_occupation_pays_little():
    o = occ()
    fast = model.simulate(o, model.FAST)
    slow = model.simulate(o, model.SLOW, early_adopters=True)
    df = model.cost_table(fast, slow, o)
    assert df.loc["Carpenters", "cost_share_of_income"] < df.loc["Accounting clerks", "cost_share_of_income"]


def test_quintiles_cover_one_to_five():
    q = model.quintiles(occ())
    assert set(q.unique()) == {1, 2, 3, 4, 5}
