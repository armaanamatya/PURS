"""Tests for the PLP schedule and Appendix-E calibration.

The Appendix-E derivation is a falsifiable number the paper hands us:
p_final ~= 0.146 for L=28, p_init=0, t_mid=0.5, r_0=0.45, target mean 0.30.
"""
import math

import numpy as np

from omnidrop.schedule import (
    PLPSchedule,
    calibrate_pfinal,
    calibrate_pfinal_paper_approx,
)


def test_sigmoid_midpoint_is_half():
    sch = PLPSchedule(p_init=0.0, p_final=0.5, L=28, t_mid=0.5, beta=20.0)
    # at l/L == t_mid the sigmoid is exactly 0.5 -> p_l = p_init + 0.5*(p_final-p_init)
    l = int(round(0.5 * 28))  # 14
    assert math.isclose(sch.pruning_ratio(l), 0.0 + 0.5 * (0.5 - 0.0), rel_tol=1e-9)


def test_penultimate_cap_no_prune_on_last_layer():
    sch = PLPSchedule(p_init=0.02, p_final=0.5, L=28)
    assert sch.pruning_ratio(27) == 0.0          # layer L-1 never prunes
    assert sch.pruning_ratio(26) > 0.0           # layer L-2 does


def test_appendix_E_paper_approx():
    # Closed-form derivation reproduces the paper's ~0.146.
    pf = calibrate_pfinal_paper_approx(target_mean=0.30, L=28, r0=0.45)
    assert abs(pf - 0.146) < 0.01, pf


def test_appendix_E_numeric_calibration_closes_to_target():
    # The exact numeric solver hits the 30% mean by construction...
    pf = calibrate_pfinal(target_mean=0.30, L=28, p_init=0.0, t_mid=0.5,
                          beta=20.0, r0=0.45)
    sch = PLPSchedule(0.0, pf, 28, 0.5, 20.0)
    assert abs(sch.mean_retained(0.45) - 0.30) < 1e-3
    # ...and the EXACT p_final (~0.206) is larger than the paper's Taylor /
    # linear approximation (0.146): the paper's closed form undershoots, so its
    # 0.146 actually realizes ~32.4% retention. See REPRODUCTION_NOTES.
    assert 0.18 < pf < 0.23, pf


def test_published_configs_realize_nominal_target():
    # The paper's adopted configs realize their nominal mean within ~0.02. The
    # paper claims the conservative 0.2 stays "strictly below 0.3"; the exact
    # mean is 0.302 -- essentially 30% (documented in REPRODUCTION_NOTES).
    sch_30 = PLPSchedule(0.0, 0.2, 28, 0.5, 20.0)
    assert abs(sch_30.mean_retained(0.45) - 0.30) < 0.02
    # 7B 20% -> (0.02, 0.5): mean retained ~ 0.217, within 0.02 of 20%.
    sch_20 = PLPSchedule(0.02, 0.5, 28, 0.5, 20.0)
    assert abs(sch_20.mean_retained(0.45) - 0.20) < 0.02


def test_num_to_prune_uses_current_count_floor():
    sch = PLPSchedule(0.0, 0.5, 28, 0.5, 20.0)
    p = sch.pruning_ratio(20)
    assert sch.num_to_prune(20, 100) == int(math.floor(p * 100))


def test_exp_variant_requires_positive_pinit():
    sch = PLPSchedule(p_init=0.0, p_final=0.5, L=28, kind="exp")
    try:
        sch.pruning_ratio(10)
        assert False, "expected ValueError for p_init=0 exp variant"
    except ValueError:
        pass
    # with p_init>0 it is monotone increasing toward p_final.
    sch2 = PLPSchedule(p_init=0.02, p_final=0.5, L=28, kind="exp")
    assert sch2.pruning_ratio(5) < sch2.pruning_ratio(20)
