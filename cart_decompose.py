"""
Two-Phase Decomposition for Large CAR-T Instances
===================================================
Phase 1: LP relaxation → identify best 2 facilities
Phase 2: Fix facilities, solve patient routing as smaller MIP

This effectively does Benders-like decomposition on the facility
selection binary variables, dramatically reducing the MIP size.
"""

import os, time as _time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyomo.environ import *
from cart_fast import (parse_dat, build_fast, solve_model, extract_results,
                       DEFAULT_URGENCY, DEFAULT_PROCESS)


def solve_two_phase(data_file, tau=0.0, urgency_config=None,
                    process_config=None, solver_name='highs',
                    time_limit=300, mip_gap=0.05, random_seed=42):
    """
    Phase 1: Solve LP relaxation to find best facilities
    Phase 2: Fix facility selection, solve patient routing MIP
    """
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = DEFAULT_PROCESS

    data = parse_dat(data_file)

    # ── Phase 1: LP relaxation ──
    print('Phase 1: LP relaxation for facility selection...', flush=True)
    t0 = _time.time()
    model_lp = build_fast(data, tau=tau, urgency_config=urgency_config,
                          process_config=process_config, random_seed=random_seed)

    # Relax binary vars
    from pyomo.core.base.var import Var
    for v in model_lp.component_objects(Var, active=True):
        for idx in v:
            if v[idx].domain is Binary:
                v[idx].domain = NonNegativeReals
                v[idx].setub(1.0)

    solver = SolverFactory(solver_name)
    solver.options['time_limit'] = min(time_limit // 3, 60)
    results = solver.solve(model_lp, tee=False, load_solutions=False)
    ok = (results.solver.status == SolverStatus.ok and
          results.solver.termination_condition in
          (TerminationCondition.optimal, TerminationCondition.feasible))

    lp_time = _time.time() - t0

    if not ok:
        return {'solved': False, 'phase': 1, 'error': 'LP relaxation failed',
                'lp_time': round(lp_time, 1)}

    model_lp.solutions.load_from(results)

    # Find top facilities by E1 value
    fac_scores = {mi: value(model_lp.E1[mi]) for mi in data['m']}
    max_fac = int(process_config.get('max_facilities', 2))
    best_facs = sorted(fac_scores, key=fac_scores.get, reverse=True)[:max_fac]
    print(f'  LP selected facilities: {best_facs} (scores: {[round(fac_scores[f],2) for f in best_facs]})')
    print(f'  LP time: {lp_time:.1f}s')

    lp_obj = value(model_lp.obj)

    # ── Phase 2: Fix facilities, solve MIP ──
    print(f'Phase 2: MIP with fixed facilities {best_facs}...', flush=True)
    t0 = _time.time()
    model_mip = build_fast(data, tau=tau, urgency_config=urgency_config,
                           process_config=process_config, random_seed=random_seed)

    # Fix facility selection
    for mi in data['m']:
        if mi in best_facs:
            model_mip.E1[mi].fix(1)
        else:
            model_mip.E1[mi].fix(0)
            # Also fix linking vars
            for c in data['c']:
                model_mip.X1[c, mi].fix(0)
            for h in data['h']:
                model_mip.X2[mi, h].fix(0)

    build_time = _time.time() - t0

    t0 = _time.time()
    results2, ok2 = solve_model(model_mip, solver_name=solver_name,
                                time_limit=time_limit, mip_gap=mip_gap)
    solve_time = _time.time() - t0

    total_time = lp_time + build_time + solve_time

    if not ok2:
        # Try with different facility combination
        print('  Phase 2 failed. Trying next best facilities...')
        # Try all pairs
        from itertools import combinations
        for fac_pair in combinations(data['m'], max_fac):
            if set(fac_pair) == set(best_facs):
                continue
            model_retry = build_fast(data, tau=tau, urgency_config=urgency_config,
                                     process_config=process_config, random_seed=random_seed)
            for mi in data['m']:
                if mi in fac_pair:
                    model_retry.E1[mi].fix(1)
                else:
                    model_retry.E1[mi].fix(0)
                    for c in data['c']:
                        model_retry.X1[c, mi].fix(0)
                    for h in data['h']:
                        model_retry.X2[mi, h].fix(0)

            results_r, ok_r = solve_model(model_retry, solver_name=solver_name,
                                          time_limit=time_limit // 2, mip_gap=mip_gap)
            if ok_r:
                res = extract_results(model_retry, tau)
                res['solved'] = True
                res['method'] = 'two_phase_retry'
                res['facilities_tried'] = list(fac_pair)
                res['lp_time'] = round(lp_time, 1)
                res['build_time'] = round(build_time, 1)
                res['solve_time'] = round(_time.time() - t0, 1)
                return res

        return {'solved': False, 'phase': 2, 'error': 'No facility combination worked',
                'lp_time': round(lp_time, 1), 'solve_time': round(solve_time, 1)}

    res = extract_results(model_mip, tau)
    res['solved'] = True
    res['method'] = 'two_phase'
    res['lp_time'] = round(lp_time, 1)
    res['build_time'] = round(build_time, 1)
    res['solve_time'] = round(solve_time, 1)
    res['total_time'] = round(total_time, 1)
    res['lp_lower_bound'] = round(lp_obj, 2)
    res['selected_facilities'] = best_facs
    return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--time-limit', type=int, default=300)
    parser.add_argument('--gap', type=float, default=0.10)
    args = parser.parse_args()

    data_file = args.data or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N50.dat')
    r = solve_two_phase(data_file, tau=args.tau, time_limit=args.time_limit, mip_gap=args.gap)

    if r['solved']:
        st = 'ALL ON TIME' if r['all_on_time'] else f'{r["num_late"]} LATE'
        print(f"\n{'='*60}")
        print(f"RESULT: {st}")
        print(f"Cost: ${r['real_cost']:,.0f} | Avg TRT: {r['avg_trt']:.1f}d | Late: {r['num_late']}")
        print(f"Facilities: {r.get('selected_facilities', r.get('facilities_open'))}")
        print(f"LP time: {r['lp_time']}s | MIP solve: {r['solve_time']}s | Total: {r.get('total_time','?')}s")
    else:
        print(f"\nFAILED at phase {r.get('phase','?')}: {r.get('error','?')}")
