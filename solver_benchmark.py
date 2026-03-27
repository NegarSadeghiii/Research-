"""
Solver & Technique Benchmark for CAR-T Supply Chain Model
==========================================================
Tests multiple solvers, decomposition approaches, and model
reformulations to find the best solution strategy.

Techniques tested:
1. HiGHS (free, bundled with Pyomo)
2. GLPK (free, open source)
3. CBC (free, COIN-OR)
4. SCIP (free, academic license)
5. LP Relaxation (continuous relaxation for speed bound)
6. Reduced formulation (fewer binary vars)
7. Warm-start / MIP start hints
"""

import sys, os, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyomo.environ import *
from cart_concrete import (parse_dat, build_concrete, solve_model,
                            extract_results, DEFAULT_URGENCY, DEFAULT_PROCESS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def check_solvers():
    """Check which solvers are available."""
    solvers = {}
    for name in ['appsi_highs', 'highs', 'glpk', 'cbc', 'scip', 'gurobi', 'cplex']:
        try:
            s = SolverFactory(name)
            solvers[name] = s.available()
        except:
            solvers[name] = False
    return solvers


def install_solvers():
    """Try to install free solvers via pip."""
    packages = {
        'highspy': 'appsi_highs',
        'pyomo[optional]': 'glpk/cbc via pyomo extras',
    }
    results = {}
    for pkg, solver_desc in packages.items():
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg, '-q'],
                          capture_output=True, timeout=120)
            results[pkg] = 'installed'
        except:
            results[pkg] = 'failed'
    return results


def test_lp_relaxation(data_file, tau=0.0, process_config=None):
    """Solve LP relaxation (all binary->continuous) for a lower bound."""
    if process_config is None:
        process_config = DEFAULT_PROCESS

    data = parse_dat(data_file)
    model = build_concrete(data, tau=tau, process_config=process_config)

    # Relax all binary variables to continuous [0,1]
    from pyomo.core.base.var import Var
    for v in model.component_objects(Var, active=True):
        for idx in v:
            if v[idx].domain is Binary:
                v[idx].domain = NonNegativeReals
                v[idx].setub(1.0)

    t0 = time.time()
    solver = SolverFactory('appsi_highs')
    solver.options['time_limit'] = 30
    results = solver.solve(model, tee=False, load_solutions=False)
    elapsed = time.time() - t0

    ok = (results.solver.status == SolverStatus.ok and
          results.solver.termination_condition in
          (TerminationCondition.optimal, TerminationCondition.feasible))

    if ok:
        model.solutions.load_from(results)
        obj = value(model.obj)
        return {'method': 'LP Relaxation', 'solved': True, 'time': round(elapsed, 2),
                'obj_value': round(obj, 2), 'note': 'Lower bound (continuous relaxation)'}
    return {'method': 'LP Relaxation', 'solved': False, 'time': round(elapsed, 2)}


def test_solver(solver_name, data_file, tau=0.0, process_config=None,
                time_limit=120, mip_gap=0.05):
    """Test a specific solver."""
    if process_config is None:
        process_config = DEFAULT_PROCESS

    data = parse_dat(data_file)

    t0 = time.time()
    model = build_concrete(data, tau=tau, process_config=process_config)
    build_time = time.time() - t0

    solver = SolverFactory(solver_name)
    if not solver.available():
        return {'solver': solver_name, 'available': False}

    # Set solver options
    if 'highs' in solver_name:
        solver.options['time_limit'] = time_limit
        solver.options['mip_rel_gap'] = mip_gap
    elif solver_name == 'glpk':
        solver.options['tmlim'] = time_limit
        solver.options['mipgap'] = mip_gap
    elif solver_name == 'cbc':
        solver.options['seconds'] = time_limit
        solver.options['ratioGap'] = mip_gap
    elif solver_name == 'scip':
        solver.options['limits/time'] = time_limit
        solver.options['limits/gap'] = mip_gap
    elif solver_name == 'gurobi':
        solver.options['TimeLimit'] = time_limit
        solver.options['MIPGap'] = mip_gap

    t0 = time.time()
    try:
        results = solver.solve(model, tee=False, load_solutions=False)
        solve_time = time.time() - t0

        ok = (results.solver.status == SolverStatus.ok and
              results.solver.termination_condition in
              (TerminationCondition.optimal, TerminationCondition.feasible))

        result = {
            'solver': solver_name,
            'available': True,
            'solved': ok,
            'build_time': round(build_time, 2),
            'solve_time': round(solve_time, 2),
            'total_time': round(build_time + solve_time, 2),
            'termination': str(results.solver.termination_condition),
        }

        if ok:
            model.solutions.load_from(results)
            res = extract_results(model, tau)
            result.update({
                'cost': res['real_cost'],
                'avg_trt': res['avg_trt'],
                'num_late': res['num_late'],
                'total_lateness': res['total_lateness'],
                'all_on_time': res['all_on_time'],
            })
        return result
    except Exception as e:
        return {
            'solver': solver_name, 'available': True, 'solved': False,
            'error': str(e), 'solve_time': round(time.time() - t0, 2),
        }


def test_reduced_time_horizon(data_file, tau=0.0, process_config=None,
                               time_limit=120):
    """Test with tighter time horizon (less variables)."""
    if process_config is None:
        process_config = DEFAULT_PROCESS

    data = parse_dat(data_file)
    max_arrival = max(t for (_, _, t) in data['INC'].keys())
    # Use minimal time horizon: max_arrival + max_due + buffer
    max_due = max(DEFAULT_URGENCY[g]['base_due'] + DEFAULT_URGENCY[g]['tolerance']
                  for g in DEFAULT_URGENCY)
    tight_horizon = max_arrival + max_due + 5

    t0 = time.time()
    model = build_concrete(data, time_horizon=tight_horizon, tau=tau,
                           process_config=process_config)
    build_time = time.time() - t0

    t0 = time.time()
    results, ok = solve_model(model, time_limit=time_limit)
    solve_time = time.time() - t0

    result = {
        'method': f'Reduced horizon (T={tight_horizon})',
        'solved': ok,
        'build_time': round(build_time, 2),
        'solve_time': round(solve_time, 2),
        'total_time': round(build_time + solve_time, 2),
    }
    if ok:
        res = extract_results(model, tau)
        result.update({
            'cost': res['real_cost'],
            'avg_trt': res['avg_trt'],
            'num_late': res['num_late'],
            'total_lateness': res['total_lateness'],
        })
    return result


def test_relaxed_processing(data_file, tau=0.0, time_limit=120):
    """Test with reduced processing times (technology improvement scenario)."""
    configs = [
        {'name': 'Original (7+7)', 'tls': 1, 'tmfe': 7, 'tqc': 7, 'max_facilities': 2},
        {'name': 'Fast QC (7+5)', 'tls': 1, 'tmfe': 7, 'tqc': 5, 'max_facilities': 2},
        {'name': 'Fast Mfg (5+7)', 'tls': 1, 'tmfe': 5, 'tqc': 7, 'max_facilities': 2},
        {'name': 'Both Fast (5+5)', 'tls': 1, 'tmfe': 5, 'tqc': 5, 'max_facilities': 2},
        {'name': 'Orig + 3 Fac', 'tls': 1, 'tmfe': 7, 'tqc': 7, 'max_facilities': 3},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop('name')
        min_trt = cfg['tls'] + 1 + cfg['tmfe'] + cfg['tqc'] + 1
        r = test_solver('appsi_highs', data_file, tau=tau,
                        process_config=cfg, time_limit=time_limit)
        r['config_name'] = name
        r['min_trt'] = min_trt
        results.append(r)

    return results


def run_full_benchmark(data_file=None, tau=0.0, time_limit=120):
    """Run the complete benchmark suite."""
    if data_file is None:
        data_file = os.path.join(BASE_DIR, 'Data_N5.dat')

    print('='*80)
    print('CAR-T Supply Chain — Solver & Technique Benchmark')
    print('='*80)
    print(f'Data: {os.path.basename(data_file)}')
    print(f'Tau: {tau}')
    print(f'Time limit per run: {time_limit}s')
    print()

    # 1. Check available solvers
    print('1. AVAILABLE SOLVERS')
    print('-'*40)
    solvers = check_solvers()
    for name, avail in solvers.items():
        print(f'  {name:20s} {"YES" if avail else "no"}')
    print()

    # 2. Test each available solver
    print('2. SOLVER COMPARISON')
    print('-'*80)
    print(f'{"Solver":20s} {"Solved":>8} {"Build":>8} {"Solve":>8} {"Total":>8} {"Cost":>12} {"AvgTRT":>8} {"Late":>6}')
    print('-'*80)
    available = [s for s, a in solvers.items() if a]
    solver_results = []
    for s in available:
        r = test_solver(s, data_file, tau=tau, time_limit=time_limit)
        solver_results.append(r)
        if r.get('solved'):
            print(f"{s:20s} {'YES':>8} {r['build_time']:>7.1f}s {r['solve_time']:>7.1f}s {r['total_time']:>7.1f}s {r.get('cost',0):>12,.0f} {r.get('avg_trt',0):>8.2f} {r.get('num_late',0):>6}")
        else:
            term = r.get('termination', r.get('error', 'unknown'))[:20]
            print(f"{s:20s} {'NO':>8} {r.get('build_time',0):>7.1f}s {r.get('solve_time',0):>7.1f}s {'—':>8} {'—':>12} {'—':>8} {'—':>6}  ({term})")
    print()

    # 3. LP Relaxation (lower bound)
    print('3. LP RELAXATION (Lower Bound)')
    print('-'*40)
    lp = test_lp_relaxation(data_file, tau=tau)
    if lp['solved']:
        print(f"  Solved in {lp['time']}s — Obj: {lp['obj_value']:,.0f}")
    else:
        print(f"  Failed in {lp['time']}s")
    print()

    # 4. Reduced time horizon
    print('4. REDUCED TIME HORIZON')
    print('-'*40)
    rth = test_reduced_time_horizon(data_file, tau=tau, time_limit=time_limit)
    if rth['solved']:
        print(f"  {rth['method']}: Solved in {rth['total_time']}s — Cost: {rth['cost']:,.0f}, Avg TRT: {rth['avg_trt']:.1f}, Late: {rth['num_late']}")
    else:
        print(f"  {rth['method']}: Failed in {rth['total_time']}s")
    print()

    # 5. Processing time scenarios
    print('5. PROCESSING TIME SCENARIOS (Technology Impact)')
    print('-'*80)
    print(f'{"Config":25s} {"MinTRT":>7} {"Solved":>8} {"Time":>8} {"Cost":>12} {"AvgTRT":>8} {"Late":>6}')
    print('-'*80)
    proc_results = test_relaxed_processing(data_file, tau=tau, time_limit=time_limit)
    for r in proc_results:
        if r.get('solved'):
            print(f"{r['config_name']:25s} {r['min_trt']:>7} {'YES':>8} {r['solve_time']:>7.1f}s {r.get('cost',0):>12,.0f} {r.get('avg_trt',0):>8.2f} {r.get('num_late',0):>6}")
        else:
            print(f"{r['config_name']:25s} {r['min_trt']:>7} {'NO':>8} {r.get('solve_time',0):>7.1f}s {'—':>12} {'—':>8} {'—':>6}")
    print()

    # Summary
    print('='*80)
    print('SUMMARY & RECOMMENDATIONS')
    print('='*80)
    best_solver = min([r for r in solver_results if r.get('solved')],
                      key=lambda x: x['solve_time'], default=None)
    if best_solver:
        print(f"  Best solver: {best_solver['solver']} ({best_solver['solve_time']}s)")
    else:
        print("  No solver found a solution within time limit")

    feasible_procs = [r for r in proc_results if r.get('solved') and r.get('num_late', 0) == 0]
    if feasible_procs:
        print(f"  Feasible configs: {', '.join(r['config_name'] for r in feasible_procs)}")
    else:
        print("  No processing config achieved full feasibility at this tau")

    # Return structured results for UI
    return {
        'solvers': solver_results,
        'lp_relaxation': lp,
        'reduced_horizon': rth,
        'processing_scenarios': proc_results,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--time-limit', type=int, default=120)
    parser.add_argument('--install-solvers', action='store_true')
    args = parser.parse_args()

    if args.install_solvers:
        print('Installing solvers...')
        r = install_solvers()
        for pkg, status in r.items():
            print(f'  {pkg}: {status}')
        print()

    data = args.data or os.path.join(BASE_DIR, 'Data_N5.dat')
    run_full_benchmark(data_file=data, tau=args.tau, time_limit=args.time_limit)
