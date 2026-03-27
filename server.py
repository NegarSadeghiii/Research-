"""
CAR-T Supply Chain — Integrated API Server
CP-SAT + HiGHS benchmark, scenario analysis, single experiments.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, json, threading, uuid, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

from cart_cpsat import (run_experiment as cpsat_run, DEFAULT_URGENCY,
                        DEFAULT_PROCESS, parse_dat)
from generate_small_data import generate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
jobs = {}


def _urg(raw):
    if not raw: return None
    return {g: {'fraction': raw[g].get('fraction', DEFAULT_URGENCY[g]['fraction']),
                'base_due': raw[g].get('base_due', DEFAULT_URGENCY[g]['base_due']),
                'tolerance': raw[g].get('tolerance', DEFAULT_URGENCY[g]['tolerance'])}
            for g in ('high','medium','low') if g in raw}

def _proc(raw):
    if not raw: return None
    return {k: raw.get(k, DEFAULT_PROCESS[k]) for k in DEFAULT_PROCESS}

def _ensure_data(n):
    f = os.path.join(BASE_DIR, f'Data_N{n}.dat')
    if not os.path.exists(f):
        generate(input_file=os.path.join(BASE_DIR,'Data200_profileA.dat'),
                 output_file=f, num_patients=n)
    return f


@app.route('/')
def index(): return send_from_directory('.', 'ui.html')

@app.route('/api/run', methods=['POST'])
def run_single():
    p = request.json or {}
    n = p.get('n', 25)
    _ensure_data(n)
    try:
        r = cpsat_run(tau=p.get('tau',0), data_file=os.path.join(BASE_DIR,f'Data_N{n}.dat'),
                      urgency_config=_urg(p.get('urgency')),
                      process_config=_proc(p.get('process')),
                      time_limit=p.get('time_limit',120))
        return jsonify(r)
    except Exception as e:
        return jsonify({'solved':False,'error':str(e)})

@app.route('/api/generate-data', methods=['POST'])
def gen_data():
    n = request.json.get('num_patients',10)
    _ensure_data(n)
    return jsonify({'ok':True,'n':n})

@app.route('/api/scenarios', methods=['POST'])
def run_scenarios():
    """Run multiple scenarios in parallel."""
    scens = request.json.get('scenarios', [])
    if not scens: return jsonify([])

    # Ensure all data files exist
    for s in scens:
        _ensure_data(s.get('n',25))

    def run_one(s):
        try:
            r = cpsat_run(
                tau=s.get('tau',0),
                data_file=os.path.join(BASE_DIR, f"Data_N{s.get('n',25)}.dat"),
                urgency_config=_urg(s.get('urgency')),
                process_config=_proc(s.get('process')),
                time_limit=s.get('time_limit',120))
            r['_label'] = s.get('label','')
            r['_idx'] = s.get('idx',0)
            return r
        except Exception as e:
            return {'solved':False,'error':str(e),'_label':s.get('label',''),'_idx':s.get('idx',0)}

    with ThreadPoolExecutor(max_workers=min(len(scens), os.cpu_count() or 4)) as ex:
        futures = {ex.submit(run_one, s): s.get('idx',i) for i,s in enumerate(scens)}
        results = {}
        for f in as_completed(futures):
            idx = futures[f]
            results[idx] = f.result()
    return jsonify([results[i] for i in sorted(results)])

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Compare CP-SAT vs HiGHS (Pyomo) across patient counts."""
    p = request.json or {}
    sizes = p.get('sizes', [5, 10, 15, 25])
    tau = p.get('tau', 0.0)
    time_limit = p.get('time_limit', 120)

    results = []
    for n in sizes:
        _ensure_data(n)
        df = os.path.join(BASE_DIR, f'Data_N{n}.dat')
        entry = {'n': n, 'tau': tau}

        # CP-SAT
        try:
            r_cp = cpsat_run(tau=tau, data_file=df, time_limit=time_limit)
            entry['cpsat'] = {
                'solved': r_cp.get('solved',False),
                'cost': r_cp.get('real_cost',0),
                'avg_trt': r_cp.get('avg_trt',0),
                'num_late': r_cp.get('num_late',0),
                'total_lateness': r_cp.get('total_lateness',0),
                'build_time': r_cp.get('build_time',0),
                'solve_time': r_cp.get('solve_time',0),
                'optimal': r_cp.get('optimal',False),
            }
        except Exception as e:
            entry['cpsat'] = {'solved':False,'error':str(e)}

        # HiGHS (Pyomo) — import here to avoid startup cost
        try:
            from cart_fast import run_experiment as highs_run
            r_hi = highs_run(tau=tau, data_file=df, time_limit=time_limit, mip_gap=0.05)
            entry['highs'] = {
                'solved': r_hi.get('solved',False),
                'cost': r_hi.get('real_cost',0),
                'avg_trt': r_hi.get('avg_trt',0),
                'num_late': r_hi.get('num_late',0),
                'total_lateness': r_hi.get('total_lateness',0),
                'build_time': r_hi.get('build_time',0),
                'solve_time': r_hi.get('solve_time',0),
            }
            # Compute optimality gap
            if entry['cpsat'].get('solved') and entry['highs'].get('solved'):
                cp_cost = entry['cpsat']['cost']
                hi_cost = entry['highs']['cost']
                if cp_cost > 0:
                    entry['gap_pct'] = round((hi_cost - cp_cost) / cp_cost * 100, 2)
        except Exception as e:
            entry['highs'] = {'solved':False,'error':str(e)}

        results.append(entry)
    return jsonify(results)


@app.route('/api/feasibility', methods=['POST'])
def feasibility_boundary():
    """Sweep tau in fine steps to find the exact feasibility boundary."""
    p = request.json or {}
    n = p.get('n', 25)
    steps = p.get('steps', 21)  # 0.00, 0.05, 0.10 ... 1.00
    uc = _urg(p.get('urgency'))
    pc = _proc(p.get('process'))

    _ensure_data(n)
    df = os.path.join(BASE_DIR, f'Data_N{n}.dat')

    taus = [round(i / (steps - 1), 4) for i in range(steps)]
    results = []

    def run_one(tau):
        try:
            r = cpsat_run(tau=tau, data_file=df, urgency_config=uc,
                          process_config=pc, time_limit=30)
            return {'tau': tau, 'solved': r.get('solved', False),
                    'cost': r.get('real_cost', 0), 'avg_trt': r.get('avg_trt', 0),
                    'num_late': r.get('num_late', 0), 'total_lateness': r.get('total_lateness', 0),
                    'all_on_time': r.get('all_on_time', False),
                    'num_patients': r.get('num_patients', n)}
        except Exception as e:
            return {'tau': tau, 'solved': False, 'error': str(e)}

    # Run all in parallel
    with ThreadPoolExecutor(max_workers=min(steps, os.cpu_count() or 4)) as ex:
        futures = {ex.submit(run_one, t): t for t in taus}
        res_map = {}
        for f in as_completed(futures):
            t = futures[f]
            res_map[t] = f.result()

    results = [res_map[t] for t in taus]

    # Find boundary
    feasible_taus = [r['tau'] for r in results if r.get('all_on_time')]
    infeasible_taus = [r['tau'] for r in results if r.get('solved') and not r.get('all_on_time')]
    boundary = max(feasible_taus) if feasible_taus else None
    first_late = min(infeasible_taus) if infeasible_taus else None

    return jsonify({
        'results': results,
        'boundary_tau': boundary,
        'first_late_tau': first_late,
        'n': n,
    })


if __name__ == '__main__':
    import multiprocessing
    cores = multiprocessing.cpu_count()
    print(f'\n  CAR-T Supply Chain Server')
    print(f'  CP-SAT ({cores} cores) + HiGHS benchmark')
    print(f'  http://localhost:5050\n')
    app.run(host='127.0.0.1', port=5050, debug=False)
