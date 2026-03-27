"""
CAR-T Supply Chain — Google OR-Tools CP-SAT Formulation
========================================================
CP-SAT is often 10-100x faster than MIP for assignment/scheduling.
Key advantages:
- Native support for "exactly one" constraints (channeling)
- Strong propagation for binary variables
- Automatic symmetry breaking
- Parallel search by default
"""

from ortools.sat.python import cp_model
import re, random, os, time as _time, json

DEFAULT_URGENCY = {
    'high':   {'fraction': 0.20, 'base_due': 16, 'tolerance': 1},
    'medium': {'fraction': 0.50, 'base_due': 18, 'tolerance': 2},
    'low':    {'fraction': 0.30, 'base_due': 20, 'tolerance': 4},
}
DEFAULT_PROCESS = {'tls': 1, 'tmfe': 7, 'tqc': 7, 'max_facilities': 2}
BIG_M_COST = 1_000_000
SCALE = 100  # CP-SAT needs integers; scale costs by 100


def parse_dat(filepath):
    with open(filepath) as f:
        txt = f.read()
    data = {}
    for s in ['c', 'h', 'j', 'm', 'p']:
        m = re.search(rf'set {s}\s*:=(.*?);', txt, re.DOTALL)
        data[s] = m.group(1).split() if m else []
    for param in ['CIM', 'CVM', 'FCAP', 'TT1', 'TT3']:
        m = re.search(rf'param {param}\s*:=(.*?);', txt, re.DOTALL)
        if m:
            pairs = re.findall(r'(\S+)\s+(\S+)', m.group(1))
            data[param] = {k: float(v) for k, v in pairs}
    m = re.search(r'param U1\s*:=(.*?);', txt, re.DOTALL)
    data['U1'] = {}
    if m:
        for c, mi, j, v in re.findall(r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', m.group(1)):
            data['U1'][(c, mi, j)] = float(v)
    m = re.search(r'param U3\s*:=(.*?);', txt, re.DOTALL)
    data['U3'] = {}
    if m:
        for mi, h, j, v in re.findall(r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', m.group(1)):
            data['U3'][(mi, h, j)] = float(v)
    m = re.search(r'param INC\s*:=(.*?);', txt, re.DOTALL)
    data['INC'] = {}
    if m:
        for p, c, t, v in re.findall(r'(\S+)\s+(\S+)\s+(\d+)\s+(\d+)', m.group(1)):
            data['INC'][(p, c, int(t))] = int(v)
    for param in ['FMAX', 'FMIN', 'TAD', 'TLS']:
        m = re.search(rf'param {param}\s*:=\s*(\S+)\s*;', txt)
        data[param] = float(m.group(1)) if m else 0
    return data


def build_cpsat(data, tau=0.0, urgency_config=None, process_config=None, random_seed=42):
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = DEFAULT_PROCESS

    P = data['p']
    C = data['c']
    H = data['h']
    J = data['j']
    M = data['m']

    TLS  = int(process_config.get('tls', 1))
    TMFE = int(process_config.get('tmfe', 7))
    TQC  = int(process_config.get('tqc', 7))
    MAX_FAC = int(process_config.get('max_facilities', 2))
    TT1  = {j: int(data['TT1'][j]) for j in J}
    TT3  = {j: int(data['TT3'][j]) for j in J}
    min_tt1 = min(TT1.values())
    min_tt3 = min(TT3.values())

    # Assign urgency
    n = len(P)
    num_high = int(round(n * urgency_config['high']['fraction']))
    num_med  = int(round(n * urgency_config['medium']['fraction']))
    random.seed(random_seed)
    shuffled = P[:]
    random.shuffle(shuffled)
    group_map, deadlines = {}, {}
    for idx, p in enumerate(shuffled):
        g = 'high' if idx < num_high else ('medium' if idx < num_high + num_med else 'low')
        cfg = urgency_config[g]
        group_map[p] = g
        deadlines[p] = cfg['base_due'] + cfg['tolerance'] * (1.0 - tau)

    # Patient data
    patient_arrival = {}
    patient_center = {}
    patient_hospital = {}
    c_to_h = {C[i]: H[i] for i in range(len(C))}
    for (p, c, t), v in data['INC'].items():
        if v == 1:
            patient_arrival[p] = t
            patient_center[p] = c
            patient_hospital[p] = c_to_h[c]

    # ── CP-SAT Model ──
    model = cp_model.CpModel()

    # Facility selection
    E = {mi: model.new_bool_var(f'E_{mi}') for mi in M}
    model.add(sum(E[mi] for mi in M) <= MAX_FAC)

    # For each patient: choose (manufacturing_site, outbound_transport, return_transport)
    # This is the key reformulation — instead of Y1[p,c,m,j,t] for all t,
    # we have one assignment variable per patient
    route_vars = {}  # route_vars[p] = {(mi, j_out, j_ret): BoolVar}
    patient_trt = {}
    patient_late = {}
    patient_cost_out = {}
    patient_cost_ret = {}

    # Compute min possible TRT (for deadline clamping)
    min_tt1 = min(TT1.values())
    min_tt3 = min(TT3.values())
    min_possible_trt = TLS + min_tt1 + TMFE + TQC + min_tt3

    # Max wait days a patient can be held at collection before shipping
    # This adds timing flexibility (like the original Pyomo model)
    MAX_WAIT = 10  # days — enough to stagger patients for capacity

    patient_wait = {}  # wait[p] = IntVar (days held at collection center)
    patient_mfg_start = {}  # when patient enters manufacturing

    for p in P:
        arr = patient_arrival[p]
        c = patient_center[p]
        h = patient_hospital[p]
        dl = deadlines[p]

        # Clamp deadline: never below physical minimum (ensures feasibility)
        dl_clamped = max(dl, min_possible_trt)

        # Wait variable: patient can be held 0..MAX_WAIT days at collection
        wait_var = model.new_int_var(0, MAX_WAIT, f'W_{p}')
        patient_wait[p] = wait_var

        options = []
        for mi in M:
            for j_out in J:
                for j_ret in J:
                    tt_out = TT1[j_out]
                    tt_ret = TT3[j_ret]
                    base_trt = TLS + tt_out + TMFE + TQC + tt_ret
                    cost_out = int(data['U1'].get((c, mi, j_out), 0) * SCALE)
                    cost_ret = int(data['U3'].get((mi, h, j_ret), 0) * SCALE)
                    rv = model.new_bool_var(f'R_{p}_{mi}_{j_out}_{j_ret}')
                    route_vars.setdefault(p, {})[( mi, j_out, j_ret)] = rv
                    options.append((rv, mi, j_out, j_ret, base_trt, cost_out, cost_ret))

        # Exactly one route per patient
        model.add_exactly_one(rv for rv, *_ in options)

        # Link to facility
        for (mi, j_out, j_ret), rv in route_vars[p].items():
            model.add_implication(rv, E[mi])

        # Base TRT (without wait) determined by route
        base_trt_var = model.new_int_var(0, 200, f'BTRT_{p}')
        model.add(base_trt_var == sum(rv * trt for rv, mi, j_out, j_ret, trt, co, cr in options))

        # Actual TRT = base_trt + wait
        trt_var = model.new_int_var(0, 200, f'TRT_{p}')
        model.add(trt_var == base_trt_var + wait_var)
        patient_trt[p] = trt_var

        # Manufacturing start day (for capacity tracking)
        # mfg_start = arrival + TLS + wait + TT1[j_out]
        # We need per-route mfg_start, but wait is shared
        # mfg_start_var tracks the actual day patient enters mfg
        mfg_start_var = model.new_int_var(0, 300, f'MS_{p}')
        # mfg_start = arr + TLS + wait + TT1_chosen
        tt1_chosen = model.new_int_var(1, max(TT1.values()), f'TT1_{p}')
        model.add(tt1_chosen == sum(rv * TT1[j_out] for rv, mi, j_out, j_ret, trt, co, cr in options))
        model.add(mfg_start_var == arr + TLS + wait_var + tt1_chosen)
        patient_mfg_start[p] = mfg_start_var

        # Lateness (clamped deadline)
        late_var = model.new_int_var(0, 200, f'LATE_{p}')
        dl_int = int(dl_clamped)
        model.add(late_var >= trt_var - dl_int)
        patient_late[p] = late_var

        # Transport costs
        cost_out_var = model.new_int_var(0, 10_000_000, f'CO_{p}')
        cost_ret_var = model.new_int_var(0, 10_000_000, f'CR_{p}')
        model.add(cost_out_var == sum(rv * co for rv, mi, j_out, j_ret, trt, co, cr in options))
        model.add(cost_ret_var == sum(rv * cr for rv, mi, j_out, j_ret, trt, co, cr in options))
        patient_cost_out[p] = cost_out_var
        patient_cost_ret[p] = cost_ret_var

    # Compute time horizon (needed for cost formula and capacity)
    max_arr = max(patient_arrival.values())
    max_lead = TLS + MAX_WAIT + max(TT1.values()) + TMFE + TQC + max(TT3.values())
    T_max = max_arr + max_lead + 5

    # Facility cost: match Pyomo formula CTM[p] = E[m]*(CIM+CVM)*len(T)/len(P)
    # Total = sum_p CTM[p] = sum_m E[m]*(CIM+CVM)*len(T)
    total_fac_cost = model.new_int_var(0, 10**9, 'FAC_COST')
    model.add(total_fac_cost == sum(
        E[mi] * int((data['CIM'][mi] + data['CVM'].get(mi, 0)) * T_max * SCALE)
        for mi in M))

    # Material + QC cost (constant per patient)
    mat_qc_per_patient = int((10476 + 9312) * SCALE)

    # Create interval variables for manufacturing at each facility
    # For each (patient, facility): optional interval present if patient uses that facility
    for mi in M:
        fcap = int(data['FCAP'][mi])
        intervals = []
        demands = []
        for p in P:
            # Bool: does patient p use facility mi?
            uses_mi = model.new_bool_var(f'U_{p}_{mi}')
            mi_routes = [rv for (m, jo, jr), rv in route_vars.get(p, {}).items() if m == mi]
            if not mi_routes:
                model.add(uses_mi == 0)
                continue
            # uses_mi = 1 iff any route using mi is selected
            model.add(sum(mi_routes) == uses_mi)

            # Optional interval: [mfg_start, mfg_start + TMFE) present iff uses_mi
            iv = model.new_optional_fixed_size_interval_var(
                patient_mfg_start[p], TMFE, uses_mi, f'IV_{p}_{mi}')
            intervals.append(iv)
            demands.append(1)

        if intervals:
            # Cumulative constraint: at most fcap patients in mfg at any time
            model.add_cumulative(intervals, demands, fcap)

    # Objective: minimize cost + Big-M * lateness
    total_transport = sum(patient_cost_out[p] + patient_cost_ret[p] for p in P)
    total_material = mat_qc_per_patient * len(P)
    total_lateness = sum(patient_late[p] for p in P)

    model.minimize(
        total_fac_cost + total_transport + total_material
        + BIG_M_COST * SCALE * total_lateness
    )

    return model, {
        'group_map': group_map, 'deadlines': deadlines,
        'patient_trt': patient_trt, 'patient_late': patient_late,
        'patient_wait': patient_wait, 'patient_mfg_start': patient_mfg_start,
        'patient_arrival': patient_arrival, 'patient_hospital': patient_hospital,
        'patient_center': patient_center,
        'patient_cost_out': patient_cost_out, 'patient_cost_ret': patient_cost_ret,
        'total_fac_cost': total_fac_cost, 'route_vars': route_vars,
        'E': E, 'P': P, 'M': M, 'process_config': process_config,
    }


def solve_cpsat(model, time_limit=300, num_workers=8):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = num_workers
    solver.parameters.log_search_progress = False
    status = solver.solve(model)
    return solver, status


def extract_cpsat_results(solver, meta, tau, status):
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {'solved': False, 'status': str(status)}

    P = meta['P']
    M = meta['M']
    gm = meta['group_map']
    dl = meta['deadlines']
    pc = meta['process_config']

    patients = []
    gs = {g: {'count': 0, 'trts': [], 'lates': [], 'deadline': 0}
          for g in ('high', 'medium', 'low')}

    route_vars = meta['route_vars']
    for p in sorted(P):
        trt = solver.value(meta['patient_trt'][p])
        late = solver.value(meta['patient_late'][p])
        wait = solver.value(meta['patient_wait'][p])
        grp = gm[p]
        arr = meta['patient_arrival'][p]

        # Find selected route
        facility = transport_out = transport_ret = '?'
        for (mi, jo, jr), rv in route_vars.get(p, {}).items():
            if solver.value(rv):
                facility = mi
                transport_out = jo
                transport_ret = jr
                break

        patients.append({
            'id': p, 'group': grp,
            'arrival_day': arr,
            'completion_day': arr + trt,
            'turnaround': trt,
            'wait': wait,
            'facility': facility,
            'transport_out': transport_out,
            'transport_ret': transport_ret,
            'deadline': round(dl[p], 2),
            'lateness': late,
            'on_time': late < 1,
        })
        if grp in gs:
            gs[grp]['count'] += 1
            gs[grp]['trts'].append(trt)
            gs[grp]['lates'].append(late)
            gs[grp]['deadline'] = round(dl[p], 2)

    fac_cost = solver.value(meta['total_fac_cost']) / SCALE
    transport_cost = sum(solver.value(meta['patient_cost_out'][p]) + solver.value(meta['patient_cost_ret'][p])
                         for p in P) / SCALE
    mat_cost = (10476 + 9312) * len(P)
    real_cost = fac_cost + transport_cost + mat_cost
    tot_late = sum(solver.value(meta['patient_late'][p]) for p in P)
    n_late = sum(1 for p in P if solver.value(meta['patient_late'][p]) > 0)

    summary = {}
    for g, info in gs.items():
        if info['count'] > 0:
            summary[g] = {
                'count': info['count'], 'deadline': info['deadline'],
                'avg_trt': round(sum(info['trts']) / len(info['trts']), 2),
                'min_trt': min(info['trts']),
                'max_trt': max(info['trts']),
                'avg_lateness': round(sum(info['lates']) / len(info['lates']), 2),
                'max_lateness': max(info['lates']),
                'num_late': sum(1 for l in info['lates'] if l > 0),
            }

    facs_open = [mi for mi in M if solver.value(meta['E'][mi]) == 1]

    tls = pc.get('tls', 1)
    tmfe = pc.get('tmfe', 7)
    tqc = pc.get('tqc', 7)
    min_trt = tls + 1 + tmfe + tqc + 1

    return {
        'tau': tau, 'num_patients': len(P), 'solver': 'CP-SAT',
        'real_cost': round(real_cost, 2), 'avg_trt': round(sum(solver.value(meta['patient_trt'][p]) for p in P) / len(P), 2),
        'total_lateness': tot_late, 'num_late': n_late,
        'all_on_time': n_late == 0,
        'facilities_open': facs_open,
        'group_summary': summary, 'patients': patients,
        'min_trt': min_trt, 'process_config': pc,
    }


def run_experiment(tau=0.0, data_file=None, urgency_config=None,
                   process_config=None, time_limit=300, random_seed=42):
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N5.dat')
    if process_config is None:
        process_config = DEFAULT_PROCESS

    t0 = _time.time()
    data = parse_dat(data_file)
    model, meta = build_cpsat(data, tau=tau, urgency_config=urgency_config,
                               process_config=process_config, random_seed=random_seed)
    build_time = _time.time() - t0

    t0 = _time.time()
    solver, status = solve_cpsat(model, time_limit=time_limit)
    solve_time = _time.time() - t0

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            'tau': tau, 'solved': False, 'solver': 'CP-SAT',
            'termination': str(status),
            'build_time': round(build_time, 1),
            'solve_time': round(solve_time, 1),
            'num_patients': len(data['p']),
        }

    res = extract_cpsat_results(solver, meta, tau, status)
    res['solved'] = True
    res['build_time'] = round(build_time, 1)
    res['solve_time'] = round(solve_time, 1)
    res['optimal'] = (status == cp_model.OPTIMAL)
    return res


if __name__ == '__main__':
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None)
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--time-limit', type=int, default=120)
    args = parser.parse_args()

    data = args.data or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N50.dat')
    r = run_experiment(tau=args.tau, data_file=data, time_limit=args.time_limit)
    if r['solved']:
        st = 'ALL ON TIME' if r['all_on_time'] else f'{r["num_late"]} LATE'
        opt = '(OPTIMAL)' if r.get('optimal') else '(feasible)'
        print(f"CP-SAT {opt} | {st} | Cost: ${r['real_cost']:,.0f} | TRT: {r['avg_trt']:.1f}d")
        print(f"Build: {r['build_time']}s | Solve: {r['solve_time']}s")
        print(f"Facilities: {r['facilities_open']}")
        for g in ('high', 'medium', 'low'):
            if g in r['group_summary']:
                s = r['group_summary'][g]
                lt = f" LATE:{s['num_late']}" if s['num_late'] > 0 else ""
                print(f"  {g:>8}: n={s['count']} DL={s['deadline']} TRT=[{s['min_trt']},{s['avg_trt']},{s['max_trt']}]{lt}")
    else:
        print(f"CP-SAT FAILED: {r.get('termination','?')} ({r['solve_time']}s)")
