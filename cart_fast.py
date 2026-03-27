"""
CAR-T Supply Chain — Time-Windowed Concrete Model (Fast)
=========================================================
Key optimization: each patient only gets variables for time periods
within their feasible window [arrival, arrival + max_deadline + buffer].
This cuts binary variables by 70-90%.

Big-M lateness penalty for feasibility analysis.
"""

from pyomo.environ import *
import re, random, json, os, time as _time

DEFAULT_URGENCY = {
    'high':   {'fraction': 0.20, 'base_due': 16, 'tolerance': 1},
    'medium': {'fraction': 0.50, 'base_due': 18, 'tolerance': 2},
    'low':    {'fraction': 0.30, 'base_due': 20, 'tolerance': 4},
}

DEFAULT_PROCESS = {
    'tls': 1, 'tmfe': 7, 'tqc': 7, 'max_facilities': 2,
}

BIG_M = 1_000_000


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


def build_fast(data, tau=0.0, urgency_config=None, process_config=None, random_seed=42):
    """Build time-windowed ConcreteModel."""
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
    fmin, fmax = data['FMIN'], data['FMAX']

    max_tt1 = max(TT1.values())
    max_tt3 = max(TT3.values())
    max_total_lead = TLS + max_tt1 + TMFE + TQC + max_tt3

    # Assign urgency groups
    n = len(P)
    num_high = int(round(n * urgency_config['high']['fraction']))
    num_med  = int(round(n * urgency_config['medium']['fraction']))
    random.seed(random_seed)
    shuffled = P[:]
    random.shuffle(shuffled)

    group_map = {}
    deadlines = {}
    max_dl = 0
    for idx, p in enumerate(shuffled):
        g = 'high' if idx < num_high else ('medium' if idx < num_high + num_med else 'low')
        cfg = urgency_config[g]
        dl = cfg['base_due'] + cfg['tolerance'] * (1.0 - tau)
        group_map[p] = g
        deadlines[p] = dl
        max_dl = max(max_dl, dl)

    # Per-patient time windows
    patient_arrival = {}
    patient_center = {}
    for (p, c, t), v in data['INC'].items():
        if v == 1:
            patient_arrival[p] = t
            patient_center[p] = c

    # Window: [arrival, arrival + max(deadline, max_total_lead) + buffer]
    patient_window = {}
    for p in P:
        arr = patient_arrival[p]
        window_end = arr + max(deadlines[p], max_total_lead) + 5
        patient_window[p] = list(range(arr, int(window_end) + 1))

    # Global time range
    all_times = set()
    for w in patient_window.values():
        all_times.update(w)
    T_global = sorted(all_times)
    t_min = min(T_global)
    t_max = max(T_global)

    # ── Build model ──
    m = ConcreteModel()

    # Facility variables (not patient-indexed, use full time range)
    m.E1 = Var(M, within=Binary)
    m.X1 = Var(C, M, within=Binary)
    m.X2 = Var(M, H, within=Binary)

    # Time-windowed patient variables
    # Y1[p,c,m,j,t] — only for t in patient's window
    y1_idx = [(p, c, mi, j, t) for p in P for c in C for mi in M for j in J for t in patient_window[p]]
    y2_idx = [(p, mi, h, j, t) for p in P for mi in M for h in H for j in J for t in patient_window[p]]

    m.Y1 = Var(y1_idx, within=Binary)
    m.Y2 = Var(y2_idx, within=Binary)

    # Flow variables (windowed)
    pt_idx = [(p, t) for p in P for t in patient_window[p]]
    pct_idx = [(p, c, t) for p in P for c in C for t in patient_window[p]]
    pmt_idx = [(p, mi, t) for p in P for mi in M for t in patient_window[p]]
    pmht_idx = [(p, mi, h, j, t) for p in P for mi in M for h in H for j in J for t in patient_window[p]]
    pcmjt_idx = [(p, c, mi, j, t) for p in P for c in C for mi in M for j in J for t in patient_window[p]]
    pht_idx = [(p, h, t) for p in P for h in H for t in patient_window[p]]

    m.INH  = Var(pht_idx, within=NonNegativeIntegers)
    m.CTM  = Var(P, within=NonNegativeReals)
    m.FTD  = Var(pmht_idx, within=NonNegativeReals)
    m.TTC  = Var(P, within=NonNegativeReals)
    m.LSA  = Var(pcmjt_idx, within=NonNegativeReals)
    m.LSR  = Var(pcmjt_idx, within=NonNegativeReals)
    m.MSO  = Var(pmht_idx, within=NonNegativeReals)
    m.OUTC = Var(pct_idx, within=NonNegativeReals)
    m.OUTM = Var(pmt_idx, within=NonNegativeReals)
    m.INM  = Var(pmt_idx, within=NonNegativeReals)
    m.DURV = Var(pmt_idx, within=NonNegativeReals)

    # Non-windowed
    mt_idx = [(mi, t) for mi in M for t in T_global]
    m.RATIO = Var(mt_idx, within=NonNegativeReals)
    m.CAP   = Var(mt_idx)

    m.TRT  = Var(P)
    m.ATRT = Var()
    m.STT  = Var(P)
    m.CTT  = Var(P)
    m.LATE = Var(P, within=NonNegativeReals)

    # Helper: check if index exists
    pw_set = {p: set(patient_window[p]) for p in P}

    def in_window(p, t):
        return t in pw_set[p]

    # ── Objective: cost + Big-M lateness ──
    m.obj = Objective(expr=(
        sum(m.CTM[p] for p in P)
        + sum(m.TTC[p] for p in P)
        + (10476 + 9312) * len(P)
        + BIG_M * sum(m.LATE[p] for p in P)
    ))

    # ── Constraints ──
    m.C1 = ConstraintList()
    for p in P:
        m.C1.add(m.CTM[p] == sum(
            m.E1[mi] * (data['CIM'][mi] + data['CVM'].get(mi, 0))
            * len(T_global) / len(P) for mi in M))

    m.C2 = ConstraintList()
    for p in P:
        m.C2.add(m.TTC[p] == (
            sum(m.Y1[p, c, mi, j, t] * data['U1'].get((c, mi, j), 0)
                for c in C for mi in M for j in J for t in patient_window[p])
            + sum(m.Y2[p, mi, h, j, t] * data['U3'].get((mi, h, j), 0)
                  for mi in M for h in H for j in J for t in patient_window[p])))

    # Facility utilisation (use global time)
    m.RATIOEQ = ConstraintList()
    for mi in M:
        fcap = data['FCAP'][mi]
        for t in T_global:
            terms = [m.DURV[p, mi, t] / fcap for p in P if in_window(p, t)]
            if terms:
                m.RATIOEQ.add(m.RATIO[mi, t] == sum(terms))

    m.MSBnew = ConstraintList()
    for p in P:
        pw = patient_window[p]
        for mi in M:
            for t in pw:
                m.MSBnew.add(m.DURV[p, mi, t] == (
                    sum(m.INM[p, mi, tt - 1] - m.OUTM[p, mi, tt]
                        for tt in pw if tt <= t and tt > pw[0] and (tt-1) in pw_set[p])
                    + m.OUTM[p, mi, t]))

    # MSB1: collection → OUTC at arrival + TLS
    m.MSB1 = ConstraintList()
    for p in P:
        arr = patient_arrival[p]
        c = patient_center[p]
        t_out = arr + TLS
        if in_window(p, t_out):
            m.MSB1.add(1 == m.OUTC[p, c, t_out])  # INC=1 → OUTC at arr+TLS
    # Zero OUTC elsewhere
    m.MSB1z = ConstraintList()
    for p in P:
        arr = patient_arrival[p]
        pc = patient_center[p]
        for c in C:
            for t in patient_window[p]:
                if not (c == pc and t == arr + TLS):
                    m.MSB1z.add(m.OUTC[p, c, t] == 0)

    # MSB3: transport to mfg
    m.MSB3 = ConstraintList()
    for p in P:
        for c in C:
            for mi in M:
                for j in J:
                    tt1 = TT1[j]
                    for t in patient_window[p]:
                        ta = t + tt1
                        if in_window(p, ta):
                            m.MSB3.add(m.LSR[p, c, mi, j, t] == m.LSA[p, c, mi, j, ta])

    # MSB7: collection outflow
    m.MSB7 = ConstraintList()
    for p in P:
        for c in C:
            for t in patient_window[p]:
                m.MSB7.add(m.OUTC[p, c, t] == sum(m.LSR[p, c, mi, j, t] for mi in M for j in J))

    # MSB5: mfg inflow
    m.MSB5 = ConstraintList()
    for p in P:
        for mi in M:
            for t in patient_window[p]:
                m.MSB5.add(m.INM[p, mi, t] == sum(m.LSA[p, c, mi, j, t] for c in C for j in J))

    # MSB2: manufacturing time
    m.MSB2 = ConstraintList()
    for p in P:
        for mi in M:
            for t in patient_window[p]:
                to = t + TMFE
                if in_window(p, to):
                    m.MSB2.add(m.INM[p, mi, t] == m.OUTM[p, mi, to])

    # MSB8: QC then ship
    m.MSB8 = ConstraintList()
    for p in P:
        for mi in M:
            for t in patient_window[p]:
                ts = t + TQC
                if in_window(p, ts):
                    m.MSB8.add(m.OUTM[p, mi, t] == sum(
                        m.MSO[p, mi, h, j, ts] for h in H for j in J))

    # MSB4: return transport
    m.MSB4 = ConstraintList()
    for p in P:
        for mi in M:
            for h in H:
                for j in J:
                    tt3 = TT3[j]
                    for t in patient_window[p]:
                        ta = t + tt3
                        if in_window(p, ta):
                            m.MSB4.add(m.MSO[p, mi, h, j, t] == m.FTD[p, mi, h, j, ta])

    # MSB6: hospital inflow
    m.MSB6 = ConstraintList()
    for p in P:
        for h in H:
            for t in patient_window[p]:
                m.MSB6.add(m.INH[p, h, t] == sum(m.FTD[p, mi, h, j, t] for mi in M for j in J))

    # Capacity
    m.CAP1 = ConstraintList()
    for mi in M:
        fcap = data['FCAP'][mi]
        for t in T_global:
            m.CAP1.add(m.CAP[mi, t] == fcap - sum(
                m.INM[p, mi, tt] for p in P for tt in patient_window[p]
                if tt < t and tt >= t - TMFE and in_window(p, tt)))

    m.CAPCON1 = ConstraintList()
    for mi in M:
        for t in T_global:
            ins = [m.INM[p, mi, t] for p in P if in_window(p, t)]
            outs = [m.OUTM[p, mi, t] for p in P if in_window(p, t)]
            if ins:
                m.CAPCON1.add(sum(ins) - sum(outs) <= m.CAP[mi, t])

    # Max facilities
    m.CON1 = Constraint(expr=sum(m.E1[mi] for mi in M) <= MAX_FAC)

    # Linking
    m.CON2 = ConstraintList()
    for c in C:
        for mi in M:
            m.CON2.add(m.X1[c, mi] <= m.E1[mi])
    m.CON3 = ConstraintList()
    for mi in M:
        for h in H:
            m.CON3.add(m.X2[mi, h] <= m.E1[mi])

    m.CON4 = ConstraintList()
    for p, c, mi, j, t in y1_idx:
        m.CON4.add(m.Y1[p, c, mi, j, t] <= m.X1[c, mi])
    m.CON5 = ConstraintList()
    for p, mi, h, j, t in y2_idx:
        m.CON5.add(m.Y2[p, mi, h, j, t] <= m.X2[mi, h])

    # Each patient: exactly one route
    m.CON6 = ConstraintList()
    for p in P:
        m.CON6.add(sum(m.Y1[p, c, mi, j, t] for c in C for mi in M for j in J for t in patient_window[p]) == 1)
    m.CON7 = ConstraintList()
    for p in P:
        m.CON7.add(sum(m.Y2[p, mi, h, j, t] for mi in M for h in H for j in J for t in patient_window[p]) == 1)

    # Demand
    m.DEM = Constraint(expr=sum(m.INH[p, h, t] for p, h, t in pht_idx) <= len(P))

    # Flow bounds
    m.CON8 = ConstraintList()
    m.CON9 = ConstraintList()
    for p, c, mi, j, t in pcmjt_idx:
        m.CON8.add(m.LSR[p, c, mi, j, t] >= m.Y1[p, c, mi, j, t] * fmin)
        m.CON9.add(m.LSR[p, c, mi, j, t] <= m.Y1[p, c, mi, j, t] * fmax)

    m.CON10 = ConstraintList()
    m.CON11 = ConstraintList()
    for p, mi, h, j, t in pmht_idx:
        m.CON10.add(m.MSO[p, mi, h, j, t] >= m.Y2[p, mi, h, j, t] * fmin)
        m.CON11.add(m.MSO[p, mi, h, j, t] <= m.Y2[p, mi, h, j, t] * fmax)

    # Hospital matching
    m.HMATCH = ConstraintList()
    for p in P:
        for ic, c in enumerate(C):
            h = H[ic]
            inc_val = sum(data['INC'].get((p, c, t), 0) for t in patient_window[p])
            m.HMATCH.add(sum(m.Y2[p, mi, h, j, t] for mi in M for j in J for t in patient_window[p]) == inc_val)

    # Timing
    m.START = ConstraintList()
    for p in P:
        m.START.add(m.STT[p] == sum(
            data['INC'].get((p, c, t), 0) * t for c in C for t in patient_window[p]))
    m.END = ConstraintList()
    for p in P:
        m.END.add(m.CTT[p] == sum(m.INH[p, h, t] * t for h in H for t in patient_window[p]))

    m.TSEQ = ConstraintList()
    for p in P:
        m.TSEQ.add(m.STT[p] <= m.CTT[p])
    m.TIME = ConstraintList()
    for p in P:
        m.TIME.add(m.TRT[p] == m.CTT[p] - m.STT[p])
    m.ATIME = Constraint(expr=m.ATRT == sum(m.TRT[p] for p in P) / len(P))

    # Deadline
    m.LATECON = ConstraintList()
    for p in P:
        m.LATECON.add(m.LATE[p] >= m.TRT[p] - deadlines[p])

    # Metadata
    m._group_map = group_map
    m._deadlines = deadlines
    m._patients = P
    m._mfg = M
    m._process = process_config

    return m


def solve_model(model, solver_name='highs', time_limit=300, mip_gap=0.05):
    solver = SolverFactory(solver_name)
    if solver_name in ('highs', 'appsi_highs'):
        solver.options['time_limit'] = time_limit
        solver.options['mip_rel_gap'] = mip_gap
    results = solver.solve(model, tee=False, load_solutions=False)
    ok = (results.solver.status == SolverStatus.ok and
          results.solver.termination_condition in
          (TerminationCondition.optimal, TerminationCondition.feasible))
    if ok:
        model.solutions.load_from(results)
    return results, ok


def extract_results(model, tau):
    P = model._patients
    M = model._mfg
    gm = model._group_map
    dl = model._deadlines
    pc = model._process

    patients = []
    gs = {g: {'count': 0, 'trts': [], 'lates': [], 'deadline': 0}
          for g in ('high', 'medium', 'low')}

    for p in sorted(P):
        trt = value(model.TRT[p])
        late = value(model.LATE[p])
        grp = gm[p]
        patients.append({
            'id': p, 'group': grp,
            'arrival_day': round(value(model.STT[p]), 1),
            'completion_day': round(value(model.CTT[p]), 1),
            'turnaround': round(trt, 2), 'deadline': round(dl[p], 2),
            'lateness': round(late, 2), 'on_time': late < 0.01,
        })
        if grp in gs:
            gs[grp]['count'] += 1
            gs[grp]['trts'].append(trt)
            gs[grp]['lates'].append(late)
            gs[grp]['deadline'] = round(dl[p], 2)

    real_cost = value(sum(model.CTM[p] for p in P) + sum(model.TTC[p] for p in P)
                      + (10476 + 9312) * len(P))
    tot_late = sum(value(model.LATE[p]) for p in P)
    n_late = sum(1 for p in P if value(model.LATE[p]) > 0.01)

    summary = {}
    for g, info in gs.items():
        if info['count'] > 0:
            summary[g] = {
                'count': info['count'], 'deadline': info['deadline'],
                'avg_trt': round(sum(info['trts']) / len(info['trts']), 2),
                'min_trt': round(min(info['trts']), 2),
                'max_trt': round(max(info['trts']), 2),
                'avg_lateness': round(sum(info['lates']) / len(info['lates']), 2),
                'max_lateness': round(max(info['lates']), 2),
                'num_late': sum(1 for l in info['lates'] if l > 0.01),
            }

    tls = pc.get('tls', 1)
    tmfe = pc.get('tmfe', 7)
    tqc = pc.get('tqc', 7)
    min_trt = tls + 1 + tmfe + tqc + 1

    return {
        'tau': tau, 'num_patients': len(P),
        'real_cost': round(real_cost, 2), 'avg_trt': round(value(model.ATRT), 2),
        'total_lateness': round(tot_late, 2), 'num_late': n_late,
        'all_on_time': n_late == 0,
        'facilities_open': [mi for mi in M if value(model.E1[mi]) > 0.5],
        'group_summary': summary, 'patients': patients,
        'min_trt': min_trt, 'process_config': pc,
    }


def run_experiment(tau=0.0, data_file=None, urgency_config=None,
                   process_config=None, solver_name='highs',
                   random_seed=42, time_limit=300, mip_gap=0.05):
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N5.dat')
    if process_config is None:
        process_config = DEFAULT_PROCESS

    t0 = _time.time()
    data = parse_dat(data_file)
    model = build_fast(data, tau=tau, urgency_config=urgency_config,
                       process_config=process_config, random_seed=random_seed)
    build_time = _time.time() - t0

    t0 = _time.time()
    results, ok = solve_model(model, solver_name=solver_name,
                              time_limit=time_limit, mip_gap=mip_gap)
    solve_time = _time.time() - t0

    if not ok:
        return {
            'tau': tau, 'solved': False,
            'termination': str(results.solver.termination_condition),
            'build_time': round(build_time, 1),
            'solve_time': round(solve_time, 1),
            'num_patients': len(data['p']),
        }

    res = extract_results(model, tau)
    res['solved'] = True
    res['build_time'] = round(build_time, 1)
    res['solve_time'] = round(solve_time, 1)
    return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--data', default=None)
    parser.add_argument('--solver', default='highs')
    parser.add_argument('--time-limit', type=int, default=120)
    parser.add_argument('--gap', type=float, default=0.05)
    args = parser.parse_args()

    r = run_experiment(tau=args.tau, data_file=args.data,
                       solver_name=args.solver, time_limit=args.time_limit,
                       mip_gap=args.gap)
    if r.get('solved'):
        st = 'ALL ON TIME' if r['all_on_time'] else f'{r["num_late"]} LATE'
        print(f"tau={args.tau} | {st} | Cost: {r['real_cost']:,.0f} | Avg TRT: {r['avg_trt']:.1f}")
        print(f"Build: {r['build_time']}s | Solve: {r['solve_time']}s | MinTRT: {r['min_trt']}d")
        print(f"Facilities: {r['facilities_open']}")
        for g in ('high', 'medium', 'low'):
            if g in r['group_summary']:
                s = r['group_summary'][g]
                lt = f", LATE: {s['num_late']}" if s['num_late'] > 0 else ""
                print(f"  {g:>8}: n={s['count']}, DL={s['deadline']}, TRT=[{s['min_trt']},{s['avg_trt']},{s['max_trt']}]{lt}")
    else:
        print(f"tau={args.tau} | FAILED: {r['termination']} ({r['solve_time']}s)")
