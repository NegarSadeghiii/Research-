"""
CAR-T Supply Chain — Concrete Model (Fast Build)
=================================================
Same mathematical formulation but built as ConcreteModel
with only the necessary constraints instantiated (no Constraint.Skip).
This dramatically reduces build time and memory.

Big-M approach for deadlines with tightness factor τ.
"""

from pyomo.environ import *
import re, random, json, sys, os, time

DEFAULT_URGENCY = {
    'high':   {'fraction': 0.20, 'base_due': 16, 'tolerance': 1},
    'medium': {'fraction': 0.50, 'base_due': 18, 'tolerance': 2},
    'low':    {'fraction': 0.30, 'base_due': 20, 'tolerance': 4},
}

BIG_M = 1_000_000


def parse_dat(filepath):
    """Parse the .dat file and return all data as dicts."""
    with open(filepath) as f:
        txt = f.read()

    data = {}

    # Parse sets
    for s in ['c', 'h', 'j', 'm', 'p']:
        m = re.search(rf'set {s}\s*:=(.*?);', txt, re.DOTALL)
        data[s] = m.group(1).split() if m else []

    # Parse indexed params (1D)
    for param in ['CIM', 'CVM', 'FCAP', 'TT1', 'TT3']:
        m = re.search(rf'param {param}\s*:=(.*?);', txt, re.DOTALL)
        if m:
            pairs = re.findall(r'(\S+)\s+(\S+)', m.group(1))
            data[param] = {k: float(v) for k, v in pairs}

    # Parse U1 (c, m, j -> cost)
    m = re.search(r'param U1\s*:=(.*?);', txt, re.DOTALL)
    data['U1'] = {}
    if m:
        triples = re.findall(r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', m.group(1))
        for c, mi, j, v in triples:
            data['U1'][(c, mi, j)] = float(v)

    # Parse U3 (m, h, j -> cost)
    m = re.search(r'param U3\s*:=(.*?);', txt, re.DOTALL)
    data['U3'] = {}
    if m:
        triples = re.findall(r'(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', m.group(1))
        for mi, h, j, v in triples:
            data['U3'][(mi, h, j)] = float(v)

    # Parse INC (p, c, t -> 1)
    m = re.search(r'param INC\s*:=(.*?);', txt, re.DOTALL)
    data['INC'] = {}
    if m:
        entries = re.findall(r'(\S+)\s+(\S+)\s+(\d+)\s+(\d+)', m.group(1))
        for p, c, t, v in entries:
            data['INC'][(p, c, int(t))] = int(v)

    # Scalars
    for param in ['FMAX', 'FMIN', 'TAD', 'TLS']:
        m = re.search(rf'param {param}\s*:=\s*(\S+)\s*;', txt)
        data[param] = float(m.group(1)) if m else 0

    return data


DEFAULT_PROCESS = {
    'tls': 1,    # leukapheresis (days)
    'tmfe': 7,   # manufacturing / cell expansion (days)
    'tqc': 7,    # quality control (days)
    'max_facilities': 2,  # max open facilities
}


def build_concrete(data, time_horizon=None, tau=0.0, urgency_config=None,
                   process_config=None, random_seed=42):
    """Build a ConcreteModel directly from parsed data."""
    if urgency_config is None:
        urgency_config = DEFAULT_URGENCY
    if process_config is None:
        process_config = DEFAULT_PROCESS

    patients = data['p']
    centers  = data['c']
    hospitals = data['h']
    transports = data['j']
    mfg_sites = data['m']

    # Auto time horizon
    if time_horizon is None:
        max_arrival = max(t for (_, _, t) in data['INC'].keys())
        time_horizon = max_arrival + 35

    T = list(range(1, time_horizon + 1))

    TLS  = int(process_config.get('tls', data['TLS']))
    TMFE = int(process_config.get('tmfe', 7))
    TQC  = int(process_config.get('tqc', 7))
    MAX_FAC = int(process_config.get('max_facilities', 2))
    TT1  = {j: int(data['TT1'][j]) for j in transports}
    TT3  = {j: int(data['TT3'][j]) for j in transports}

    # Assign urgency groups
    n = len(patients)
    num_high = int(round(n * urgency_config['high']['fraction']))
    num_med  = int(round(n * urgency_config['medium']['fraction']))
    random.seed(random_seed)
    shuffled = patients[:]
    random.shuffle(shuffled)

    group_map = {}
    deadlines = {}
    for idx, p in enumerate(shuffled):
        if idx < num_high:
            g = 'high'
        elif idx < num_high + num_med:
            g = 'medium'
        else:
            g = 'low'
        cfg = urgency_config[g]
        group_map[p] = g
        deadlines[p] = cfg['base_due'] + cfg['tolerance'] * (1.0 - tau)

    # ── Build model ──
    m = ConcreteModel()

    # Index sets
    P = patients
    C = centers
    H = hospitals
    J = transports
    M = mfg_sites

    # Variables
    m.E1   = Var(M, within=Binary)
    m.X1   = Var(C, M, within=Binary)
    m.X2   = Var(M, H, within=Binary)
    m.Y1   = Var(P, C, M, J, T, within=Binary)
    m.Y2   = Var(P, M, H, J, T, within=Binary)
    m.INH  = Var(P, H, T, within=NonNegativeIntegers)
    m.CTM  = Var(P, within=NonNegativeReals)
    m.FTD  = Var(P, M, H, J, T, within=NonNegativeReals)
    m.TTC  = Var(P, within=NonNegativeReals)
    m.LSA  = Var(P, C, M, J, T, within=NonNegativeReals)
    m.LSR  = Var(P, C, M, J, T, within=NonNegativeReals)
    m.MSO  = Var(P, M, H, J, T, within=NonNegativeReals)
    m.OUTC = Var(P, C, T, within=NonNegativeReals)
    m.OUTM = Var(P, M, T, within=NonNegativeReals)
    m.INM  = Var(P, M, T, within=NonNegativeReals)
    m.DURV = Var(P, M, T, within=NonNegativeReals)
    m.RATIO= Var(M, T, within=NonNegativeReals)
    m.CAP  = Var(M, T)
    m.TRT  = Var(P)
    m.ATRT = Var()
    m.STT  = Var(P)
    m.CTT  = Var(P)
    m.LATE = Var(P, within=NonNegativeReals)

    # ── Objective ──
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
            * len(T) / len(P) for mi in M))

    m.C2 = ConstraintList()
    for p in P:
        m.C2.add(m.TTC[p] == (
            sum(m.Y1[p, c, mi, j, t] * data['U1'].get((c, mi, j), 0)
                for c in C for mi in M for j in J for t in T)
            + sum(m.Y2[p, mi, h, j, t] * data['U3'].get((mi, h, j), 0)
                  for mi in M for h in H for j in J for t in T)))

    # Facility utilisation
    m.RATIOEQ = ConstraintList()
    for mi in M:
        fcap = data['FCAP'][mi]
        for t in T:
            m.RATIOEQ.add(m.RATIO[mi, t] == sum(m.DURV[p, mi, t] / fcap for p in P))

    m.MSBnew = ConstraintList()
    for p in P:
        for mi in M:
            for t in T:
                m.MSBnew.add(m.DURV[p, mi, t] == (
                    sum(m.INM[p, mi, tt - 1] - m.OUTM[p, mi, tt]
                        for tt in T if tt <= t and tt > 1)
                    + m.OUTM[p, mi, t]))

    # MSB1: collection delay (only where INC=1)
    m.MSB1 = ConstraintList()
    for (p, c, t_arr), v in data['INC'].items():
        t_out = t_arr + TLS
        if t_out in T:
            m.MSB1.add(v == m.OUTC[p, c, t_out])
        # Also: INC at non-arrival times is 0 → OUTC is 0
    # Zero out OUTC for non-arrival combos
    m.MSB1_zero = ConstraintList()
    for p in P:
        for c in C:
            for t in T:
                if (p, c, t - TLS) not in data['INC']:
                    m.MSB1_zero.add(m.OUTC[p, c, t] == 0)

    # MSB3: transport to manufacturing (only for valid time shifts)
    m.MSB3 = ConstraintList()
    for p in P:
        for c in C:
            for mi in M:
                for j in J:
                    tt1 = TT1[j]
                    for t in T:
                        t_arr = t + tt1
                        if t_arr in T:
                            m.MSB3.add(m.LSR[p, c, mi, j, t] == m.LSA[p, c, mi, j, t_arr])

    # MSB7: outflow from collection
    m.MSB7 = ConstraintList()
    for p in P:
        for c in C:
            for t in T:
                m.MSB7.add(m.OUTC[p, c, t] == sum(m.LSR[p, c, mi, j, t] for mi in M for j in J))

    # MSB5: inflow to manufacturing
    m.MSB5 = ConstraintList()
    for p in P:
        for mi in M:
            for t in T:
                m.MSB5.add(m.INM[p, mi, t] == sum(m.LSA[p, c, mi, j, t] for c in C for j in J))

    # MSB2: manufacturing time
    m.MSB2 = ConstraintList()
    for p in P:
        for mi in M:
            for t in T:
                t_out = t + TMFE
                if t_out in T:
                    m.MSB2.add(m.INM[p, mi, t] == m.OUTM[p, mi, t_out])

    # MSB8: QC time then ship out
    m.MSB8 = ConstraintList()
    for p in P:
        for mi in M:
            for t in T:
                t_ship = t + TQC
                if t_ship in T:
                    m.MSB8.add(m.OUTM[p, mi, t] == sum(
                        m.MSO[p, mi, h, j, t_ship] for h in H for j in J))

    # MSB4: return transport
    m.MSB4 = ConstraintList()
    for p in P:
        for mi in M:
            for h in H:
                for j in J:
                    tt3 = TT3[j]
                    for t in T:
                        t_arr = t + tt3
                        if t_arr in T:
                            m.MSB4.add(m.MSO[p, mi, h, j, t] == m.FTD[p, mi, h, j, t_arr])

    # MSB6: hospital inflow
    m.MSB6 = ConstraintList()
    for p in P:
        for h in H:
            for t in T:
                m.MSB6.add(m.INH[p, h, t] == sum(m.FTD[p, mi, h, j, t] for mi in M for j in J))

    # Capacity
    m.CAP1 = ConstraintList()
    for mi in M:
        fcap = data['FCAP'][mi]
        for t in T:
            m.CAP1.add(m.CAP[mi, t] == fcap - sum(
                m.INM[p, mi, tt] for p in P for tt in T
                if tt < t and tt >= t - TMFE))

    m.CAPCON1 = ConstraintList()
    for mi in M:
        for t in T:
            m.CAPCON1.add(
                sum(m.INM[p, mi, t] for p in P)
                - sum(m.OUTM[p, mi, t] for p in P) <= m.CAP[mi, t])

    # Max open facilities
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
    for p in P:
        for c in C:
            for mi in M:
                for j in J:
                    for t in T:
                        m.CON4.add(m.Y1[p, c, mi, j, t] <= m.X1[c, mi])

    m.CON5 = ConstraintList()
    for p in P:
        for mi in M:
            for h in H:
                for j in J:
                    for t in T:
                        m.CON5.add(m.Y2[p, mi, h, j, t] <= m.X2[mi, h])

    # Each patient uses exactly one route
    m.CON6 = ConstraintList()
    for p in P:
        m.CON6.add(sum(m.Y1[p, c, mi, j, t] for c in C for mi in M for j in J for t in T) == 1)

    m.CON7 = ConstraintList()
    for p in P:
        m.CON7.add(sum(m.Y2[p, mi, h, j, t] for mi in M for h in H for j in J for t in T) == 1)

    # Demand
    m.DEM = Constraint(expr=sum(m.INH[p, h, t] for p in P for h in H for t in T) <= len(P))

    # Flow bounds
    m.CON8 = ConstraintList()
    m.CON9 = ConstraintList()
    fmin = data['FMIN']
    fmax = data['FMAX']
    for p in P:
        for c in C:
            for mi in M:
                for j in J:
                    for t in T:
                        m.CON8.add(m.LSR[p, c, mi, j, t] >= m.Y1[p, c, mi, j, t] * fmin)
                        m.CON9.add(m.LSR[p, c, mi, j, t] <= m.Y1[p, c, mi, j, t] * fmax)

    m.CON10 = ConstraintList()
    m.CON11 = ConstraintList()
    for p in P:
        for mi in M:
            for h in H:
                for j in J:
                    for t in T:
                        m.CON10.add(m.MSO[p, mi, h, j, t] >= m.Y2[p, mi, h, j, t] * fmin)
                        m.CON11.add(m.MSO[p, mi, h, j, t] <= m.Y2[p, mi, h, j, t] * fmax)

    # Patient-hospital matching (patient from c_i returns to h_i)
    m.HOSP_MATCH = ConstraintList()
    for p in P:
        for idx_c, c in enumerate(C):
            h = H[idx_c]  # c1->h1, c2->h2, etc.
            inc_val = sum(data['INC'].get((p, c, t), 0) for t in T)
            m.HOSP_MATCH.add(
                sum(m.Y2[p, mi, h, j, t] for mi in M for j in J for t in T) == inc_val)

    # Timing
    m.START = ConstraintList()
    for p in P:
        m.START.add(m.STT[p] == sum(
            data['INC'].get((p, c, t), 0) * t for c in C for t in T))

    m.END = ConstraintList()
    for p in P:
        m.END.add(m.CTT[p] == sum(m.INH[p, h, t] * t for h in H for t in T))

    m.TSEQ = ConstraintList()
    for p in P:
        m.TSEQ.add(m.STT[p] <= m.CTT[p])

    m.TIME = ConstraintList()
    for p in P:
        m.TIME.add(m.TRT[p] == m.CTT[p] - m.STT[p])

    m.ATIME = Constraint(expr=m.ATRT == sum(m.TRT[p] for p in P) / len(P))

    # Deadline constraint (Big-M: LATE[p] >= TRT[p] - DEADLINE[p])
    m.LATECON = ConstraintList()
    for p in P:
        m.LATECON.add(m.LATE[p] >= m.TRT[p] - deadlines[p])

    # Store metadata
    m._group_map = group_map
    m._deadlines = deadlines
    m._patients = P
    m._mfg = M

    return m


def solve_model(model, solver_name='appsi_highs', time_limit=300):
    solver = SolverFactory(solver_name)
    if solver_name in ('appsi_highs', 'highs'):
        solver.options['time_limit'] = time_limit
        solver.options['mip_rel_gap'] = 0.05
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
    group_map = model._group_map
    deadlines_map = model._deadlines

    patients = []
    group_summary = {g: {'count': 0, 'trts': [], 'lates': [], 'deadline': 0}
                     for g in ('high', 'medium', 'low')}

    for p in sorted(P):
        trt  = value(model.TRT[p])
        dl   = deadlines_map[p]
        late = value(model.LATE[p])
        grp  = group_map[p]

        patients.append({
            'id': p, 'group': grp,
            'arrival_day': round(value(model.STT[p]), 1),
            'completion_day': round(value(model.CTT[p]), 1),
            'turnaround': round(trt, 2),
            'deadline': round(dl, 2),
            'lateness': round(late, 2),
            'on_time': late < 0.01,
        })

        if grp in group_summary:
            group_summary[grp]['count'] += 1
            group_summary[grp]['trts'].append(trt)
            group_summary[grp]['lates'].append(late)
            group_summary[grp]['deadline'] = round(dl, 2)

    real_cost = value(
        sum(model.CTM[p] for p in P)
        + sum(model.TTC[p] for p in P)
        + (10476 + 9312) * len(P)
    )
    total_lateness = sum(value(model.LATE[p]) for p in P)
    num_late = sum(1 for p in P if value(model.LATE[p]) > 0.01)

    summary = {}
    for g, info in group_summary.items():
        if info['count'] > 0:
            summary[g] = {
                'count': info['count'],
                'deadline': info['deadline'],
                'avg_trt': round(sum(info['trts']) / len(info['trts']), 2),
                'min_trt': round(min(info['trts']), 2),
                'max_trt': round(max(info['trts']), 2),
                'avg_lateness': round(sum(info['lates']) / len(info['lates']), 2),
                'max_lateness': round(max(info['lates']), 2),
                'num_late': sum(1 for l in info['lates'] if l > 0.01),
            }

    return {
        'tau': tau,
        'num_patients': len(P),
        'real_cost': round(real_cost, 2),
        'avg_trt': round(value(model.ATRT), 2),
        'total_lateness': round(total_lateness, 2),
        'num_late': num_late,
        'all_on_time': num_late == 0,
        'facilities_open': [mi for mi in M if value(model.E1[mi]) > 0.5],
        'group_summary': summary,
        'patients': patients,
    }


def run_experiment(tau=0.0, data_file=None, urgency_config=None,
                   process_config=None, solver_name='highs',
                   random_seed=42, time_limit=300):
    if data_file is None:
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data_N5.dat')
    if process_config is None:
        process_config = DEFAULT_PROCESS

    t0 = time.time()
    data = parse_dat(data_file)
    model = build_concrete(data, tau=tau, urgency_config=urgency_config,
                           process_config=process_config, random_seed=random_seed)
    build_time = time.time() - t0

    t0 = time.time()
    results, ok = solve_model(model, solver_name=solver_name, time_limit=time_limit)
    solve_time = time.time() - t0

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
    # Compute min theoretical TRT
    tls = process_config.get('tls', 1)
    tmfe = process_config.get('tmfe', 7)
    tqc = process_config.get('tqc', 7)
    min_tt = min(int(data['TT1'].get('j1', 1)), int(data['TT1'].get('j2', 2)))
    min_tt3 = min(int(data['TT3'].get('j1', 1)), int(data['TT3'].get('j2', 2)))
    res['min_trt'] = tls + min_tt + tmfe + tqc + min_tt3
    res['process_config'] = process_config
    return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=0.0)
    parser.add_argument('--data', default=None)
    parser.add_argument('--solver', default='appsi_highs')
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--time-limit', type=int, default=120)
    args = parser.parse_args()

    if args.sweep:
        print(f"{'tau':>5}  {'OnTime':>6}  {'Late#':>5}  {'TotLate':>8}  "
              f"{'Cost':>12}  {'AvgTRT':>7}  {'H_DL':>5}  {'M_DL':>5}  {'L_DL':>5}  {'Build':>6}  {'Solve':>6}")
        print('-' * 90)
        for tau_val in [i / 10.0 for i in range(11)]:
            r = run_experiment(tau=tau_val, data_file=args.data,
                               solver_name=args.solver, time_limit=args.time_limit)
            cfg = DEFAULT_URGENCY
            h_dl = cfg['high']['base_due'] + cfg['high']['tolerance'] * (1 - tau_val)
            m_dl = cfg['medium']['base_due'] + cfg['medium']['tolerance'] * (1 - tau_val)
            l_dl = cfg['low']['base_due'] + cfg['low']['tolerance'] * (1 - tau_val)
            if r['solved']:
                ot = 'YES' if r['all_on_time'] else 'NO'
                print(f"{tau_val:>5.1f}  {ot:>6}  {r['num_late']:>5}  {r['total_lateness']:>8.1f}  "
                      f"{r['real_cost']:>12,.0f}  {r['avg_trt']:>7.2f}  {h_dl:>5.1f}  {m_dl:>5.1f}  {l_dl:>5.1f}  "
                      f"{r['build_time']:>5.1f}s  {r['solve_time']:>5.1f}s")
            else:
                print(f"{tau_val:>5.1f}  {'FAIL':>6}  {'—':>5}  {'—':>8}  "
                      f"{'—':>12}  {'—':>7}  {h_dl:>5.1f}  {m_dl:>5.1f}  {l_dl:>5.1f}  "
                      f"{r['build_time']:>5.1f}s  {r['solve_time']:>5.1f}s")
    else:
        r = run_experiment(tau=args.tau, data_file=args.data,
                           solver_name=args.solver, time_limit=args.time_limit)
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            if r['solved']:
                status = 'ALL ON TIME' if r['all_on_time'] else f'{r["num_late"]} LATE'
                print(f"tau={args.tau} | {status} | Cost: {r['real_cost']:,.0f} | Avg TRT: {r['avg_trt']:.2f}")
                print(f"Build: {r['build_time']}s | Solve: {r['solve_time']}s")
                print(f"Facilities: {r['facilities_open']}")
                for g in ('high', 'medium', 'low'):
                    if g in r['group_summary']:
                        s = r['group_summary'][g]
                        late_str = f", LATE: {s['num_late']}/{s['count']}" if s['num_late'] > 0 else ""
                        print(f"  {g:>8}: n={s['count']}, DL={s['deadline']}, "
                              f"TRT=[{s['min_trt']},{s['avg_trt']},{s['max_trt']}]{late_str}")
            else:
                print(f"tau={args.tau} | SOLVER FAILED: {r['termination']}")
