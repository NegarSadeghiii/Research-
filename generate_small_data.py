"""
Generate a reduced .dat file for faster testing.
Takes the first N patients (by arrival day) and shortens the time horizon.
"""
import re, sys

def generate(input_file='Data200_profileA.dat', output_file=None,
             num_patients=15, time_horizon=None):
    with open(input_file) as f:
        txt = f.read()

    # Parse INC to find patients sorted by arrival
    inc_matches = re.findall(r'(p\d+)\s+(c\d+)\s+(\d+)\s+1', txt)
    arrivals = sorted(inc_matches, key=lambda x: int(x[2]))

    # Take first N patients
    selected = arrivals[:num_patients]
    patient_ids = [p for p, c, d in selected]
    max_arrival = max(int(d) for _, _, d in selected)

    if time_horizon is None:
        time_horizon = max_arrival + 30  # enough room for turnaround

    if output_file is None:
        output_file = f'Data_N{num_patients}.dat'

    # Figure out which centers and hospitals are needed
    centers_used = sorted(set(c for _, c, _ in selected))
    # Keep all hospitals (patient returns to matching hospital)
    hospitals = ['h1', 'h2', 'h3', 'h4']

    lines = []
    lines.append('# Auto-generated reduced data')
    lines.append(f'# {num_patients} patients, time horizon {time_horizon}')
    lines.append('')
    lines.append(f'set c := {" ".join(["c1","c2","c3","c4"])};')
    lines.append(f'set h := {" ".join(hospitals)};')
    lines.append('set j := j1 j2;')
    lines.append('set m := m1 m2 m3 m4 m5 m6;')
    lines.append(f'set p := {" ".join(patient_ids)};')
    lines.append('')

    # Copy fixed params from original
    for param in ['CIM', 'CVM', 'FCAP', 'TT1', 'TT3']:
        m = re.search(rf'param {param}\s*:=(.*?);', txt, re.DOTALL)
        if m:
            lines.append(f'param {param} :={m.group(1)};')
            lines.append('')

    # U1 and U3 (transport costs)
    for param in ['U1', 'U3']:
        m = re.search(rf'param {param}\s*:=(.*?);', txt, re.DOTALL)
        if m:
            lines.append(f'param {param} :={m.group(1)};')
            lines.append('')

    # INC — only selected patients
    lines.append('param INC :=')
    for p, c, d in selected:
        lines.append(f'{p} {c} {d} 1')
    lines.append(';')
    lines.append('')

    # Scalar params
    for param_line in re.findall(r'param (FMAX|FMIN|TAD|TLS)\s*:=\s*\S+\s*;', txt):
        m = re.search(rf'param {param_line}\s*:=.*?;', txt)
    for scalar in ['FMAX', 'FMIN', 'TAD', 'TLS']:
        m = re.search(rf'param {scalar}\s*:=\s*(\S+)\s*;', txt)
        if m:
            lines.append(f'param {scalar} := {m.group(1)} ;')

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Generated {output_file}: {num_patients} patients, '
          f'arrivals day {selected[0][2]}–{selected[-1][2]}, '
          f'time horizon {time_horizon}')
    return output_file


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    generate(num_patients=n)
