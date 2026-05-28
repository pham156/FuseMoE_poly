import argparse
import csv
import math
import re
from pathlib import Path

summary_re = re.compile(r"^(?P<epoch>\d+)\s+(?P<flag>True|False)$|^(?P<split>Train|Validation|Test) router summary (?P<body>.*)$")
entry_re = re.compile(r"(?P<label>L\d+(?:M\d+)?|last):mass\[(?P<mass>[^\]]+)\] sel\[(?P<sel>[^\]]+)\]")
metric_re = re.compile(r"^Current (?P<metric>auc|auprc|f1|macro_f1|acc) (?P<value>[-+0-9.eE]+)")

def parse_vec(s):
    return [float(x) for x in s.split(',')]

def entropy_norm(v):
    total = sum(v)
    if total <= 0:
        return float('nan')
    p = [x / total for x in v]
    k = len(p)
    return -sum(x * math.log(max(x, 1e-12)) for x in p) / math.log(k)

def l1_uniform(v):
    total = sum(v)
    if total <= 0:
        return float('nan')
    p = [x / total for x in v]
    u = 1.0 / len(p)
    return sum(abs(x - u) for x in p)

def max_share(v):
    total = sum(v)
    if total <= 0:
        return float('nan')
    return max(v) / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rows = []
    metric_rows = []
    for log in args.logs:
        path = Path(log)
        epoch = None
        for line in path.read_text(errors='replace').splitlines():
            m_epoch = re.match(r"^(\d+)\s+(True|False)$", line.strip())
            if m_epoch:
                epoch = int(m_epoch.group(1))
                continue
            m = re.match(r"^(Train|Validation|Test) router summary (.*)$", line.strip())
            if m:
                split, body = m.group(1), m.group(2)
                for e in entry_re.finditer(body):
                    mass = parse_vec(e.group('mass'))
                    sel = parse_vec(e.group('sel'))
                    rows.append({
                        'log': str(path),
                        'job_id': path.name.split('_')[0],
                        'epoch': epoch,
                        'split': split,
                        'label': e.group('label'),
                        'mass': ','.join(f'{x:.6f}' for x in mass),
                        'sel': ','.join(f'{x:.6f}' for x in sel),
                        'mass_entropy_norm': entropy_norm(mass),
                        'mass_l1_from_uniform': l1_uniform(mass),
                        'mass_max_share': max_share(mass),
                        'sel_entropy_norm': entropy_norm(sel),
                        'sel_l1_from_uniform': l1_uniform(sel),
                        'sel_max_share': max_share(sel),
                    })
                continue
            mm = metric_re.match(line.strip())
            if mm:
                metric_rows.append({
                    'log': str(path),
                    'job_id': path.name.split('_')[0],
                    'epoch': epoch,
                    'metric': mm.group('metric'),
                    'value': float(mm.group('value')),
                })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        fieldnames = ['log','job_id','epoch','split','label','mass','sel','mass_entropy_norm','mass_l1_from_uniform','mass_max_share','sel_entropy_norm','sel_l1_from_uniform','sel_max_share']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    metric_out = out.with_name(out.stem + '_metrics.csv')
    with metric_out.open('w', newline='') as f:
        fieldnames = ['log','job_id','epoch','metric','value']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(metric_rows)
    print(f'wrote {len(rows)} router rows to {out}')
    print(f'wrote {len(metric_rows)} metric rows to {metric_out}')
    if rows:
        worst_mass = max(rows, key=lambda r: r['mass_l1_from_uniform'])
        worst_sel = max(rows, key=lambda r: r['sel_l1_from_uniform'])
        print('worst mass l1 row:', worst_mass)
        print('worst sel l1 row:', worst_sel)

if __name__ == '__main__':
    main()
