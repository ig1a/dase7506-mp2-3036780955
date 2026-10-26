"""Replay original experiment scripts in a fresh output directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-mlp',action='store_true')
    args=p.parse_args()
    out=args.output.resolve()
    if out.exists():raise ValueError('Choose a new output directory')
    if not (ROOT/'upstream/code/data/candidates/gsm8k_train.npz').exists():
        raise ValueError('Restore teacher data before training')
    out.mkdir(parents=True)
    for path in ROOT.glob('*.py'):shutil.copy2(path,out/path.name)
    shutil.copy2(ROOT/'SOURCE.json',out/'SOURCE.json')
    shutil.copytree(ROOT/'upstream',out/'upstream',ignore=shutil.ignore_patterns('data','__pycache__'))
    (out/'upstream/code/data').symlink_to(ROOT/'upstream/code/data',target_is_directory=True)
    scripts=['run_round.py','run_pairwise.py','run_pairwise_c001.py']
    if args.include_mlp:scripts.append('run_mlp_ranker.py')
    for script in scripts:subprocess.run([sys.executable,'-B',str(out/script)],cwd=out,check=True)
    diagnostic=out/'results/arithmetic-20261026'
    subprocess.run([sys.executable,'-B',str(out/'diagnose_arithmetic.py'),'--out',str(diagnostic)],cwd=out,check=True)
    # The parser is the exact audited implementation; reuse the documented review.
    for name in ['PLAN.json','manual-audit.json','DEVELOPMENT.md']:
        shutil.copy2(ROOT/'records/arithmetic-20261026'/name,diagnostic/name)
    subprocess.run([sys.executable,'-B',str(out/'run_arithmetic.py')],cwd=out,check=True)
    print('Experiment reproduction:',out)


if __name__=='__main__':main()
