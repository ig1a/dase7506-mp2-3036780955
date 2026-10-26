"""Official prediction plus independent scoring in two fresh CPU processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',required=True,choices=['validation','test','audit'])
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoint.npz')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    checkpoint=args.checkpoint.resolve()
    env=os.environ.copy()
    threads={key:'4' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}
    env.update(threads);env['PYTHONDONTWRITEBYTECODE']='1'
    commands=[
        [sys.executable,'-B',str(ROOT/'upstream/code/predict.py'),'--method',str(ROOT/'submission_arithmetic.py'),'--checkpoint',str(checkpoint),'--split',args.split,'--output',str(out/'predictions.json')],
        [sys.executable,'-B',str(ROOT/'upstream/code/score_predictions.py'),'--split',args.split,'--predictions',str(out/'predictions.json'),'--output',str(out/'official-score.json')]]
    processes=[]
    for name,cmd in zip(['prediction','scoring'],commands):
        start=time.perf_counter()
        result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
        processes.append({'name':name,'wall_seconds':time.perf_counter()-start,'command':cmd,
                          'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        (out/'processes.json').write_text(json.dumps(processes,indent=2)+'\n')
        if result.returncode:raise RuntimeError(result.stderr)
    result={'official_score':json.loads((out/'official-score.json').read_text()),
            'total_prediction_plus_scoring_wall_seconds':sum(x['wall_seconds'] for x in processes),
            'platform':platform.platform(),'threads':threads,'asset_file_bytes':checkpoint.stat().st_size,
            'asset_uncompressed_bytes':sum(x.file_size for x in zipfile.ZipFile(checkpoint).infolist()),
            'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (out/'measurement.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
