"""Reproduce the final fixed model from teacher training data only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream/code'))
sys.path.insert(0,str(ROOT))
from common import load_inputs, read_jsonl, parse_answer
import submission_arithmetic as method


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('Choose a new output checkpoint')
    t=time.perf_counter()
    bank,arrays=load_inputs('train')
    gold=read_jsonl(ROOT/'upstream/code/data/gsm8k_train.jsonl')
    assert [r['id'] for r in bank]==[r['id'] for r in gold]
    answers=method.parsed_answers(bank)
    valid=np.array([[v is not None for v in aa] for aa in answers])
    y=np.array([[v is not None and v==parse_answer(g['answer']) for v in aa] for aa,g in zip(answers,gold)])
    x=method.make_features(bank,arrays,'arithmetic')
    scaler=StandardScaler().fit(x.reshape(-1,x.shape[-1]))
    z=scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape)
    differences,weights=[],[]
    informative=0
    for q in range(len(bank)):
        pos=np.flatnonzero(y[q]);neg=np.flatnonzero(valid[q]&~y[q])
        count=len(pos)*len(neg)
        if not count:continue
        informative+=1
        for i in pos:
            for j in neg:
                differences.append(z[q,i]-z[q,j]);weights.append(1/(2*count))
    d=np.asarray(differences);w=np.asarray(weights)
    model=LogisticRegression(C=.01,fit_intercept=False,solver='lbfgs',max_iter=2000,tol=1e-6,random_state=17)
    tick=time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(np.concatenate((d,-d)),np.r_[np.ones(len(d)),np.zeros(len(d))],sample_weight=np.r_[w,w])
    fit=time.perf_counter()-tick
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,feature_set=np.array('arithmetic'),mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_[0],intercept=np.array(0.0))
    info={'training_questions':len(bank),'informative_questions':informative,'positive_negative_pairs':len(d),
          'C':.01,'seed':17,'iterations':int(model.n_iter_[0]),'fit_seconds':fit,
          'total_training_seconds':time.perf_counter()-t,
          'checkpoint_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
          'numpy':np.__version__,'sklearn':__import__('sklearn').__version__}
    args.output.with_suffix('.training.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
