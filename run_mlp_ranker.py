"""One fixed 32-unit pairwise ranker; train and validation only."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import json
import sys
import time
import copy
import zipfile
import hashlib
import subprocess
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / 'upstream/code'
sys.path.insert(0, str(OFFICIAL))
from common import load_inputs
sys.path.insert(0, str(ROOT))
import submission
import submission_mlp
from run_pairwise import labels, accuracy, digest, write

PLAN = {'seed':17, 'hidden':32, 'activation':'tanh', 'input_dimensions':1028,
        'optimizer':'Adam', 'learning_rate':0.001, 'betas':[0.9,0.999], 'epsilon':1e-8,
        'batch_questions':64, 'max_epochs':100, 'validation_every_epochs':5,
        'early_stopping_patience_epochs':20, 'selection':'highest hard validation correct count; ties keep earliest epoch',
        'l2_lambda':0.001, 'l2_definition':'0.5 * lambda * (sum(w1^2) + sum(w2^2)); no bias penalty',
        'objective':'mean over informative questions of mean correct-vs-valid-wrong pair softplus(-(s_pos-s_neg)), plus L2',
        'dtype':'float32', 'threads':4, 'configurations':1, 'fits':1, 'dropout':0,
        'scaler':'Reuse identical train-only scaler from pairwise C=0.01; features unchanged',
        'test_accessed':False, 'baseline':'artifacts/pairwise-C001-20261026.npz',
        'source':'https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf (pairwise objective, section 3; not exact reproduction)'}


def loss_grad(z, y, valid, p, lam):
    flat = z.reshape(-1, z.shape[-1])
    h = np.tanh(flat @ p['w1'] + p['b1'])
    scores = (h @ p['w2']).reshape(len(z),4)
    mask = y[:,:,None] & (valid & ~y)[:,None,:]
    counts = mask.sum(axis=(1,2))
    assert np.all(counts > 0)
    weight = mask.astype(z.dtype) / counts[:,None,None].astype(z.dtype) / len(z)
    delta = scores[:,:,None] - scores[:,None,:]
    data_loss = np.sum(weight * np.logaddexp(0, -delta))
    # d softplus(-delta) / d delta = -sigmoid(-delta), stable for either sign.
    gd = -weight * np.exp(-np.logaddexp(0, delta))
    gs = (gd.sum(axis=2) - gd.sum(axis=1)).reshape(-1)
    gh = gs[:,None] * p['w2'][None,:] * (1-h*h)
    grads = {'w1':flat.T @ gh + lam*p['w1'], 'b1':gh.sum(axis=0),
             'w2':h.T @ gs + lam*p['w2']}
    loss = data_loss + 0.5*lam*(np.sum(p['w1']**2)+np.sum(p['w2']**2))
    return float(loss), grads


def preflight():
    rng=np.random.default_rng(913)
    z=rng.normal(size=(3,4,5))
    y=np.array([[1,1,0,0],[1,0,0,0],[0,1,1,0]],dtype=bool)
    valid=np.array([[1,1,1,1],[1,0,1,1],[1,1,1,0]],dtype=bool)
    p={'w1':rng.normal(scale=.2,size=(5,3)), 'b1':rng.normal(scale=.1,size=3), 'w2':rng.normal(scale=.2,size=3)}
    value,g=loss_grad(z,y,valid,p,.001)
    errors=[]
    for key in p:
        for index in np.ndindex(p[key].shape):
            old=p[key][index];eps=1e-6
            p[key][index]=old+eps;a,_=loss_grad(z,y,valid,p,.001)
            p[key][index]=old-eps;b,_=loss_grad(z,y,valid,p,.001)
            p[key][index]=old
            errors.append(abs((a-b)/(2*eps)-g[key][index]))
    assert max(errors)<1e-7
    perm=[2,0,3,1]
    moved,mg=loss_grad(z[:,perm],y[:,perm],valid[:,perm],p,.001)
    np.testing.assert_allclose(value,moved,atol=1e-12)
    for key in g: np.testing.assert_allclose(g[key],mg[key],atol=1e-12)
    changed=z.copy();changed[~valid]=1000
    ignored,_=loss_grad(changed,y,valid,p,.001)
    np.testing.assert_allclose(value,ignored,atol=1e-12)
    return {'gradient_max_abs_error':max(errors), 'permutation_loss_and_gradient':'passed',
            'invalid_candidates_excluded':'passed', 'synthetic_only':True, 'training_runs':0}


def selections(scores, valid):
    s=scores.copy();s[~valid]=-np.inf
    return s.argmax(axis=1)


def main():
    out=ROOT/'results/mlp32-20261026';out.mkdir(exist_ok=False)
    ckpt=ROOT/'artifacts/mlp32-20261026.npz';assert not ckpt.exists()
    baseline=ROOT/PLAN['baseline']
    frozen={str(p.relative_to(ROOT)):digest(p) for p in [baseline,ROOT/'submission.py',ROOT/'run_pairwise_c001.py']}
    write(out/'PLAN.json',dict(PLAN,frozen=frozen))
    write(out/'preflight.json',preflight())
    tb,ta=load_inputs('train');ty,tvalid,th,_=labels(tb,'train')
    vb,va=load_inputs('validation');vy,vvalid,vh,answers=labels(vb,'validation')
    with np.load(baseline) as m: mean=m['mean'].copy();scale=m['scale'].copy()
    tz=((submission.make_features(tb,ta,'context')-mean)/scale).astype(np.float32)
    vz=((submission.make_features(vb,va,'context')-mean)/scale).astype(np.float32)
    informative=np.flatnonzero((ty[:,:,None] & (tvalid & ~ty)[:,None,:]).any(axis=(1,2)))
    pair_count=int((ty[:,:,None] & (tvalid & ~ty)[:,None,:]).sum())
    assert len(informative)==2114 and pair_count==6516
    rng=np.random.default_rng(PLAN['seed']);d=tz.shape[-1];h=PLAN['hidden']
    p={'w1':rng.uniform(-np.sqrt(6/(d+h)),np.sqrt(6/(d+h)),size=(d,h)).astype(np.float32),
       'b1':np.zeros(h,dtype=np.float32),
       'w2':rng.uniform(-np.sqrt(6/(h+1)),np.sqrt(6/(h+1)),size=h).astype(np.float32)}
    moments={k:np.zeros_like(v) for k,v in p.items()};variances=copy.deepcopy(moments)
    best=None;best_count=-1;best_epoch=0;step=0;curve=[]
    started=time.perf_counter();update_seconds=0.0
    for epoch in range(1,PLAN['max_epochs']+1):
        order=rng.permutation(informative)
        tick=time.perf_counter()
        for start in range(0,len(order),PLAN['batch_questions']):
            ix=order[start:start+PLAN['batch_questions']]
            loss,grads=loss_grad(tz[ix],ty[ix],tvalid[ix],p,PLAN['l2_lambda'])
            assert np.isfinite(loss)
            step+=1
            for key in p:
                moments[key]=.9*moments[key]+.1*grads[key]
                variances[key]=.999*variances[key]+.001*grads[key]**2
                p[key]-=PLAN['learning_rate']*(moments[key]/(1-.9**step))/(np.sqrt(variances[key]/(1-.999**step))+PLAN['epsilon'])
        update_seconds+=time.perf_counter()-tick
        if epoch % PLAN['validation_every_epochs']==0:
            vs=submission_mlp.network(vz,p);ts=submission_mlp.network(tz,p)
            vm,_=accuracy(vy,vh,selections(vs,vvalid));tm,_=accuracy(ty,th,selections(ts,tvalid))
            objective,_=loss_grad(tz[informative],ty[informative],tvalid[informative],p,PLAN['l2_lambda'])
            row={'epoch':epoch,'train':tm,'validation':vm,'train_objective':objective}
            curve.append(row)
            with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if vm['hard_correct']>best_count:
                best_count=vm['hard_correct'];best_epoch=epoch;best={k:v.copy() for k,v in p.items()}
            if epoch-best_epoch>=PLAN['early_stopping_patience_epochs']:break
    fit_wall=time.perf_counter()-started
    assert best is not None
    np.savez(ckpt,mean=mean,scale=scale,**best)
    expected=submission_mlp.network(vz,best)
    exported=submission_mlp.decision_scores(vb,va,ckpt)
    np.testing.assert_allclose(expected,exported,atol=1e-6,rtol=1e-6)
    perm=[2,0,3,1]
    shuffled=[dict(row,candidates=[dict(row['candidates'][i],index=j) for j,i in enumerate(perm)]) for row in vb[:20]]
    shuffled_scores=submission_mlp.decision_scores(shuffled,{'embeddings':va['embeddings'][:20,perm]},ckpt)
    np.testing.assert_allclose(shuffled_scores,exported[:20,perm],atol=2e-6,rtol=2e-6)
    original_choices=submission_mlp.predict(vb[:20],{'embeddings':va['embeddings'][:20]},ckpt)
    moved_choices=submission_mlp.predict(shuffled,{'embeddings':va['embeddings'][:20,perm]},ckpt)
    assert original_choices==[perm[i] for i in moved_choices]
    predictions=ROOT/'results/predictions/mlp32-20261026.json'
    commands=[
      [sys.executable,'-B',str(OFFICIAL/'predict.py'),'--method',str(ROOT/'submission_mlp.py'),'--checkpoint',str(ckpt),'--split','validation','--output',str(predictions)],
      [sys.executable,'-B',str(OFFICIAL/'score_predictions.py'),'--split','validation','--predictions',str(predictions),'--output',str(out/'official-score.json')]]
    processes=[]
    for command in commands:
        tick=time.perf_counter();result=subprocess.run(command,cwd=OFFICIAL,capture_output=True,text=True,check=True)
        processes.append({'command':command,'seconds':time.perf_counter()-tick,'stdout':result.stdout,'stderr':result.stderr})
    choices=np.array(json.loads(predictions.read_text())['selected']);metrics,ok=accuracy(vy,vh,choices)
    assert choices.tolist()==submission.choose(vb,expected)
    assert json.loads((out/'official-score.json').read_text())['correct']==metrics['correct']
    train_metrics,_=accuracy(ty,th,selections(submission_mlp.network(tz,best),tvalid))
    old_choices=np.array(submission.predict(vb,va,baseline));old_metrics,old_ok=accuracy(vy,vh,old_choices)
    cases=[]
    for category,mask in [('recovered',~old_ok&ok),('regressed',old_ok&~ok)]:
        for n in np.flatnonzero(mask)[:5]:
            cases.append({'category':category,'id':vb[n]['id'],'question':vb[n]['question'],'answers':answers[n],
                          'correct_indices':np.flatnonzero(vy[n]).tolist(),'old_choice':int(old_choices[n]),'new_choice':int(choices[n]),
                          'candidates':[c['text'] for c in vb[n]['candidates']]})
    assert frozen=={name:digest(ROOT/name) for name in frozen}
    summary={'plan':PLAN,'parameter_count':sum(v.size for v in best.values()),'informative_questions':len(informative),'pairs_per_epoch':pair_count,
             'epochs_run':epoch,'best_epoch':best_epoch,'validation_checks':len(curve),'stopped_early':epoch<PLAN['max_epochs'],
             'train':train_metrics,'validation':metrics,'baseline_validation':old_metrics,'last_check':curve[-1],
             'paired':{'recovered':int((~old_ok&ok).sum()),'regressed':int((old_ok&~ok).sum()),'both_wrong':int((~old_ok&~ok).sum())},
             'training_update_seconds':update_seconds,'fit_wall_including_validation_seconds':fit_wall,'processes':processes,
             'fresh_prediction_and_scoring_seconds':sum(v['seconds'] for v in processes),'asset_bytes':ckpt.stat().st_size,
             'asset_uncompressed_bytes':sum(v.file_size for v in zipfile.ZipFile(ckpt).infolist()),'checkpoint_sha256':digest(ckpt),
             'export_max_error':float(np.max(np.abs(exported-expected))),'candidate_permutation_check':'passed','baseline_unchanged':True,
             'hardware':'Apple M5; local CPU, not reference Xeon qualification','numpy':np.__version__,'test_accessed':False,
             'interpretation_limit':'One nonlinear recipe versus fixed linear baseline; optimizer, regularization and validation early selection also differ.'}
    write(out/'summary.json',summary);write(out/'error-cases.json',cases)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
