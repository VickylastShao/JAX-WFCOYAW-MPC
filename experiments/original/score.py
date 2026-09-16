"""Offline scoring only; no data from this module enters command formation."""
import json
from pathlib import Path
from contract import require,save,sha

def measure(path,key):
    import numpy as np
    with np.load(path) as z:
        y=np.asarray(z[key],dtype=np.float64)
    require(y.shape==(11400,9),'trace shape')
    d=np.diff(np.concatenate((np.zeros((1,9)),y)),axis=0)
    moving=np.abs(d)>1e-6
    starts=int(np.sum(moving & ~np.concatenate((np.zeros((1,9),dtype=bool),moving[:-1]))))
    reversals=0
    for col in d.T:
        s=np.sign(col[np.abs(col)>1e-6]);reversals+=int(np.sum(s[1:]!=s[:-1]))
    from jax_yaw_net_energy import motion_hours
    hours=float(motion_hours(y[:,None,:],np.zeros((1,9)),xp=np))
    require(abs(hours-float(np.abs(d).sum())/(.3*3600))<1e-10,'shared accounting identity')
    return dict(moving_hours=hours,whole_step_moving_hours=float(moving.sum())*.2/3600,starts=starts,reversals=reversals,travel_deg=float(np.abs(d).sum()),distance_equivalent_hours=float(np.abs(d).sum())/(.3*3600))

def run(stage,config,output):
    import numpy as np
    metadata=json.loads((config/'BASELINES.json').read_text())
    results=json.loads((output/'RESULT.json').read_text())
    require(len(results['pairs'])==len(metadata),'cell count mismatch')
    rows=[]
    for r in results['pairs']:
        cid=r['cell'];b=metadata[cid]
        tr=output/cid/'TRACES.npz'
        require(r['technical_pass'],'technical endpoint')
        new=measure(tr,'mpc_yaw_deg')
        with np.load(tr) as z:
            gen=float(np.asarray(z['mpc_power_watts'],dtype=np.float64).sum())*.2/3.6e6
            zero=float(np.asarray(z['noyaw_power_watts'],dtype=np.float64).sum())*.2/3.6e6
        require(abs(gen-r['mpc_mwh']*1000)<1e-6,'new power integral')
        row={'cell':cid,'E':dict(new,generation_kwh=gen),'zero_kwh':zero}
        for arm in ('G','L'):
            if arm not in b: continue
            bp=Path(b[arm]['path'])
            require(sha(bp)==b[arm]['sha256'],'baseline trace hash')
            old=measure(bp,'mpc_yaw_deg' if arm=='G' else 'yaw_deg')
            with np.load(bp) as z:
                oldgen=float(np.asarray(z['mpc_power_watts' if arm=='G' else 'lookup_power_watts'],dtype=np.float64).sum())*.2/3.6e6
                if arm=='G':
                    oldzero=float(np.asarray(z['noyaw_power_watts'],dtype=np.float64).sum())*.2/3.6e6
                    require(abs(zero-oldzero)<1e-5,'replayed zero baseline mismatch')
            require(abs(oldgen-b[arm]['generation_kwh'])<1e-6,'baseline power integral')
            row[arm]=dict(old,generation_kwh=oldgen)
        rows.append(row)
    scenarios=[]
    for power in (0,10,30,60,100):
        effects=[]
        for r in rows:
            nets={a:r[a]['generation_kwh']-power*r[a]['moving_hours'] for a in ('E','G','L') if a in r}
            nets['zero']=r['zero_kwh']
            effects.append({'cell':r['cell'],'net_kwh':nets,'E_minus_G_kwh':nets['E']-nets['G']})
        sums={a:sum(x['net_kwh'][a] for x in effects) for a in effects[0]['net_kwh']}
        roots={}
        for r in effects:
            root=r['cell'].split('__')[0]
            roots.setdefault(root,[]).append(r)
        root_gains={root:{a:sum(100*(x['net_kwh']['E']/x['net_kwh'][a]-1) for x in group)/len(group) for a in sums if a!='E'} for root,group in roots.items()}
        scenarios.append({'power_kw':power,'net_totals_kwh':sums,'gain_percent':{a:100*(sums['E']/sums[a]-1) for a in sums if a!='E'},'root_mean_gain_percent':root_gains,'cells':effects})
    primary=next(s for s in scenarios if s['power_kw']==30)
    totals={a:{k:sum(r[a][k] for r in rows) for k in ('starts','reversals','travel_deg','moving_hours')} for a in ('E','G','L') if a in rows[0]}
    if stage=='development':
        checks={'net_positive_vs_zero':primary['gain_percent']['zero']>0,
                'three_of_four_positive_vs_zero':sum(x['net_kwh']['E']>x['net_kwh']['zero'] for x in primary['cells'])>=3}
    else:
        checks={'net_positive_vs_zero_and_lookup':all(primary['gain_percent'][a]>0 for a in ('zero','L')),
                'six_root_means_positive':len(primary['root_mean_gain_percent'])==6 and all(v[a]>0 for v in primary['root_mean_gain_percent'].values() for a in ('zero','L'))}
    # Conservative legacy whole-step billing is a prespecified sensitivity, not a new trial.
    whole_step=[]
    for power in (0,10,30,60,100):
        sums={a:sum(r[a]['generation_kwh']-power*r[a]['whole_step_moving_hours'] for r in rows) for a in ('E','G','L') if a in rows[0]}
        sums['zero']=sum(r['zero_kwh'] for r in rows)
        whole_step.append({'power_kw':power,'net_totals_kwh':sums,'gain_percent':{a:100*(sums['E']/sums[a]-1) for a in sums if a!='E'}})
    passed=all(checks.values())
    report={'status':'pass' if passed else 'not_passed','checks':checks,'stage':stage,'technical_pass':True,
            'independent_confirmation':False,'model':'constant input during fractional within-step movement until target reached; shared exact absolute-distance formula; 30 kW scenario, not measurement', 'whole_step_sensitivity':whole_step,
            'totals':totals,'scenarios':scenarios,'rows':rows}
    save(output/'NET_SCORE.json',report)
    return passed
