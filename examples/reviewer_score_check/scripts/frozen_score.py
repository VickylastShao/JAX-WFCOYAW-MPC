"""Predefined joint advancement test on complete fixed F2 development matrix."""
from pathlib import Path
import json,hashlib,sys,math
import numpy as np
base=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cells=[];sources={};technical=[]
for family in ['T1','T2']:
 arms={}
 for arm in ['MPC','LOOKUP']:
  run=base/('B2_F2_'+arm+'_'+family);e=json.loads((run/'EXIT.json').read_text());assert e['exit_code']==0,(family,arm,'exit')
  p=json.loads((run/'PROTOCOL.json').read_text());r=json.loads((run/'results/RESULT.json').read_text());assert r['status']=='completed'
  assert r['protocol_sha256']==sha(run/'PROTOCOL.json') and p['forecast_method']=='F2'
  assert p['cell']['unit']=='OC-73090801' and p['cell']['family']==family
  z=np.load(run/'results/TRACES.npz');power=z['mpc_power_watts' if arm=='MPC' else 'lookup_power_watts'];yaw=z['mpc_yaw_deg' if arm=='MPC' else 'yaw_deg'];zero=z['noyaw_power_watts']
  assert power.shape==yaw.shape==zero.shape==(11400,9) and all(np.isfinite(x).all() for x in [power,yaw,zero])
  change=np.diff(np.concatenate((np.zeros((1,9)),yaw)),axis=0);travel=float(np.abs(change).sum());generation=float(power.astype('float64').sum()*.2/3.6e6);cost=30*travel/(.3*3600);noyaw=float(zero.astype('float64').sum()*.2/3.6e6)
  for key,value in [('generation_kwh',generation),('yaw_motion_kwh',cost),('net_kwh',generation-cost),('noyaw_kwh',noyaw),('yaw_travel_deg',travel)]:assert math.isclose(r[key],value,rel_tol=1e-10,abs_tol=1e-8),(family,arm,key)
  records=[json.loads((run/'results'/f'decision_{t:04d}.json').read_text()) for t in range(0,1800,200)]
  ok=all(x['online_seconds']<=180 and not x['fallback_reasons'] for x in records) and r['physical_checks_pass'] and np.max(np.abs(yaw))<=30.0001 and np.max(np.abs(change))/.2<=.3001
  technical.append(bool(ok));arms[arm]={'result':r,'protocol':p,'records':records,'zero':zero}
  for f in [run/'EXIT.json',run/'PROTOCOL.json',run/'results/RESULT.json',run/'results/TRACES.npz']+list((run/'results').glob('decision_*.json')):sources[str(f)]=sha(f)
 a,b=arms['MPC'],arms['LOOKUP'];assert a['protocol']['cell']==b['protocol']['cell'];assert np.array_equal(a['zero'],b['zero']),'no-yaw paths must be bitwise paired'
 assert [x['forecast_sha256'] for x in a['records']]==[x['forecast_sha256'] for x in b['records']],'forecast packets differ'
 r=a['result'];l=b['result'];cells.append({'family':family,'mpc_net_kwh':r['net_kwh'],'lookup_net_kwh':l['net_kwh'],'noyaw_kwh':r['noyaw_kwh'],'yaw_travel_deg':r['yaw_travel_deg'],'maximum_online_seconds':r['maximum_online_seconds'],'delta_noyaw_kwh':r['net_kwh']-r['noyaw_kwh'],'delta_lookup_kwh':r['net_kwh']-l['net_kwh']})
sums={k:sum(x[k] for x in cells) for k in ['mpc_net_kwh','lookup_net_kwh','noyaw_kwh','yaw_travel_deg','delta_noyaw_kwh','delta_lookup_kwh']}
for name in ['noyaw','lookup']:sums['gain_'+name+'_percent']=100*sums['delta_'+name+'_kwh']/sums[name+'_kwh' if name=='noyaw' else 'lookup_net_kwh']
passed=all(technical) and sums['delta_noyaw_kwh']>0 and sums['delta_lookup_kwh']>0
(out/'RESULT.json').write_text(json.dumps({'status':'complete','candidate':'F2','eligible_for_C':bool(passed),'technical_endpoints_pass':all(technical),'cells':cells,'pooled':sums,'evidence_class':'outcome_exposed_root_fixed_candidate_development','independent_confirmation':False,'F1':'disqualified at T2 numerical qualification; no score','sources':sources},indent=2))
print(json.dumps({'eligible_for_C':bool(passed),'pooled':sums}))
