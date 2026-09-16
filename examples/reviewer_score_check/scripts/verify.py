"""Verify package identity, replay frozen trace accounting, compare scientific outputs."""
from pathlib import Path
import json,hashlib,subprocess,sys,tempfile,math
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((P/'MANIFEST.json').read_text())
for n,v in manifest['files'].items():
 f=P/n
 if not f.is_file() or sha(f)!=v['sha256']:raise SystemExit('Integrity failure: '+n)
with tempfile.TemporaryDirectory(prefix='les-mpc-score-check-') as tmp:
 out=Path(tmp)/'score';subprocess.run([sys.executable,str(P/'scripts/frozen_score.py'),str(P/'data'),str(out)],check=True)
 got=json.loads((out/'RESULT.json').read_text());expected=json.loads((P/'results/expected/RESULT.json').read_text())
 def compare(a,b):
  if isinstance(b,dict):
   assert set(a)==set(b)
   for k in b:compare(a[k],b[k])
  elif isinstance(b,list):
   assert len(a)==len(b)
   for aa,bb in zip(a,b):compare(aa,bb)
  elif isinstance(b,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)
  else:assert a==b,(a,b)
 for key in ['status','candidate','eligible_for_C','technical_endpoints_pass','cells','pooled','evidence_class','independent_confirmation','F1']:compare(got[key],expected[key])
 print(json.dumps({'verification':'pass','trace_arms':4,'scientific_outcome':'F2 technical pass; negative pooled contrasts; no advancement','solver_rerun':False,'python':sys.version.split()[0]},indent=2))
