"""Synthetic failure injection; these timings are never performance evidence."""
import json
from pathlib import Path
from unittest.mock import patch
from cats_npu.algorithms import MethodRunner
from cats_npu.benchmark import bench
from cats_npu.statistics import read_observations, analyze

out=Path(__file__).resolve().parent/'harness_reproduction'
calls=0
original=MethodRunner.run
def inject(self,scores):
    global calls
    calls+=1
    # Preflight=1, first timed observation=2, second timed inference=3.
    if calls==3: raise RuntimeError('SYNTHETIC injected runtime failure')
    return original(self,scores)

error=None
try:
    with patch.object(MethodRunner,'run',inject):
        bench(methods=['cpu_exact'],ns=[16],ks=[2],distributions=['normal'],dtype='f16',
              seed=42,tile=16,batch=1,initial_budget=1,warmup=0,repeats=2,blocks=1,
              out=out,session_id='synthetic_failure',allow_skips=False)
except Exception as exc: error=str(exc)
report=analyze(read_observations([out]))
result=dict(purpose='synthetic failure-handling demonstration; not benchmark data',
            benchmark_error=error,analysis=report)
(out.parent/'harness_reproduction.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
