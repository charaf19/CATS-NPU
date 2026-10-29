"""Small, recorded counterexamples; no production source changes."""
import json
from pathlib import Path
import numpy as np
import openvino.opset13 as ops
from cats_npu.algorithms import MethodRunner
from cats_npu.reference import exact_topk

out = Path(__file__).resolve().parent
rows = []
for method, n, k, tile, initial in [
    ('ov_cpu_native_hier', 65, 10, 16, 1),
    ('ov_cpu_native_adaptive', 65, 10, 16, 3),
    ('ov_npu_native_adaptive', 64, 4, 16, 1),
]:
    x = np.arange(n, dtype=np.float16)
    if 'npu' in method:
        x = (np.arange(n)-n//2).astype(np.float16)*np.nextafter(np.float16(0),np.float16(1))
    row = dict(method=method, n=n, k=k, tile=tile, initial_budget=initial)
    try:
        runner=MethodRunner(method,n,k,tile=tile,batch=2,dtype='f16',initial_budget=initial)
        runner.prepare()
        got=runner.run(x)
        row.update(actual=got.indices.tolist(),expected=exact_topk(x,k).tolist(),
                   correct=bool(np.array_equal(got.indices,exact_topk(x,k))),
                   **got.metadata())
    except Exception as exc:
        row['error']=f'{type(exc).__name__}: {exc}'
    rows.append(row)
for name,a in [('xor',np.arange(16,dtype=np.int32)^1),
               ('cast',(np.arange(16,dtype=np.int32)^1).astype(np.int32))]:
    row=dict(case=name,dtype=str(a.dtype),dtype_char=a.dtype.char,dtype_num=a.dtype.num)
    try: ops.constant(a); row['constructed']=True
    except Exception as exc: row.update(constructed=False,error=str(exc))
    rows.append(row)
(out/'minimal_counterexamples.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
print(json.dumps(rows,indent=2))
