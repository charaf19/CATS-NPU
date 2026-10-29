"""Audit evidence only. Run from the repository root using the existing venv."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from cats_npu.algorithms import MethodRunner
from cats_npu.backends import _native_model, _ov_import
from cats_npu.reference import exact_topk
from cats_npu.workloads import DISTRIBUTIONS, scores_for

OUT = Path(__file__).resolve().parent

def digest(x):
    return hashlib.sha256(x.tobytes()).hexdigest()

def main():
    records = []
    # B is tiles per device call, not independent query batch size.
    configs = [(64, 4, 16, 1, 1), (65, 10, 16, 2, 3),
               (33, 33, 16, 2, 1), (17, 1, 16, 2, 1)]
    for dtype in ('f16', 'f32'):
        for method in ('ov_cpu_native_hier', 'ov_npu_native_direct',
                       'ov_npu_native_hier', 'ov_npu_native_adaptive'):
            for n, k, tile, batch, initial in configs:
                try:
                    runner = MethodRunner(method, n, k, tile, batch, dtype, initial)
                    runner.prepare()
                except Exception as exc:
                    records.append(dict(method=method, dtype=dtype, n=n, k=k,
                                        stage='prepare', error=f'{type(exc).__name__}: {exc}'))
                    continue
                for dist in (*DISTRIBUTIONS, 'all_equal', 'signed_zero', 'subnormal', 'extreme'):
                    x = scores_for(n, k, dist if dist in DISTRIBUTIONS else 'normal', 471, dtype)
                    if dist == 'all_equal': x[:] = 1
                    if dist == 'signed_zero': x[:] = 0; x[::2] = -0.0
                    if dist == 'subnormal':
                        step = np.nextafter(x.dtype.type(0), x.dtype.type(1))
                        x = (np.arange(n) - n//2).astype(x.dtype) * step
                    if dist == 'extreme':
                        x = np.linspace(-1, 1, n).astype(x.dtype) * np.finfo(x.dtype).max
                    row = dict(method=method, dtype=dtype, n=n, k=k, tile=tile,
                               tile_batch=batch, initial_budget=initial, distribution=dist,
                               input_sha256=digest(x), stage='infer', device_info=runner.device_info)
                    expected = exact_topk(x, k)
                    try:
                        result = runner.run(x)
                        row.update(correct=bool(np.array_equal(result.indices, expected)),
                                   actual=result.indices.tolist(), expected=expected.tolist(),
                                   **result.metadata())
                    except Exception as exc:
                        row.update(correct=False, error=f'{type(exc).__name__}: {exc}')
                    records.append(row)
                    if not row.get('correct', False):
                        np.savez(OUT / f"failure_{method}_{dtype}_N{n}_K{k}_{dist}.npz", scores=x, expected=expected)
    ov, ops = _ov_import()
    core = ov.Core()
    raw = []
    for dtype in ('f16', 'f32'):
        model = _native_model(ov, ops, 1, 16, 4, dtype)
        cm = core.compile_model(model, 'NPU')
        req = cm.create_infer_request()
        for dist in ('normal', 'tied', 'all_equal', 'one_ulp'):
            x = scores_for(16, 4, dist if dist in DISTRIBUTIONS else 'normal', 471, dtype)
            if dist == 'all_equal': x[:] = 1
            if dist == 'one_ulp':
                x[:] = 1
                for i in range(1,16): x[i] = np.nextafter(x[i-1], x.dtype.type(2))
            req.infer({0:x.reshape(1,16)})
            values = np.array(req.get_output_tensor(0).data, copy=True).ravel()
            ids = np.array(req.get_output_tensor(1).data, copy=True).ravel()
            expected = exact_topk(x,4)
            raw.append(dict(dtype=dtype, distribution=dist,
                            execution_devices_raw=cm.get_property('EXECUTION_DEVICES'),
                            indices=ids.tolist(), expected=expected.tolist(),
                            deterministic_correct=bool(np.array_equal(ids,expected)),
                            selected_values_match=bool(np.array_equal(values,x[ids]))))
    reshapes = []
    for dtype in (np.float16, np.float32):
        x = np.arange(16, dtype=dtype).reshape(1,16) / 8 - 1
        p = ops.parameter([1,16], dtype=dtype)
        model = ov.Model([ops.reshape(p,ops.constant(np.array([2,8],np.int64)),False)],[p])
        row = dict(dtype=str(x.dtype), operation='Reshape', device_requested='NPU')
        try:
            cm = core.compile_model(model,'NPU')
            req = cm.create_infer_request()
            req.infer({0:x})
            y = np.array(req.get_output_tensor(0).data, copy=True)
            row.update(compiled=True,inferred=True,correct=bool(np.array_equal(y,x.reshape(2,8))),
                       execution_devices_raw=cm.get_property('EXECUTION_DEVICES'))
        except Exception as exc: row['error'] = f'{type(exc).__name__}: {exc}'
        reshapes.append(row)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  purpose='correctness audit; timings are not performance comparisons',
                  observations=records, raw_topk=raw, reshape=reshapes,
                  summary=dict(tested=len(records),correct=sum(r.get('correct',False) for r in records),
                               failed=sum(not r.get('correct',False) for r in records)))
    (OUT/'targeted_correctness.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps(report['summary']))
    for row in records:
        if not row.get('correct',False):
            print(json.dumps({key:row.get(key) for key in ('method','dtype','n','k','distribution','error')},default=str))
    print(json.dumps(dict(raw_topk=raw,reshape=reshapes),indent=2,default=str))

if __name__ == '__main__': main()
