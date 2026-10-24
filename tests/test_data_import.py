import numpy as np
from cats_npu.data_import import make_scores_from_embeddings
from cats_npu.reference import exact_topk


def test_import_local_embeddings(tmp_path):
    emb = np.array([[1,0], [0,1], [0.6,0.8], [-1,0]], dtype=np.float32)
    q = np.array([[0.8,0.6]], dtype=np.float32)
    np.save(tmp_path/'x.npy', emb)
    np.save(tmp_path/'q.npy', q)
    meta = make_scores_from_embeddings(tmp_path/'x.npy', tmp_path/'q.npy', 0,
                                       tmp_path/'scores.npy', dtype='f32', chunk=2)
    x = np.load(tmp_path/'scores.npy')
    assert meta['n']==4
    np.testing.assert_allclose(x, emb @ q[0], atol=1e-6)
    assert exact_topk(x, 1).tolist() == [2]
