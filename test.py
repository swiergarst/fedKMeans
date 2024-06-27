import pytest
from fed_kmeans import run
from common import *





def test_FEMNIST():
    pass

# @pytest.mark.parametrize("ppc", [50, 100, 200])
# @pytest.mark.parametrize("noise", [1, 1.1, 1.2, 1.3, 1.4, 1.5])
# @pytest.mark.parametrize("beta", [0.1, 1, 10])
# def test_orig_abl(ppc, noise, beta):
#     crounds = 2
#     dset = "abl"
#     k_global = 16
    
#     dset_config = load_dset_config(beta = beta, ppc = ppc, noise = noise)
#     config = load_config(n_runs = 1, k_global = k_global, crounds = crounds, dset = dset, init = 'k-means++', iter_local = 1, drop = True, weighted_agg = True, dset_options = dset_config)

#     scores = run(config)
#     assert np.all(scores < 1) and np.all(scores > 0)


@pytest.mark.parametrize("init", ['k-means++', 'random'])
@pytest.mark.parametrize("iter_local", [1, 10, 20])
@pytest.mark.parametrize("drop", [True, False])
@pytest.mark.parametrize('weighted_agg', [True, False])
def test_extra_abl(init, iter_local, drop, weighted_agg):
    crounds = 2
    dset = "abl"
    k_global = 16
    
    dset_config = load_dset_config(beta = 1, ppc = 50, noise = 1)
    config = load_config(n_runs = 1, k_global = k_global, crounds = crounds, dset = dset, init = init, iter_local = iter_local, drop = drop, weighted_agg = weighted_agg, dset_options = dset_config)

    scores = run(config)
    assert np.all(scores < 1) and np.all(scores > 0)

