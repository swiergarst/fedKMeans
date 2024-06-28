# import fed_kmeans_V2 as fkm
from fed_kmeans import run
from common import *
import argparse
parser = argparse.ArgumentParser(description="running fed kmeans")
parser.add_argument("-r", type=int, default = 200, help ="number of runs")
parser.add_argument("-k", type=int, default = 16, help="k global for k means")
parser.add_argument("-c", type=int, default = 100, help = "number of communication rounds")
parser.add_argument("-b", type=int, default = 0.1, help = "beta, used for selecting dataset")
parser.add_argument("-p", type=int, default = 50, help = "points per cluster, used for selecting some datasets")
parser.add_argument("-n", type=int, default = 1, help = "noise level, used for selecting some datasets")
parser.add_argument("-d", type=str, default = "abl", help = "dataset selection")
parser.add_argument("-i", type=str, default="k-means++", help = "initialization type (kmeans++ or random)")
parser.add_argument("-l", type=int, default = 1,help="amount of local iterations")
parser.add_argument("-e", type=int, default=1, help = "whether to drop empty clusters or not")
parser.add_argument("-w", type=int, default=1, help="whether to run weighted or unweighted aggregation")

args = parser.parse_args()

# convert from int to bool
if args.e == 1:
    drop = True
elif args.e == 0:
    drop = False

if args.w == 1:
    weighted = True
elif args.w == 0:
    weighted = False



dset_config = load_dset_config(beta = 1, ppc = 50, noise = 1)
config = load_config(n_runs = 1, k_global = args.k, crounds = args.c, dset = args.d, init = args.i, iter_local = args.l, drop = drop, weighted_agg = weighted, dset_options = dset_config)


results = run(config)

print(results)
