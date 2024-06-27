import fed_kmeans_V2 as fkm
import argparse
parser = argparse.ArgumentParser(description="running fed kmeans")
parser.add_argument("-r", type=int, default = 200, help ="number of runs")
parser.add_argument("-k", type=int, default = 16, help="k global for k means")
parser.add_argument("-c", type=int, default = 100, help = "number of communication rounds")
parser.add_argument("-b", type=int, default = 0.1, help = "beta, used for selecting dataset")
parser.add_argument("-p", type=int, default = 50, help = "points per cluster, used for selecting some datasets")
parser.add_argument("-n", type=int, default = 1, help = "noise level, used for selecting some datasets")
parser.add_argument("-d", type=str, default = "abl", help = "dataset selection")

args = parser.parse_args()


cluster_wise_V2 = fkm.run_V2(args.k, n_runs= args.r, crounds = args.c, beta = args.b, ppc=args.p, noise = args.n,  dset=args.d)

print(cluster_wise_V2)