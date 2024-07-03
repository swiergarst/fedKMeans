from fkms.algorithms import kfed
from common import *
import argparse

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


parser = argparse.ArgumentParser(description="running kfed (dennis et. al.")
parser.add_argument("-d", type = str, default = 'abl', help = "which dataset to run: abl, cancer_iid, cancer_niid, FEMNIST")
parser.add_argument("-k", type = int, default = 4, help = "global k")
parser.add_argument("-l", type = int, default = 4, help="local clusters")
# for ablation datasets
parser.add_argument("-b", type=float, default = 0.1, help = "beta, used for selecting some datasets")
parser.add_argument("-p", type=int, default = 50, help = "points per cluster, used for selecting some datasets")
parser.add_argument("-n", type=int, default = 1, help = "noise level, used for selecting some datasets")


args = parser.parse_args()


n_clients = det_n_clients(args.d)

print(f'dataset: {args.d}')


datasets = []
labels = []





for i in range(n_clients):
    dataset, label = load_data(i, args.d, args.b)
    datasets.append(dataset)
    labels.append(label)

_, centers_fed = kfed(datasets, args.l, args.k, useSKLearn = True, sparse = False )
if args.d in ['FEMNIST', "cancer_iid", "cancer_niid"]:
    silh_score = calc_silhouette_score(np.vstack(datasets), centers_fed)
    simpl_silh_score = calc_ssilh_score2(datasets, centers_fed)

    print(silh_score, simpl_silh_score)

else:
    scores = calc_ARI(centers_fed, datasets, labels )    
    print(scores)



