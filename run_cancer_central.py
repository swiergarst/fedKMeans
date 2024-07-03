from sklearn.cluster import KMeans
from common import *
import numpy as np




n_clients = 3
datasets_fourcancers = []
for i in range(n_clients):
    dset_fourcancers, _ = load_data(i, 'cancer_iid')
    datasets_fourcancers.append(dset_fourcancers)
    
    
dataset_fourcancers = np.vstack(datasets_fourcancers)


k = 4
km = KMeans(n_clusters= k).fit(dataset_fourcancers)
centers_cent = np.copy(km.cluster_centers_)
silh_score_C = calc_silhouette_score(dataset_fourcancers, centers_cent)
simpl_silh_score_C = calc_simpl_silh_score(datasets_fourcancers, centers_cent)

print (silh_score_C, simpl_silh_score_C)