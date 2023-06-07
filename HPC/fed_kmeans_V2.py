import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.cluster import adjusted_rand_score
from common import *








class client_V2():
    def __init__(self, data, labels, n_clusters) : 
        self.data = data
        self.labels = labels #ground-truth labels; purely used for validation
        self.n_clusters = min(data.shape[0], n_clusters)
        self.means, self.sample_amts = self.init_means()
        

    def init_means(self):
        local_clusters, _ = kmeans_plusplus(self.data, self.n_clusters)
        km = KMeans(n_clusters = local_clusters.shape[0]).fit(self.data)
        km.cluster_centers_ = np.copy(local_clusters)

        labels = km.predict(self.data)
        sample_amts = np.array([len(labels[labels == i]) for i in range(self.n_clusters)])
        
        cluster_mask = [sample > 1 for sample in sample_amts]
        local_clusters = local_clusters[cluster_mask]
        sample_amts = sample_amts[cluster_mask]


        return local_clusters, sample_amts
    
    def det_local_clusters(self, score=True):

        km = KMeans(n_clusters = min(self.data.shape[0], self.n_clusters), max_iter = 1).fit(self.data)

        # need to 'unfit' the data
        #km.n_clusters_ = self.n_clusters
        km.cluster_centers_ = np.copy(self.means)

        cluster_labels = km.predict(self.data)  
        
        if score:
            score = adjusted_rand_score(self.labels, cluster_labels)
        else:
            score = None
            
        non_empty_clusters = [ np.where(cluster_labels == i)[0].shape[0] > 0 for i in range(self.n_clusters)]    

        self.means = self.means[non_empty_clusters]
        self.n_clusters = len(self.means)
        km = KMeans(n_clusters = self.n_clusters, init = self.means, max_iter = 1, n_init = 1).fit(self.data)
        self.means = np.copy(km.cluster_centers_)
        sample_amts = np.array([len(km.labels_[km.labels_ == i]) for i in range(self.n_clusters)])
        
        # assert clusters are larger than size 1 (privacy issues)
        cluster_mask = [sample > 1 for sample in sample_amts]
        self.means = self.means[cluster_mask]
        sample_amts = sample_amts[cluster_mask]
        #print(self.data.shape[0], self.n_clusters, sample_amts)

        return(np.copy(self.means), sample_amts, score)

    def calc_local_ssilh_score(self, global_centers):
        km = KMeans(n_clusters = global_centers.shape[0])
        km.cluster_centers_ = np.copy(global_centers)
        km._n_threads = 8
        labels = km.predict(self.data)
        score = 0
        for i, label in enumerate(labels):
            bi = np.min(np.linalg.norm(self.data[i,:] - global_centers[np.arange(global_centers.shape[0]) != label, :], axis = 1))
            ai = np.linalg.norm(self.data[i,:] - global_centers[label,:])
            score_i = (bi - ai)/max(bi,ai)
            score += score_i   
        score /= self.data.shape[0]

        return score, self.data.shape[0] 


class server_V2():
    def __init__(self, n_global):
        self.n_global = n_global
        
    def aggregate(self, local_clusters, samples):
        cluster_aggregator = KMeans(n_clusters = self.n_global)
        avg_means = np.zeros((self.n_global, 2))

        means_res = np.array(local_clusters)
        #cluster_aggregator.fit(means_res, sample_weight =  samples.reshape(-1))
        cluster_aggregator.fit(means_res, sample_weight = samples)
        
        return cluster_aggregator.cluster_centers_
 

def run_V2(n_global,n_runs = 1, crounds = 10, beta = 0.1, dset = 'regular', ppc = 50, noise = 1 ):
    
    if (dset == "FEMNIST"):
        n_clients = 10
        fed_score = False
    else:
        n_clients = 5
        fed_score = True

    scores = np.zeros((n_clients, crounds, n_runs))
    avg_scores = np.zeros((crounds, n_runs))
    for r in range(n_runs):
        local_clusters = []
        clients = []

        cluster_sizes = []

        # create server object
        server = server_V2(n_global)

        # initialize clients, including first versions for cluster means (using n_global)
        for i in range(n_clients):
            data, labels = load_data(i, dset, beta=beta, ppc = ppc, noise=noise)
            client = client_V2(data, labels, n_global)
            clients.append(client)
            local_clusters.append(client.means)
            cluster_sizes.append(client.sample_amts)

        local_clusters = np.concatenate(local_clusters)


        cluster_sizes = np.concatenate(cluster_sizes)

        #print(cluster_sizes)
        for c in range(crounds):
            #print("round:" , c)
            global_clusters = server.aggregate(local_clusters, cluster_sizes)

            local_clusters = []
            cluster_sizes = []
            for i, client in enumerate(clients):
                #client.set_clusters(global_clusters)
                client.means = np.copy(global_clusters)
                client.n_clusters = global_clusters.shape[0]
                local_cluster, cluster_size, scores[i,c,r] = client.det_local_clusters(score = fed_score)
                local_clusters.append(local_cluster)
                cluster_sizes.append(cluster_size)

            local_clusters = np.concatenate(local_clusters)
            cluster_sizes = np.concatenate(cluster_sizes)

    # for FEMNIST, we look at silhouette score (centrally calcualted)        
        if (dset == "FEMNIST"):
            full_dset, _ = load_stacked_data(dset, n_clients)
            avg_scores[0,r] = calc_silhouette_score(full_dset, global_clusters)
            avg_scores[1,r] = calc_simpl_silh_score_fed(clients, global_clusters)
    # calculate the (weighted) mean ARI for all clients combined
    if (dset != "FEMNIST"):
        tot_samples = 0
        #fed_mean = np.zeros(n_runs)
        for client_i, client_o in enumerate(clients):
            n_s = len(client_o.labels)
            avg_scores += n_s * scores[client_i, :, :]
            tot_samples += n_s

        avg_scores /= tot_samples
    return(avg_scores)