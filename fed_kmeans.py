import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.metrics.cluster import adjusted_rand_score
from common import load_data


# this is basically everything that happens 'client-side'
class client():
    def __init__(self, data, labels, n_centroids):
        self.data = data
        self.labels = labels #ground-truth labels; purely used for validation
        #self.n_centroids = n_centroids
        self.means = np.zeros((n_centroids, 2))
        self.n_c_local = 0
    def predict_means(self, r, global_centers, scores_first = False):
        
        # no global centers in first iteration yet, so just take our own initialization
        if r == 0:
            local_centers = self.means
        else:    
            local_centers = self.det_new_local_centers(global_centers)
            #print(local_centers.shape)
        km = KMeans(n_clusters = self.n_c_local, init = local_centers, max_iter = 1).fit(self.data)
        mean_vals = np.copy(km.cluster_centers_)
        self.means = np.copy(km.cluster_centers_)
        samples = self.calc_cluster_amts(km)

        # safety check in case we send a single sample values back as a mean
        inds = np.where(samples == 1)
        mask = np.ones_like(samples, dtype = bool)
        mask[inds] = False
        samples = samples[mask]
        mean_vals = mean_vals[mask]
        
        # restore last global centers if we use those for calculating ARI
        if scores_first == False:
            km.cluster_centers_ = np.copy(local_centers)
            labels = km.predict(self.data)
        else:
            labels = np.copy(km.labels_)
            
        if isinstance(self.labels, np.ndarray):
            score = adjusted_rand_score(self.labels, labels)
        else: 
            score = None

            
        return mean_vals, score, samples
                                
    def calc_cluster_amts(self, km):
        return np.array([len(km.labels_[km.labels_ == i]) for i in range(self.n_c_local)])
        
    def det_new_local_centers(self, global_centers):
        #simple option: calculate distances from data mean
        
        dmean = np.mean(self.data, axis = 0)
        #print(global_centers, dmean)
        dists = np.linalg.norm(global_centers - dmean, axis = 1)
        dist_inds = np.argsort(dists)
        
        gcenter_mat = np.array([global_centers for i in range(self.n_c_local)])
        
        
        # calculate distances between old and new clusters
        distMat = np.sum((global_centers[:, None,:] - self.means[None,:,:])**2, -1)
        
        #print(distMat.shape, self.n_c_local)
        
        selected_means = np.zeros((self.n_c_local, self.data.shape[1]))
        #select the n_c_local amount of clusters from the global clusters
        inds_log = []
        for i in range(distMat.shape[1]):
            ind = np.argmin(distMat[:,i])
            selected_means[i,:] = global_centers[ind,:]
            distMat[ind, :] = 100000000
        #lcenter_mat = np.array([self.means for i in range(global_centers.shape[0])])
        
        #print("centers selected with mean: ", dist_inds[0:self.n_c_local])
        #print("centres selected with cluster distance: ", )
        #return global_centers[dist_inds[0:self.n_c_local]]
        return selected_means
  
    
    def calc_cluster_distance(self, local_clusters):
        # output: a n-1 by n-1 array of distances (could make it n by n with diagonal 0)
        a = np.zeros((local_clusters.shape[0], local_clusters.shape[0]))
        # this is super unoptimized but w/e
        for i in range(local_clusters.shape[0]):
            for j in range(local_clusters.shape[0]):
                a[i,j] = np.linalg.norm(local_clusters[i,:] -  local_clusters[j,:])
        return a   
    
    def init_local_clusters(self, n_cluster_g, thr):

        #check if we have more datapoints than n_cluster_g, otherwise we start with the amt of datapoints as clusters
        
        if self.data.shape[0] >= n_cluster_g:
            local_cluster_amt = n_cluster_g
        else:
            local_cluster_amt = self.data.shape[0]
        #thr = 1.5625
        for j in range(n_cluster_g - 1):
                clusters_l, _  = kmeans_plusplus(self.data, local_cluster_amt)
                a = self.calc_cluster_distance(clusters_l)
                if (a<thr).sum() > local_cluster_amt:
                    local_cluster_amt -= 1
                else:
                    break
        self.n_c_local = local_cluster_amt         
        local_clusters, _ = kmeans_plusplus(self.data, local_cluster_amt)
        self.means = local_clusters
        
        return local_clusters
    
    def det_var(self):
        mean = np.mean(self.data, axis = 0)
        mean2 = np.mean(self.data**2, axis = 0)
        s = self.data.shape[0]
        
        return(mean, mean2, s)

#'server-side' implementation
class server():
    def __init__(self, n_clusters, n_dim):
        self.n_clusters = n_clusters
        self.n_dim = n_dim
        
    def aggregate(self, means, samples):
        
        cluster_aggregator = KMeans(n_clusters = self.n_clusters)
        avg_means = np.zeros((self.n_clusters, self.n_dim))
        
        means_res = means.reshape((-1,self.n_dim))
        #cluster_aggregator.fit(means_res, sample_weight =  samples.reshape(-1))
        cluster_aggregator.fit(means_res, sample_weight = samples)
        
        return cluster_aggregator.cluster_centers_
                         
    
    def det_t(self, means, means2, s):
        var = self.det_global_var(means, means2, s)
        
        std = np.sqrt(var)
        np.place(std, std == 0, 1)
        power = 1/self.n_dim
        
        top = np.prod((2* std)**power)
        t = top / (self.n_clusters**power)
        #power = 0.5
        #print(std* 2)
        #print(std)
        #t = (np.prod(std * 2 )/ (self.n_clusters))**power
        return t
    
    def det_global_var(self, means, means2, s):
        
        # weighted average for both means and means squared
        s_tot = np.sum(s)
        means_tot = np.sum(s * means, axis = 0) / s_tot
        means2_tot = np.sum(s * means2, axis = 0) / s_tot
        
        var = means2_tot - means_tot**2
        return var


#to run everything smoothly
#def run(clients, server, crounds, nclients, n_cluster_global , alpha, Xmin, Xmax, Ymin, Ymax):

def run(clients, server, crounds, nclients, n_cluster_global, dim =2, t = None, scores_first=False):# , alpha, Xmin, Xmax, Ymin, Ymax):

    #initialize: determine global variance, use to calculate t 
    samples = np.zeros((len(clients), 1))
    means = np.zeros((len(clients), dim))
    means2 = np.zeros_like(means)
    for client_i, client in enumerate(clients):
        means[client_i,:] , means2[client_i, :], samples[client_i,:] = client.det_var()
      
    if t == None:
        t = server.det_t(means, means2, samples)
 
    #print(t)
    #var = server.det_global_var(means, means2, samples)
    
    
    #initialize: determine local amount of clusters, and give initial local cluster means
    for client_i, client in enumerate(clients):

        #client.init_local_clusters(n_cluster_global, alpha, Xmin, Xmax, Ymin, Ymax)
        client.init_local_clusters(n_cluster_global, t)
        #print(client.n_c_local)
        ''''
        init_clusters_local = cl
        if client_i == 0:,
            init_clusters = init_clusters_local
        else:
            init_clusters = np.concatenate((init_clusters, init_clusters_local))
        #print(init_clusters_local.shape)
    '''
    #local_clusters = init_clusters
    #samples = np.ones(local_clusters.shape[0])

    scores = np.zeros((len(clients), crounds))
    global_clusters = None
    # communication loop
    for r in range(crounds):
        # central aggregation
                #local cluster selection and kmeans
        for client_i, client in enumerate(clients):
            # this function first determines which global clusters to select, and then does local kmeans with those
            local_cluster, scores[client_i, r], sample = client.predict_means(r, global_clusters, scores_first = scores_first)

            #print(local_cluster.shape)
            #client concatenation 'at the server'
            if client_i == 0:
                local_clusters = local_cluster
                samples = sample
            else:
                local_clusters = np.concatenate((local_clusters, local_cluster), axis = 0)
                samples = np.concatenate((samples, sample), axis = 0)
                
        global_clusters = server.aggregate(local_clusters, samples)
        

        #print(local_clusters.shape, samples.shape)
    #One final aggregation
    global_clusters =  server.aggregate(local_clusters, samples)
    
    return global_clusters, scores

def run_synthetic_experiment(n_cluster_global, crounds, n_runs, data_choice, t = None, beta = None, ppc = None, noise = None, scores_first=False):
    n_clients = 5

    full_scores = np.zeros((crounds, n_clients, n_runs))
    avg_scores = np.zeros((crounds, n_runs))
    all_means = np.zeros((n_cluster_global, 2, n_runs))

    # init clients and server
    #clients = [client(np.genfromtxt(datafiles[i], delimiter=','), np.genfromtxt(labelfiles[i], delimiter=','), n_cluster_global) for i in range(n_clients)]
    clients = []
    for i in range(n_clients):
        data, labels = load_data(i, data_choice, beta = beta, ppc=ppc, noise = noise)
        clients.append(client(data, labels, n_cluster_global))

    server_obj = server(n_cluster_global, 2)


    for r in range(n_runs):
        means_federated, scores = run(clients, server_obj, crounds, n_clients, n_cluster_global, t = t, scores_first = scores_first)#, alpha, Xmin, Xmax, Ymin, Ymax)
        full_scores[:,:,r] = scores.T
        all_means[:,:,r] = means_federated

    # calculate the (weighted) mean ARI for all clients combined
    tot_samples = 0
    #fed_mean = np.zeros(n_runs)
    for client_i, client_o in enumerate(clients):
        n_s = len(client_o.labels)
        avg_scores += n_s * full_scores[:, client_i, :]
        tot_samples += n_s

    avg_scores /= tot_samples
    
    return {
        "scores" : avg_scores,
        "crounds" : crounds,
        "n_runs" : n_runs,
        "t" : t,
        "beta" : beta,
        "ppc" : ppc,
        "noise" : noise,
        "sf" : scores_first}