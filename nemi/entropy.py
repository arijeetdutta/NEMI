"""
Takes cluster labels of shape (n_ens,npts)
Calculates entropy and overlap
"""

import pandas as pd
import numpy as np
import os

dummy = True # data switch

# --------------------------------------------------------
# Chose which data to work with
# --------------------------------------------------------
if dummy:

    """
    Use a dummy cluster label data with nc = 4, nens = 50, npts = 10
    """
    np.random.seed(42)  # Set seed for reproducibility
    arrays = []
    while len(arrays) < 50:
        arr = np.random.choice([0, 1, 2, 3], size=10)
        # print(set(arr))
        if len(set(arr)) == 4:  # Ensure all labels 0,1,2,3 are present
            arrays.append(arr)
    lab = np.array(arrays)

else:
    """
    Use real data
    """
    # filename = './nemi_pack_nclust_4_nbr_100_mind_0.0.npy'
    filename = './test_clusters.npy'
    if os.path.exists(filename):
        lab = np.load(filename)
        print("File loaded successfully.")
    else:
        raise FileNotFoundError(f"Error: '{filename}' does not exist. Use dummy=True for dummy data to proceed")    

# --------------------------------------------------------

print(f"NEMI pack shape:'{lab.shape}' ")
print(f"Unique labels:'{np.unique(lab)}' ")
# print(f"Ensemble size:'{lab.shape[0]}' ")
nens = lab.shape[0]

# --------------------------------------------------------

def get_overlap(ensembles,id=0,n_ens=3,max_clusters=None):


    base_id = id
    base_labels = ensembles[base_id]
    compare_ids = [i for i in range(n_ens)]
    compare_ids.pop(base_id)

    num_clusters = int(np.max(base_labels) + 1)


    # if not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap=np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0]))*np.nan

    # print(num_clusters, max_clusters)
    summaryStats=np.zeros((num_clusters, max_clusters))

    # compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?


    # dataVector=[nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]
    dataVector=[ensembles[id] for id, nemi in enumerate(ensembles) if id != base_id]

    # loop over ensemble members, not including the base member
    for compare_cnt, compare_id in enumerate(compare_ids):
        # grab clusters of ensemble member
        compare_labels= dataVector[compare_cnt]

        # go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area) 
        for c1 in range(max_clusters): 
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            # mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1==base_labels)] = 1 
            # # Count numer of entries [Why?] 
            summaryStats[0, c1]=np.sum(data1_M) 

            # go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int) 

                # mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2==compare_labels)] = 1    

                # Sum of flags where the two datasets of that cluster are both present
                num_overlap=np.sum(data1_M*data2_M)       

                #Sum of where they overlap
                num_total=np.sum(data1_M | data2_M)       

                #Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)       
                summaryStats[c2, c1]=(num_overlap / num_total)*100 # Add percentage of coverage

            #Filled in 'summaryStatistics' matrix results of percentage overlaps

        usedClusters = set() # Used to mak sure clusters don't get selected twice
        #Clusters are already sorted by size
        
        sortedOverlapForOneCluster=np.zeros(base_labels.shape, dtype=int)*np.nan
        # go through clusters from (biggest to smallest since they are sorted)
        for c1 in range(max_clusters):  
            sortedOverlapForOneCluster=np.zeros(base_labels.shape, dtype=int)*np.nan
            #print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

            # find biggest cluster in first column, making sure it has not been used
            sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
            biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]

            # record it for later
            usedClusters.add(biggestCluster)

            # Initialize dummy array
            data2_M = np.zeros(base_labels.shape, dtype=int)

            # Select which country is being assessed
            data2_M[np.where(biggestCluster == compare_labels)]=1 # Select cluster being assessed

            sortedOverlapForOneCluster[np.where(data2_M==1)]=1
            sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

    # fill in the base entry in the sorted overlap
    for c1 in range(max_clusters):  
        sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

    return  sortedOverlap

def _entropy(row,i=0):

    data = row['counts']
    L = sum(data)
    n = len(data)
    
    if n != 1:
        ress = 0
        for i in range(n):
            ress =  ress + (data[i]/L * np.log2(data[i]/L))
                            
    else:
                            
        ress = (data[i]/L * np.log2(data[i]/L))
    
    return ress*(-1)

    

def entropy_for_baseids(lab,bid=0,n_ens=50):
    sortedOverlap = get_overlap(lab,id=bid,n_ens=nens,max_clusters=None) # calculates relabelled clusters for a given base_id
    ov = np.nan_to_num(sortedOverlap)
    df = pd.DataFrame(np.argmax(ov,axis=1)).T
    df = df.astype('int64')
    df_c = pd.DataFrame(
        df.stack().groupby(level=0).apply(lambda x: np.unique(x, return_inverse=True, return_counts=True)[2])
    )
    df_c.columns = ['counts']
    ent =  df_c.apply(_entropy, axis=1) # entropy
    EntMax = (-1) * nc * (1/nc) * np.log2(1/nc)
    entropy = np.asanyarray((ent * 100)/EntMax)
    
    return entropy

    



