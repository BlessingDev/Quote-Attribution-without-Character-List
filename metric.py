from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    DBSCAN, 
    AffinityPropagation, 
    HDBSCAN,
    BisectingKMeans
)
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import visualize

def dbscan_cluster(features):
    # features는 numpy 배열로 넘어왔다고 가정
    X = StandardScaler().fit_transform(features)
    
    db = DBSCAN(eps=35, min_samples=5).fit(X)
    
    return db

def affinity_cluster(features):
    X = StandardScaler().fit_transform(features)
    
    ap = AffinityPropagation(preference=-80).fit(X)
    
    return ap

def hdbscan_cluster(features):
    # features는 numpy 배열로 넘어왔다고 가정
    #X = StandardScaler().fit_transform(features)
    
    hdb = HDBSCAN(store_centers="centroid", cluster_selection_method="leaf", cluster_selection_epsilon=0.0, alpha=1.0).fit(features)
    
    return hdb

def bikmeans_cluster(features, centroids):
    
    bik = BisectingKMeans(n_clusters=centroids).fit(features)
    
    return bik

def calc_v_measure_with_af(features, labels, draw_fig=False):
    cluster = affinity_cluster(features)
    labels_to_idx = dict()
    #n_clusters = len(set(cluster.labels_)) - (1 if -1 in cluster.labels_ else 0)
    n_clusters = len(set(cluster.labels_))
    
    for idx in cluster.cluster_centers_indices_:
        if labels_to_idx.get(labels[idx], None) is None:
            labels_to_idx[labels[idx]] = cluster.labels_[idx]

    # 할당 안 된 애들은 클러스터 숫자에 겹치지 않게 레이블 부여
    curl = n_clusters
    for l in labels:
        if labels_to_idx.get(l, None) is None:
            labels_to_idx[l] = curl
            curl += 1
    
    label_indices = [labels_to_idx[l] for l in labels]
    
    fig = visualize.plot_affinity_cluster(cluster, features, labels)
    result = metrics.homogeneity_completeness_v_measure(labels_true=np.array(label_indices), labels_pred=cluster.labels_)
    return *result, fig

def calc_v_measure_with_hdb(features, labels, draw_fig=False):
    cluster = hdbscan_cluster(features)
    labels_to_idx = dict()
    n_clusters = cluster.centroids_.shape[0]
    #n_clusters = len(set(cluster.labels_))
    
    # 각 군집당 가장 많이 등장한 샘플 구하기
    num_cluster_list = []
    for c_idx in range(n_clusters):
        indices = [i for i, c_i in enumerate(cluster.labels_) if c_i == c_idx]

        num_cluster_list.append((c_idx, len(indices)))
    
    num_cluster_list.sort(key=lambda x: x[1], reverse=True)
    
    # 군집에 많은 샘플이 포함된 것부터 할당.
    for c_idx, n in num_cluster_list:
        cluster_indices = cluster.labels_ == c_idx
        clsuter_labels = [l for l, b in zip(labels, cluster_indices) if b]
        cluster_count = Counter(clsuter_labels)
        cluster_counter_list = [(k, v) for k, v in cluster_count.items()]
        cluster_counter_list.sort(key=lambda x: x[1], reverse=True)
        
        # 각 군집에 가장 많이 할당된 레이블을 할당
        assigned_cluster = -1
        idx = 0
        while assigned_cluster == -1 and idx < len(cluster_counter_list):
            label, cluster_count = cluster_counter_list[idx]
            if labels_to_idx.get(label, -1) == -1:
                labels_to_idx[label] = c_idx
                assigned_cluster = c_idx
            
            idx += 1

    # 할당 안 된 애들은 클러스터 숫자에 겹치지 않게 레이블 부여
    curl = n_clusters
    for l in labels:
        if labels_to_idx.get(l, None) is None:
            labels_to_idx[l] = curl
            curl += 1
    
    label_indices = [labels_to_idx[l] for l in labels]
    
    fig = None
    if draw_fig:
        fig = visualize.plot_hdbs_cluster(cluster, features, labels)
    
    result = metrics.homogeneity_completeness_v_measure(labels_true=np.array(label_indices), labels_pred=cluster.labels_)
    return *result, fig

def calc_adjusted_rand_with_hdb(features, labels, draw_fig=False):
    cluster = hdbscan_cluster(features)
    labels_to_idx = dict()
    n_clusters = cluster.centroids_.shape[0]
    #n_clusters = len(set(cluster.labels_))
    
    # 각 군집당 가장 많이 등장한 샘플 구하기
    num_cluster_list = []
    for c_idx in range(n_clusters):
        indices = [i for i, c_i in enumerate(cluster.labels_) if c_i == c_idx]

        num_cluster_list.append((c_idx, len(indices)))
    
    num_cluster_list.sort(key=lambda x: x[1], reverse=True)
    
    # 군집에 많은 샘플이 포함된 것부터 할당.
    for c_idx, n in num_cluster_list:
        cluster_indices = cluster.labels_ == c_idx
        clsuter_labels = [l for l, b in zip(labels, cluster_indices) if b]
        cluster_count = Counter(clsuter_labels)
        cluster_counter_list = [(k, v) for k, v in cluster_count.items()]
        cluster_counter_list.sort(key=lambda x: x[1], reverse=True)
        
        # 각 군집에 가장 많이 할당된 레이블을 할당
        assigned_cluster = -1
        idx = 0
        while assigned_cluster == -1 and idx < len(cluster_counter_list):
            label, cluster_count = cluster_counter_list[idx]
            if labels_to_idx.get(label, -1) == -1:
                labels_to_idx[label] = c_idx
                assigned_cluster = c_idx
            
            idx += 1

    # 할당 안 된 애들은 클러스터 숫자에 겹치지 않게 레이블 부여
    curl = n_clusters
    for l in labels:
        if labels_to_idx.get(l, None) is None:
            labels_to_idx[l] = curl
            curl += 1
    
    label_indices = [labels_to_idx[l] for l in labels]
    
    fig = None
    if draw_fig:
        fig = visualize.plot_hdbs_cluster(cluster, features, labels)
    
    result = metrics.adjusted_rand_score(labels_true=np.array(label_indices), labels_pred=cluster.labels_)
    return result, fig

def calc_adjusted_rand_with_bik(features, labels, draw_fig=False):
    speaker_list = list(set(labels))
    labels_to_idx = dict()
    n_labels = len(speaker_list)
    
    for i, l in enumerate(speaker_list):
        labels_to_idx[l] = i
    
    
    cluster = bikmeans_cluster(features, n_labels)
    
    label_indices = [labels_to_idx[l] for l in labels]
    
    fig = None
    if draw_fig:
        fig = visualize.plot_bikm_cluster(cluster, features, labels)
    
    result = metrics.adjusted_rand_score(labels_true=np.array(label_indices), labels_pred=cluster.labels_)
    return result, fig

def get_pred_accuracies(pred, labels, threshold=0.5):
    batch_size = pred.shape[0]
    sample_size = pred.shape[1]
    
    num_labels = []
    for b in range(batch_size):
        batch_labels = []
        for s in range(sample_size):
            if labels[b][s] == '':
                batch_labels.append(1)
            else:
                batch_labels.append(0)
            
        num_labels.append(batch_labels)
    
    num_labels = torch.Tensor(num_labels).to(torch.int64).to(pred.device)
    
    prob = F.sigmoid(pred)
    pred_labels = (prob > threshold).to(torch.int32).squeeze(2)
    
    corr_num = torch.sum((num_labels == pred_labels).to(torch.int32)).item()
    total_num = torch.numel(pred)
    
    return corr_num / total_num