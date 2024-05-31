from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    DBSCAN, 
    AffinityPropagation, 
    HDBSCAN
)
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

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

def plot_affinity_cluster(af, features, labels):
    n_clusters = len(set(af.labels_))
    label_set = set(labels)
    label_to_idx = dict()
    for l_idx, l in enumerate(label_set):
        label_to_idx[l] = l_idx

    # PCA 3차원 축소
    norm_x = StandardScaler().fit_transform(features)
    pca = PCA(n_components=3)
    principal_component = pca.fit_transform(norm_x)
    principal_df = pd.DataFrame(data=principal_component, columns = ['component1', 'component2', 'component3'])
    principal_df["labels"] = labels
    principal_df["cluster"] = af.labels_
    explained_ratio = sum(pca.explained_variance_ratio_)

    # Plot 그리기
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(16,9), dpi=300)
    ax = fig.add_subplot(projection='3d')
    label_cmap = ListedColormap(sns.color_palette("Set2", n_colors=len(label_set)).as_hex())
    #cluster_cmap = ListedColormap(sns.color_palette("tab10", n_colors=n_clusters).as_hex())
    label_list = sorted(list(label_set))
    
    for l in label_list:
        l_df = principal_df.loc[principal_df["labels"] == l]
        
        display_label = l
        if l == '':
            display_label = "narrative"
        
        sc = ax.scatter(
            l_df["component1"],
            l_df["component2"],
            l_df["component3"],
            c=label_cmap.colors[label_to_idx[l]],
            label=display_label,
            alpha=0.2,
            s=8
        )

    for k in range(n_clusters):
        class_members = (af.labels_ == k)
        cluster_center = principal_df.iloc[af.cluster_centers_indices_[k]]
        '''ax.scatter(
            principal_df.iloc[af.cluster_centers_indices[k]]["component1"], 
            principal_df.iloc[af.cluster_centers_indices[k]]["component2"],
            principal_df.iloc[af.cluster_centers_indices[k]]["component3"],
            color=label_cmap.colors[label_to_idx[principal_df.iloc[af.cluster_centers_indices[k]]["labels"]]], 
            marker=".",
            alpha=0.2
        )'''

        cluster_df = principal_df[principal_df["cluster"] == k]
        majority_label = cluster_df["labels"].value_counts().index[0]
        l_col = label_cmap.colors[label_to_idx[majority_label]]
        ax.scatter(
            cluster_center["component1"], 
            cluster_center["component2"], 
            cluster_center["component3"], 
            s=10, 
            color=l_col, 
            marker="o"
        )

        for x_idx in principal_df[class_members].index:
            x = principal_df.loc[x_idx]
            plt.plot(
                [cluster_center["component1"], x["component1"]], 
                [cluster_center["component2"], x["component2"]], 
                [cluster_center["component3"], x["component3"]], 
                color=l_col,
                alpha=0.2
            )
    
    ax.set_title("estimated cluster={0}, real labels={1}".format(n_clusters, len(label_set)))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

    #plt.savefig("scatter_af", bbox_inches='tight')
    return fig

def plot_hdbs_cluster(hdbs, features, labels):
    n_clusters = hdbs.centroids_.shape[0]
    cluster_labels = hdbs.labels_
    label_set = set(labels)
    label_to_idx = dict()
    for l_idx, l in enumerate(label_set):
        label_to_idx[l] = l_idx
    
    unclustered_samples = len(cluster_labels[cluster_labels < 0])

    # PCA 2차원 축소
    features_cent = np.vstack([features, hdbs.centroids_])
    norm_x = StandardScaler().fit_transform(features_cent)
    pca = PCA(n_components=2)
    principal_component = pca.fit_transform(norm_x)
    principal_df = pd.DataFrame(data=principal_component[:-n_clusters], columns = ['component1', 'component2'])
    centroid_df = pd.DataFrame(data=principal_component[-n_clusters:], columns = ['component1', 'component2'])
    principal_df["labels"] = labels
    principal_df["cluster"] = cluster_labels
    explained_ratio = sum(pca.explained_variance_ratio_)

    # Plot 그리기
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(16,9), dpi=300)
    ax = fig.add_subplot()
    label_cmap = ListedColormap(sns.color_palette("tab10", n_colors=len(label_set)).as_hex())
    label_list = sorted(list(label_set))
    
    for l in label_list:
        l_df = principal_df.loc[principal_df["labels"] == l]
        
        display_label = l
        if l == '':
            display_label = "narrative"
        
        not_n = l_df[l_df["cluster"] >= 0]
        noise = l_df[l_df["cluster"] < 0]
        '''ax.scatter(
            not_n["component1"],
            not_n["component2"],
            not_n["component3"],
            c=label_cmap.colors[label_to_idx[l]],
            label=display_label,
            alpha=0.0,
            s=8
        )'''
        
        
        if len(not_n) > 0:
            ax.scatter(
                noise["component1"],
                noise["component2"],
                c=label_cmap.colors[label_to_idx[l]],
                alpha=0.2,
                marker="X",
                s=8
            )
        else:
            ax.scatter(
                noise["component1"],
                noise["component2"],
                c=label_cmap.colors[label_to_idx[l]],
                label=display_label,
                alpha=0.2,
                marker="X",
                s=8
            )

    displayed_label = set()
    for k in range(n_clusters):
        cluster_center = centroid_df.iloc[k]
        '''ax.scatter(
            principal_df.iloc[af.cluster_centers_indices[k]]["component1"], 
            principal_df.iloc[af.cluster_centers_indices[k]]["component2"],
            principal_df.iloc[af.cluster_centers_indices[k]]["component3"],
            color=label_cmap.colors[label_to_idx[principal_df.iloc[af.cluster_centers_indices[k]]["labels"]]], 
            marker=".",
            alpha=0.2
        )'''
        
        cluster_df = principal_df[principal_df["cluster"] == k]
        cluster_label_counts = cluster_df["labels"].value_counts()
        majority_label = cluster_label_counts.index[0]
        l_col = label_cmap.colors[label_to_idx[majority_label]]
        
        incluster_ratio = cluster_label_counts.iloc[0] / len(cluster_df)
        
        display_label = majority_label
        if display_label == '':
            display_label = "narrative"
        
        if display_label in displayed_label:
            ax.scatter(
                cluster_center["component1"], 
                cluster_center["component2"], 
                s=incluster_ratio * 20,
                color=l_col, 
                marker="o",
                alpha=0.5
            )
        else:
            ax.scatter(
                cluster_center["component1"], 
                cluster_center["component2"], 
                s=incluster_ratio * 20,
                label=display_label,
                color=l_col, 
                marker="o",
                alpha=0.5
            )
            displayed_label.add(display_label)

        '''for x_idx in cluster_df.index:
            x = principal_df.loc[x_idx]
            plt.plot(
                [cluster_center["component1"], x["component1"]], 
                [cluster_center["component2"], x["component2"]], 
                [cluster_center["component3"], x["component3"]], 
                color=l_col,
                alpha=0.2
            )'''
    
    ax.set_title("estimated cluster={0}, real labels={1}\nunclustered samples={2}".format(n_clusters, len(label_set), unclustered_samples))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

    #plt.savefig("scatter_hdbs", bbox_inches='tight')
    return fig

def calc_v_measure_with_af(features, labels):
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
    
    fig = plot_affinity_cluster(cluster, features, labels)
    result = metrics.homogeneity_completeness_v_measure(labels_true=np.array(label_indices), labels_pred=cluster.labels_)
    return *result, fig

def calc_v_measure_with_hdb(features, labels):
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
    
    fig = plot_hdbs_cluster(cluster, features, labels)
    result = metrics.homogeneity_completeness_v_measure(labels_true=np.array(label_indices), labels_pred=cluster.labels_)
    return *result, fig