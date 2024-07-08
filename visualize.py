import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from collections import Counter

def plot_speaker_pie(labels, title):
    label_counter = Counter(labels)
    
    label_counter_list = [(k, v) for k, v in label_counter.items()]
    label_counter_list.sort(key=lambda x: x[1], reverse=True)
    
    label_counts = [v for _, v in label_counter_list]
    labels_list = [k for k, _ in label_counter_list]
    
    fig = plt.figure(figsize=(16,9), dpi=300)
    ax = fig.add_subplot()
    
    ax.pie(label_counts, labels=labels_list, autopct='%.1f%%', counterclock=False)
    ax.set_title("{0}\ntotal samples: {1}".format(title, len(labels)))
    
    return fig

def plot_scatter_fig(features, labels):
    label_set = set(labels)
    label_list = sorted(list(label_set))
    label_to_idx = dict()

    for l_idx, l in enumerate(label_list):
        label_to_idx[l] = l_idx
    
    norm_x = StandardScaler().fit_transform(features)
    pca = PCA(n_components=3)
    principal_component = pca.fit_transform(norm_x)
    principal_df = pd.DataFrame(data=principal_component, columns = ['component1', 'component2', 'component3'])
    principal_df["labels"] = labels
    explained_ratio = sum(pca.explained_variance_ratio_)

    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(16,9), dpi=300)
    ax = fig.add_subplot(projection='3d')
    cmap = ListedColormap(sns.color_palette("terrain", n_colors=len(label_set)).as_hex())

    for l in label_list:
        l_df = principal_df.loc[principal_df["labels"] == l]
        
        display_label = l
        if l == '':
            display_label = "narrative"
            
        sc = ax.scatter(
            l_df["component1"],
            l_df["component2"],
            l_df["component3"],
            c=cmap.colors[label_to_idx[l]],
            label=display_label,
            alpha=0.2
        )

    ax.set_title("3D Scatter (PCA Exp. Rate={0})".format(round(explained_ratio, 2)))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    
    return fig

def plot_hdbs_cluster(hdbs, features, labels):
    n_clusters = hdbs.centroids_.shape[0]
    cluster_labels = hdbs.labels_
    label_set = set(labels)
    label_list = sorted(list(label_set))
    label_to_idx = dict()
    for l_idx, l in enumerate(label_list):
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
    #explained_ratio = sum(pca.explained_variance_ratio_)

    # Plot 그리기
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(16,9), dpi=300)
    ax = fig.add_subplot()
    label_cmap = ListedColormap(sns.color_palette("terrain", n_colors=len(label_list)).as_hex())
    
    for l in label_list:
        l_df = principal_df.loc[principal_df["labels"] == l]
        
        display_label = l
        if l == '':
            display_label = "narrative"
        
        not_n = l_df[l_df["cluster"] >= 0]
        noise = l_df[l_df["cluster"] < 0]
        ax.scatter(
            not_n["component1"],
            not_n["component2"],
            c=label_cmap.colors[label_to_idx[l]],
            label=display_label,
            alpha=0.5,
            s=8,
            zorder=5
        )
        
        
        if len(not_n) > 0:
            ax.scatter(
                noise["component1"],
                noise["component2"],
                c=label_cmap.colors[label_to_idx[l]],
                alpha=0.2,
                marker="X",
                s=8,
                zorder=0
            )
        else:
            ax.scatter(
                noise["component1"],
                noise["component2"],
                c=label_cmap.colors[label_to_idx[l]],
                label=display_label,
                alpha=0.2,
                marker="X",
                s=8,
                zorder=0
            )

    for k in range(n_clusters):
        cluster_center = centroid_df.iloc[k]
        
        cluster_df = principal_df[principal_df["cluster"] == k]
        cluster_label_counts = cluster_df["labels"].value_counts()
        majority_label = cluster_label_counts.index[0]
        l_col = label_cmap.colors[label_to_idx[majority_label]]
        
        incluster_ratio = cluster_label_counts.iloc[0] / len(cluster_df)
        
        display_label = majority_label
        if display_label == '':
            display_label = "narrative"
        
        ax.scatter(
            cluster_center["component1"], 
            cluster_center["component2"], 
            s=incluster_ratio * 30,
            color=l_col, 
            marker="*",
            alpha=0.9,
            zorder=10
        )
    
    ax.set_title("estimated cluster={0}, real labels={1}\nunclustered samples={2}".format(n_clusters, len(label_set), unclustered_samples))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

    #plt.savefig("scatter_hdbs", bbox_inches='tight')
    return fig

def plot_bikm_cluster(bikm, features, labels):
    n_clusters = bikm.cluster_centers_.shape[0]
    cluster_labels = bikm.labels_
    label_set = set(labels)
    label_list = sorted(list(label_set))
    label_to_idx = dict()
    for l_idx, l in enumerate(label_list):
        label_to_idx[l] = l_idx
    
    unclustered_samples = len(cluster_labels[cluster_labels < 0])

    # PCA 2차원 축소
    features_cent = np.vstack([features, bikm.cluster_centers_])
    norm_x = StandardScaler().fit_transform(features_cent)
    pca = PCA(n_components=2)
    principal_component = pca.fit_transform(norm_x)
    principal_df = pd.DataFrame(data=principal_component[:-n_clusters], columns = ['component1', 'component2'])
    centroid_df = pd.DataFrame(data=principal_component[-n_clusters:], columns = ['component1', 'component2'])
    principal_df["labels"] = labels
    principal_df["cluster"] = cluster_labels
    #explained_ratio = sum(pca.explained_variance_ratio_)

    # Plot 그리기
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(16,9), dpi=300)
    ax = fig.add_subplot()
    label_cmap = ListedColormap(sns.color_palette("terrain", n_colors=len(label_list)).as_hex())
    
    for l in label_list:
        l_df = principal_df.loc[principal_df["labels"] == l]
        
        display_label = l
        if l == '':
            display_label = "narrative"
        
        not_n = l_df[l_df["cluster"] >= 0]
        noise = l_df[l_df["cluster"] < 0]
        ax.scatter(
            not_n["component1"],
            not_n["component2"],
            c=label_cmap.colors[label_to_idx[l]],
            label=display_label,
            alpha=0.5,
            s=8,
            zorder=5
        )
        
        
        if len(not_n) > 0:
            ax.scatter(
                noise["component1"],
                noise["component2"],
                c=label_cmap.colors[label_to_idx[l]],
                alpha=0.2,
                marker="X",
                s=8,
                zorder=0
            )
        else:
            ax.scatter(
                noise["component1"],
                noise["component2"],
                c=label_cmap.colors[label_to_idx[l]],
                label=display_label,
                alpha=0.2,
                marker="X",
                s=8,
                zorder=0
            )

    for k in range(n_clusters):
        cluster_center = centroid_df.iloc[k]
        
        cluster_df = principal_df[principal_df["cluster"] == k]
        cluster_label_counts = cluster_df["labels"].value_counts()
        majority_label = cluster_label_counts.index[0]
        l_col = label_cmap.colors[label_to_idx[majority_label]]
        
        incluster_ratio = cluster_label_counts.iloc[0] / len(cluster_df)
        
        display_label = majority_label
        if display_label == '':
            display_label = "narrative"
        
        ax.scatter(
            cluster_center["component1"], 
            cluster_center["component2"], 
            s=incluster_ratio * 30,
            color=l_col, 
            marker="*",
            alpha=0.9,
            zorder=10
        )
    
    ax.set_title("estimated cluster={0}, real labels={1}\nunclustered samples={2}".format(n_clusters, len(label_set), unclustered_samples))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

    #plt.savefig("scatter_hdbs", bbox_inches='tight')
    return fig

def plot_affinity_cluster(af, features, labels):
    n_clusters = len(set(af.labels_))
    label_set = set(labels)
    label_list = sorted(list(label_set))
    label_to_idx = dict()
    for l_idx, l in enumerate(label_list):
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
    label_cmap = ListedColormap(sns.color_palette("Set2", n_colors=len(label_list)).as_hex())
    #cluster_cmap = ListedColormap(sns.color_palette("tab10", n_colors=n_clusters).as_hex())
    
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