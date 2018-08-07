'''
  name: cluster.py
  last modified: 14 mar 18

  cluster analysis using k-means
'''

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import random
import mpld3
import sys 
import time
from utility import write_file
from tqdm import tqdm

def run_cluster(all_files, param_dict, out_file, verbose_level):
    """
    interfaces with the second model 
    """
    if out_file is not None:
        write_file(out_file + "_cluster.txt", "\n\n\n" + time.ctime() + "\n===\n")

    total = 0
    for i in range(0, param_dict['num_runs']):
        total += sermons_cluster(all_files, param_dict, out_file, verbose_level)

    overall_avg = total/param_dict['num_runs'] 

    if verbose_level > 0:
        if out_file is None: 
            print("overall average: {}".format(overall_avg))
        else: 
            write_file(out_file + "_cluster.txt", 
                "overall average: {}".format(overall_avg))

    return (overall_avg)

def format_output(all_files, cluster_list, num_k, kmeans_out, out_file, verbose_level):
    
    # dictionary of dictionaries 
    # ex: 'pure in heart': {0: [0, []], 1: [1, [10, 15]], ...}
    doc_to_clusters = {}
    # dictionary with key sermon to value theme    
    doc_to_theme = {}
    out = ""

    if verbose_level > 0:
        if out_file is None: 
            print(kmeans_out)
        else:
            write_file(out_file + "_cluster.txt", kmeans_out)

    for i in range(0, len(all_files.filenames)):
        # get document name with path removed, e.g. true_saints_part_3
        doc_name = all_files.filenames[i]
        doc_name = doc_name[doc_name.rfind("/") + 1:doc_name.rfind(".")]
        # "whole" document without any subdivison, e.g. true_saints
        doc_name_whole = doc_name[:doc_name.rfind("_part")]
        # find the part of this document, e.g. part 3 of true saints 
        doc_part_num = int(doc_name[doc_name.rfind("_")+1:])
        # we have visited this sermon before. that means this must be another 
        # part of the document, e.g. part 3 or part 8 
        if doc_name_whole in doc_to_clusters:
            cluster_to_freq_list = doc_to_clusters[doc_name_whole]
            # add 1 to the frequency 
            cluster_to_freq_list[cluster_list[i]][0] += 1
            # also append this part number to the list 
            cluster_to_freq_list[cluster_list[i]][1].append(doc_part_num)
        # we have not yet seen this sermon before 
        else:
            # initialize a dictionary entry for this sermon where the key 
            # is the cluster number and the value is an array of size 2 
            # where the first element is the number of parts found in the 
            # cluster and the second element is a list of the exact parts 
            # that are in the cluster 
            # ex: 1: [1, [10]] --> in cluster 1, 1 document from this sermon 
            #                      has been found with name part 10                     
            cluster_to_freq_list = {}
            for j in range(0,num_k):
                cluster_to_freq_list[j] = [0,[]]
            cluster_to_freq_list[cluster_list[i]] = [1,[doc_part_num]] 
            doc_to_clusters[doc_name_whole] = cluster_to_freq_list
            # we also want to get a handle on the theme for each full sermon 
            # for reporting
            doc_to_theme[doc_name_whole] = all_files.target_names[all_files.target[i]]

    cluster_dictionary = {}
    doc_to_best_len = {}
    doc_to_total_len = {}

    for i in range(0,num_k):
        cluster_dictionary[i] = []
    # go through each (full) document 
    for doc, clusters in doc_to_clusters.items():
        # rank the clusters by their frequency, using the first value in the 
        # clusters dictionary, i.e., the frequency 
        sorted_list = sorted(clusters.items(), 
                        key=lambda freq: freq[1][0], reverse= True)
        # get the number of parts in the best cluster for this document 
        doc_to_best_len[doc] = sorted_list[0][1][0]
        # get the total number of parts for this document
        doc_to_total_len[doc] = sorted_list[0][1][0] 
        for i in range(1, num_k):
            doc_to_total_len[doc] += sorted_list[i][1][0] 
        cluster_dictionary[sorted_list[0][0]].append([doc,sorted_list])
    
    # get a list of clusters that definitely have documents in it. this is to 
    # compensate for a rare case where some cluster does not have any full 
    # documents in it, because the majority vote left this cluster empty 
    valid_k_list = []
    for i in range(0, num_k): 
        # list is not empty
        if cluster_dictionary[i]: 
            valid_k_list.append(i)

    # calculate the average
    average_list = []

    for i in range(0, num_k):
        overall_total = 0
        num_in_cluster = 0
        for doc_list in cluster_dictionary[i]:
            num_in_cluster += 1
            overall_total += (doc_to_best_len[doc_list[0]] / doc_to_total_len[doc_list[0]]) * 100
        if i in valid_k_list:
            average_list.append(overall_total/num_in_cluster)
        else:
            # compensate for rare case mentioned above by marking it 
            average_list.append(-1)  
            
    true_average_list = list(filter(lambda a: a != -1, average_list))

    overall_average = sum(true_average_list)/ float(len(true_average_list))
    out += ">> total average: {0:.2f}%\n".format(overall_average)

    # print the results 
    for i in range(0,num_k):
        out += '[Cluster {}]\n'.format(i) 
        for doc_list in cluster_dictionary[i]:
            out += '\t'+doc_list[0]+' ('+doc_to_theme[doc_list[0]]+')'
            out += "({0:.2f}%)\n".format(
                (doc_to_best_len[doc_list[0]] / doc_to_total_len[doc_list[0]])* 100)
            for partial_cluster_list in doc_list[1]:
                out += '\t  - Cluster {}: ['.format(partial_cluster_list[0])
                sorted_parts = sorted(partial_cluster_list[1][1])
                for parts in sorted_parts:
                    out += 'part {} '.format(parts)
                out += ']\n'
        out += "\n"
        average = average_list[i]
        if average != -1: 
            out += " > average: {0:.2f}%\n\n".format(average)
        else: 
            out += " > average: N/A\n\n"

    if out_file is None:
        print(out)
    else:
        write_file(out_file + "_cluster.txt", out)

    return overall_average


def sermons_cluster(all_files, param_dict, out_file, verbose_level):
    """
    much credit owed to brandon rose from http://brandonrose.org/clustering
    """

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(all_files.data)

    # X_train_counts is a sparse matrix to save space.
    # (0, 120) 1 --> means the 120th word in this document
    # appears 1 time. the 0 refers to the fact that the
    # feature dimension is only 1, so it will always be 0.

    # print(count_vect.get_feature_names()[0:100])
    # print(X_train_counts.shape)
    # print(X_train_counts[0,0:100])

    # test_list = list(X_train_counts[0,0:100])
    # print(test_list)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
   
    num_clusters = param_dict['k']

    km = KMeans(n_clusters=num_clusters, n_init= param_dict['n_init'],
        max_iter= param_dict['max_iter'], tol= param_dict['tol'])
    km.fit(X_train_tf)
    clusters = km.labels_.tolist()

    dist = 1 - cosine_similarity(X_train_tf)
    
    # set up colors per clusters using a dict
    cluster_colors = {}
    for i in range(0, num_clusters):
        # generate random hex color
        cluster_colors[i] = ("#%06x" % random.randint(0, 0xFFFFFF))
    #cluster_colors = {0: '#bf80ff', 1: '#d95f02',
    #                  2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

    MDS()

    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    # get all the top words
    #

    out = ""
    # order_centroids is ordered by descending order (::-1), given by index 
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    cluster_names = {}
    for i in tqdm(range(num_clusters)):
        out += "Cluster {} words:".format(i)
        # go for the word
        first_5_words = ''
        for k in range(0,100):
            ind = order_centroids[i,k]
            if k < 5 :
                first_5_words = first_5_words+count_vect.get_feature_names()[ind] + ','
            #word_index = km.cluster_centers_[i,ind]
            total_count = 0
            # go through all the documents and check for this word
            for j in range(0, len(km.labels_)):
                # only increment total count if the document is part of the cluster i
                if km.labels_[j] == i:
                    #total_count += X_train_tf[j, ind]
                    total_count += X_train_counts[j, ind]
            out += ' {}({}) '.format(
                  count_vect.get_feature_names()[ind], total_count)
        cluster_names[i] = str(i)+':'+first_5_words[:-1]
        out += "\n\n"
        
    xs, ys = pos[:, 0], pos[:, 1]

    titles = []
    for i in range(0, len(all_files.target)):
        doc_name = all_files.filenames[i]
        # document name includes path and extension; remove both
        doc_name = doc_name[doc_name.rfind("/") + 1:doc_name.rfind(".")]
        doc_name = doc_name[:doc_name.rfind("_part_")]
        titles.append(doc_name)

    #create data frame that has the result of the MDS plus the cluster numbers and titles
    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

    # group by cluster
    groups = df.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(13, 7))  # set size
    # ax.margins(0)  # Optional, just adds 5% padding to the autoscaling
    #ax.legend(numpoints=5)  # show legend with only 1 point
    count = 0
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                color=cluster_colors[name], label=cluster_names[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')

        ax.tick_params(
            axis='y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

        count += 1

    ax.legend(numpoints=1,fontsize='x-small',loc = 3)

    for i in range(len(df)):
        ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=5)

    if out_file is None: 
        plt.savefig('output_cluster.pdf')
    else:
        plt.savefig(out_file + "_cluster.pdf")
        
    plt.close()

    return format_output(all_files,clusters, num_clusters, 
        out, out_file, verbose_level)