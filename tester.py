"""
name: tester.py
date last modified: 8 apr 2018

testing suite to measure performance of philograph models 
results of these tests are included in the report 

note: this module is not part of the official application 
      and must be tested programatically 
"""
import matplotlib.pyplot as plt 
from sklearn.datasets import load_files
import numpy as np
import preprocessor
from svm import run_svm
from cluster import run_cluster

# plots for kmeans 

def plot_cluster_tol(data_list, tol_list):
  plt.gcf().clear()
  plt.plot(tol_list, data_list, 'r-')
  plt.yticks(np.arange(min(data_list) - 3, max(data_list) + 3, 1))
  plt.legend(['Average Performance'])
  plt.xlabel('Tolerance')
  plt.ylabel('Performance')
  plt.title('Tolerance vs Performance')

  plt.savefig('cluster_tol.png')
  plt.gcf().clear()

def plot_cluster_iter(data_list, iter_list):
  plt.gcf().clear()
  plt.plot(iter_list, data_list, 'g-')
  plt.yticks(np.arange(min(data_list) - 3, max(data_list) + 3, 1))
  plt.legend(['Average Performance'])
  plt.xlabel('Max Iterations')
  plt.ylabel('Performance')
  plt.title('Max Iterations vs Performance')

  plt.savefig('cluster_iter.png')
  plt.gcf().clear()

def plot_cluster_init(data_list, init_list):
  plt.gcf().clear()
  plt.plot(init_list, data_list, 'b-')
  plt.yticks(np.arange(min(data_list) - 3, max(data_list) + 3, 1))
  plt.legend(['Average Performance'])
  plt.xlabel('n_init')
  plt.ylabel('Performance')
  plt.title('n_init vs Performance')

  plt.savefig('cluster_init.png')
  plt.gcf().clear()

def plot_cluster_clusters(data_list, k_list):
  plt.gcf().clear()
  plt.plot(k_list, data_list, 'b-')
  #plt.yticks(np.arange(min(data_list) - 3, max(data_list) + 3, 1))
  plt.legend(['Average Performance'])
  plt.xlabel('k')
  plt.ylabel('Performance')
  plt.title('k vs Performance')

  plt.savefig('cluster_k.png')
  plt.gcf().clear()

# plots for svm 

def plot_split_num(test_error_list, split_num_list):
  plt.plot(split_num_list, test_error_list, 'r-')

  plt.legend(['Testing Set Error'])
  plt.xlabel('Lines per Split')
  plt.ylabel('Error Rate')
  plt.title('Lines per Split vs Error Rate')

  plt.savefig('svm_split_{0}.png'.format(split_num_list[0]))
  plt.gcf().clear()

def plot_split_num_no_filtering(test_error_list, split_num_list):
  plt.plot(split_num_list, test_error_list, 'r-')

  plt.legend(['Testing Set Error'])
  plt.xlabel('Lines per Split')
  plt.ylabel('Error Rate')
  plt.title('Lines per Split vs Error Rate (no stopwords removed)')

  plt.savefig('svm_split_{0}.png'.format(split_num_list[0]))
  plt.gcf().clear()

def plot_sliding_window(test_error_list, overlap_list):
  plt.plot(overlap_list, test_error_list, 'r-')

  plt.legend(['Testing Set Error'])
  plt.xlabel('Overlapped Lines')
  plt.ylabel('Error Rate')
  plt.title('Sliding Window vs Error Rate')

  plt.savefig('svm_overlap_{0}.png'.format(overlap_list[0]))
  plt.gcf().clear()

# testers for svm 

def test_sliding_window():
  # list of testing set accuracies
  test_error_list = []
  overlap_list = []

  overlap = 20
  while overlap >= 0: 
    print("overlap num : {}".format(overlap))
    preprocessor.format_corpus("sermons")
    preprocessor.clean_directory("sermons_partitioned")
    preprocessor.split_files(38, overlap, True, "sermons", "sermons_partitioned")

    all_files = load_files("./sermons_partitioned/")
    test_error_list.append(100 - run_svm(all_files, 4, None, 3))
    overlap_list.append(overlap)
    overlap -= 2

  print(test_error_list)
  plot_sliding_window(test_error_list, overlap_list)

def test_split_num_no_filtering():

  # list of testing set accuracies
  test_error_list = []
  split_num_list = []

  split_num = 38
  while split_num >= 4: 
    print("split num : {}".format(split_num))
    preprocessor.format_corpus("sermons")
    preprocessor.clean_directory("sermons_partitioned")
    preprocessor.split_files(split_num, 0, False, "sermons", "sermons_partitioned")

    all_files = load_files("./sermons_partitioned/")
    test_error_list.append(100 - run_svm(all_files, 4, None, 3))
    split_num_list.append(split_num)
    split_num -= 2
  
  print(test_error_list)
  plot_split_num(test_error_list, split_num_list)

def test_split_num():

  # list of testing set accuracies
  test_error_list = []
  split_num_list = []

  split_num = 38
  while split_num >= 4: 
    print("split num : {}".format(split_num))
    preprocessor.format_corpus("sermons")
    preprocessor.clean_directory("sermons_partitioned")
    preprocessor.split_files(split_num, 0, True, "sermons", "sermons_partitioned")

    all_files = load_files("./sermons_partitioned/")
    test_error_list.append(100 - run_svm(all_files, 4, None, 3))
    split_num_list.append(split_num)
    split_num -= 2
  
  print(test_error_list)
  plot_split_num(test_error_list, split_num_list)

def test_svm(): 
    """
    runs all svm testers together 
    """
    test_split_num()
    test_sliding_window()
    test_split_num_no_filtering()

# tester for kmeans 

def test_cluster():
  #n_init_list = [10, 20, 30]
  n_init_list = [10, 30, 50, 70, 90, 110]
  max_iter_list = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
  #tol_list = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
  tol_list = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2]
  k_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

  results_n_init = []
  results_max_iter = []
  results_tol_list = []
  results_k = []
  
  preprocessor.format_corpus("sermons")
  preprocessor.clean_directory("sermons_partitioned")
  preprocessor.split_files(10, 3, True, "sermons", "sermons_partitioned")
  all_files = load_files("./sermons_partitioned/")
  
  print("k...")
  for value in k_list:
    param_dict = {}
    param_dict['n_init'] = n_init_list[1]
    param_dict['max_iter'] = max_iter_list[1]
    param_dict['tol'] = tol_list[7]
    param_dict['num_runs'] = 10
    param_dict['k'] = value

    results_k.append(run_cluster(all_files, param_dict, "tester", 0))
  
  
  print("n_init...")
  for value in n_init_list:
    param_dict = {}
    param_dict['n_init'] = value
    param_dict['max_iter'] = max_iter_list[1]
    param_dict['tol'] = tol_list[7]
    param_dict['num_runs'] = 10
    param_dict['k'] = 5

    results_n_init.append(run_cluster(all_files, param_dict, "tester", 0))
  
  print("max_iter...")
  for value in max_iter_list:
    param_dict = {}
    param_dict['n_init'] = n_init_list[1]
    param_dict['max_iter'] = value
    param_dict['tol'] = tol_list[7]
    param_dict['num_runs'] = 10
    param_dict['k'] = 5

    results_max_iter.append(run_cluster(all_files, param_dict, "tester", 0))
  
  print("tol...")
  for value in tol_list:
    param_dict = {}
    param_dict['n_init'] = n_init_list[1]
    param_dict['max_iter'] = max_iter_list[1]
    param_dict['tol'] = value
    param_dict['num_runs'] = 10
    param_dict['k'] = 5

    results_tol_list.append(run_cluster(all_files, param_dict, "tester", 0))
  

  print(results_n_init)
  print(results_max_iter)
  print(results_tol_list)
  print(results_k)

  plot_cluster_init(results_n_init, n_init_list)
  plot_cluster_iter(results_max_iter, max_iter_list)
  plot_cluster_tol(results_tol_list, tol_list)
  plot_cluster_clusters(results_k, k_list)


# uncomment to launch test suite for svm 
#test_svm()
# uncomment to launch test suite for kmeans
#test_cluster()
