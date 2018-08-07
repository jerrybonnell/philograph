"""
  name: philograph.py
  last modified: 29 mar 18

  main file to launch application
"""

from sklearn.datasets import load_files
from svm import run_svm
from cluster import run_cluster
from utility import write_file
import argparse
import sys
import json
import preprocessor

PROG = "philograph"
DESCRIPTION = "philograph. textual analysis tools in the digital humanities."
SETTINGS_FILENAME = "settings.json"
SVM = 0
KMEANS = 1


def preprocess(corpus_folder, mode, settings, out_file, verbose_level):
    preprocessor.format_corpus(corpus_folder)
    preprocessor.clean_directory(corpus_folder + "_partitioned")

    filter_words = True
    if settings['filter_words'] == 0:
        filter_words = False

    preprocessor.split_files(settings['num_lines_split'],
                             settings['sliding_window_size'],
                             filter_words, corpus_folder,
                             corpus_folder + "_partitioned")

    all_files = load_files(corpus_folder + "_partitioned")

    if verbose_level > 1:
        print("mode : {} filter: {} window size: {} num_lines_split: {}".
              format(
                  mode, filter_words, settings['sliding_window_size'],
                  settings['num_lines_split']
              ))

    if mode == SVM:
        run_svm(all_files, settings['svm'][
                'num_runs'], out_file, verbose_level)
    elif mode == KMEANS:
        param_dict = {}
        param_dict['n_init'] = settings['kmeans']['n_init']
        param_dict['max_iter'] = settings['kmeans']['max_iter']
        param_dict['tol'] = settings['kmeans']['tol']
        param_dict['num_runs'] = settings['kmeans']['num_runs']
        param_dict['k'] = settings['kmeans']['k']

        run_cluster(all_files, param_dict, out_file, verbose_level)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('-v', '--verbose', help='verbose mode', action="count")
    group.add_argument('-p', '--predict', metavar='corpus_folder',
                       help='launches predicting function, SVM')
    group.add_argument('-c', '--cluster', metavar='corpus_folder',
                       help='launches clustering function, k-means')
    parser.add_argument('-o', '--output', nargs="?", help='output file',
                        metavar='filename')
    args = parser.parse_args()

    output = args.output
    verbosity = args.verbose
    if verbosity is None:
        verbosity = 0

    # fetch user-preferred paramter values for models from json file
    json_file = open(SETTINGS_FILENAME)
    json_str = json_file.read()
    json_data = json.loads(json_str)

    if args.predict is not None:
        corpus_folder = args.predict
        preprocess(corpus_folder, SVM, json_data, output, verbosity)
    elif args.cluster is not None:
        corpus_folder = args.cluster
        preprocess(corpus_folder, KMEANS, json_data, output, verbosity)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# test_split_num()
# test_sliding_window()
# test_cluster()
# run()
