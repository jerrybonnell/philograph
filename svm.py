'''
  name: svm.py
  last modified: 14 mar 18

  scikit support vector machine
'''

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utility import write_file
from tqdm import tqdm
import random
import operator
import time 

doc_to_num_incorrect = {}
doc_to_themes_incorrect = {}
documents = set([])

def run_svm(all_files, num_runs, out_file, verbose_level):
    """
    interfaces with the first model 
    """
    out = ""
    if out_file is not None:
        write_file(out_file + "_svm.txt", time.ctime() + "\n===\n")

    for i in range(0, len(all_files.target)):
        doc_name = all_files.filenames[i]
        doc_name = doc_name[doc_name.rfind("/") + 1:doc_name.rfind(".")]
        doc_name_whole = doc_name[:doc_name.rfind("_part")]
        # each document name serves as a key. the number of times
        # it is incorrect when it serves in the testing set
        # will be its value
        documents.add(doc_name_whole)
        doc_to_num_incorrect[doc_name] = 0
        doc_to_themes_incorrect[doc_name] = []

    # run the SVM classifier for a user-specified number of times
    (avg_accuracy_rate, out) = avg_run(num_runs, all_files, out, verbose_level)

    sorted_dict = sorted(doc_to_num_incorrect.items(),
                         key=operator.itemgetter(1), reverse=True)

    if verbose_level > 0:
        if out_file is None:
            print(out)
        else:
            write_file(out_file + "_svm.txt", out)

    format_output(all_files, out_file)

    return avg_accuracy_rate

def format_output(all_files, out_file):
    # a dictionary of dictionaries 
    theme_to_doc = {}
    out = ""

    # go through the entire corpus  
    for i in range(0, len(all_files.filenames)):
        # get document name with path removed, e.g. true_saints_part_3
        doc_name = all_files.filenames[i]
        doc_name = doc_name[doc_name.rfind("/") + 1:doc_name.rfind(".")]
        # "whole" document without any subdivison, e.g. true_saints
        doc_name_whole = doc_name[:doc_name.rfind("_part")]
        # get this document's target index 
        theme_index = all_files.target[i]
        if theme_index in theme_to_doc:
        # we have visited a document from this theme already. check to see if 
        # this document is part of a sermon that exists in its sub-directory 
            doc_to_array = theme_to_doc[theme_index]
            if doc_name_whole in doc_to_array:
                # if it is, then increment this sermon's number of parts 
                doc_to_array[doc_name_whole][0] += 1
                # and keep a tally of the total # of times this sermon was wrong
                # when any of its subdivisions served in the testing set 
                doc_to_array[doc_name_whole][1] += doc_to_num_incorrect[doc_name]
            else:
                # if it hasn't, initialize a new array 
                doc_to_array[doc_name_whole] = [1,doc_to_num_incorrect[doc_name]]
        # if the target index is not in the dictionary, this means we have not yet 
        # visited any sermons from this theme. initialize a new dictionary and add
        # a new key (this document's name) whose value is an array of size 2 
        # where 
        #   [# parts, total # times wrong when its parts served in testing set]
        else:
            doc_to_array = {}
            doc_to_array[doc_name_whole] = [1,doc_to_num_incorrect[doc_name]]
            theme_to_doc[theme_index] = doc_to_array
    
    # now print out the data 
    for theme_index, documents in theme_to_doc.items():
        # sort from best to worst; note that we're sorting according 
        # to the second element in the array 
        sorted_dict = sorted(documents.items(), 
                            key=lambda doc: doc[1][1], reverse=False)
        # print theme name, e.g. sinners 
        out += all_files.target_names[theme_index] + "\n"
        for document in sorted_dict:
            # print the full sermon name 
            out += "{:>3s} {}".format(">",document[0]) + "\n"
            for i in range(0,document[1][0]):
                # finally, print out the subdivision information 
                doc_part = document[0] + "_part_" + str(i)
                out += "{:>10s}part {:<6d} {:<2d}/{:>2d}{:>3s}".format("", i, 
                        doc_to_num_incorrect[doc_part], 
                        len(doc_to_themes_incorrect[doc_part]), "")
                if doc_to_num_incorrect[doc_part] > 0:
                    out += str(doc_to_themes_incorrect[doc_part]) + "\n"
                else:
                    out += "\n"
            # print out total number of times this sermon was incorrect; note this
            # is the value that is sorted by 
            out += '{:>3s} total {:<3d}\n'.format("*", document[1][1]) + "\n"
    
    if out_file is None:
        print(out)
    else:
        write_file(out_file + "_svm.txt", out)

def avg_run(avg_num, all_files, out, verbose_level):
    total_accuracy = 0
    # run the process for a user-specified number of times
    for i in tqdm(range(0, avg_num)):
        # run the classifier and obtain accuracy measures
        (train_accuracy, 
         test_accuracy, out) = sermons_predict_param_search(all_files, out, verbose_level)
        out += "\n run {}: train {} test {}\n".format(i + 1, train_accuracy, test_accuracy)
        out += "------------------------\n"
        total_accuracy += test_accuracy
    print()
    avg_accuracy_rate = total_accuracy / float(avg_num)

    out += "\n"
    out += "avg accuracy on testing set => {}\n".format(avg_accuracy_rate)
    return (avg_accuracy_rate, out)

def sermons_predict_param_search(all_files, out, verbose_level):

    # list of indices matching documents to serve in the testing set
    #test_set_index = random.sample(range(0, len(all_files.target)), 40)
    test_set_index = random.sample(range(0, len(all_files.target)),
                                   int(0.4 * len(all_files.target)))
    # list of indices matching documents to serve in the training set
    # by taking the set difference of all numbers between 0 and the
    # total number of documents with the testing set indices, we obtain
    # the indices for the training set
    training_set_index = list(
        set(range(0, len(all_files.target))).difference(set(test_set_index)))

    # list containing the content for each document in the test set
    test_set_data = [all_files.data[i] for i in test_set_index]
    # list containing the (target) label for each document in the test set
    test_set_target = [all_files.target[i] for i in test_set_index]

    # likewise for the training set
    training_set_data = [all_files.data[i] for i in training_set_index]
    training_set_target = [all_files.target[i] for i in training_set_index]

    # calculate how many parts of a sermon serve in the training set
    # (only for verbose mode)
    doc_to_parts_in_training = {}
    for i in range(0, len(all_files.target)):
        doc_name = all_files.filenames[i]
        doc_name = doc_name[doc_name.rfind("/") + 1:doc_name.rfind("_part")]
        doc_to_parts_in_training[doc_name] = 0

    for i in range(0, len(training_set_index)):
        doc_name = all_files.filenames[training_set_index[i]]
        doc_name = doc_name[doc_name.rfind("/") + 1:doc_name.rfind("_part")]
        doc_to_parts_in_training[doc_name] += 1


    # train SVM
    # scikit-learn provides a pipeline to put everything together; performs
    # all the steps of sermons_predict() at once
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SVC(kernel='linear'))])

    # select parameters for testing in SVM classifier
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__C': (1, .5, .1),
        'clf__gamma': (10, 1, .1)}

    # for parameter tuning, use Grid Search to find optimal combination;
    # the operation can be computationally prohibitive
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

    # fit the classifier according to training set
    gs_clf = gs_clf.fit(training_set_data, training_set_target)

    # predict testing set
    predicted = gs_clf.predict(test_set_data)

    # calculate accuracy on the testing set
    total_wrong = 0
    for i in range(0, len(predicted)):
        # get name of documents that were mis-classified
        doc_name = all_files.filenames[test_set_index[i]]
        # document name includes path and extension; remove both
        doc_name = doc_name[doc_name.rfind("/") + 1:doc_name.rfind(".")]
        # add this misclassified theme to list of all misclassified for
        # this document
        doc_to_themes_incorrect[doc_name].append(
            all_files.target_names[predicted[i]])
        if predicted[i] != test_set_target[i]:
            total_wrong += 1
            # increment the ``incorrectness'' value for this document
            doc_to_num_incorrect[doc_name] += 1
            if verbose_level > 1:
                out += "\"{}\" ({}) ==> {}\n".format(doc_name,
                                                       all_files.target_names[test_set_target[i]],
                                                       all_files.target_names[predicted[i]])

    test_accuracy = ((len(predicted) - total_wrong) * 100 / len(predicted))

    predicted = gs_clf.predict(training_set_data)

    # calculate accuracy on the training set
    total_wrong = 0
    for i in range(0, len(predicted)):
        if predicted[i] != training_set_target[i]:
            total_wrong += 1

    train_accuracy = ((len(predicted) - total_wrong) * 100 / len(predicted))

    return (train_accuracy, test_accuracy, out)