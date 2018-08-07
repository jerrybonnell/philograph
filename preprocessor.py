"""
  name: preprocessor.py
  last modified: 29 mar 18

  module for preprocessing a corpus
"""
import re
import glob
import os
import shutil
from nltk.corpus import stopwords
from utility import convert_numerals


def format_text(filename):
    """
    formats a document pointed to by a filename by enforcing
    all characters to be alphanumeric
    (this squashes format errors like non-UTF8 characters)

    filename: filename as a string
    """

    with open(filename, 'r') as file:
        text = file.readlines()

    i = 0
    while i < len(text):
        text[i] = re.sub('[^a-zA-Z0-9 \n]', '', text[i])
        i += 1

    with open(filename, 'w') as file:
        file.writelines(text)


def format_corpus(directory):
    """
    formats a corpus pointed to by a directory name.
    the function assumes documents to be in .txt format

    directory: directory name as a string
    """
    if os.path.isdir(directory) is False:
        raise ValueError(
            'invalid directory name! perhaps it was misspelled...')

    for filename in glob.iglob(directory + "/**/*.txt", recursive=True):
        format_text(filename)


def clean_directory(directory):
    """
    deletes all subdirectories pointed to by a directory name;
    if it does not exists, creates a directory with name directory

    directory: directory name as a string.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for subdir_path in glob.iglob(directory + "/**/"):
            shutil.rmtree(subdir_path, ignore_errors=True)


def split_files(num_lines_split, sliding_window, filter_words,
                read_dir, write_dir):
    """
    populates directory pointed to by write_dir with partitioned documents

    num_lines_split: number of lines to read before splitting
    sliding_window: number of lines to overlap between parts
    read_dir: directory containing the original documents
    write_dir: directory containing the documents after partitioning
    """

    # list containing names of all sub-directories. the first element
    # is omitted because it contains the name of the "." directory.
    # ex: './sermons/sinners', './sermons/rewards', etc.
    all_sub_dirs = [x[0] for x in os.walk(read_dir)][1:]
    # toggle removal of stopwords from the corpus
    if filter_words is True:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set()

    # for each sub-directory in the list of all sub-directories
    for sub_dir in all_sub_dirs:
        # subdirectory name has format ./sermons/xxxx
        # remove the ./sermons piece
        sub_dir_name = sub_dir.replace(read_dir, "")
        # replace that piece with "./sermons_partitioned" instead,
        new_sub = write_dir + sub_dir_name + "/"
        os.mkdir(new_sub)  # creates the new sub-directory

        # get all the file names in that sub-directory
        files = glob.glob(sub_dir + "/*.txt")
        # for each file in the list of files
        for file in files:
            # get the name of this file without the attached path
            file_name_only = file.replace(sub_dir + "/", "")
            f_data = open(file, 'r')
            text = f_data.readlines()
            # some elements may consist of only a "\n"; remove these
            text = list(filter(lambda x: x != '\n', text))
            i = 0
            j = 0
            while i < len(text):
                # each new document will consist of 10 lines from
                # the original document
                # if we have reached a 10th line, open a new file
                if j % num_lines_split == 0:
                    # if this isn't the first portion of the document, then
                    # be sure to close the old file pointer
                    if i > 0:
                        f_0.close()
                        i -= sliding_window
                    # open a new file containing this portion of the document
                    f_0 = open(new_sub + file_name_only.replace(".txt", "") +
                               "_part_" + str(int(j / num_lines_split)) +
                               ".txt", "w")
                # strip the line of trailing characters and convert roman
                # numerals to ints for easier removal
                line = convert_numerals(text[i].strip())
                # remove digits from the line since this may influence
                # performance
                line = "".join([c for c in line if not c.isdigit()])
                # make sure all characters are alphabetical
                # regex = re.compile('[^a-zA-Z  \n]')
                line = re.sub('[^a-zA-Z \n]', '', line)
                # stopwords
                words = " ".join(stop_words).strip()
                words = words.split()
                line_splitted = line.lower().split()
                line_splitted = [x for x in line_splitted if x not in words]
                line = " ".join(line_splitted)
                # write a line to the file
                f_0.write(line + "\n")
                i += 1
                j += 1

            # close the file
            if f_0.closed is False:
                f_0.close()
