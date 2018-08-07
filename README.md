# Philograph: Textual Analysis Tools in the Digital Humanities

Philograph is an application built to assist literary scholars in their
research. The application offers usability for analyzing connections
in and between texts based on themes and clusters of keywords. 

Makes use of the supervised machine-learning technique Support Vector
Machines and the unsupervised clustering algorithm k-means.

Tailored for use in analyzing a corpus of sermons written by 
18th-century Puritan minister, Jonathan Edwards. See `sermons` for the 
dataset used in this project. 

The directory `research-data` contains results generated from this dataset. 

## Prerequisites

__Disclaimer__: _Philograph works best on macOS and has not been tested 
on other operating systems.

* Unix-like operating system (macOS or Linux)
* Basic understanding of command-line
* `python` should be installed 
* `pip3` should be installed. Philograph has many dependencies and it is
likely you will have to install several packages, e.g. `numpy` or
`sklearn`. 

## Getting Started 

1. At the command prompt, `git clone` this repository (or download it
as a zip archive). 

        $ git clone https://github.com/jerrybonnell/philograph.git

2. Change directory to `philograph` and launch the application.

        $ cd philograph
        $ python philograph.py 

    You should see a help screen that displays suggested options. If not,
    there are likely missing packages. Use `pip3` to install them.

## Options 

* `-p <corpus folder>` `--predict <corpus folder>`
  
  launches predicting function, SVM  

* `-c <corpus folder>` `--cluster <corpus folder>`

  launches clustering function, k-means

* `-o <filename>` `--output <filename>`

  writes results to a file with name `<filename>`

* `-v` `--verbose`

  runs in verbose mode. `-vv` runs with more verbosity 
  and `-vvv` runs with highest verbosity. 

* `-h` `--help`

  shows the help screen! 

Note that the parameters used by these models can be customized 
by editing `settings.json`. 

## Notes 

Developed as part of the CSC410/CSC411 project course at the University of Miami. 



