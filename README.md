ACRV Robotic Vision Challenge 1 Starter Kit
===========================================

Welcome to the Robotic Vision Challenge.
If you didn't come from there, you can find the challenge website at:
https://competitions.codalab.org/competitions/19742

This starter kit contains tools to help you submit your algorithm to the challenge.


Repository Contents: 
- README.md : This file!
- submission_builder.py : some helpful python code to generate submissions in the correct format, see Submission Format
- download_test_data.sh : Bash script to download the test images into a folder called 'test_data'
- class_list.txt : List of the classes used in this challenge
- tests : Unit tests for the submission builder. This requires the evaluation code (see below)      


Test data
---------

The test data for this challenge consists of 18 sequences of images, numbered 000000 to 000017.
It can be downloaded using the included "download_test_data.sh" or using the links therein.
There is no training data released for this challenge, train on whatever data seems appropriate.

Class List
----------

This challenge uses a selection of the Microsoft Common Objects in Context class list,
reduced to indoor objects. The full class list is:

'none', 'bottle', 'cup', 'knife', 'bowl', 'wine glass', 'fork', 'spoon', 'banana', 'apple', 'orange', 'cake',
'potted plant', 'mouse', 'keyboard', 'laptop', 'cell phone', 'book', 'pen', 'clock',
'chair', 'potted plant', 'dining table', 'couch', 'bed', 'toilet', 'television', 'microwave', 'toaster',
'refrigerator', 'oven', 'sink', 'person'


Submission Format
-----------------

The test data consists of images grouped into sequences ('000000', '000001', etc..)
The submission format is a folder containing a json file for each sequence ('000000.json', '000001.json', etc..)
Each json file should contain the following structure:
```
{
  "classes": [<an ordered list of class names>],
  "detections": [
    [
      {
        "bbox": [x1, y1, x2, y2],
        "covars": [
          [[xx1, xy1],[xy1, yy1],
          [[xx2, xy2],[xy2, yy2]
        ],
        "label_probs": [<an ordered list of probabilities for each class>]
      },
      {
      }
    ],
    [],
    []
    ...
  ]
}
```
There must be an entry in the "detections" list for every image in the sequence, even if it is an empty list.
The "label_probs" list in each detection must match the "classes" list at the top level, such that the i-th entry
in "label_probs" is the probability of the i-th class in the class list.
"label_probs" should be a normalized probability distribution, summing to 1.
If it sums to more than 1, it will be normalized. If it sums to less than 1, it is assumed the remaining probability
is assigned to a background class, and will be ignored.
The scoring program will re-order and match classes based on their string names.
If your system by default outputs more classes, or in a different order, specify your order in the submission json,
and the evaluation script will re-order your classes appropriately.

'submission_builder.py' contains helper code to generate json files in this format, see the comments there for more
exmaples.

Evaluation Code
---------------

The code used to score submissions is also available, at:
https://github.com/jskinn/rvchallenge-evaluation

This code is required to run the unit tests in the tests directory,
simply check it out in an adjacent directory called 'scoring_program'
