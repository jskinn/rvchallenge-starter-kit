#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BSD 3-Clause License

Copyright (c) 2018, John Skinner, David Hall, Niko Sünderhauf, and Feras Dayoub,
ARC Centre of Excellence for Robotic Vision, Queensland University of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


-----------------------------------

usage: submission_validator.py [-h] [-q] submission_directory

Validator script for submissions to the challenge. Call this script on a
submission to check for errors and invalid values in your submission before
uploading.

positional arguments:
  submission_directory  The folder containing the submission to validate. Zip
                        up and submit this folder when done.

optional arguments:
  -h, --help            show this help message and exit
  -q, --quiet           Suppress warning messages, which may occur when class
                        probabilities are not normalized or detections are
                        ignored. These are not errors, and may produce
                        excessive output

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys
import os.path
import warnings
import numpy as np
import json


# This is the list of valid classes for this challenge, in order
# The class id is the index in this list
CLASSES = [
    'none',
    'bottle',
    'cup',
    'knife',
    'bowl',
    'wine glass',
    'fork',
    'spoon',
    'banana',
    'apple',
    'orange',
    'cake',
    'potted plant',
    'mouse',
    'keyboard',
    'laptop',
    'cell phone',
    'book',
    'clock',
    'chair',
    'dining table',
    'couch',
    'bed',
    'toilet',
    'television',
    'microwave',
    'toaster',
    'refrigerator',
    'oven',
    'sink',
    'person'
]


# A simple map to go back from class id to class name
CLASS_IDS = {class_name: idx for idx, class_name in enumerate(CLASSES)}


# Some helper synonyms, to handle cases where multiple words mean the same class
# This list is used when loading the ground truth to map it to the list above
SYNONYMS = {
    'tv': 'television',
    'tvmonitor': 'television',
    'computer monitor': 'television',    # They're approximately the same, right?
    'stool': 'chair',
    'diningtable': 'dining table',
    'pottedplant': 'potted plant',
    'cellphone': 'cell phone',
    'wineglass': 'wine glass',
    'background': 'none',
    'bg': 'none',
    '__background__': 'none'
}


def validate_submission(directory, sequence_ids=np.arange(18)):
    """
    Validate all the submissions for all the sequences outlined in the given folder.
    Each sequence's detections are provided in a file ending with 'detections.json'.
    Each detections.json file contains a dictionary which has a key 'detections' containing a list of list of
    detections for each image.
    Individual detections are given as dictionaries which have the keys:
        'img_size': (height x width)
        'img_num': int identifying which image the detection is a part of
        'label_probs': full list of probabilities that the detection is describing each of the classes
        'bbox': coordinates of the bounding box corners [left, top, right, bottom]
        'covars': covariances for the top-left and bottom-right corners respectively.
            Each with format [[xx, xy], [yx, yy]]. Covariances must be positive semi-definite
            or all zeros (regular BBox).
    Order of list of lists should correspond with ground truth image order.
    If an image does not have any detections, entry should be an empty list.
    Users can choose to only test a subset of sequences identified by sequence_ids.
    Note that they must have valid submissions for all sequences before submission to the competition.

    :param directory: location of each sequence's submission json file.
    :param sequence_ids: list of sequence identification numbers for all sequences to be validated.
    Defaults to all 18 sequence ids needed for submission to competition ([0,1,2, ..., 17]).
    """
    if not os.path.isdir(directory):
        raise ValueError("Submission directory {0} does not exist".format(directory))

    expected_sequence_names = {'{0:06}'.format(idx) for idx in sequence_ids}
    sequences = {}
    for root, _, files in os.walk(directory):
        for sequence_name in expected_sequence_names:
            json_file = sequence_name + '.json'
            if json_file in files:
                if sequence_name in sequences:
                    raise ValueError("{0} : more than one json file found for sequence, {1} and {2}".format(
                        sequence_name,
                        os.path.relpath(sequences[sequence_name], directory),
                        os.path.relpath(os.path.join(root, json_file), directory)
                    ))
                else:
                    sequences[sequence_name] = os.path.join(root, json_file)

    missing = expected_sequence_names - set(sequences.keys())
    if len(missing) > 0:
        raise ValueError("The following sequences do not have any detections submitted: {0}".format(sorted(missing)))

    for sequence_name in sorted(sequences.keys()):
        print("Validating submission for sequence {0}...".format(sequence_name))
        validate_sequence(sequences[sequence_name])


def validate_sequence(sequence_json):
    """
    Read and validate a sequence's detections json file.
    json file contains a dictionary which has a key 'detections' containing a list of list of
    detections for each image.
    Individual detections are given as dictionaries which have the keys:
        'img_size': (height x width)
        'img_num': int identifying which image the detection is a part of
        'label_probs': full list of probabilities that the detection is describing each of the classes
        'bbox': coordinates of the bounding box corners [left, top, right, bottom]
        'covars': covariances for the top-left and bottom-right corners respectively.
            Each with format [[xx, xy], [yx, yy]]. Covariances must be positive semi-definite
            or all zeros (regular BBox).
    Order of list of lists should correspond with ground truth image order.
    If an image does not have any detections, entry should be an empty list.
    :param sequence_json:
    :return: generator of generator of DetectionInstances for each image
    """
    with open(sequence_json, 'r') as f:
        data_dict = json.load(f)
    sequence_name = os.path.basename(sequence_json)

    # Validate
    if 'classes' not in data_dict:
        raise KeyError("{0} : Missing key \'classes\'".format(sequence_name))
    if 'detections' not in data_dict:
        raise KeyError("{0} : Missing key \'detections\'".format(sequence_name))
    if len(set(data_dict['classes']) & (set(CLASS_IDS) | set(SYNONYMS.keys()))) <= 0:
        raise ValueError("{0} : classes does not contain any recognized classes".format(sequence_name))

    # Work out which of the submission classes correspond to which of our classes
    our_class_ids = []
    sub_class_ids = []
    for sub_class_id, class_name in enumerate(data_dict['classes']):
        our_class_id = get_class_id(class_name)
        if our_class_id is not None:
            our_class_ids.append(our_class_id),
            sub_class_ids.append(sub_class_id)

    # create a detection instance for each detection described by dictionaries in dict_dets
    dict_dets = data_dict['detections']
    next_progress = 0
    for img_idx, img_dets in enumerate(dict_dets):
        validate_detections(img_dets, (our_class_ids, sub_class_ids),
                            num_classes=len(data_dict['classes']), img_idx=img_idx, sequence_name=sequence_name)
        progress = img_idx / len(dict_dets)
        if progress > next_progress:
            print_progress(progress)
            next_progress += 0.05
    print('\r  Complete!                  ')    # Padding to remove previous lines


def validate_detections(img_dets, class_mapping, num_classes=len(CLASSES), img_idx=-1,
                        sequence_name='unknown'):
    """
    Validate detections for a given image.

    :param img_dets: list of detections given as dictionaries.
    Individual detection dictionaries have the keys:
        'img_num': int identifying which image the detection is a part of
        'label_probs': full list of probabilities that the detection is describing each of the classes
        'bbox': coordinates of the bounding box corners [left, top, right, bottom]
        'covars': covariances for the top-left and bottom-right corners respectively.
            Each with format [[xx, xy], [yx, yy]]. Covariances must be positive semi-definite
            or all zeros (regular BBox).
    :param class_mapping: A pair of lists of indexes, the first to our class list, and the second to theirs
    :param num_classes: The number of classes to expect
    :param img_idx: The current image index, for error reporting
    :param sequence_name: The current image name, for error reporting
    :return: generator of DetectionInstances
    """
    for det_idx, det in enumerate(img_dets):
        if 'label_probs' not in det:
            raise KeyError(make_error_msg("missing key \'label_probs\'", sequence_name, img_idx, det_idx))
        if 'bbox' not in det:
            raise KeyError(make_error_msg("missing key \'bbox\'", sequence_name, img_idx, det_idx))
        if len(det['label_probs']) != num_classes:
            raise KeyError(make_error_msg("The number of class probabilities doesn't match the number of classes",
                                          sequence_name, img_idx, det_idx))
        if len(det['bbox']) != 4:
            raise ValueError(make_error_msg("The bounding box must contain exactly 4 entries",
                                            sequence_name, img_idx, det_idx))
        if det['bbox'][2] < det['bbox'][0]:
            raise ValueError(make_error_msg("The x1 coordinate must be less than the x2 coordinate",
                                            sequence_name, img_idx, det_idx))
        if det['bbox'][3] < det['bbox'][1]:
            raise ValueError(make_error_msg("The y1 coordinate must be less than the y2 coordinate",
                                            sequence_name, img_idx, det_idx))

        # Use numpy list indexing to move specific indexes from the submission
        label_probs = np.zeros(len(CLASSES), dtype=np.float32)
        label_probs[class_mapping[0]] = np.array(det['label_probs'])[class_mapping[1]]
        total_prob = np.sum(label_probs)

        if total_prob > 0.5:  # Arbitrary theshold for classes we care about.
            # Normalize the label probability
            if total_prob > 1:
                warnings.warn(make_error_msg("The class probabilities were greater than 1, and were normalized",
                                             sequence_name, img_idx, det_idx))
                label_probs /= total_prob
            if 'covars' in det and det['covars'] != [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]:
                covars = np.array(det['covars'])
                if covars.shape != (2, 2, 2):
                    raise ValueError(make_error_msg("Key 'covars' must contain 2 2x2 matrices",
                                                    sequence_name, img_idx, det_idx))
                if not np.allclose(covars.transpose((0, 2, 1)), covars):
                    raise ValueError(make_error_msg("Given covariances are not symmetric",
                                                    sequence_name, img_idx, det_idx))
                if not is_positive_semi_definite(covars[0]):
                    raise ValueError(make_error_msg("The upper-left covariance is not positive semi-definite",
                                                    sequence_name, img_idx, det_idx))
                if not is_positive_semi_definite(covars[1]):
                    raise ValueError(make_error_msg("The lower-right covariance is not positive semi-definite",
                                                    sequence_name, img_idx, det_idx))
        else:
            warnings.warn(make_error_msg("The detection was ignored as it's total probability across "
                                         "all known classes was {0}, which is less than 0.5".format(total_prob),
                                         sequence_name, img_idx, det_idx))


def is_positive_semi_definite(mat):
    """
    Check if a matrix is positive semi-definite, that is, all it's eigenvalues are positive.
    All covariance matrices must be positive semi-definite.
    Only works on symmetric matrices (due to eigh), so check that first
    :param mat:
    :return: True iff the matrix is positive semi-definite
    """
    eigvals, _ = np.linalg.eigh(mat)
    return np.all(eigvals >= 0)


def make_error_msg(msg, sequence_name, img_idx, det_idx):
    """
    Make an error message for a particular detection, so that the particular problem can be identified.
    :param msg: The raw error message
    :param sequence_name: The name of the sequence containing the detection
    :param img_idx: The index of the image the detection was made on
    :param det_idx: The index of the detection within that image
    :return:
    """
    return "{0}, image index {1}, detection index {2} : {3}".format(sequence_name, img_idx, det_idx, msg)


def get_class_id(class_name):
    """
    Given a class string, find the id of that class
    This handles synonym lookup as well
    :param class_name:
    :return:
    """
    class_name = class_name.lower()
    if class_name in CLASS_IDS:
        return CLASS_IDS[class_name]
    elif class_name in SYNONYMS:
        return CLASS_IDS[SYNONYMS[class_name]]
    return None


def print_progress(progress):
    """
    Print a progress bar, see https://stackoverflow.com/questions/3002085/python-to-print-out-status-bar-and-percentage
    :param progress: The progress so far, as a fraction
    :return:
    """
    sys.stdout.write('\r')
    sys.stdout.write("  [{0: <20}] {1}%".format('=' * int(20 * progress), int(100 * progress)))
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validator script for submissions to the challenge. '
                                                 'Call this script on a submission to check for errors and invalid '
                                                 'values in your submission before uploading.')
    parser.add_argument('submission_directory', type=str, help='The folder containing the submission to validate. '
                                                               'Zip up and submit this folder when done.')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress warning messages, which may occur when class probabilities are '
                             'not normalized or detections are ignored. These are not errors, '
                             'and may produce excessive output')
    args = parser.parse_args()

    if args.quiet:
        warnings.simplefilter('ignore')

    validate_submission(args.submission_directory)
