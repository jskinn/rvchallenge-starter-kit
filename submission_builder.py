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

Helper code to make submissions to the challenge.

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
The scoring program will re-order and match classes based on their string names (see scoring_program).

The following functions and classes are designed to help create these json files in the correct format,
They are here to make submission easier, but are not required, feel free to modify or use some or all of this
"""
from __future__ import absolute_import, division, print_function

import os
import os.path
import json

# Load numpy if it is available
try:
    import numpy as np
except ImportError:
    np = None


def make_detection(class_probabilities, xmin, ymin, xmax, ymax, upper_left_cov=None, lower_right_cov=None):
    """
    Record a single detection, call this for each detection in each image.
    Requires a probability for each class, an upper-left and lower-right corner.
    May also optionally include covariance matrices for both corners,
    where the given coordinates are then interpreted as means

    The most complex argument is the covariance, each of these should be a 2x2 matrix as a list of lists,
    with the form:
    [[x^2, xy],
     [xy, y^2]]
    The matrix must be positive semi-definite, and the diagonal elements must be greater than 1

    :param class_probabilities: A list of class confidences as floating points,
    must match the list of classes (see below) and sum to at most 1.
    :param xmin: The x coordinate of the upper left corner (mean)
    :param ymin: The y coordinate of the upper left corner (mean)
    :param xmax: The x coordinate of the lower right corner (mean)
    :param ymax: The y coordinate of the lower right corner (mean)
    :param upper_left_cov: 2x2 covariance matrix for the upper left corner, as a list of lists. Optional.
    :param lower_right_cov: 2x2 covariance matrix for the lower right corner, as a list of lists. Optionall
    :return:
    """
    if xmax < xmin:
        raise ValueError("xmax is less than xmin")
    if ymax < ymin:
        raise ValueError("ymax is less than ymin")
    if sum(class_probabilities) > 1 + 1e-14:
        raise ValueError("The class probabilities sum to more than 1")
    if np is not None and isinstance(class_probabilities, np.ndarray):
        class_probabilities = class_probabilities.tolist()

    detection = {
        'label_probs': class_probabilities,
        'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
    }
    if upper_left_cov is not None and lower_right_cov is not None:
        # Validate the covariances
        if not is_2x2_matrix(upper_left_cov):
            raise ValueError("The upper-left covariance is not a 2x2 matrix")
        if not is_symmetric(upper_left_cov):
            raise ValueError("The upper-left covariance is not symmetric")
        if not is_positive_definite(upper_left_cov):
            raise ValueError("The upper-left covariance is not positive definite")

        if not is_2x2_matrix(lower_right_cov):
            raise ValueError("The lower-right covariance is not a 2x2 matrix")
        if not is_symmetric(lower_right_cov):
            raise ValueError("The lower-right covariance is not symmetric")
        if not is_positive_definite(lower_right_cov):
            raise ValueError("The lower-right covariance is not positive definite")

        if np is not None and isinstance(upper_left_cov, np.ndarray):
            upper_left_cov = upper_left_cov.tolist()
        if np is not None and isinstance(lower_right_cov, np.ndarray):
            lower_right_cov = lower_right_cov.tolist()

        detection['covars'] = [upper_left_cov, lower_right_cov]
    elif upper_left_cov is not None and lower_right_cov is None:
        raise ValueError("Got covariance for upper left corner but not lower right")
    elif upper_left_cov is None and lower_right_cov is not None:
        raise ValueError("Got covariance for lower right corner but not upper left")
    return detection


def make_detection_height_width(class_probabilities, x, y, width, height, upper_left_cov=None, lower_right_cov=None):
    """
    Alternative form of make detection, above. instead taking the bounding box as x, y, width, height
    :param class_probabilities: A list of probabilities for each class
    :param x: The x coordinate of the upper-left corner
    :param y: The y coordinate of the upper-left corner
    :param width: The width of the box (in the x dimension)
    :param height: The height of the box (in the y dimension)
    :param upper_left_cov: Covariance matrix for the upper left corner
    :param lower_right_cov: Covariance matrix for the lower right corner
    :return:
    """
    return make_detection(class_probabilities, x, y, x + width, y + height, upper_left_cov, lower_right_cov)


def make_sequence_output(detections, classes):
    """
    Create the output object for an entire sequence

    :param detections: A list of lists of detections. Must contain an entry for each image in the sequence
    :param classes: The list of classes in the order they appear in the label probabilities
    :return:
    """
    return {
        'detections': detections,
        'classes': classes
    }


def make_simple_covariance(xvar, yvar):
    """
    Make simple spherical covariance, as can be passed as upper_left_cov or lower_right_cov.
    The resulting covariance is elliptical in 2-d and axis-aligned, there is no correlation component
    :param xvar: Horizontal covariance
    :param yvar: Vertical covariance
    :return: A 2x2 covariance matrix, as a list of lists.
    """
    return [
        [xvar, 0],
        [0, yvar]
    ]


def is_2x2_matrix(mat):
    """
    A quick check to ensure a value is a 2x2 matrix
    :param mat:
    :return:
    """
    return len(mat) == 2 and len(mat[0]) == 2 and len(mat[1]) == 2


def is_symmetric(mat):
    """
    Check if a 2x2 matrix is symmetric
    :param mat:
    :return:
    """
    return mat[0][1] == mat[1][0]


def is_positive_definite(mat):
    """
    Check if a matrix is positive semi-definite, that is, all it's eigenvalues are positive.
    All covariance matrices must be positive semi-definite.
    Only works on symmetric matrices (due to eigh), so check that first
    :param mat:
    :return:
    """
    if np is not None:
        mat = np.asarray(mat)
        eigvals, _ = np.linalg.eigh(mat)
        return np.all(eigvals >= 0)
    # Numpy is unavailable, assume the matrix is valid
    return True


class SubmissionWriter(object):
    """
    A helper class to handle writing ACRV Robotic Vision Challenge 1 submissions in the correct format.
    Simply create the object pointing to the desired output directory,
    and then call 'add_detection' for each detection, 'next_image' after each image,
    and 'save_sequence' after each sequence. e.g.:
    ```
    writer = submission_builder.SubmissionWriter('submission', classes)
    for sequence_name in os.listdir('test_dir'):
        if not sequence_name.endswith('.zip'):
            for image_file in os.listdir(os.path.join('test_dir', sequence_name)):
                detections = do_detection(image_file, ...)
                for detection in detections:
                    writer.add_detection(...)
                writer.next_image()
            writer.save_sequence(sequence_name)
    ```


    To create the final submission zip file (on linux/unix), simply cd into the submission folder and run
    ```
    zip -r submission.zip ./*
    ```
    """

    def __init__(self, submission_folder, class_list):
        self.submission_folder = submission_folder
        self.class_list = class_list
        self._all_detections = []
        self._current_detections = []

    def add_detection(self, class_probabilities, xmin, ymin, xmax, ymax, upper_left_cov=None, lower_right_cov=None):
        """
        Add a detection for the current image.
        The parameters are the same as make_detection, see above.

        :param xmin: The x coordinate of the upper left corner (mean)
        :param ymin: The y coordinate of the upper left corner (mean)
        :param xmax: The x coordinate of the lower right corner (mean)
        :param ymax: The y coordinate of the lower right corner (mean)
        :param upper_left_cov: 2x2 covariance matrix for the upper left corner, as a list of lists
        :param lower_right_cov: 2x2 covariance matrix for the lower right corner, as a list of lists
        :param class_probabilities: A list of class confidences as floats,
                                    which correspond to the matching entry in the class list
        :return:
        """
        if len(class_probabilities) != len(self.class_list):
            raise RuntimeError("Class probabilities are not the same length as the class list")
        self._current_detections.append(make_detection(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            upper_left_cov=upper_left_cov,
            lower_right_cov=lower_right_cov,
            class_probabilities=class_probabilities
        ))

    def next_image(self):
        """
        Move to the next image. Call this after each image, including the last one in each sequence,
        even if there are no detections.
        :return:
        """
        self._all_detections.append(self._current_detections)
        self._current_detections = []

    def save_sequence(self, sequence_name):
        """
        Finish a particular image sequence, writing image ids and class confidences to file.
        This clears all accumulated detections for the given sequence, ready for the next one.
        Call this exactly once per sequence, after each sequence is complete
        Do not call this more than once per sequence name, since it will overwrite previously saved.

        :param sequence_name: The name of the folder containing the images (not the full path)
        :return: None
        """
        # If there are outstanding detections, add them as another image
        if len(self._current_detections) > 0:
            self.next_image()

        # Create the output folder if it doesn't exist
        if not os.path.exists(self.submission_folder):
            os.makedirs(self.submission_folder)

        # Write all the accumulated detections to file
        with open(os.path.join(self.submission_folder, '{0}.json'.format(sequence_name)), 'w') as fp:
            json.dump(make_sequence_output(self._all_detections, self.class_list), fp)

        self._all_detections = []
