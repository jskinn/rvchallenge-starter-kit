import unittest
import os.path
import shutil
import numpy as np

import scoring_program.tests.test_helpers as th
import scoring_program.submission_loader as submission_loader
import scoring_program.class_list as class_list
import starter_kit.submission_builder as submission_builder


class TestMakeDetection(unittest.TestCase):

    def test_makes_valid_detection_without_covars(self):
        confidences = [0.1, 0.2, 0.3, 0.4]
        det = submission_builder.make_detection(confidences, 1, 3, 12, 14)
        self.assertIn('bbox', det)
        self.assertIn('label_probs', det)
        self.assertNotIn('covars', det)
        self.assertEqual([1, 3, 12, 14], det['bbox'])
        self.assertEqual(confidences, det['label_probs'])

    def test_makes_valid_detection_with_covars(self):
        confidences = [0.1, 0.2, 0.3, 0.4]
        upper_left = [[3, 1], [1, 4]]
        lower_right = [[10, 0], [0, 15]]
        det = submission_builder.make_detection(confidences, 1, 3, 12, 14, upper_left, lower_right)
        self.assertIn('bbox', det)
        self.assertIn('label_probs', det)
        self.assertIn('covars', det)
        self.assertEqual([1, 3, 12, 14], det['bbox'])
        self.assertEqual(confidences, det['label_probs'])
        self.assertEqual([upper_left, lower_right], det['covars'])

    def test_errors_if_xmax_less_than_xmin(self):
        with self.assertRaises(ValueError) as cm:
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4], 15, 3, 2, 14)
        msg = str(cm.exception)
        self.assertIn('xmax', msg)
        self.assertIn('xmin', msg)

    def test_errors_if_ymax_less_than_ymin(self):
        with self.assertRaises(ValueError) as cm:
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4], 1, 31, 12, 14)
        msg = str(cm.exception)
        self.assertIn('ymax', msg)
        self.assertIn('ymin', msg)

    def test_errors_if_probabilities_greater_than_1(self):
        with self.assertRaises(ValueError) as cm:
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14)
        msg = str(cm.exception)
        self.assertIn('probabilities', msg)

    def test_errors_if_only_one_covar_given(self):
        cov = [[3, 1], [1, 4]]
        with self.assertRaises(ValueError):
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14, upper_left_cov=cov)
        with self.assertRaises(ValueError):
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14, lower_right_cov=cov)

    def test_errors_if_covar_is_not_2x2(self):
        cov1 = [[3, 1, 2], [1, 4, 3]]
        cov2 = [[3, 1], [1, 4]]
        with self.assertRaises(ValueError):
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14,
                                              upper_left_cov=cov1, lower_right_cov=cov2)
        with self.assertRaises(ValueError):
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14,
                                              upper_left_cov=cov2, lower_right_cov=cov1)

    def test_errors_if_covar_is_not_symmetric(self):
        cov1 = [[3, 2], [1, 3]]
        cov2 = [[3, 1], [1, 4]]
        with self.assertRaises(ValueError):
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14,
                                              upper_left_cov=cov1, lower_right_cov=cov2)
        with self.assertRaises(ValueError):
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14,
                                              upper_left_cov=cov2, lower_right_cov=cov1)

    def test_errors_if_covar_is_not_postitive_definite(self):
        cov1 = [[1, 4], [4, 1]]
        cov2 = [[3, 1], [1, 4]]
        with self.assertRaises(ValueError):
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14,
                                              upper_left_cov=cov1, lower_right_cov=cov2)
        with self.assertRaises(ValueError):
            submission_builder.make_detection([0.1, 0.2, 0.3, 0.4, 0.5], 1, 3, 12, 14,
                                              upper_left_cov=cov2, lower_right_cov=cov1)


class TestSubmissionBuilder(th.ExtendedTestCase):
    temp_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'temp')

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_integration(self):
        # Make an example submission
        submission = {
            '000000': [
                [{
                    'classes': [0.1, 0.4, 0.2, 0.3],
                    'bbox': [1, 2, 14, 15]
                }, {
                    'classes': [0.8, 0.1, 0.05, 0.05],
                    'bbox': [1, 2, 14, 15],
                    'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
                }],
                [],
                [{
                    'classes': [0.4, 0.1, 0.3, 0.2],
                    'bbox': [1, 2, 14, 15],
                    'covars': [[[10, 2], [2, 10]], [[1, 0], [0, 100]]]
                }, {
                    'classes': [0.1, 0.7, 0.1, 0.1],
                    'bbox': [1, 2, 14, 15],
                    'covars': [[[5, 0], [0, 15]], [[16, 1], [1, 8]]]
                }],
                [{
                    'classes': [0.4, 0.1, 0.4, 0.1],
                    'bbox': [11, 12, 44, 55],
                    'covars': [[[15, 0], [0, 21]], [[126, 2], [2, 18]]]
                }],
                [{
                    'classes': [0.9, 0.01, 0.04, 0.05],
                    'bbox': [13, 14, 46, 57],
                    'covars': [[[51, 0], [0, 15]], [[1, 1], [1, 1]]]
                }]
            ],
            '000006': [
                [],
                [{
                    'classes': [0.1, 0.4, 0.2, 0.3],
                    'bbox': [1, 2, 14, 15],
                    'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
                }, {
                    'classes': [0.8, 0.1, 0.05, 0.05],
                    'bbox': [1, 2, 14, 15]
                }],
                [],
                [{
                    'classes': [0.4, 0.1, 0.3, 0.2],
                    'bbox': [11, 12, 44, 55],
                    'covars': [[[10, 2], [2, 10]], [[1, 0], [0, 100]]]
                }, {
                    'classes': [0.1, 0.7, 0.1, 0.1],
                    'bbox': [1, 2, 14, 15],
                    'covars': [[[5, 0], [0, 15]], [[16, 1], [1, 8]]]
                }],
                [],
                [{
                    'classes': [0.4, 0.1, 0.4, 0.1],
                    'bbox': [1, 2, 14, 15]
                }, {
                    'classes': [0.9, 0.01, 0.04, 0.05],
                    'bbox': [13, 14, 46, 57],
                    'covars': [[[51, 0], [0, 15]], [[2, 1], [1, 2]]]
                }],
                []
            ]
        }
        classes = class_list.CLASSES[1:5]

        # Write our submission to fle
        writer = submission_builder.SubmissionWriter(self.temp_dir, classes)
        for sequence_name, sequence_data in submission.items():
            for detections in sequence_data:
                for detection in detections:
                    if 'covars' in detection:
                        writer.add_detection(
                            class_probabilities=detection['classes'],
                            xmin=detection['bbox'][0],
                            ymin=detection['bbox'][1],
                            xmax=detection['bbox'][2],
                            ymax=detection['bbox'][3],
                            upper_left_cov=detection['covars'][0],
                            lower_right_cov=detection['covars'][1]
                        )
                    else:
                        writer.add_detection(
                            class_probabilities=detection['classes'],
                            xmin=detection['bbox'][0],
                            ymin=detection['bbox'][1],
                            xmax=detection['bbox'][2],
                            ymax=detection['bbox'][3]
                        )
                writer.next_image()
            writer.save_sequence(sequence_name)

        # Read that submission using the submission loader, and check that it's the same
        loaded_sequences = submission_loader.read_submission(self.temp_dir)
        self.assertEqual(set(submission.keys()), set(loaded_sequences.keys()))
        for sequence_name, generator in loaded_sequences.items():
            detections = list(generator)
            self.assertEqual(len(submission[sequence_name]), len(detections))
            for img_idx in range(len(detections)):
                img_detections = list(detections[img_idx])
                self.assertEqual(len(submission[sequence_name][img_idx]), len(img_detections))
                for det_idx in range(len(img_detections)):
                    sub_det = submission[sequence_name][img_idx][det_idx]
                    expected_classes = np.zeros(len(class_list.CLASSES))
                    expected_classes[1:5] = sub_det['classes']
                    self.assertNPEqual(expected_classes, img_detections[det_idx].class_list)
                    self.assertNPEqual(sub_det['bbox'], img_detections[det_idx].box)
                    if 'covars' in sub_det:
                        self.assertNPEqual(sub_det['covars'], img_detections[det_idx].covs)
