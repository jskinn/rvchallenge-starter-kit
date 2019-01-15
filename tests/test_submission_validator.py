from __future__ import absolute_import, division, print_function, unicode_literals

import os.path
import warnings
import shutil
import json

import tests.test_helpers as th
import submission_validator


class TestSubmissionValidatorValidateDetections(th.ExtendedTestCase):

    def test_errors_for_missing_label_probs(self):
        with self.assertRaises(KeyError) as cm:
            submission_validator.validate_detections([{
                # 'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def test_errors_for_label_probs_wrong_length(self):
        with self.assertRaises(KeyError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46]
            }], [list(range(5)), list(range(5))], num_classes=6, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def test_warns_for_label_probs_ignored(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            submission_validator.validate_detections([{
                'label_probs': [0.01, 0.01, 0.01, 0.01, 0.02],
                'bbox': [12, 14, 55, 46]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
            self.assertEqual(1, len(w))
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assert_contains_indexes(str(w[-1].message), 'test.json', 13, 0)

    def test_warns_for_label_probs_normalized(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.5, 0.5, 0.2],
                'bbox': [12, 14, 55, 46]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
            self.assertEqual(1, len(w))
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assert_contains_indexes(str(w[-1].message), 'test.json', 13, 0)

    def test_errors_for_missing_bbox(self):
        with self.assertRaises(KeyError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2]
                # 'bbox': [12, 14, 55, 46]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def test_errors_for_bbox_wrong_size(self):
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46, 47]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def test_errors_for_bbox_xmax_less_than_xmin(self):
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [55, 14, 12, 46]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def test_errors_for_bbox_ymax_less_than_ymin(self):
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 46, 55, 14]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def test_errors_for_invalid_cov(self):
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[1, 0], [0, 1]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def test_errors_for_assymetric_cov(self):
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [3, 4]], [[1, 0], [0, 1]]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[5, 6], [7, 8]]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def test_errors_for_non_positive_definite_cov(self):
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [2, 1]], [[1, 0], [0, 1]]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_detections([{
                'label_probs': [0.1, 0.1, 0.1, 0.1, 0.2],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 2], [2, 1]]]
            }], [list(range(5)), list(range(5))], num_classes=5, img_idx=13, sequence_name='test.json')
        self.assert_contains_indexes(str(cm.exception), 'test.json', 13, 0)

    def assert_contains_indexes(self, msg, sequence_name, img_num, det_idx):
        self.assertIn(str(sequence_name), msg)
        self.assertIn(str(img_num), msg)
        self.assertIn(str(det_idx), msg)


class TestSubmissionValidatorValidateSequence(th.ExtendedTestCase):
    temp_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'temp')

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def make_sequence(self, detections):
        os.makedirs(self.temp_dir, exist_ok=True)
        json_file = os.path.join(self.temp_dir, 'test.json')
        patch_classes(detections)
        with open(json_file, 'w') as fp:
            json.dump({
                'classes': submission_validator.CLASSES,
                'detections': detections
            }, fp)
        return json_file

    def test_errors_contain_correct_image_numbers(self):
        detections = [
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }],
            [],
            [],
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }, {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [2, 1]], [[1, 0], [0, 1]]]
            }],
        ]
        sequence_json = self.make_sequence(detections)
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_sequence(sequence_json)
        msg = str(cm.exception)
        self.assertIn(os.path.basename(sequence_json), msg)
        self.assertIn('3', msg)
        self.assertIn('1', msg)

    def test_errors_if_classes_missing(self):
        detections = [
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }],
            [],
            [],
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }, {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [2, 1]], [[1, 0], [0, 1]]]
            }],
        ]
        patch_classes(detections)

        os.makedirs(self.temp_dir, exist_ok=True)
        json_file = os.path.join(self.temp_dir, 'test.json')
        with open(json_file, 'w') as fp:
            json.dump({
                # 'classes': submission_validator.CLASSES,
                'detections': detections
            }, fp)

        with self.assertRaises(KeyError) as cm:
            submission_validator.validate_sequence(json_file)
        msg = str(cm.exception)
        self.assertIn('test.json', msg)

    def test_errors_if_detections_missing(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        json_file = os.path.join(self.temp_dir, 'test.json')
        with open(json_file, 'w') as fp:
            json.dump({
                'classes': submission_validator.CLASSES,
                # 'detections': detections
            }, fp)

        with self.assertRaises(KeyError) as cm:
            next(submission_validator.validate_sequence(json_file))
        msg = str(cm.exception)
        self.assertIn('test.json', msg)

    def test_errors_if_no_valid_classes(self):
        detections = [
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }],
            [],
            [],
            [{
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
            }, {
                'label_probs': [0.1, 0.2, 0.3, 0.4],
                'bbox': [12, 14, 55, 46],
                'covars': [[[1, 2], [2, 1]], [[1, 0], [0, 1]]]
            }],
        ]
        patch_classes(detections)

        os.makedirs(self.temp_dir, exist_ok=True)
        json_file = os.path.join(self.temp_dir, 'test.json')
        with open(json_file, 'w') as fp:
            json.dump({
                'classes': [str(idx) for idx in range(len(submission_validator.CLASSES))],
                'detections': detections
            }, fp)

        with self.assertRaises(ValueError) as cm:
            next(submission_validator.validate_sequence(json_file))
        msg = str(cm.exception)
        self.assertIn('test.json', msg)


class TestSubmissionLoaderReadSubmission(th.ExtendedTestCase):
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')

    def tearDown(self):
        if os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def make_submission(self, detections_map, subfolder=None):
        root = self.temp_dir
        if subfolder is not None:
            root = os.path.join(self.temp_dir, subfolder)
        os.makedirs(root, exist_ok=True)
        for sequence_name, detections in detections_map.items():
            json_file = os.path.join(root, '{0}.json'.format(sequence_name))
            patch_classes(detections)
            with open(json_file, 'w') as fp:
                json.dump({
                    'classes': submission_validator.CLASSES,
                    'detections': detections
                }, fp)

    def test_raises_error_if_directory_doesnt_exist(self):
        not_a_dir = os.path.join(self.temp_dir, 'not', 'a', 'dir')
        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_submission(not_a_dir)
        msg = str(cm.exception)
        self.assertIn(not_a_dir, msg)

    def test_raises_error_if_missing_sequences(self):
        all_idx = set(range(18))
        excluded = {3, 4, 5, 13, 17}
        self.make_submission({'{0:06}'.format(idx): [] for idx in all_idx - excluded})

        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_submission(self.temp_dir)
        msg = str(cm.exception)
        for idx in excluded:
            self.assertIn('{0:06}'.format(idx), msg)

    def test_raises_error_if_duplicate_sequence(self):
        self.make_submission({'{0:06}'.format(idx): [] for idx in range(18)}, 'folder_a')
        self.make_submission({'000000': []}, 'folder_b')

        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_submission(self.temp_dir)
        msg = str(cm.exception)
        self.assertIn('folder_a/000000.json', msg)
        self.assertIn('folder_b/000000.json', msg)

    def test_no_error_if_testing_subset(self):
        all_idx = set(range(18))
        excluded = {3, 4, 5, 13, 17}
        subset = [idx for idx in all_idx - excluded]
        self.make_submission({'{0:06}'.format(idx): [] for idx in subset})

        submission_validator.validate_submission(self.temp_dir, sequence_ids=[idx for idx in subset])

    def test_raise_error_if_missing_sequence_in_subset(self):
        all_idx = set(range(18))
        excluded = {3, 4, 5, 13, 17}
        subset = [idx for idx in all_idx - excluded]
        self.make_submission({'{0:06}'.format(idx): [] for idx in subset[:-1]})

        with self.assertRaises(ValueError) as cm:
            submission_validator.validate_submission(self.temp_dir,
                                                     sequence_ids=subset)
        msg = str(cm.exception)
        self.assertIn('{0:06}'.format(16), msg)


def patch_classes(detections):
    # Patch the label probabilities to be the right length
    for img_dets in detections:
        for det in img_dets:
            if len(det['label_probs']) < len(submission_validator.CLASSES):
                det['label_probs'] = make_probs(det['label_probs'])


def make_probs(probs):
    full_probs = [0.0] * len(submission_validator.CLASSES)
    full_probs[0:len(probs)] = probs
    return full_probs
