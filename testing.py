# -*- coding: utf-8 -*-
"""
Unit testing for package.
"""

import unittest
import numpy as np
import preprocessing as pp

class TestPreprocessor(unittest.TestCase):
    """Tests for the pre-processing function"""

    def setUp(self):
        """Load bible dataset for testing"""
        self.pre = pp.Preprocessor()
        self.pre.preprocess("input/testing.txt", window=3, shuffle=False)


    def test_preprocess(self):
        """Test the main preprocssing routine"""

        # Check exceptions on bad input
        sample_data = ["!@#$%^", "AAAAAA", "bbBBbb", "Cc^- c"]
        with self.assertRaises(ValueError) as context:
            self.pre.preprocess(None, 3, data=sample_data)
        self.assertEqual(str(context.exception),
                "Letters present in names missing in encoding.")

        sample_data = ["AAAAAA", "BBBBBB", "CCCCCC"]
        with self.assertRaises(ValueError) as context:
            self.pre.preprocess(None, 3, data=sample_data)
        self.assertEqual(str(context.exception),
                "At least 4 samples needed to encode.")

        # Good input
        sample_data = ["AAAAAA", "AAAAAA", "bbBBbb", "Cc^- c"]
        self.pre.preprocess(None, 3, data=sample_data)

        # Check string cleanup
        tgts = ["AAAAAA", "BBBBBB", "CC-_C"]
        check = [t in self.pre.get_targets() for t in tgts]
        self.assertTrue(all(check))

        # Make sure duplicate was dropped
        self.assertEqual(len(tgts), 3)


    def test_get_rnn_format(self):
        """Test functions to reshape for RNNs in Tensorflow. Format is:
            i = samples
            j = time/window
            k = features
        """
        self.pre.preprocess("input/testing.txt", window=3, shuffle=False)

        x_train_rnn, y_train_rnn, x_test_rnn, y_test_rnn = \
            self.pre.get_rnn_format()

        # Data for trainnig should be
        # 0 - ^^^
        # 1 - ^^A
        # 2 - ^AA
        # 3 - AAA ...

        # Test masks
        mask_caret = [t=="^" for t in pp.LETTERS]
        mask_a = [t=="A" for t in pp.LETTERS]
        mask_b = [t=="B" for t in pp.LETTERS]
        mask_d = [t=="D" for t in pp.LETTERS]

        # For all carets (begining of sequence), each letter should
        # have equal probability.
        prob_caret = np.zeros(len(pp.LETTERS))
        for idx in [pp.LETTERS.index(t) for t in ["A", "B", "C", "D"]]:
            prob_caret[idx] = 0.25

        # Element 2 0 should be carent
        check_caret = [t==1 for t in x_train_rnn[2][0]]
        self.assertEqual(check_caret, mask_caret)

        # Element 2 1 should be "A"
        check_a = [t>0 for t in x_train_rnn[2][1]]
        self.assertEqual(check_a, mask_a)

        # Element 5 & 6 should be the same
        self.assertTrue(np.all(x_train_rnn[5]==x_train_rnn[6]))

        # 10th Element is "BBB"
        for i in [0, 1, 2]:
            check_b = [t==1 for t in x_train_rnn[10+i][1]]
            self.assertEqual(check_b, mask_b)

        # test set - element 3 0 to 2 are "D"
        for i in [0, 1, 2]:
            check_d = [t==1 for t in x_test_rnn[3][i]]
            self.assertEqual(check_d, mask_d)

        # First test input/output is "^^^" --> "A, B, C, D"
        self.assertTrue(np.all(np.isclose(y_train_rnn[0][0], prob_caret)))

        # 4th train y is "D"
        self.assertTrue(np.all(x_test_rnn[3][0]==mask_d))

        # 2nd test y is "D"
        self.assertTrue(np.all(y_test_rnn[1][0]==mask_d))


    def test_statistical_prob(self):
        """Test frequency calculations for single letters and letter pairs"""

        test_words = [
        "DAA",
        "DAB",
        "DBC",
        "DBD",
        "DCE",
        "DBF",
        "BBB",
        "BBC",
        "BED",
        "CBB"]

        stat_prob=pp.StatisticalProb()
        stat_prob.calc_stats(test_words)
        prob_1=stat_prob.get_first_prob()
        prob_2=stat_prob.get_second_prob("D")

        test_p1=np.zeros(len(pp.LETTERS))
        test_p1[pp.LETTERS.index("D")]=0.6
        test_p1[pp.LETTERS.index("B")]=0.3
        test_p1[pp.LETTERS.index("C")]=0.1

        self.assertTrue(all(np.equal(prob_1,test_p1)))

        test_p2=np.zeros(len(pp.LETTERS))
        test_p2[pp.LETTERS.index("A")]=2/6
        test_p2[pp.LETTERS.index("B")]=3/6
        test_p2[pp.LETTERS.index("C")]=1/6

        self.assertTrue(all(np.equal(prob_2, test_p2)))


    def test_create_input_output(self):
        """Test create input/output methods of preprocessor"""

        human, test_x, test_y = self.pre._create_input_output(["PHILA'_DELPHIA"]) # pylint: disable=protected-access

        self.assertEqual(''.join([t[-1] for t in human['input']]),
                         "^PHILA'_DELPHIA")
        self.assertEqual(''.join([t for t in human['target']]),
                         "PHILA'_DELPHIA$")

        # Check position indicator in X vector
        check_x0=np.array([t/self.pre.get_max_length() for t in
                        range(len("PHILA' DELPHIA$"))])
        self.assertTrue(all(check_x0==test_x[:,0]))
        self.assertEqual(test_y.shape, (len("PHILA' DELPHIA")+1, len(pp.LETTERS)))


    def test_encode_in_out(self):
        """Test positional marker and one-hot encoding"""

        # A vector of the LETTERS should return the identity matrix
        x_enc, y_enc = self.pre._encode_in_out(pp.LETTERS,"B", 9) # pylint: disable=protected-access

        # Check positional marker
        self.assertEqual(x_enc[0], 9/self.pre.get_max_length())

        # Check one-hot encoding
        x_onehot = x_enc[1:]

        self.assertTrue(np.array_equal(x_onehot,
                        np.eye(len(pp.LETTERS)).reshape(-1)))

        # Check that "B" is highlighted
        y_test=np.zeros(len(pp.LETTERS))
        y_test[pp.LETTERS.index("B")]=1
        self.assertTrue(np.array_equal(y_enc, y_test))

    def test_serialization(self):
        """Test serialization and de-serialization code"""

        pre_orig = pp.Preprocessor()
        pre_test = pp.Preprocessor()

        # Populate orig from input file, create new object
        # from serialized JSON
        pre_orig.preprocess("input/bible_characters.txt", 5)
        txt=pre_orig.to_json()
        pre_test.from_json(txt)

        # Check non-numeric attributes
        self.assertEqual(pre_orig.get_max_length(), pre_test.get_max_length())
        self.assertEqual(pre_orig.filename, pre_test.filename)
        self.assertEqual(pre_orig.window, pre_test.window)
        self.assertEqual(pre_orig.get_targets(), pre_test.get_targets())

        #  Check numeric
        self.assertTrue(np.all(np.isclose(pre_orig.x_train, pre_test.x_train)))
        self.assertTrue(np.all(np.isclose(pre_orig.y_train, pre_test.y_train)))
        self.assertTrue(np.all(np.isclose(pre_orig.x_test, pre_test.x_test)))
        self.assertTrue(np.all(np.isclose(pre_orig.y_test, pre_test.y_test)))

        # Check statistics - order
        self.assertTrue(all(pre_test.statistics.get_second_df().index==\
                                pre_orig.statistics.get_second_df().index))
        self.assertTrue(all(pre_test.statistics.get_second_df().columns==\
                                pre_orig.statistics.get_second_df().columns))

        # Check statistics - values
        self.assertTrue(np.all(np.isclose(
                                    pre_orig.statistics.get_first_prob(),
                                    pre_test.statistics.get_first_prob()
                        )))
        for test_letter in pp.LETTERS:
            self.assertTrue(
                np.all(np.isclose(
                    pre_orig.statistics.get_second_prob(test_letter),
                    pre_test.statistics.get_second_prob(test_letter)
            )))


if __name__ == '__main__':
    unittest.main()
