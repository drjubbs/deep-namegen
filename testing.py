# -*- coding: utf-8 -*-
"""
Test suite for code.
"""

import unittest
import numpy as np
import preprocessing as pp

class TestPreprocessing(unittest.TestCase):

    def test_create_input_output(self):
        a, b, c = pp.create_input_output(["PHILA' DELPHIA"])
        
        self.assertEqual(''.join([t[-1] for t in a['input']]),
                         "^PHILA' DELPHIA")
        self.assertEqual(''.join([t for t in a['target']]),
                         "PHILA' DELPHIA$")
        
        self.assertEqual(c.shape, (len("PHILA' DELPHIA")+1, len(pp.LETTERS)))
        
    
    def test_encode_in_out(self):
        
        # A vector of the LETTERS should return the identity matrix
        x, y = pp.encode_in_out(pp.LETTERS,"B")
        self.assertTrue(np.array_equal(x,
                        np.eye(len(pp.LETTERS)).reshape(-1)))
        
        # Check that "B" is highlighted
        y_test=np.zeros(len(pp.LETTERS))
        y_test[pp.LETTERS.index("B")]=1
        self.assertTrue(np.array_equal(y,y_test))

if __name__ == '__main__':
    unittest.main()