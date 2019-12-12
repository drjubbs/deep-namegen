# -*- coding: utf-8 -*-
"""
Test suite for code.
"""

import unittest
import numpy as np
import preprocessing as pp

class TestPreprocessing(unittest.TestCase):

    def test_create_input_output(self):
        human, Xs, ys = pp.create_input_output(["PHILA' DELPHIA"])
        
        self.assertEqual(''.join([t[-1] for t in human['input']]),
                         "^PHILA' DELPHIA")
        self.assertEqual(''.join([t for t in human['target']]),
                         "PHILA' DELPHIA$")
        
        # Check position indicator in X vector
        check_X0=np.array([t/pp.MAX_LENGTH for t in 
                        range(len("PHILA' DELPHIA$"))])
        self.assertTrue(all(check_X0==Xs[:,0]))        
        self.assertEqual(ys.shape, (len("PHILA' DELPHIA")+1, len(pp.LETTERS)))
        
    
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