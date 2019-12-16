# -*- coding: utf-8 -*-
"""
Test suite for code.
"""

import unittest
import numpy as np
import preprocessing as pp

class TestPreprocessing(unittest.TestCase):
   
    def setUp(self):
        """Load bible dataset for testing"""
        self.pre = pp.Preprocessor("in/bible_characters.txt")
    
    def test_StatisticalProb(self):
        
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

        sp=pp.StatisticalProb(test_words)
        p1=sp.get_first_prob()
        p2=sp.get_second_prob("D")
        
        test_p1=np.zeros(len(pp.LETTERS))
        test_p1[pp.LETTERS.index("D")]=0.6
        test_p1[pp.LETTERS.index("B")]=0.3
        test_p1[pp.LETTERS.index("C")]=0.1
        
        self.assertTrue(all(np.equal(p1,test_p1)))
        
        test_p2=np.zeros(len(pp.LETTERS))
        test_p2[pp.LETTERS.index("A")]=2/6
        test_p2[pp.LETTERS.index("B")]=3/6
        test_p2[pp.LETTERS.index("C")]=1/6
        
        self.assertTrue(all(np.equal(p2, test_p2)))


    def test_create_input_output(self):
                
        human, Xs, ys = self.pre._create_input_output(["PHILA'_DELPHIA"])
        
        self.assertEqual(''.join([t[-1] for t in human['input']]),
                         "^PHILA'_DELPHIA")
        self.assertEqual(''.join([t for t in human['target']]),
                         "PHILA'_DELPHIA$")
        
        # Check position indicator in X vector
        check_X0=np.array([t/pp.MAX_LENGTH for t in 
                        range(len("PHILA' DELPHIA$"))])
        self.assertTrue(all(check_X0==Xs[:,0]))        
        self.assertEqual(ys.shape, (len("PHILA' DELPHIA")+1, len(pp.LETTERS)))
        
    
    def test_encode_in_out(self):
        
        # A vector of the LETTERS should return the identity matrix
        x, y = self.pre._encode_in_out(pp.LETTERS,"B", 9)
        
        # Check positional marker
        self.assertEqual(x[0], 9/pp.MAX_LENGTH)
        xr = x[1:]
        
        self.assertTrue(np.array_equal(xr,
                        np.eye(len(pp.LETTERS)).reshape(-1)))
        
        # Check that "B" is highlighted
        y_test=np.zeros(len(pp.LETTERS))
        y_test[pp.LETTERS.index("B")]=1
        self.assertTrue(np.array_equal(y,y_test))

if __name__ == '__main__':
    unittest.main()