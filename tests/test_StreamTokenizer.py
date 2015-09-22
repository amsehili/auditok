'''
@author: Amine Sehili <amine.sehili@gmail.com>
September 2015

'''

import unittest
from auditok import StreamTokenizer, StringDataSource, DataValidator


class AValidator(DataValidator):
    
    def is_valid(self, frame):
        return frame == "A"


class TestStreamTokenizerInitParams(unittest.TestCase):
    
    
    def setUp(self):
        self.A_validator = AValidator()
        
    # Completely deactivate init_min and init_max_silence
    # The tokenizer will only rely on the other parameters
    # Note that if init_min = 0, the value of init_max_silence
    # will have no effect
    def test_init_min_0_init_max_silence_0(self):
        
        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=20,
                                     max_continuous_silence=4, init_min = 0,
                                     init_max_silence = 0, mode=0)
        
        
        data_source = StringDataSource("aAaaaAaAaaAaAaaaaaaaAAAAAAAA")
        #                            ^              ^   ^      ^
        #                            2              16  20     27
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 2, msg="wrong number of tokens, expected: 2, found: {0} ".format(len(tokens)))
        tok1, tok2 = tokens[0], tokens[1]
        
        # tok1[0]: data
        # tok1[1]: start frame (included)
        # tok1[2]: end frame (included)
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AaaaAaAaaAaAaaaa",
                        msg="wrong data for token 1, expected: 'AaaaAaAaaAaAaaaa', found: {0} ".format(data))
        self.assertEqual(start, 1, msg="wrong start frame for token 1, expected: 1, found: {0} ".format(start))
        self.assertEqual(end, 16, msg="wrong end frame for token 1, expected: 16, found: {0} ".format(end))
        
        
        data = ''.join(tok2[0])
        start = tok2[1]
        end = tok2[2]
        self.assertEqual(data, "AAAAAAAA",
                        msg="wrong data for token 1, expected: 'AAAAAAAA', found: {0} ".format(data))
        self.assertEqual(start, 20, msg="wrong start frame for token 2, expected: 20, found: {0} ".format(start))
        self.assertEqual(end, 27, msg="wrong end frame for token 2, expected: 27, found: {0} ".format(end))
    
    
        
    # A valid token is considered iff the tokenizer encounters
    # at least valid frames (init_min = 3) between witch there
    # are at most 0 consecutive non valid frames (init_max_silence = 0)
    # The tokenizer will only rely on the other parameters
    # In other words, a valid token must start with 3 valid frames
    def test_init_min_3_init_max_silence_0(self):
        
        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=20,
                                     max_continuous_silence=4, init_min = 3,
                                     init_max_silence = 0, mode=0)
        
        
        #data_source = StringDataSource("aAaaaAaAaaAaAaaaaaAAAAAAAAAaaaaaAAAAA")
        #                                             ^       ^     ^   ^
        #                                             18      26    32  36
        
        data_source = StringDataSource("aAaaaAaAaaAaAaaaaaAAAAAAAAAaaaaaaAAAAA")
        #                                             ^           ^  ^   ^
        #                                             18          30 33  37
        
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 2, msg="wrong number of tokens, expected: 2, found: {0} ".format(len(tokens)))
        tok1, tok2 = tokens[0], tokens[1]
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AAAAAAAAAaaaa",
                        msg="wrong data for token 1, expected: 'AAAAAAAAAaaaa', found: '{0}' ".format(data))
        self.assertEqual(start, 18, msg="wrong start frame for token 1, expected: 18, found: {0} ".format(start))
        self.assertEqual(end, 30, msg="wrong end frame for token 1, expected: 30, found: {0} ".format(end))
        
        
        data = ''.join(tok2[0])
        start = tok2[1]
        end = tok2[2]
        self.assertEqual(data, "AAAAA",
                        msg="wrong data for token 1, expected: 'AAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 33, msg="wrong start frame for token 2, expected: 33, found: {0} ".format(start))
        self.assertEqual(end, 37, msg="wrong end frame for token 2, expected: 37, found: {0} ".format(end))
        
    
    # A valid token is considered iff the tokenizer encounters
    # at least valid frames (init_min = 3) between witch there
    # are at most 2 consecutive non valid frames (init_max_silence = 2)
    def test_init_min_3_init_max_silence_2(self):
        
        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=20,
                                     max_continuous_silence=4, init_min = 3,
                                     init_max_silence = 2, mode=0)
        
        
        data_source = StringDataSource("aAaaaAaAaaAaAaaaaaaAAAAAAAAAaaaaaaaAAAAA")
        #                                ^          ^  ^           ^   ^   ^
        #                                5          16 19          31  35  39
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 3, msg="wrong number of tokens, expected: 3, found: {0} ".format(len(tokens)))
        tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AaAaaAaAaaaa",
                        msg="wrong data for token 1, expected: 'AaAaaAaA', found: '{0}' ".format(data))
        self.assertEqual(start, 5, msg="wrong start frame for token 1, expected: 5, found: {0} ".format(start))
        self.assertEqual(end, 16, msg="wrong end frame for token 1, expected: 16, found: {0} ".format(end))
        
        
        data = ''.join(tok2[0])
        start = tok2[1]
        end = tok2[2]
        self.assertEqual(data, "AAAAAAAAAaaaa",
                        msg="wrong data for token 2, expected: 'AAAAAAAAAaaaa', found: '{0}' ".format(data))
        self.assertEqual(start, 19, msg="wrong start frame for token 2, expected: 19, found: {0} ".format(start))
        self.assertEqual(end, 31, msg="wrong end frame for token 2, expected: 31, found: {0} ".format(end))
        
        
        data = ''.join(tok3[0])
        start = tok3[1]
        end = tok3[2]
        self.assertEqual(data, "AAAAA",
                        msg="wrong data for token 3, expected: 'AAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 35, msg="wrong start frame for token 2, expected: 35, found: {0} ".format(start))
        self.assertEqual(end, 39, msg="wrong end frame for token 2, expected: 39, found: {0} ".format(end))    
               
        
    
class TestStreamTokenizerMinMaxLength(unittest.TestCase):
  
    def setUp(self):
        self.A_validator = AValidator()
    
    
    def test_min_length_6_init_max_length_20(self):
    
        tokenizer = StreamTokenizer(self.A_validator, min_length = 6, max_length=20,
                                     max_continuous_silence=2, init_min = 3,
                                     init_max_silence = 3, mode=0)
        
        
        data_source = StringDataSource("aAaaaAaAaaAaAaaaaaAAAAAAAAAaaaaaAAAAA")
        #                            ^            ^   ^         ^
        #                            1            14  18        28
        
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 2, msg="wrong number of tokens, expected: 2, found: {0} ".format(len(tokens)))
        tok1, tok2 = tokens[0], tokens[1]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AaaaAaAaaAaAaa",
                        msg="wrong data for token 1, expected: 'AaaaAaAaaAaAaa', found: '{0}' ".format(data))
        self.assertEqual(start, 1, msg="wrong start frame for token 1, expected: 1, found: {0} ".format(start))
        self.assertEqual(end, 14, msg="wrong end frame for token 1, expected: 14, found: {0} ".format(end))
        
        
        data = ''.join(tok2[0])
        start = tok2[1]
        end = tok2[2]
        self.assertEqual(data, "AAAAAAAAAaa",
                        msg="wrong data for token 2, expected: 'AAAAAAAAAaa', found: '{0}' ".format(data))
        self.assertEqual(start, 18, msg="wrong start frame for token 2, expected: 18, found: {0} ".format(start))
        self.assertEqual(end, 28, msg="wrong end frame for token 2, expected: 28, found: {0} ".format(end))
    
    
    def test_min_length_1_init_max_length_1(self):
    
        tokenizer = StreamTokenizer(self.A_validator, min_length = 1, max_length=1,
                                     max_continuous_silence=0, init_min = 0,
                                     init_max_silence = 0, mode=0)
        
        
        data_source = StringDataSource("AAaaaAaaaAaAaaAaAaaaaaAAAAAAAAAaaaaaAAAAA")
        
        tokens = tokenizer.tokenize(data_source)
                        
        self.assertEqual(len(tokens), 21, msg="wrong number of tokens, expected: 21, found: {0} ".format(len(tokens)))
        
        
    def test_min_length_10_init_max_length_20(self):
    
        tokenizer = StreamTokenizer(self.A_validator, min_length = 10, max_length=20,
                                     max_continuous_silence=4, init_min = 3,
                                     init_max_silence = 3, mode=0)
        
        
        data_source = StringDataSource("aAaaaAaAaaAaAaaaaaaAAAAAaaaaaaAAAAAaaAAaaAAA")
        #                            ^              ^             ^            ^
        #                            1              16            30           45
        
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 2, msg="wrong number of tokens, expected: 2, found: {0} ".format(len(tokens)))
        tok1, tok2 = tokens[0], tokens[1]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AaaaAaAaaAaAaaaa",
                        msg="wrong data for token 1, expected: 'AaaaAaAaaAaAaaaa', found: '{0}' ".format(data))
        self.assertEqual(start, 1, msg="wrong start frame for token 1, expected: 1, found: {0} ".format(start))
        self.assertEqual(end, 16, msg="wrong end frame for token 1, expected: 16, found: {0} ".format(end))
        
        
        data = ''.join(tok2[0])
        start = tok2[1]
        end = tok2[2]
        self.assertEqual(data, "AAAAAaaAAaaAAA",
                        msg="wrong data for token 2, expected: 'AAAAAaaAAaaAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 30, msg="wrong start frame for token 2, expected: 30, found: {0} ".format(start))
        self.assertEqual(end, 43, msg="wrong end frame for token 2, expected: 43, found: {0} ".format(end))
    
    
        
    def test_min_length_4_init_max_length_5(self):
    
        tokenizer = StreamTokenizer(self.A_validator, min_length = 4, max_length=5,
                                     max_continuous_silence=4, init_min = 3,
                                     init_max_silence = 3, mode=0)
        
        
        data_source = StringDataSource("aAaaaAaAaaAaAaaaaaAAAAAAAAaaaaaaAAAAAaaaaaAAaaAaa")
        #                                             ^   ^^   ^    ^   ^     ^   ^
        #                                             18 2223  27   32  36    42  46
        
        tokens = tokenizer.tokenize(data_source)
               
        self.assertEqual(len(tokens), 4, msg="wrong number of tokens, expected: 4, found: {0} ".format(len(tokens)))
        tok1, tok2, tok3, tok4 = tokens[0], tokens[1], tokens[2], tokens[3]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AAAAA",
                        msg="wrong data for token 1, expected: 'AAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 18, msg="wrong start frame for token 1, expected: 18, found: {0} ".format(start))
        self.assertEqual(end, 22, msg="wrong end frame for token 1, expected: 22, found: {0} ".format(end))
        
        
        data = ''.join(tok2[0])
        start = tok2[1]
        end = tok2[2]
        self.assertEqual(data, "AAAaa",
                        msg="wrong data for token 1, expected: 'AAAaa', found: '{0}' ".format(data))
        self.assertEqual(start, 23, msg="wrong start frame for token 1, expected: 23, found: {0} ".format(start))
        self.assertEqual(end, 27, msg="wrong end frame for token 1, expected: 27, found: {0} ".format(end))
        
        
        data = ''.join(tok3[0])
        start = tok3[1]
        end = tok3[2]
        self.assertEqual(data, "AAAAA",
                        msg="wrong data for token 1, expected: 'AAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 32, msg="wrong start frame for token 1, expected: 1, found: {0} ".format(start))
        self.assertEqual(end, 36, msg="wrong end frame for token 1, expected: 7, found: {0} ".format(end))
        
        
        data = ''.join(tok4[0])
        start = tok4[1]
        end = tok4[2]
        self.assertEqual(data, "AAaaA",
                        msg="wrong data for token 2, expected: 'AAaaA', found: '{0}' ".format(data))
        self.assertEqual(start, 42, msg="wrong start frame for token 2, expected: 17, found: {0} ".format(start))
        self.assertEqual(end, 46, msg="wrong end frame for token 2, expected: 22, found: {0} ".format(end))
        
        
class TestStreamTokenizerMaxContinuousSilence(unittest.TestCase):
    
    def setUp(self):
        self.A_validator = AValidator()
    
    
    def test_min_5_max_10_max_continuous_silence_0(self):

        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=10,
                                    max_continuous_silence=0, init_min = 3,
                                    init_max_silence = 3, mode=0)
        
        data_source = StringDataSource("aaaAAAAAaAAAAAAaaAAAAAAAAAa")
        #                              ^   ^ ^    ^  ^       ^
        #                              3   7 9   14 17      25
        
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 3, msg="wrong number of tokens, expected: 3, found: {0} ".format(len(tokens)))
        tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AAAAA",
                        msg="wrong data for token 1, expected: 'AAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 3, msg="wrong start frame for token 1, expected: 3, found: {0} ".format(start))
        self.assertEqual(end, 7, msg="wrong end frame for token 1, expected: 7, found: {0} ".format(end))
        
        
        data = ''.join(tok2[0])
        start = tok2[1]
        end = tok2[2]
        self.assertEqual(data, "AAAAAA",
                        msg="wrong data for token 1, expected: 'AAAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 9, msg="wrong start frame for token 1, expected: 9, found: {0} ".format(start))
        self.assertEqual(end, 14, msg="wrong end frame for token 1, expected: 14, found: {0} ".format(end))
        
        
        data = ''.join(tok3[0])
        start = tok3[1]
        end = tok3[2]
        self.assertEqual(data, "AAAAAAAAA",
                        msg="wrong data for token 1, expected: 'AAAAAAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 17, msg="wrong start frame for token 1, expected: 17, found: {0} ".format(start))
        self.assertEqual(end, 25, msg="wrong end frame for token 1, expected: 25, found: {0} ".format(end))
        
        
        
        
    def test_min_5_max_10_max_continuous_silence_1(self):

        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=10,
                                    max_continuous_silence=1, init_min = 3,
                                    init_max_silence = 3, mode=0)
        
        data_source = StringDataSource("aaaAAAAAaAAAAAAaaAAAAAAAAAa")
        #                              ^        ^^ ^ ^        ^
        #                              3       12131517      26
        #                                     (12 13 15 17)
        
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 3, msg="wrong number of tokens, expected: 3, found: {0} ".format(len(tokens)))
        tok1, tok2, tok3 = tokens[0], tokens[1], tokens[2]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AAAAAaAAAA",
                        msg="wrong data for token 1, expected: 'AAAAAaAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 3, msg="wrong start frame for token 1, expected: 3, found: {0} ".format(start))
        self.assertEqual(end, 12, msg="wrong end frame for token 1, expected: 10, found: {0} ".format(end))
        
        
        data = ''.join(tok2[0])
        start = tok2[1]
        end = tok2[2]
        self.assertEqual(data, "AAa",
                        msg="wrong data for token 1, expected: 'AAa', found: '{0}' ".format(data))
        self.assertEqual(start, 13, msg="wrong start frame for token 1, expected: 9, found: {0} ".format(start))
        self.assertEqual(end, 15, msg="wrong end frame for token 1, expected: 14, found: {0} ".format(end))
        
        
        data = ''.join(tok3[0])
        start = tok3[1]
        end = tok3[2]
        self.assertEqual(data, "AAAAAAAAAa",
                        msg="wrong data for token 1, expected: 'AAAAAAAAAa', found: '{0}' ".format(data))
        self.assertEqual(start, 17, msg="wrong start frame for token 1, expected: 17, found: {0} ".format(start))
        self.assertEqual(end, 26, msg="wrong end frame for token 1, expected: 26, found: {0} ".format(end))
        
        
class TestStreamTokenizerModes(unittest.TestCase):
    
    def setUp(self):
        self.A_validator = AValidator()
    
    def test_STRICT_MIN_LENGTH(self):
        
        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=8,
                                    max_continuous_silence=3, init_min = 3,
                                    init_max_silence = 3, mode=StreamTokenizer.STRICT_MIN_LENGTH)
        
        data_source = StringDataSource("aaAAAAAAAAAAAA")
        #                             ^      ^
        #                             2      9
        
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 1, msg="wrong number of tokens, expected: 1, found: {0} ".format(len(tokens)))
        tok1 = tokens[0]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AAAAAAAA",
                        msg="wrong data for token 1, expected: 'AAAAAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 2, msg="wrong start frame for token 1, expected: 2, found: {0} ".format(start))
        self.assertEqual(end, 9, msg="wrong end frame for token 1, expected: 9, found: {0} ".format(end))
    
    
    def test_DROP_TAILING_SILENCE(self):
        
        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=10,
                                    max_continuous_silence=2, init_min = 3,
                                    init_max_silence = 3, mode=StreamTokenizer.DROP_TAILING_SILENCE)
        
        data_source = StringDataSource("aaAAAAAaaaaa")
        #                             ^   ^
        #                             2   6
        
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 1, msg="wrong number of tokens, expected: 1, found: {0} ".format(len(tokens)))
        tok1 = tokens[0]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AAAAA",
                        msg="wrong data for token 1, expected: 'AAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 2, msg="wrong start frame for token 1, expected: 2, found: {0} ".format(start))
        self.assertEqual(end, 6, msg="wrong end frame for token 1, expected: 6, found: {0} ".format(end))
        
        
    def test_STRICT_MIN_LENGTH_and_DROP_TAILING_SILENCE(self):
        
        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=8,
                                    max_continuous_silence=3, init_min = 3,
                                    init_max_silence = 3, mode=StreamTokenizer.STRICT_MIN_LENGTH | StreamTokenizer.DROP_TAILING_SILENCE)
        
        data_source = StringDataSource("aaAAAAAAAAAAAAaa")
        #                             ^      ^
        #                             2      8
        
        tokens = tokenizer.tokenize(data_source)
                
        self.assertEqual(len(tokens), 1, msg="wrong number of tokens, expected: 1, found: {0} ".format(len(tokens)))
        tok1 = tokens[0]
        
        
        data = ''.join(tok1[0])
        start = tok1[1]
        end = tok1[2]
        self.assertEqual(data, "AAAAAAAA",
                        msg="wrong data for token 1, expected: 'AAAAAAAA', found: '{0}' ".format(data))
        self.assertEqual(start, 2, msg="wrong start frame for token 1, expected: 2, found: {0} ".format(start))
        self.assertEqual(end, 9, msg="wrong end frame for token 1, expected: 9, found: {0} ".format(end))
        
    
class TestStreamTokenizerCallback(unittest.TestCase):
    
    def setUp(self):
        self.A_validator = AValidator()
    
    def test_callback(self):
        
        tokens = []
        
        def callback(data, start, end):
            tokens.append((data, start, end))
            
        
        tokenizer = StreamTokenizer(self.A_validator, min_length = 5, max_length=8,
                                    max_continuous_silence=3, init_min = 3,
                                    init_max_silence = 3, mode=0)
        
        data_source = StringDataSource("aaAAAAAAAAAAAAa")
        #                             ^      ^^   ^
        #                             2      910  14
        
        tokenizer.tokenize(data_source, callback=callback)
        
        self.assertEqual(len(tokens), 2, msg="wrong number of tokens, expected: 1, found: {0} ".format(len(tokens)))
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
