import unittest

from src.sabre_own import initSabre

#import sys
#from os.path import dirname, abspath
#sys.path.append(dirname(abspath(__file__)))

class TestMainFunction(unittest.TestCase):
    def test_dictionary_values(self):
        result = initSabre()  # assuming the main_function returns the result dictionary
        expected_values = {
            "buffer_size": 25000,
            "total_played_utility": 484.9654794493675,
            "time_average_played_utility": 2.4359830955119453,
            "total_played_bitrate": 579714,
            "time_average_played_bitrate": 2911.905205778773,
            "total_play_time": 597.252272,
            "total_play_time_chunks": 199.08409066666667,
            "total_rebuffer": 0.0,
            "rebuffer_ratio": 0.0,
            "time_average_rebuffer": 0.0,
            "total_rebuffer_events": 0,
            "time_average_rebuffer_events": 0.0,
            "total_bitrate_change": 80654,
            "time_average_bitrate_change": 405.1252901721904,
            "total_log_bitrate_change": 30.65922013412642,
            "time_average_log_bitrate_change": 0.15400135707207366,
            "time_average_score": 2.4359830955119453,
            "over_estimate_count": 88,
            "over_estimate": 334.3471958953972,
            "leq_estimate_count": 110,
            "leq_estimate": 689.183814243396,
            "estimate": -234.28114307059911,
            "rampup_time": 3.252272,
            "total_reaction_time": 61.26135999999942
        }    
        self.assertDictEqual(result, expected_values)

if __name__ == '__main__':
    unittest.main()
