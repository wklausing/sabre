import unittest
import sys
sys.path.append('/Users/prabu/git/sabre')

from src.sabreOriginal import init as initSabreOriginal
from src.sabre import init as initSabre
from src.sabreNew2 import init as initSabreNew

class TestMainFunction(unittest.TestCase):

    originalResult = {
            'buffer_size': 25000,
            'total_played_utility': 484.9654794493675,
            'time_average_played_utility': 2.4359830955119453,
            'total_played_bitrate': 579714,
            'time_average_played_bitrate': 2911.905205778773,
            'total_play_time': 597.252272,
            'total_play_time_chunks': 199.08409066666667,
            'total_rebuffer': 0.0,
            'rebuffer_ratio': 0.0,
            'time_average_rebuffer': 0.0,
            'total_rebuffer_events': 0,
            'time_average_rebuffer_events': 0.0,
            'total_bitrate_change': 80654,
            'time_average_bitrate_change': 405.1252901721904,
            'total_log_bitrate_change': 30.65922013412642,
            'time_average_log_bitrate_change': 0.15400135707207366,
            'time_average_score': 2.4359830955119453,
            'total_reaction_time': 61.26135999999942,
            'estimate': -234.28114307059911
        }

    # def testCheckValuesWithOriginal(self):
    #     '''
    #     Testing original values against sabreOriginal.py
    #     '''
    #     resultSabreOriginal = initSabreOriginal()
    #     for key in self.originalResult:
    #         self.assertEqual(self.originalResult[key], resultSabreOriginal[key])

    # def testCheckValuesWithSabre(self):
    #     '''
    #     Testing original values against sabre.py
    #     '''
    #     resultSabreOriginal = initSabre()
    #     for key in self.originalResult:
    #         self.assertEqual(self.originalResult[key], resultSabreOriginal[key])

    def testCheckValuesAgainstSabreNew(self):
        '''
        Testing sabre.py against sabreNew.py
        '''
        resultSabre = initSabre()
        resultSabreNew = initSabreNew()
        for key in resultSabre:
            self.assertEqual(resultSabre[key], resultSabreNew[key])

    def testCheckValuesAgainstSabreNewAllOptions(self):
        '''
        Testing sabre.py against sabreNew.py
        '''
        abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
        averageList = ['ewma', 'sliding']
        for abr in abrList:
            for average in averageList:
                print('Testing: ', abr, average)
                resultSabreNew = initSabreNew(abr=abr, moving_average=average, verboseInput=False)
                resultSabre = initSabre(abr=abr, moving_average=average, verboseInput=False)
                
                for key in resultSabre:
                    self.assertEqual(resultSabre[key], resultSabreNew[key])

if __name__ == '__main__':
    unittest.main()
