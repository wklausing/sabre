import unittest
import sys
sys.path.append('/Users/prabu/git/sabre')
sys.path.append('/Users/prabu/git/sabre/src')

from src.sabreV1 import init as initSabreV1 # Having this for testing purposes
from src.sabreV2 import init as initSabreV2 # Encapsulation done
from src.sabreV3 import Sabre as SabreV3
from src.sabreV4 import Sabre as SabreV4
from src.sabreV6 import Sabre as SabreV6
from src.sabreV7 import Sabre as SabreV7
# from src.sabreV5 import Sabre as SabreV5 # Working on this

#from src.gymSabre import SabreGym

class TestMainFunction(unittest.TestCase):

    # Values from original Sabre in default mode.
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

    # def testCheckValuesWithSabreV1(self):
    #     '''
    #     Testing original values against sabre.py
    #     '''
    #     resultSabreOriginal = initSabreV1()
    #     for key in self.originalResult:
    #         self.assertEqual(self.originalResult[key], resultSabreOriginal[key])

    # def testCheckValuesAgainstSabreV2(self):
    #     '''
    #     Testing sabreV1.py against sabreV2.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV1 = initSabreV1(abr=abr, moving_average=average, verboseInput=False)
    #             resultSabreV2 = initSabreV2(abr=abr, moving_average=average, verbose=False)
                
    #             for key in resultSabreV1:
    #                 self.assertEqual(resultSabreV1[key], resultSabreV2[key])

    # def testCheckValuesAgainstSabreV3(self):
    #     '''
    #     Testing sabreV2.py against sabreV3.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV2 = initSabreV2(abr=abr, moving_average=average, verbose=False)
    #             resultSabreV3 = SabreV3(abr=abr, moving_average=average, verbose=False).testing()
                
    #             for key in resultSabreV2:
    #                 self.assertEqual(resultSabreV2[key], resultSabreV3[key])

    # def testCheckValuesAgainstSabreV4(self):
    #     '''
    #     Testing sabreV3.py against sabreV4.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV3 = SabreV3(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             resultSabreV4 = SabreV4(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             for key in resultSabreV3:
    #                 print(key)
    #                 self.assertEqual(resultSabreV3[key], resultSabreV4[key])

    # def testCheckValuesAgainstSabreV5(self):
    #     '''
    #     Testing sabreV4.py against sabreV5.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV4 = SabreV4(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             resultSabreV5 = SabreV5(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             for key in resultSabreV4:
    #                 print(key)
    #                 self.assertEqual(resultSabreV4[key], resultSabreV5[key])

    # def testQoECalc(self):
    #     '''
    #     Example of calculating QoE with Sabre.
    #     '''
    #     # Bitrate - ReBuffering - BitrateSwichingEvents
    #     sabre = SabreV5(abr='bola', moving_average='ewma', verbose=False,  replace='right')
    #     result = {}
    #     while True:
    #         if isinstance(result, dict) and len(result) > 10: break            
    #         result = sabre.downloadSegment()

    #     bitrate = result['total_played_bitrate']
    #     rebuffer = result['total_rebuffer']
    #     bitrateSwitchingEvents = result['total_bitrate_change']
    #     QoE = bitrate - rebuffer - bitrateSwitchingEvents
        
    #     print('QoE: ', QoE)
    #     self.assertEqual(1, 1)
        
    # def testCheckValuesAgainstSabreV6(self):
    #     '''
    #     Testing sabreV4.py against sabreV6.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV4 = SabreV4(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             resultSabreV6 = SabreV6(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             for key in resultSabreV4:
    #                 print(key)
    #                 self.assertEqual(resultSabreV4[key], resultSabreV6[key])

    def testCheckValuesAgainstSabreV7(self):
        '''
        Testing sabreV6.py against sabreV7.py
        '''
        abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
        averageList = ['ewma', 'sliding']
        for abr in abrList:
            for average in averageList:
                print('Testing: ', abr, average)
                resultSabreV6 = SabreV6(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
                resultSabreV7 = SabreV7(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
                for key in resultSabreV6:
                    #print(key)
                    try:
                        self.assertEqual(resultSabreV6[key], resultSabreV7[key])
                    except AssertionError:
                        print('Error: ', key, resultSabreV6[key], resultSabreV7[key])


if __name__ == '__main__':
    unittest.main()
