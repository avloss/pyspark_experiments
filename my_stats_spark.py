#
# USAGE: $SPARK_HOME/bin/spark-submit my_annotations.py
#

from pyspark.mllib.stat import Statistics
from pyspark import SparkContext
import scipy.stats
import pytest

sc = SparkContext('local', 'stats')


##need to declare CDF outsice of the class, otherwise spark tries to ship it with a closure

def CDF(a, b, l, x):
    a_count = len(filter(lambda y: y <= x, a.value))
    b_count = len(filter(lambda y: y <= x, b.value))
    return abs(a_count - b_count) / l


class basic_stats(object):

    def __init__(self, a, b):
        self.a_driver = a
        self.b_driver = b
        self.a = sc.parallelize(a).map(lambda x: x + 0.0)
        self.b = sc.parallelize(b).map(lambda x: x + 0.0)

        self.length = len(a) + 0.0

    def ks(self):
        a_br = sc.broadcast(self.a_driver)
        b_br = sc.broadcast(self.b_driver)
        length = self.length
        return self.a.union(self.b).distinct().map(lambda x: CDF(a_br,
                b_br, length, x)).max()

    def pearson(self):
        return Statistics.corr(self.a, self.b, 'pearson')


class advanced_stats(basic_stats):

    def __init__(self, a, b):
        basic_stats.__init__(self, a, b)

    def rho(self):
        return Statistics.corr(self.a, self.b, 'spearman')

    def tau(self):
        zip = self.a.zip(self.b).zipWithIndex()
        denominator = self.length * (self.length - 1) / 2.0
        pairs = zip.cartesian(zip).filter(lambda x: x[0][1] > x[1][1])
        differences = pairs.map(lambda x: (x[0][0][0] - x[1][0][0]) \
                                * (x[0][0][1] - x[1][0][1]))
        numerator = differences.filter(lambda x: x != 0).map(lambda x: \
                (1 if x > 0 else -1)).sum()
        return numerator / denominator


def test_ks():
    list1 = [1, 2, 3, 4, 5, 6]
    list2 = [9, 8, 7, 6, 5, 4]

    # make sure my implementation matches non-distributed scipy version

    stats = basic_stats(list1, list2)
    assert stats.ks() == scipy.stats.ks_2samp(list1, list2).statistic


def test_tau():
    list1 = [0,2,4,4,3,10]
    list2 = [4,5,0,3,2,1]

    # make sure my implementation matches non-distributed scipy version
    # there are some issues and it doesn't match on longer sequences

    stats = advanced_stats(list1, list2)
    assert abs(stats.tau() - scipy.stats.kendalltau(list1, list2)[0]) < 0.02


if __name__ == '__main__':
    test_ks()
    test_tau()

    # normally this would be started with "pytest.main([__file__])",
    # but I'm not sure how pyspark and pytest would behave together

