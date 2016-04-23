#
# USAGE: $SPARK_HOME/bin/spark-submit my_annotations.py
#

from pyspark.mllib.stat import Statistics
from pyspark import SparkContext

sc = SparkContext('local', 'stats')


def compute_pearson_with_spark(func):

    def spark_pearson(a, b):
        rdd_a = sc.parallelize(a)
        rdd_b = sc.parallelize(b)
        g = func.func_globals
        g['pearson'] = Statistics.corr(rdd_a, rdd_b, 'pearson')
        g['rho'] = Statistics.corr(rdd_a, rdd_b, 'spearman')
        func(a, b)

    return spark_pearson


@compute_pearson_with_spark
def print_summary(a, b):
    print 'pearson is {}'.format(pearson)
    print 'rho is {}'.format(rho)
    return

list1 = [1, 2, 3, 11, 1]
list2 = [4, 5, 3, 8, 8]
print_summary(list1, list2)
