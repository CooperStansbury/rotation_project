
# imports 
import os


# build spark context
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, SparkSession
conf = SparkConf()
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()


# globals
HDFS_ROOT_DIR = '/user/cstansbu/D1-M/'
TEST_FILE = 'D1-M_0_BRR_D1-M-001.adap.txt.results.tsv'


################################################### FUNCTIONS

def unionAll(*dfs):
    """A function to combine multiple dataframes"""
    return reduce(DataFrame.union, dfs)


if __name__ == '__main__':
    
    # hdfs config
    hadoop = sc._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration() 
    path = hadoop.fs.Path(HDFS_ROOT_DIR)
    
    seqs = []

    for f in fs.get(conf).listStatus(path):
        print(f"------------------------------------------------------------------------ ", f.getPath(), f.getLen())
        f_path =  f.getPath()
        tmp_df = spark.read.csv(str(f_path), sep=r'\t', header=True).select('nucleotide')
        seqs.append(tmp_df)
        
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    print("------------------------------------------------------------------------")
    df = unionAll(seqs)
    print("------------------------------------------------------------------------", df.count())
    
        
    
        
        
        
   
    # f_path = f"{HDFS_ROOT_DIR}{TEST_FILE}"
    # df = spark.read.csv(f_path, sep=r'\t', header=True).select('nucleotide')
    # print(f"------------------------------------------------------------------------ {df.count()}")
