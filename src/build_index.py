import pandas as pd
from pyspark import SparkConf, SparkContext
from operator import add
import nltk
import sys
assert sys.version_info >= (3, 5)

tokenizer = nltk.tokenize.RegexpTokenizer('\w+')
def text2counts(p, stopwords=set(nltk.corpus.stopwords.words('english'))):
  idx, t = p
  return [((w, idx), 1) for w in tokenizer.tokenize(t) if w not in stopwords]

def main(inPath, outPath):
    df = pd.read_csv(inPath, dtype="str").rename(columns={"Unnamed: 0" : "id"})
    rows = [(i, str(t)+"\n"+str(c)) for i, t, c in zip(df['id'].values, df['title'].values, df['content'].values)]
    rdd = sc.parallelize(rows, 1024)


    rdd = rdd.flatMap(text2counts).reduceByKey(add)
    def keyByWord(p):
      (w, idx), c = p
      return (w, (idx, c))
    def mergeValue(d, v):
      idx, c = v
      if idx not in d:
        d[idx] = 0
      d[idx] += c
      return d
    def mergeDict(d1, d2):
      for idx, c in d2.items():
        if idx not in d1:
          d1[idx] = 0
        d1[idx] += c
      return d1

    rdd = rdd.map(keyByWord).aggregateByKey({}, mergeValue, mergeDict, numPartitions=1024)
    rdd.saveAsSequenceFile(outPath)

if __name__ == '__main__':
    conf = SparkConf().setAppName('news index builder')
    sc = SparkContext(conf=conf)
    assert sc.version >= '2.3'  # make sure we have Spark 2.3+
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
