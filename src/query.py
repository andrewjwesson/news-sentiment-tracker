from pyspark import SparkConf, SparkContext
import sys
import json
from lark import Lark, Transformer

grammar = """
    ?start: query
    ?query: pred
          | query "&" pred -> qand
          | query "|" pred -> qor
          | query "-" pred -> minus
          
    ?pred: term
         | WORD ">" NUMBER -> gt
         | "(" query ")"
         
    term: WORD
         
    %import common.WORD
    %import common.NUMBER
    %import common.WS
    %ignore WS"""

class Query2Pred(Transformer):
        
    def term(self, args):
        word = args[0]
        return (lambda d: word in d)
    
    def gt(self, args):
        term, count = args
        return (lambda d: word in d and d[word] > count)
    
    def qand(self, args):
        p1, p2 = args
        return (lambda d: p1(d) and p2(d))

    def qor(self, args):
        p1, p2 = args
        return (lambda d: p1(d) or p2(d))

    def minus(self, args):
        p1, p2 = args
        return (lambda d: p1(d) and not p2(d))

def tree2terms(t):
    return set([d.children[0] for d in t.find_data("term")])

def pivot(p):
    word, counts = p
    return [(doc, {word: count}) for doc, count in counts.items()]



class BooleanSearch():
    def __init__(self, path, sc=None):
        self.lark = Lark(grammar)
        self.q2p = Query2Pred()
        if not sc:
            conf = SparkConf().setAppName('boolean search')
            self.sc = SparkContext(conf=conf)
        assert self.sc.version >= '2.3'  # make sure we have Spark 2.3+
        self.rdd = self.sc.sequenceFile(path)

    def runQueries(self, queries):
        trees = [self.lark.parse(query) for query in queries]
        functions = [self.q2p.transform(tree) for tree in trees]
        terms = [term for tree in trees for term in tree2terms(tree)]
        rdd = self.rdd.filter(lambda p: p[0] in terms)

        rdd = rdd.flatMap(pivot) \
          .reduceByKey(lambda d1, d2: {**d1, **d2})

        def multiQueries(p):
            i, d = p
            return [(q, i) for q, f in zip(queries, functions) if f(d)]
        rdd = rdd.flatMap(multiQueries) \
                .groupByKey() \
                .map(lambda p: (p[0], list(p[1]))) 
        return rdd.collectAsMap()

def main(path, output, queries):
    bs = BooleanSearch(path)
    d = bs.runQueries(queries)
    with open(output, "w") as f:
        json.dump(d, f)

if __name__ == '__main__':
    path = sys.argv[1]
    output = sys.argv[2]
    queries = sys.argv[3:]
    main(path, output, queries)
