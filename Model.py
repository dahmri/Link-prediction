import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import SVMWithSGD, SVMModel, LogisticRegressionWithSGD
from pyspark.mllib.util import MLUtils
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
import pandas as pd



def parsePoint(line):
    values = line.split(' ')
    return LabeledPoint(values[7], values[0:7])


conf = SparkConf().set('spark.driver.memory', '50G').set('spark.executor.memory', '8G').setMaster("local[4]")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


Data="./TrainTest/dtrain_Trades_Network.csv"
data1 = sc.textFile(Data)
trainData = data1.map(parsePoint)

trainData.persist()
print ("We start training")
#model = SVMWithSGD.train(trainData, iterations=1)
model = LogisticRegressionWithSGD.train(trainData, 1)
model.clearThreshold()


Test="./TrainTest/dtest_Trades_Network.csv"
data2 = sc.textFile(Test)
testData = data2.map(parsePoint)

print ("We start testing")
score = testData.map(lambda p: (float(model.predict(p.features)), p.label))

metrics = BinaryClassificationMetrics(score)
print ("areaUnderROC ",metrics.areaUnderROC)
print ("areaUnderPR  ",metrics.areaUnderPR)
