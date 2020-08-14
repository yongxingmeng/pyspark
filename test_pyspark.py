#!/Users/qmeng//anaconda3/bin/python

"""
Description:   pyspark 
Date       :   2020-08-13

"""

import pyspark

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col,array_contains


#sc = pyspark.SparkContext( 'local[*]' )

#spark = SparkSession.builder.appName('localhost') \
#            .getOrCreate()


import pandas as pd

df = pd.read_csv("/Users/qmeng/hebo/pyspark/train.csv" )

df = df[['feature_1', 'feature_2', 'feature_3', 'target' ]]

df.fillna( 0, inplace = True )

df.to_csv( "/Users/qmeng/hebo/pyspark/train0.csv", index = False )



import pyspark
spark = pyspark.sql.SparkSession.builder.appName("MyApp") \
            .config("spark.jars.packages", "com.microsoft.ml.spark:mmlspark_2.11:1.0.0-rc1") \
            .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven") \
            .getOrCreate()

from mmlspark.lightgbm import LightGBMRegressor



schema = StructType() \
      .add("feature_1",DoubleType(),True) \
      .add("feature_2",DoubleType(),True) \
      .add("feature_3",DoubleType(),True) \
      .add("target",DoubleType(),True)       
      #.add("card_id",StringType(),True) \


df  = spark.read.format("csv") \
      .options(header='True', delimiter=',') \
      .schema(schema) \
      .load("/Users/qmeng/hebo/pyspark/train0.csv" )
 

print("records read: " + str(df.count()))
print("Schema: ")
df.printSchema()
print( df.limit(10).toPandas().head() )


train, test = df.randomSplit([0.85, 0.15], seed=1)


# light gbm

model = LightGBMRegressor(objective='quantile',
                          alpha=0.2,
                          learningRate=0.3,
                          numLeaves=31)


print ( 'fit model' )
model.fit(train)


print(model.getFeatureImportances())



scoredData = model.transform(test)
scoredData.limit(10).toPandas()

from mmlspark.train import ComputeModelStatistics
metrics = ComputeModelStatistics(evaluationMetric='regression',
                                 labelCol='label',
                                 scoresCol='prediction') \
            .transform(scoredData)
metrics.toPandas()


def test():


    pass




 
 

if __name__ == "__main__":
    test()
