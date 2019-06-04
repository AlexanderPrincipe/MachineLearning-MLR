from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from decimal import *

def preprocesar_rls(sqlContext):
	rdd	= sqlContext.read.csv("kc_house_data.csv", header=True).rdd
	#rdd_columnas = rdd.map(
	#	lambda x : (Decimal(x[2]), int(x[5]) + int(x[6]) + int(x[12]) + int(x[13])) )
	rdd_columnas = rdd.map(
		lambda x : (Decimal(x[2]), int(x[5])) )
	return rdd_columnas.toDF(["price", "sqft_total"])

def preprocesar_rlm(sqlContext):
	rdd	= sqlContext.read.csv("kc_house_data.csv", header=True).rdd
	rdd_columnas = rdd.map(
		lambda x : (Decimal(x[2]), int(x[5]), int(x[6]), int(x[12]) , int(x[13])) )
	return rdd_columnas.toDF(["price", "sqft_living", "sqft_lot", "sqft_above", "sqft_basement"])

def entrenar(df_casas):
	assembler = VectorAssembler(
		inputCols=["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"], outputCol="features")
	training_df = assembler.transform(df_casas)
	lr = LinearRegression(featuresCol="features", labelCol="price")
	lr_model = lr.fit(training_df)

	print("b:{} Ws:{}".format(
		lr_model.intercept, lr_model.coefficients))
	
	return training_df, lr_model


def validar(lr_model, training_df):
	predictions_df = lr_model.transform(training_df)
	evaluador = RegressionEvaluator(
		labelCol="price", 
		predictionCol="prediction",
		metricName="rmse")
	metrica = evaluador.evaluate(predictions_df)
	print("RMSE: {}".format(metrica))

	evaluador = RegressionEvaluator(
		labelCol="price", 
		predictionCol="prediction",
		metricName="r2")
	metrica = evaluador.evaluate(predictions_df)
	print("R2: {}".format(metrica))
	

def main():
	# Configurar la conexion a spark
	conf = SparkConf().setAppName("PredPreciosCasas").setMaster("local")
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	df_casas = preprocesar_rlm(sqlContext)
	training_df, lr_model = entrenar(df_casas)
	validar(lr_model, training_df)

if __name__ == "__main__":
	main()


