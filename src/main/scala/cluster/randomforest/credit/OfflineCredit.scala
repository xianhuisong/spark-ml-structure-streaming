package cluster.randomforest.credit

import org.apache.spark.SparkConf
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator

object OfflineCredit {
  case class Credit(
    creditability: Double,
    balance: Double, duration: Double, history: Double, purpose: Double, amount: Double,
    savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double,
    residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double,
    credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double)

  def parseCredit(line: Array[Double]): Credit = {
    Credit(
      line(0),
      line(1) - 1, line(2), line(3), line(4), line(5),
      line(6) - 1, line(7) - 1, line(8), line(9) - 1, line(10) - 1,
      line(11) - 1, line(12) - 1, line(13), line(14) - 1, line(15) - 1,
      line(16) - 1, line(17) - 1, line(18) - 1, line(19) - 1, line(20) - 1)
  }

  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).map(_.map(_.toDouble))
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("CreditOffline").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext._
    import sqlContext.implicits._

    val creditDF = parseRDD(sc.textFile("data/germancredit.csv")).map(parseCredit).toDF().cache()

    val featureCols = Array("balance", "duration", "history", "purpose", "amount",
      "savings", "employment", "instPercent", "sexMarried", "guarantors",
      "residenceDuration", "assets", "age", "concCredit", "apartment",
      "credits", "occupation", "dependents", "hasPhone", "foreign")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val labelIndexer = new StringIndexer().setInputCol("creditability").setOutputCol("label").fit(creditDF)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    val Array(trainingData, testData) = creditDF.randomSplit(Array(0.8, 0.2), 5043)

    val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(8).setNumTrees(30).setFeatureSubsetStrategy("auto").setSeed(5043)

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.maxBins, Array(25, 31))
      .addGrid(classifier.maxDepth, Array(5, 10))
      .addGrid(classifier.numTrees, Array(20, 60))
      .addGrid(classifier.impurity, Array("entropy", "gini"))
      .build()

    val pipeline = new Pipeline().setStages(Array(assembler, labelIndexer, classifier, labelConverter))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val pipelineFittedModel = cv.fit(trainingData)

    val predictions = pipelineFittedModel.transform(testData)
    val accuracy = evaluator.evaluate(predictions)
    println("accuracy  pipeline fitting" + accuracy)

    println(pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))

    println(
      pipelineFittedModel
        .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
        .stages(0)
        .extractParamMap)

    val rm2 = new RegressionMetrics(
      predictions.select("prediction", "label").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println("MSE: " + rm2.meanSquaredError)
    println("MAE: " + rm2.meanAbsoluteError)
    println("RMSE Squared: " + rm2.rootMeanSquaredError)
    println("R Squared: " + rm2.r2)
    println("Explained Variance: " + rm2.explainedVariance + "\n")

    val model = pipelineFittedModel.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]

    model.write.overwrite().save("data/credit")
    sc.stop

  }
}