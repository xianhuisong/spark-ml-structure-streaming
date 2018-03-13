package cluster.randomforest.credit

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.rdd.RDD

case class Credit(
  balance: Double, duration: Double, history: Double, purpose: Double, amount: Double,
  savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double,
  residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double,
  credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double)

object OnlineCredit {

  def parseCredit(line: Array[Double]): Credit = {
    Credit(
      line(0) - 1, line(1), line(2), line(3), line(4),
      line(5) - 1, line(6) - 1, line(7), line(8) - 1, line(9) - 1,
      line(10) - 1, line(11) - 1, line(12), line(13) - 1, line(14) - 1,
      line(15) - 1, line(16) - 1, line(17) - 1, line(18) - 1, line(19) - 1)
  }

  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).map(_.map(_.toDouble))
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("CreditOnline")
      .master("local[2]")
      .getOrCreate()

    import spark.implicits._

    val model = PipelineModel.load("data/credit")

    val df = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "credit")
      .load()
      .selectExpr("CAST(value as STRING)")
      .as[String]
      .transform(ds => ds.map(_.split(",")).map(_.map(_.toDouble)).map(parseCredit))
      .transform(ds => model.transform(ds).select("features", "prediction", "predictedLabel"))
      .writeStream
      .format("console")
      .start()
      .awaitTermination()
  }
}