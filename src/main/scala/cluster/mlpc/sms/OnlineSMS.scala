package cluster.mlpc.sms

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.PipelineModel

case class SMS(message: Array[String])

object OnlineSMS {
  final val VECTOR_SIZE = 100
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("OnlineSMS")
      .master("local[2]")
      .getOrCreate()

    import spark.implicits._

    val model = PipelineModel.load("data/sms")

    val df = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "sms")
      .load()
      .selectExpr("CAST(value as STRING)")
      .as[String]
      .map(value => {
        val line = value.split(" ");
        SMS(line)
      })
      .transform(ds => model.transform(ds))
      .writeStream
      .format("console")
      .start()
      .awaitTermination()

  }
}