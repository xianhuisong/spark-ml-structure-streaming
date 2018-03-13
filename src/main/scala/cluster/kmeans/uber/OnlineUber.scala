package cluster.kmeans.uber

import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler

case class Uber(dt: String, lat: Double, lon: Double, base: String) extends Serializable

object OnlineUber {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("OnlineUber")
      .master("local[2]")
      .getOrCreate()

    import spark.implicits._

    val featureCols = Array("lat", "lon")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val model = KMeansModel.load("data/uber")
    model.clusterCenters.foreach(println)

    val df = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "uber")
      .load()
      .selectExpr("CAST(value as STRING)")
      .as[String]
      .map(value => {
        val line = value.split(",");
        Uber(line(0), line(1).toDouble, line(2).toDouble, line(3))
      })
      .transform {
        ds =>
          {
            val df = assembler.transform(ds)
            model.transform(df)
          }
      }
      .writeStream
      .format("console")
      .start()
      .awaitTermination()

  }
}