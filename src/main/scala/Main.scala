
// Goal

// The species of an Iris plant can be determined from the length and width of its sepal and petal. Different Iris
// species have petals and sepals of different lengths and widths. Our goal is to train a multi-nominal classifier
// that can predict the species of an Iris plant given its sepal length, sepal width, petal length, and petal width.
// We will use sepal length, sepal width, petal length, and petal width of an Iris plant as features. The
// species to which a plant belongs is the label or the class of a plant.

import org.apache.spark._
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.linalg.{ Vector, Vectors }
//import org.apache.spark.mllib.classification.NaiveBayes
//import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

object Main extends App {

  override def main(arg: Array[String]): Unit = {

    var sparkConf = new SparkConf().setMaster("local").setAppName("IrisFlowerRecognisationSystem")
    var sc = new SparkContext(sparkConf)
    val spark = SparkSession.builder().appName("test").master("local[*]").getOrCreate()

    sc.setLogLevel("ERROR")

    val lines = sc.textFile("file:///home/suket/case_studies/IrisPlantClassification/src/main/resourses/iris.data")
    lines.persist()

    // To remove empty lines from the dataset, let’s filter it.
    val nonEmpty = lines.filter(_.nonEmpty)

    // Next, we extract the features and labels. The data is in CSV format, so let’s split each line.
    val parsed = nonEmpty map { _.split(",") }

    // The MLlib algorithms operate on RDD of LabeledPoint, so we need to transform parsed to an RDD
    // of LabeledPoint. You may recollect that both the features and label fields in a LabeledPoint are of type
    // Double. However, the input dataset has both the features and label in string format. Fortunately, the features
    // are numerical values stored as strings, so converting the features to Double values is straightforward.
    // However, the labels are stored as alphabetical strings, which need to be converted to numerical labels. To
    // convert the name of a species to a Double typed value, we will map a species name to a number using a Map
    // data structure. We find unique values in the species column in the dataset and assign a 0-based index to
    // each species.

    val distinctSpecies = parsed.map { a => a(4) }.distinct.collect
    val textToNumeric = distinctSpecies.zipWithIndex.toMap

    // Now we are ready to create an RDD of LabeledPoint from parsed.

    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.mllib.linalg.{ Vector, Vectors }

    val labeledPoints = parsed.map { a =>
      LabeledPoint(
        textToNumeric(a(4)),
        Vectors.dense(a(0).toDouble, a(1).toDouble, a(2).toDouble, a(3).toDouble))
    }
    // let’s split the dataset into training and test data. We will use 80% of the data for training a model and 20% for testing it

    val dataSplits = labeledPoints.randomSplit(Array(0.8, 0.2))
    val trainingData = dataSplits(0)
    val testData = dataSplits(1)

    // Now we are ready to train a model. You can use any classification algorithm at this step. We will use the
    // NaiveBayes algorithm.

    import org.apache.spark.mllib.classification.NaiveBayes
    val model = NaiveBayes.train(trainingData)

    // The model trained here can be used to classify an Iris plant. Given the features of an Iris plant, it can
    // predict or tell its species.

    // let’s evaluate our model on the test dataset. The first step in evaluating a model is to have it predict
    // a label for each observation in the test dataset.

    val predictionsAndLabels = testData.map { d => (model.predict(d.features), d.label) }

    // The preceding code creates an RDD of actual and predicted labels. With this information, we can
    // calculate various model evaluation metrics. For example, we can calculate the accuracy of the model by
    // dividing the number of correct predictions by the number of observations in the test dataset. Alternatively,
    // we can use the MulticlassMetrics class to find the precision, recall, and F-measure of our model.

    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    val metrics = new MulticlassMetrics(predictionsAndLabels)

    val recall = metrics.recall
    // recall: Double = 0.9354838709677419

    val precision = metrics.precision
    // precision: Double = 0.9354838709677419

    val fMeasure = metrics.fMeasure
    // fMeasure: Double = 0.9354838709677419

    //        import spark.implicits._

    sc.stop()

  }

}