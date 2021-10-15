package data

import scala.io.Source
import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv._

final case class IrisDataSource(path: String) extends DataSource {

  val data = loadData(path)

  def loadData(fileName: String): Dataset = {
    // Read dataset
    val data = CSVReader.open(Source.fromResource(fileName)).all()
    // Format xs
    val xsData = data map { (r: List[String]) =>
      DenseVector(r: _*).slice(0, 4)
    }
    val xs = DenseMatrix(xsData: _*).mapValues(_.toDouble)
    // Format ys
    val ysData = data
      .map { (r: List[String]) => DenseVector(r: _*) }
      .map { v => v(-1) }
      .map {
        case "Iris-setosa"     => Array(1.0, 0.0, 0.0)
        case "Iris-versicolor" => Array(0.0, 1.0, 0.0)
        case "Iris-virginica"  => Array(0.0, 0.0, 1.0)
      }
    val ys = DenseMatrix(ysData: _*)
    // Check data size
    assert(xs.rows == 150 && xs.cols == 4)
    assert(ys.rows == 150 && ys.cols == 3)
    // Labels
    val xLabels = Seq(
      "sepal length in cm",
      "sepal width in cm",
      "petal length in cm",
      "petal width in cm"
    )
    val yLabels = Seq(
      "iris class"
    )
    // Return
    Dataset(xs, ys, xLabels, yLabels)
  }

}
