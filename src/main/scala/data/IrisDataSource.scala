package data

import scala.io.Source
import com.github.tototoshi.csv._
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.linalg._
import breeze.numerics._
import breeze.stats.mean

final case class IrisDataSource(path: String) extends DataSource {

  val data = loadData(path)

  def loadData(fileName: String): Dataset = {
    // Read dataset
    // val data = CSVReader.open(Source.fromResource(fileName)).all()
    val data = CSVReader
      .open(
        "/home/calcifer/git/marco/machine-learning-algorithms/src/main/resources/iris.csv"
      )
      .all()
    // Format xs
    val xsData = data map { (r: List[String]) =>
      DenseVector(r: _*).slice(0, 4)
    }
    val xs = DenseMatrix(xsData: _*).mapValues(_.toDouble)
    // Normalize xs
    val means = mean(xs(::, *))
    val xsCentered = xs(*, ::) - mean(xs(::, *)).t
    // println(xsCentered(0 to 2, ::))
    val maxByCol = max(xsCentered(::, *)).t
    val minByCol = abs(min(xsCentered(::, *))).t
    val maxAndMinByCol = DenseVector.horzcat(maxByCol, minByCol).toDenseMatrix.t
    // println(maxAndMinByCol)
    val largestAbsByCol = max(maxAndMinByCol(::, *)).t.toDenseVector
    // println(largestAbsByCol)
    val xsBounded = xsCentered(*, ::) / largestAbsByCol
    // println(xsBounded(0 to 2, ::))
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
    Dataset(xsBounded, ys, xLabels, yLabels)
  }

}
