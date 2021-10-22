package data

import scala.io.Source
import breeze.linalg.{DenseMatrix, DenseVector}
import com.github.tototoshi.csv._

final case class PimaDataSource(path: String) extends DataSource {

  val data = loadData(path)

  def loadData(fileName: String): Dataset = {
    // Read dataset
    val raw: List[List[String]] =
      CSVReader.open(Source.fromResource(fileName)).all()
    // Get labels
    val labels  = raw.head
    val xLabels = labels.reverse.tail.reverse.toSeq
    val yLabels = Seq(labels.last)
    // Format data
    val vectors = raw.tail
      .map { _.map(_.toDouble) }
      .map { (r: List[Double]) =>
        DenseVector(r.toArray: _*)
      }
    val data = DenseMatrix(vectors: _*)
    val xs   = data(::, 0 to -1).toDenseMatrix
    val ys   = data(::, -1).toDenseMatrix.t
    // Check data size
    assert(xs.rows == 768 && xs.cols == 9)
    assert(ys.rows == 768 && ys.cols == 1)
    // Return
    Dataset(xs, ys, xLabels, yLabels)
  }

}
