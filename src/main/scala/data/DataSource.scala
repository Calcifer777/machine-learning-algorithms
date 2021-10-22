package data

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import scala.util.Random

case class Dataset(
    xs: BDM[Double],
    ys: BDM[Double],
    xLabels: Seq[String],
    yLabels: Seq[String]
)

trait DataSource {

  val data: Dataset

  def loadData(fileName: String): Dataset

  def trainTestSplit(testRatio: Double): (Dataset, Dataset) = {
    require(data.xs.rows == data.ys.rows)
    val xs                  = data.xs
    val ys                  = data.ys
    val idx                 = Random.shuffle(0 to xs.rows - 1)
    val (testIdx, trainIdx) = idx.splitAt((xs.rows * testRatio).toInt)
    (
      Dataset(
        xs(trainIdx, ::).toDenseMatrix,
        ys(trainIdx, ::).toDenseMatrix,
        data.xLabels,
        data.yLabels
      ),
      Dataset(
        xs(testIdx, ::).toDenseMatrix,
        ys(testIdx, ::).toDenseMatrix,
        data.xLabels,
        data.yLabels
      )
    )
  }

}
