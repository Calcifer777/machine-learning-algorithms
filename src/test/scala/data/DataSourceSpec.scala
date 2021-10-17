package data

import org.scalatest._
import flatspec._
import matchers._

import breeze.linalg._

class DataSourceSpec extends AnyFlatSpec with should.Matchers {

  val d = (1 to 100).map((i: Int) => Array(i.toDouble)).toSeq

  val dataset = Dataset(
    xs = DenseMatrix(d: _*),
    ys = DenseMatrix(d: _*),
    xLabels = Seq("x"),
    yLabels = Seq("y")
  )

  "A DataSource" should "split a dataset in train and test" in {

    val ds = new DataSource {
      val data = dataset
      def loadData(fileName: String): Dataset = dataset
    }

    val (trainDs, testDs) = ds.trainTestSplit(0.3)

    assert(trainDs.xs.rows == 70 && trainDs.ys.rows == 70)
    assert(testDs.xs.rows == 30 && testDs.ys.rows == 30)

    assert(trainDs.xs == trainDs.ys)
    assert(testDs.xs == testDs.ys)

  }

}
