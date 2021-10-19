package samples

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.plot._

import data.IrisDataSource
import ml.la.LA._

object IrisLDA extends App {

  val irisData = IrisDataSource("iris.csv")
  val (xs, ys) = (irisData.data.xs, irisData.data.ys)

  val components = lda(xs, ys)
  val f = Figure()
  val p = f.subplot(0)
  components foreach { c =>
    p += plot(c(::, 0), c(::, 1), '.')
  }

}
