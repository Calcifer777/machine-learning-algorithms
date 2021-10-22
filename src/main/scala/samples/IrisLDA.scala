package samples

import breeze.linalg._
import breeze.plot._
import breeze.plot.PaintScale.GreenYelloOrangeRed

import data.IrisDataSource
import ml.la.LA._

object IrisLDA extends App {

  val irisData = IrisDataSource("iris.csv")
  val (xs, ys) = (irisData.data.xs, irisData.data.ys)

  val components = lda(xs, ys)
  val f          = Figure()
  val p          = f.subplot(0)
  (components.zipWithIndex) foreach { case (c, idx) =>
    val color = GreenYelloOrangeRed(idx * 220)
    p += scatter(
      c(::, 0),
      c(::, 1),
      DenseVector.fill(c.rows)(0.05).apply,
      DenseVector.fill(c.rows)(color).apply
    )
  }

}
