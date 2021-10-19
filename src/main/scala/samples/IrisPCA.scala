package samples

import breeze.linalg._
import breeze.plot._
import breeze.plot.PaintScale.GreenYelloOrangeRed

import data.IrisDataSource
import ml.la.LA._

object IrisPCA extends App {

  val irisData = IrisDataSource("iris.csv")

  val (components, y) = pca(irisData.data.xs)
  val f = Figure()
  val p = f.subplot(0)
  p += scatter(
    components(::, 0),
    components(::, 1),
    DenseVector.fill(components.rows)(0.05).apply,
    DenseVector.fill(components.rows)(GreenYelloOrangeRed(220)).apply
  )

}
