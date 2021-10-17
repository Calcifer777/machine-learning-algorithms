package samples

import scala.util.Random
import breeze.linalg._
import breeze.plot._
import breeze.numerics._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.stats.regression._
import breeze.interpolation._

object SplineInterpolator extends App {

  val normal = breeze.stats.distributions.Gaussian(0.0, 1.0)

  val f = Figure()
  val x = linspace(-3.0, 10.0, 500)

  // Gaussian process
  val v1 = exp(-pow(x, 2) /:/ 9.0)
  val v2 = exp(-pow((x - 0.5), 2) /:/ 4.0)
  val y = 2.5 * v1 + 3.2 * v2 + DenseVector.rand(x.length, normal)

  // Input matrix
  val data = DenseVector.horzcat(v1, v2)

  // OLS
  val lsResults = leastSquares(data, y)
  val predictions = data * lsResults.coefficients

  val p1 = f.subplot(0)
  p1.title = "OLS"
  p1 += plot(x, y, '.')
  p1 += plot(x, predictions)

  // Plot cubic spline
  val knotsNum = 10
  val knots = (0 to x.length - 1 by x.length / knotsNum) appended (x.length - 1)
  val ciInter = CubicInterpolator(x(knots), y(knots))

  val p2 = f.subplot(2, 1, 1)
  p2.title = "Cubic Spline"
  p2 += plot(x, y, '.')
  p2 += plot(x, ciInter(x))

}
