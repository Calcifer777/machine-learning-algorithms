package networks

import scala.annotation
import com.typesafe.scalalogging.LazyLogging
import breeze.linalg._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import spire.math.Algebraic.Expr.Mul
import scala.math.exp

case class MultiLayerPerceptron(
    weights: Seq[MD],
    eta: Double,
    beta: Double
) extends Network {

  def makeBias(size: Int): MD = DenseMatrix.fill(size, 1)(-1)

  def addBias(m: MD): MD =
    DenseMatrix.horzcat(makeBias(m.rows), m)

  /** 
   * TODO: generalize into case class
   */
  def activationFunction(x: Double): Double = 1 / (1 + exp(-beta * x))

  def activate(input: MD): MD =
    weights
      .foldLeft(addBias(input)) { (xs: MD, ws: MD) =>
        (xs * ws).map(activationFunction)
      }

  def trainIteration(
      input: MD,
      output: MD
  ): MultiLayerPerceptron = ???

}

object MultiLayerPerceptron {

  val uniform01 = breeze.stats.distributions.Uniform(-0.1, 0.1)

  def makeWeights(rows: Int, cols: Int): MD =
    DenseMatrix.rand(rows, cols, uniform01)

  /**
   * TODO: move to tensors
   * TODO: use bias only if specified
   */
  def mlp(
      dimensions: Seq[Int],
      eta: Double,
      beta: Double
  ): MultiLayerPerceptron = {
    val weightsAdj = dimensions.zipWithIndex.map { case (x, idx) =>
      if (idx == dimensions.size - 1) x
      else x + 1
    }
    val weigthsDims = weightsAdj zip weightsAdj.tail
    MultiLayerPerceptron(
      weigthsDims.map { case (r, c) => makeWeights(r, c) },
      eta,
      beta
    )
  }
}
