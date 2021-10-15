package ml.networks

import breeze.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.numerics.round
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import com.typesafe.scalalogging.LazyLogging
import scala.annotation
import scala.math.exp
import spire.math.Algebraic.Expr.Mul

/** 
 * TODO: eta is a training hyperparameter, should not be tied to the network
 */
case class MultiLayerPerceptron(
    weights: Seq[BDM[Double]],
    eta: Double,
    beta: Double
) extends Network {

  require(weights.size == 2)

  def makeBias(size: Int): BDM[Double] = DenseMatrix.fill(size, 1)(-1)

  def addBias(m: BDM[Double]): BDM[Double] =
    DenseMatrix.horzcat(m, makeBias(m.rows))

  /** * TODO: generalize into case class */
  def activationFunction(x: Double): Double = 1.0 / (1.0 + exp(-beta * x))

  def act2(x: Double, actType: String = "sigmoid"): Double = {
    if (actType == "sigmoid") 1 / (1 + exp(-beta * x))
    else if (actType == "relu") if (x < 0) 0 else x
    else
      throw new RuntimeException(
        s"activation function type $actType not supported "
      )
  }

  // assume sigmoid activation for outputs
  def predict(input: BDM[Double]): BDM[Double] = {
    val h = (addBias(input) * weights(0)).map(activationFunction)
    (addBias(h) * weights(1)).map(activationFunction)
  }

  def predictWithTrace(input: BDM[Double]): Seq[BDM[Double]] = {
    val h = addBias((addBias(input) * weights(0)).map(activationFunction))
    val o = (h * weights(1)).map(activationFunction)
    Seq(h, o)
  }

  def trainIteration(
      inputs: BDM[Double],
      targets: BDM[Double]
  ): MultiLayerPerceptron = {

    val activations = predictWithTrace(inputs)
    val h = activations(0)
    val o = activations(1)

    val d_output = beta * (o - targets) *:* o *:* (1.0 - o)
    val d_hidden = beta * h *:* (1.0 - h) *:* (d_output * weights(1).t)

    val w_output = weights(1) - eta * (h.t * d_output)
    val w_hidden =
      weights(0) - eta * (addBias(inputs).t * d_hidden( // magic here
        ::,
        0 to weights(0).cols - 1
      ))

    MultiLayerPerceptron(
      Seq(w_hidden, w_output),
      eta,
      beta
    )
  }

}

object MultiLayerPerceptron {

  val uniform01 = breeze.stats.distributions.Uniform(-0.1, 0.1)

  def makeWeights(rows: Int, cols: Int): BDM[Double] =
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
    val weigthsDims = dimensions zip dimensions.tail
    val weights = Seq(
      makeWeights(weigthsDims(0)._1 + 1, weigthsDims(0)._2),
      makeWeights(weigthsDims(1)._1 + 1, weigthsDims(1)._2)
    )
    MultiLayerPerceptron(
      weights,
      eta,
      beta
    )
  }
}
