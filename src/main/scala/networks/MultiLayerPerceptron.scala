package networks

import scala.annotation
import com.typesafe.scalalogging.LazyLogging
import breeze.linalg._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import spire.math.Algebraic.Expr.Mul
import scala.math.exp
import breeze.numerics.round

/** 
 * TODO: eta is a training hyperparameter, should not be tied to the network
 */
case class MultiLayerPerceptron(
    weights: Seq[MD],
    eta: Double,
    beta: Double
) extends Network {

  require(weights.size == 2)

  def makeBias(size: Int): MD = DenseMatrix.fill(size, 1)(-1)

  def addBias(m: MD): MD = DenseMatrix.horzcat(m, makeBias(m.rows))

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
  def activate(input: MD): MD = {
    val h = (addBias(input) * weights(0)).map(activationFunction)
    (addBias(h) * weights(1)).map(activationFunction)
  }

  def activateWithTrace(input: MD): Seq[MD] = {
    val h = addBias((addBias(input) * weights(0)).map(activationFunction))
    val o = (h * weights(1)).map(activationFunction)
    Seq(h, o)
  }

  def trainIteration(inputs: MD, targets: MD): MultiLayerPerceptron = {

    val activations = activateWithTrace(inputs)
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
