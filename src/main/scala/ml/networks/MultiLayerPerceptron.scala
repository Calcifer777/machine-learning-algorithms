package ml.networks

import breeze.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.numerics._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import com.typesafe.scalalogging.LazyLogging
import scala.annotation
import scala.math.exp

/** 
 * TODO: eta is a training hyperparameter, should not be tied to the network
 * TODO: add softmax activation function for output layer
 */
case class MultiLayerPerceptron(
    weights: Seq[BDM[Double]],
    eta: Double = 1.0, // learning rate
    beta: Double = 1.0 // sigmoid coefficient
) extends Network[MultiLayerPerceptron] {

  require(weights.size == 2)

  def makeBias(size: Int): BDM[Double] = DenseMatrix.fill(size, 1)(-1)

  def addBias(m: BDM[Double]): BDM[Double] =
    DenseMatrix.horzcat(m, makeBias(m.rows))

  /** * TODO: generalize into case class */
  def activationFunction(x: Double): Double = 1.0 / (1.0 + exp(-beta * x))

  def predict(inputs: BDM[Double]): BDM[Double] = {
    val hidden_outputs = addBias(inputs) * weights(0)
    val hidden         = addBias(hidden_outputs.map(activationFunction))
    val outputs        = (hidden * weights(1)).map(activationFunction)
    outputs
  }

  def predictWithTrace(inputs: BDM[Double]): Seq[BDM[Double]] = {
    val hidden_outputs = addBias(inputs) * weights(0)
    val hidden         = addBias(hidden_outputs.map(activationFunction))
    val outputs        = (hidden * weights(1)).map(activationFunction)
    Seq(hidden, outputs)
  }

  def trainIteration(
      inputs: BDM[Double],
      targets: BDM[Double]
  ): MultiLayerPerceptron = {

    val activations = predictWithTrace(inputs)
    val hidden      = activations(0)
    val outputs     = activations(1)

    val delta_out     = (outputs - targets) *:* (beta * outputs *:* (1.0 - outputs))
    val delta_hid     = hidden *:* (beta * (1.0 - hidden)) *:* (delta_out * weights(1).t)
    val delta_hid_adj = delta_hid(::, 0 to weights(0).cols - 1)

    val w_output = weights(1) - eta * (hidden.t * delta_out)
    val w_hidden = weights(0) - eta * (addBias(inputs).t * delta_hid_adj)

    MultiLayerPerceptron(Seq(w_hidden, w_output), eta, beta)

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
      makeWeights(weigthsDims(0)._1 + 1, weigthsDims(0)._2) / pow(dimensions(0), 0.5),
      makeWeights(weigthsDims(1)._1 + 1, weigthsDims(1)._2) / pow(dimensions(1), 0.5)
    )
    MultiLayerPerceptron(
      weights,
      eta,
      beta
    )
  }
}
