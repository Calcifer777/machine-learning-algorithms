package ml.networks

import breeze.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import com.typesafe.scalalogging.LazyLogging
import scala.annotation

case class Perceptron(
    inputSize: Int,
    outputSize: Int,
    eta: Double,
    weights: Seq[BDM[Double]]
) extends Network[Perceptron] {

  def bias(size: Int): BDM[Double] = DenseMatrix.fill(size, 1)(-1)

  def activationFunction(x: Double): Double = if (x > 0) 1.0 else 0.0

  def predict(inputs: BDM[Double]): BDM[Double] =
    (DenseMatrix.horzcat(inputs, bias(inputs.rows)) * weights(0))
      .map(activationFunction)

  /** I: sample size
    * M: number of inputs
    * N: number of outputs
    * Inputs: I x M
    * Weights: M x N
    * Targets: I x N
    * Errors: I x N
    */
  def trainIteration(
      inputs: BDM[Double],
      targets: BDM[Double]
  ): Perceptron = {
    assert(inputs.rows == targets.rows)
    assert(inputs.cols + 1 == weights(0).rows)
    val inputsWithBias = DenseMatrix.horzcat(inputs, bias(inputs.rows))
    val errors = predict(inputs) - targets
    val newWeights = Seq(this.weights(0) - eta * (inputsWithBias.t * errors))
    Perceptron(this.inputSize, this.outputSize, eta, newWeights)
  }

}

object Perceptron extends LazyLogging {

  val uniform01 = breeze.stats.distributions.Uniform(-0.1, 0.1)

  /** Create a perceptron, add the a bias entry in the weights matrix */
  def perceptron(inputSize: Int, outputSize: Int, eta: Double): Perceptron = {
    Perceptron(
      inputSize,
      outputSize,
      eta,
      Seq(DenseMatrix.rand(inputSize + 1, outputSize, uniform01))
    )
  }

}
