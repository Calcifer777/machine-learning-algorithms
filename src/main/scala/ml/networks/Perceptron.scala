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
) extends Network {

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

  def train(
      p: Perceptron,
      inputs: BDM[Double],
      targets: BDM[Double],
      epochs: Int
  ): Perceptron = {

    logger.debug("Starting training")
    // logger.debug("Initial weights:\n" + p.weights(0).toString(10, 10))

    @annotation.tailrec
    def loop(
        p: Perceptron,
        inputs: BDM[Double],
        targets: BDM[Double],
        loops: Int
    ): Perceptron = {
      if (loops % (epochs / 10).toInt == 0)
        val outputs = p.predict(inputs)
        val precision = p.precision(outputs, targets)
        logger.debug(s"Training epoch $loops; Precision $precision\n")
      if (loops <= epochs)
        loop(p.trainIteration(inputs, targets), inputs, targets, loops + 1)
      else p
    }
    loop(p, inputs, targets, 1)

  }

}
