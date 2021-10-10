package networks

import scala.annotation
import com.typesafe.scalalogging.LazyLogging
import breeze.linalg._
import breeze.stats.distributions.Rand.FixedSeed.randBasis

case class Perceptron(
    inputSize: Int,
    outputSize: Int,
    eta: Double,
    weights: Seq[DenseMatrix[Double]]
) extends Network {

  def bias(size: Int): DenseMatrix[Double] = DenseMatrix.fill(size, 1)(-1)

  def activationFunction(x: Double): Double = if (x > 0) 1 else 0

  def activate(inputs: DenseMatrix[Double]): DenseMatrix[Double] =
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
      inputs: DenseMatrix[Double],
      targets: DenseMatrix[Double]
  ): Perceptron = {
    val activations = activate(inputs)
    val errors = activations - targets
    val adj = DenseMatrix.horzcat(inputs, bias(inputs.rows)).t * errors
    val newWeights = Seq(this.weights(0) - eta * adj)
    logger.debug("Weights:\n" + newWeights(0).toString(10, 10))
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
      inputs: DenseMatrix[Double],
      targets: DenseMatrix[Double],
      epochs: Int
  ): Perceptron = {

    logger.debug("Starting training")
    logger.debug("Initial weights:\n" + p.weights(0).toString(10, 10))

    @annotation.tailrec
    def loop(
        p: Perceptron,
        inputs: DenseMatrix[Double],
        targets: DenseMatrix[Double],
        loops: Int
    ): Perceptron = {
      logger.debug(s"Training epoch $loops")
      if (loops <= epochs)
        loop(p.trainIteration(inputs, targets), inputs, targets, loops + 1)
      else p
    }
    loop(p, inputs, targets, 1)

  }

}
