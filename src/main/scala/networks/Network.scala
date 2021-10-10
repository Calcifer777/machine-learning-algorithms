package networks

import com.typesafe.scalalogging.LazyLogging
import breeze.linalg.DenseMatrix

trait Network extends LazyLogging {

  def eta: Double
  def weights: Seq[DenseMatrix[Double]]

  def activationFunction(x: Double): Double

  def activate(input: DenseMatrix[Double]): DenseMatrix[Double]

  def trainIteration(
      input: DenseMatrix[Double],
      output: DenseMatrix[Double]
  ): Network

}

object Network extends LazyLogging {

  def train(
      net: Network,
      inputs: DenseMatrix[Double],
      targets: DenseMatrix[Double],
      epochs: Int
  ): Network = {

    logger.debug("Starting training")
    logger.debug("Initial weights:\n" + net.weights(0).toString(10, 10))

    @annotation.tailrec
    def loop(
        net: Network,
        inputs: DenseMatrix[Double],
        targets: DenseMatrix[Double],
        loops: Int
    ): Network = {
      if ((epochs - loops) % 10000 == 0) logger.debug(s"Training epoch $loops")
      if (loops <= epochs)
        loop(net.trainIteration(inputs, targets), inputs, targets, loops + 1)
      else net
    }
    loop(net, inputs, targets, 1)

  }

}
