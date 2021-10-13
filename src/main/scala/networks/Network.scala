package networks

import com.typesafe.scalalogging.LazyLogging
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}

trait Network extends LazyLogging {

  def eta: Double
  def weights: Seq[BDM[Double]]

  def activationFunction(x: Double): Double

  def activate(input: BDM[Double]): BDM[Double]

  def trainIteration(
      input: BDM[Double],
      output: BDM[Double]
  ): Network

}

object Network extends LazyLogging {

  def train[T <: Network](
      net: T,
      inputs: BDM[Double],
      targets: BDM[Double],
      epochs: Int
  ): Network = {

    logger.debug("Starting training")
    logger.debug("Initial weights:\n" + net.weights(0).toString(10, 10))

    @annotation.tailrec
    def loop(
        net: Network,
        inputs: BDM[Double],
        targets: BDM[Double],
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
