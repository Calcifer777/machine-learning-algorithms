package ml.networks

import com.typesafe.scalalogging.LazyLogging
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._

import ml.Model

// Type signature: https://www.youtube.com/watch?v=Wki2B6iW1oA
trait Network[T <: Network[T]] extends Model with LazyLogging {

  self: T =>

  def eta: Double
  def weights: Seq[BDM[Double]]

  def activationFunction(x: Double): Double

  def predict(input: BDM[Double]): BDM[Double]

  def trainIteration(
      input: BDM[Double],
      output: BDM[Double]
  ): T

}

object Network extends LazyLogging {

  def train[T <: Network[T]](
      net: T,
      inputs: BDM[Double],
      targets: BDM[Double],
      epochs: Int
  ): T = {

    logger.debug("\nStarting training")

    @annotation.tailrec
    def loop(
        net: T,
        inputs: DenseMatrix[Double],
        targets: DenseMatrix[Double],
        loops: Int
    ): T = {
      if (loops % (epochs / 10).toInt == 0)
        val outputs = net.predict(inputs)
        val precision = net.precision(outputs, targets)
        logger.debug(
          "\n" + f"Training epoch $loops; Precision ${precision}%.0f%%"
        )
      // println(s"ITERATION: $loops")
      if (loops <= epochs)
        loop(net.trainIteration(inputs, targets), inputs, targets, loops + 1)
      else net
    }
    loop(net, inputs, targets, 1)

  }

}
