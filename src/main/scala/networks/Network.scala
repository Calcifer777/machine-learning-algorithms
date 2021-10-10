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
