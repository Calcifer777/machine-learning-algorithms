package ml.rbf

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._
import scala.util.Random
import scala.math

trait RBFLayer {
  val nHidden: Int
  def weights(inputs: BDM[Double]): BDM[Double]
  def predict(inputs: BDM[Double]): BDM[Double]
}

case class Sample(
    sigma: Double,
    nHidden: Int,
    seed: Int = 42,
    hidden: Option[BDM[Double]] = None
) extends RBFLayer {

  /** 
   * Weights are a sample of the inputs
   */
  def weights(inputs: BDM[Double]): BDM[Double] = {
    require(inputs.rows >= nHidden)
    Random.setSeed(seed)
    val idx = Random.shuffle(0 to inputs.rows - 1).take(nHidden)
    inputs(idx, ::).toDenseMatrix.t
  }

  /**
   * Inputs:                SampleSize x nInputs
   * Hidden Layer Weights:  nInputs x nHidden
   * Hidden Layer Nodes:    sameplSize x nHidden
   */
  def predict(inputs: BDM[Double]): BDM[Double] = {
    val w = weights(inputs)
    val predictionData = (0 to nHidden - 1) map { (idx: Int) =>
      // For each row in inputs, subtract a vector of weights;
      // take the square for each element
      // Dimensions: SampleSize x nInputs
      val squaredDiff = pow(inputs(*, ::) - w(::, idx), 2)
      // Sum the squared diffs by column
      // (Vector) SampleSize x 1
      val sumOfSquaredDiff = -sum(squaredDiff(*, ::))
      exp(sumOfSquaredDiff / (2 * math.pow(sigma, 2)))
    }
    val predictionBDM = BDM(predictionData: _*).t
    // Columnwise normalization: sum hiddenLayerNodes by column, divide each column by this sum
    val normalizedBDM = predictionBDM(::, *) map { c =>
      c /:/ sum(predictionBDM(*, ::))
    }
    normalizedBDM
  }
}

case class KMeans(nHidden: Int) extends RBFLayer {
  def weights(inputs: BDM[Double]): BDM[Double] = ???
  def predict(inputs: BDM[Double]): BDM[Double] = ???
}
