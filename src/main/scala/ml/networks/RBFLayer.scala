package ml.networks

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._
import scala.util.Random
import scala.math

trait RBFLayer {
  val nHidden: Int
  var weights: Option[BDM[Double]]
  def predict(inputs: BDM[Double], resetLayer: Boolean = false): BDM[Double]
}

case class SamplingLayer(
    sigma: Double,
    nHidden: Int,
    seed: Int = 42,
    hidden: Option[BDM[Double]] = None
) extends RBFLayer {

  var weights: Option[BDM[Double]] = None

  /** 
   * Weights are a sample of the inputs
   */
  def getWeights(inputs: BDM[Double]): BDM[Double] = {
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
  def predict(
      inputs: BDM[Double],
      resetWeights: Boolean = false
  ): BDM[Double] = {
    if (resetWeights)
      weights = Some(getWeights(inputs))
    val w = weights.get
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

object SamplingLayer {}
