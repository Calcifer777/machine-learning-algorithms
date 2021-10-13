package rbf

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._
import scala.util.Random
import scala.math

trait RBFLayer {
  val nHidden: Int
  def train(inputs: BDM[Double]): RBFLayer
}

case class Sample(
    sigma: Double,
    nHidden: Int,
    seed: Int = 42,
    weights: Option[BDM[Double]] = None,
    hidden: Option[BDM[Double]] = None
) extends RBFLayer {

  /**
   * Inputs:                SampleSize x nInputs
   * Hidden Layer Weights:  nInputs x nHidden
   * Hidden Layer Nodes:    sameplSize x nHidden
   */
  def train(inputs: BDM[Double]): Sample = {
    require(inputs.rows >= nHidden)
    val idx = Random.shuffle(0 to inputs.rows).take(nHidden)
    val weights = inputs(idx, ::).toDenseMatrix.t
    val hiddenLayerNodes: Seq[BDV[Double]] = (0 to nHidden) map { (idx: Int) =>
      // For each row in inputs, subtract a vector of weights;
      // take the square for each element
      val squaredDiff =
        pow(inputs(*, ::) - weights(::, idx), 2) // SampleSize x nInputs
      // Sum the squared diffs
      val sumOfSquaredDiff = sum(squaredDiff(*, ::)) // (Vector) SampleSize x 1
      exp(sumOfSquaredDiff) / (2 * math.pow(sigma, 2))
    // TODO: normalize
    }
    Sample(sigma, nHidden, seed, Some(weights), Some(BDM(hiddenLayerNodes: _*)))
  }
}

case class KMeans(nHidden: Int) extends RBFLayer {
  def train(inputs: BDM[Double]): KMeans = ???
}
