package ml.networks

import org.scalatest._
import flatspec._
import matchers._
import Perceptron._
import Network.train

import breeze.linalg._
import breeze.numerics._

class ActivationSpec extends AnyFlatSpec with should.Matchers {

  "SoftMax" should "compute the correct return value" in {
    val inputs = DenseMatrix(
      Array(1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0),
      Array(2.0, 1.0, 3.0, 4.0, 1.0, 2.0, 3.0)
    )
    val expected = DenseMatrix(
      Array(
        0.02364, 0.06426, 0.17468, 0.47483, 0.02364, 0.06426, 0.17468
      ),
      Array(
        0.06426, 0.02364, 0.17468, 0.47483, 0.02364, 0.06426, 0.17468
      )
    )
    val diffs = abs(SoftMax(inputs) - expected).toDenseVector.toScalaVector
    assert(diffs.filter(_ > 1e-4).size == 0)
  }

}
