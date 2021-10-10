package networks

import org.scalatest._
import flatspec._
import matchers._
import Perceptron._
import MultiLayerPerceptron._

import breeze.linalg._
import breeze.numerics.abs

class MultiLayerPerceptronSpec extends AnyFlatSpec with should.Matchers {

  "An MLP" should "be created with a smart constructor" in {
    val nn = mlp(Seq(5, 4, 3), 0.4, 0.3)
    assert(nn.eta == 0.4)
    assert(nn.beta == 0.3)
    assert(nn.weights(0).rows == 6)
    assert(nn.weights(0).cols == 5)
    assert(nn.weights(1).rows == 5)
    assert(nn.weights(1).cols == 3)
  }

  it should "activate via sigmoid function" in {
    val nn = mlp(Seq(5, 4, 3), 0.4, 0.3)
    val result = nn.activationFunction(3.0)
    assert(abs(result - 0.7109) < 0.001)
  }

  it should "yield the a correctly sized Matrix when activated" in {
    val nn = mlp(Seq(5, 4, 3), 0.4, 0.3)
    val sampleSize = 10
    val inputs = DenseMatrix.fill[Double](sampleSize, 5)(42)
    val result = nn.activate(inputs)
    assert(result.rows == 10)
    assert(result.cols == 3)
  }

  it should "update its weights after each training iteration" ignore {
    val sampleSize = 10
    val inputSize = 3
    val outputSize = 2
    val inputs = DenseMatrix.fill[Double](sampleSize, inputSize)(42)
    val outputs = DenseMatrix.fill[Double](sampleSize, outputSize)(42)
    val p = perceptron(inputSize, outputSize, 0.4)
    val result = p.trainIteration(inputs, outputs)
    assert(result.weights != p.weights)
    assert(result.weights(0).cols == p.weights(0).cols)
    assert(result.weights(0).rows == p.weights(0).rows)
  }

}
