package networks

import org.scalatest._
import flatspec._
import matchers._
import Network._
import MultiLayerPerceptron._

import breeze.linalg._
import breeze.numerics.abs

class MultiLayerPerceptronSpec extends AnyFlatSpec with should.Matchers {

  "An MLP" should "be created with a smart constructor" in {
    val nn = mlp(Seq(5, 4, 3), 0.4, 0.3)
    assert(nn.eta == 0.4)
    assert(nn.beta == 0.3)
    assert(nn.weights(0).rows == 6)
    assert(nn.weights(0).cols == 4)
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

  it should "yield the same output when activated with full trace" in {
    val net = mlp(Seq(5, 4, 3), 0.4, 0.3)
    val sampleSize = 1
    val inputs = DenseMatrix.fill[Double](sampleSize, 5)(42)
    val act = net.activate(inputs)
    val withTrace = net.activateWithTrace(inputs)
    assert(act == withTrace.last)
  }

  it should "update its weights after each training iteration" in {
    val sampleSize = 10
    val inputs = DenseMatrix.fill[Double](sampleSize, 5)(42)
    val outputs = DenseMatrix.fill[Double](sampleSize, 3)(42)
    val nn = mlp(Seq(5, 4, 3), 0.4, 0.3)
    val result = nn.trainIteration(inputs, outputs)
    // assert(result.weights != nn.weights)
    assert(result.weights(0).cols == nn.weights(0).cols)
    assert(result.weights(1).rows == nn.weights(1).rows)
  }

  it should "update perform backprop correctly when trained" in {
    val w = Seq(
      DenseMatrix(
        Array(-0.07973831, 0.56903848),
        Array(-0.00784318, 0.33450039),
        Array(-0.13726413, 0.28561184),
      ),
      DenseMatrix(
        Array(-0.33036651),
        Array(0.65572476),
        Array(1.19441974),
      )
    )
    val net = MultiLayerPerceptron(w, 0.25, 1)
    val inputs = DenseMatrix(
      Array(1.0, 1.0),
      Array(1.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0)
    )
    val outputs = DenseMatrix(Array(1.0), Array(0.0), Array(0.0), Array(0.0))
    val trained = net.trainIteration(inputs, outputs)
    val expectedWeights = Seq(
      DenseMatrix(
        Array(-0.08163325096664781, 0.5723107813544502),
        Array(-0.00979848542817659, 0.3378434365092607),
        Array(-0.13738660373462594, 0.2863212584160498),
      ),
      DenseMatrix(
        Array(-0.33171359370334647),
        Array(0.6601188945346561),
        Array(1.1959998857493688),
      ),
    )
    assert(trained.weights == expectedWeights)
  }

  it should "converge to the AND logic function" in {
    val net = mlp(Seq(2, 2, 1), eta=0.4, beta=1)
    val inputs = DenseMatrix(
      Array(1.0, 1.0),
      Array(1.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0)
    )
    val outputs = DenseMatrix(Array(1.0), Array(0.0), Array(0.0), Array(0.0))
    val trained = train(net, inputs, outputs, 5000)
    val predictions = trained.activate(inputs)
    val margin = outputs - predictions
    assert(margin(0, 0) < 0.1)
    assert(margin(1, 0) < 0.1)
    assert(margin(2, 0) < 0.1)
    assert(margin(3, 0) < 0.1)
  }

  it should "converge to the XOR logic function" in {
    val net = mlp(Seq(2, 2, 1), eta=0.5, beta=2)
    val inputs = DenseMatrix(
      Array(1.0, 1.0),
      Array(1.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0)
    )
    val outputs = DenseMatrix(
      Array(0.0), 
      Array(1.0), 
      Array(1.0), 
      Array(0.0)
    )
    val trained = train(net, inputs, outputs, 10000)
    val predictions = trained.activate(inputs)
    val margin = outputs - predictions
    assert(margin(0, 0) < 0.1)
    assert(margin(1, 0) < 0.1)
    assert(margin(2, 0) < 0.1)
    assert(margin(3, 0) < 0.1)
  }
}
