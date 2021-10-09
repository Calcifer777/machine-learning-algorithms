package nn

import org.scalatest._
import flatspec._
import matchers._
import Perceptron._

import breeze.linalg._

class PerceptronSpec extends AnyFlatSpec with should.Matchers {

  "A Perceptron" should "return a result of the correct size when fed an input vector" in {
    val inputs = 3
    val outputs = 2
    val sampleSize = 10
    val p = perceptron(inputs, outputs, 0.4)    
    val result = p.activate(DenseMatrix.fill(sampleSize, inputs)(1.0))
    assert(result.rows == sampleSize)
    assert(result.cols == outputs)
  }

  it should "update its weights after each training iteration" in {
    val sampleSize = 10
    val inputSize = 3
    val outputSize = 2
    val inputs = DenseMatrix.fill[Double](sampleSize, inputSize)(42)
    val outputs = DenseMatrix.fill[Double](sampleSize, outputSize)(42)
    val p = perceptron(inputSize, outputSize, 0.4)
    val result = p.trainIteration(inputs, outputs)
    assert(result.weights != p.weights)
    assert(result.weights.cols == p.weights.cols)
    assert(result.weights.rows == p.weights.rows)
  }

  "A two-inputs perceptron" should "converge to the logic AND function" in {
    val p = perceptron(2, 1, 0.4)
    val inputs = DenseMatrix(
      Array(1.0, 1.0),
      Array(1.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0)
    )
    val outputs = DenseMatrix(
      Array(1.0),
      Array(0.0),
      Array(0.0),
      Array(0.0)
    )
    val result = train(p, inputs, outputs, 20)
    assert(result.activate(inputs) == outputs)
  }

  "it" should "converge to the logic OR function" in {
    val p = perceptron(2, 1, 0.4)
    val inputs = DenseMatrix(
      Array(1.0, 1.0),
      Array(1.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0)
    )
    val outputs = DenseMatrix(
      Array(1.0),
      Array(1.0),
      Array(1.0),
      Array(0.0)
    )
    val result = train(p, inputs, outputs, 20)
    assert(result.activate(inputs) == outputs)
  }

  "it" should "not converge to the logic XOR function" in {
    val p = perceptron(2, 1, 0.4)
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
    val result = train(p, inputs, outputs, 20)
    assert(result.activate(inputs) != outputs)
  }

  "A 3-input Perceptron" should "converge to the logic XOR function" in {
    val p = perceptron(3, 1, 0.4)
    val inputs = DenseMatrix(
      Array(0.0, 0.0, 1.0),
      Array(0.0, 1.0, 0.0),
      Array(1.0, 0.0, 0.0),
      Array(1.0, 1.0, 0.0)
    )
    val outputs = DenseMatrix(
      Array(0.0),
      Array(1.0),
      Array(1.0),
      Array(0.0)
    )
    val result = train(p, inputs, outputs, 20)
    assert(result.activate(inputs) == outputs)
  }

}