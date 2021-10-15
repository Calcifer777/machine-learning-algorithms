package ml

import org.scalatest._
import flatspec._
import matchers._
import Model._

import breeze.linalg._

class ModelSpec extends AnyFlatSpec with should.Matchers {

  "A Model" should "calculate the precision index of a prediction" in {
    val outputs = DenseMatrix(
      Array(1.0, 1.0),
      Array(1.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0)
    )

    val targets = DenseMatrix(
      Array(1.0, 1.0),
      Array(0.0, 0.0),
      Array(0.0, 1.0),
      Array(0.0, 0.0)
    )
    assert(precision(outputs, targets) == 75.0)
  }

  it should "compute the confusion matrix" in {
    val outputs = DenseMatrix(
      Array(1.0, 0.0, 0.0),
      Array(1.0, 0.0, 0.0),
      Array(0.0, 1.0, 0.0),
      Array(0.0, 1.0, 0.0),
      Array(0.0, 0.0, 1.0),
      Array(0.0, 0.0, 1.0)
    )
    val targets = DenseMatrix(
      Array(1.0, 0.0, 0.0), // ok
      Array(1.0, 0.0, 0.0), // ok
      Array(1.0, 0.0, 0.0), // ko: +1 in 2 x 0
      Array(0.0, 1.0, 0.0), // ok
      Array(0.0, 0.0, 1.0), // ok
      Array(0.0, 1.0, 0.0) // ko: +1 in 3 x 2
    )
    val expected = DenseMatrix(
      Array(2, 0, 0),
      Array(1, 1, 0),
      Array(0, 1, 1)
    )
    assert(confMatrix(outputs, targets) == expected)
  }

}
