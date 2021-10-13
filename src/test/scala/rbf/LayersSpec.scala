package rbf

import org.scalatest._
import flatspec._
import matchers._
import org.scalactic.{ TolerantNumerics, Equality }


import breeze.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.numerics.abs

class LayersSpec extends AnyFlatSpec with should.Matchers {

  val epsilon = 1e-4f

  implicit val doubleEq: Equality[Double] = TolerantNumerics.tolerantDoubleEquality(epsilon)

  val inputs = DenseMatrix(
    Array(3.0, 9.0, 5.0, 6.0, 0.0),
    Array(4.0, 6.0, 2.0, 4.0, 5.0),
    Array(0.0, 5.0, 7.0, 8.0, 6.0),
    Array(0.0, 8.0, 2.0, 7.0, 4.0),
    Array(0.0, 3.0, 2.0, 6.0, 1.0),
    Array(9.0, 9.0, 5.0, 8.0, 5.0),
    Array(6.0, 0.0, 7.0, 7.0, 7.0),
    Array(6.0, 5.0, 9.0, 5.0, 8.0),
    Array(8.0, 5.0, 6.0, 8.0, 4.0),
    Array(6.0, 2.0, 2.0, 4.0, 9.0),
  )

  "An RBF sampling hidden layer" should "generate weights based on inputs" in {
    val layer = Sample(4.0, 3, 42)
    val expected = DenseMatrix(
      Array(0.0, 6.0, 0.0),  
      Array(3.0, 0.0, 5.0),  
      Array(2.0, 7.0, 7.0),  
      Array(6.0, 7.0, 8.0),  
      Array(1.0, 7.0, 6.0), 
    )
    assert(layer.weights(inputs) == expected)
  }

  it should "generate weights and nodes data when trained" in {
    val layer = Sample(4.0, 3, 42)
    val trained = layer.predict(inputs)
    val expected = DenseMatrix(
      Array(0.58561397, 0.03628523, 0.3781008 ),
      Array(0.49944823, 0.17808375, 0.32246802),
      Array(0.1252962 , 0.10717152, 0.76753228),
      Array(0.51871833, 0.02351435, 0.45776732),
      Array(0.83434161, 0.02945586, 0.13620252),
      Array(0.10753985, 0.4671318 , 0.42532835),
      Array(0.03004787, 0.85111047, 0.11884165),
      Array(0.02363848, 0.6289966 , 0.34736492),
      Array(0.10711055, 0.6359446 , 0.25694485),
      Array(0.10494769, 0.75160583, 0.14344648),
    )
    val diffs = (trained - expected).findAll( math.abs(_) > epsilon)
    assert(diffs.size == 0)
  }

}
