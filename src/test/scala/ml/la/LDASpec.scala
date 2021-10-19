package ml.la

import org.scalatest._
import flatspec._
import matchers._
import org.scalactic.{TolerantNumerics, Equality}

import breeze.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.numerics._

class LASpec extends AnyFlatSpec with should.Matchers {

  val epsilon = 1e-4f

  implicit val doubleEq: Equality[Double] =
    TolerantNumerics.tolerantDoubleEquality(epsilon)

  val m = DenseMatrix(
    Array(-2.1, 3.0),
    Array(-1.0, 1.1),
    Array(4.3, 0.12)
  )

  "LDA" should "estimate a covariance matrix" in {
    val expected = DenseMatrix(
      Array(11.71, -4.286),
      Array(-4.286, 2.144133)
    )
    val result = LA.cov(m, true)
    val diffs = (result - expected).findAll(abs(_) > 0.1)
    assert(diffs.size == 0)
  }

}
