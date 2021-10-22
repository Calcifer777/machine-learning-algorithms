package ml.networks

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._
import scala.math.exp

trait Activation extends Function1[BDM[Double], BDM[Double]] {
  def gradient(d: BDM[Double]): BDM[Double]
}

case class Sigmoid(beta: Double) extends Activation {
  def sigmoid(d: Double)    = 1.0 / (1.0 + exp(-beta * d))
  def apply(m: BDM[Double]) = m map sigmoid
  def gradient(m: BDM[Double]): BDM[Double] = m map { (d: Double) =>
    beta * d * (1 - d)
  }
}

case object Linear extends Activation {
  def apply(m: BDM[Double])                 = m
  def gradient(m: BDM[Double]): BDM[Double] = m
}

case object SoftMax extends Activation {
  def apply(m: BDM[Double]) = {
    val exps = m map exp
    exps(::, *) / sum(exps(*, ::))
  }
  def gradient(d: BDM[Double]): BDM[Double] = ???
}
