package utils

import breeze.linalg._

class Matrix[A, B](val mat: DenseMatrix[Double]) {

  def *[C](other: Matrix[B, C]): Matrix[A, C] =
    new Matrix[A, C](mat * other.mat)

  def t: Matrix[B, A] = new Matrix[B, A](mat.t)

  def +(other: Matrix[A, B]): Matrix[A, B] = new Matrix[A, B](mat + other.mat)

  def *:*(other: Matrix[A, B]): Matrix[A, B] =
    new Matrix[A, B](mat *:* other.mat)

  def *(scalar: Double): Matrix[A, B] = new Matrix[A, B](mat * scalar)
}

object Matrix {

  def readcsv[A, B](filename: String) =
    new Matrix[A, B](csvread(new java.io.File(filename)))

  def inverse[A](x: Matrix[A, A]): Matrix[A, A] = new Matrix[A, A](inv(x.mat))

  def ident[D](d: Int): Matrix[D, D] = new Matrix[D, D](DenseMatrix.eye(d))

}
