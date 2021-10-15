package ml

import com.typesafe.scalalogging.LazyLogging
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV }
import breeze.linalg._
import scala.math.abs

trait Model {

  def confMatrix(outputs: BDM[Double], targets: BDM[Double]): BDM[Int] = {
    // Ensure outputs and targets have the same dimensions
    require(outputs.rows == targets.rows)
    require(outputs.cols == targets.cols)
    // Ensure outputs and targets encode categories
    require((sum(outputs(*, ::)) :!= 1.0).toScalaVector.filter((b: Boolean) => b).size == 0)
    require((sum(targets(*, ::)) :!= 1.0).toScalaVector.filter((b: Boolean) => b).size == 0)
    val outputIdx = argmax(outputs(*, ::))
    val targetIdx = argmax(targets(*, ::))
    val numClasses = outputs.cols
    val results: Seq[Array[Int]] = (0 to numClasses - 1)
      .map { (x: Int) =>
        val l = (0 to numClasses - 1)
          .map { (y: Int) =>
            val xs = outputIdx.mapValues((a: Int) => if (a == x) 1 else 0)
            val ys = targetIdx.mapValues((b: Int) => if (b == y) 1 else 0)
            sum(xs * ys)
          }
        Array(l: _*)
      }
    DenseMatrix(results: _*)
  }

  def precision(outputs: BDM[Double], targets: BDM[Double]): Double = {
    require(outputs.rows == targets.rows)
    require(outputs.cols == targets.cols)
    val diffs = (outputs - targets)
    val correct = diffs(*, ::)
      .map((row: Vector[Double]) => math.abs(sum(row)) < 1.0E-2)
      .map(b => if (b == true) 1 else 0)
    (sum(correct).toDouble / outputs.rows) * 100
  }

}

object Model extends Model {
}