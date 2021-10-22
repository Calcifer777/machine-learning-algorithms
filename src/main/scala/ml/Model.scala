package ml

import com.typesafe.scalalogging.LazyLogging
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import scala.math.abs

trait Model extends LazyLogging {

  def confMatrix(predictions: BDM[Double], targets: BDM[Double]): BDM[Int] = {
    // Ensure outputs and targets have the same dimensions
    assert(
      predictions.rows == targets.rows,
      s"Outputs rows (${predictions.rows}) differ from targets rows (${targets.rows})"
    )
    assert(
      predictions.cols == targets.cols,
      s"Outputs cols (${predictions.cols}) differ from targets cols (${targets.cols})"
    )
    // Ensure outputs and targets encode categories
    val outputsNotOHE = (sum(predictions(*, ::)) :!= 1.0).toScalaVector
      .filter((b: Boolean) => b)
      .size != 0
    if (outputsNotOHE)
      logger.warn("Outputs are not OHE of a categorical variable")
    assert(
      (sum(targets(*, ::)) :!= 1.0).toScalaVector
        .filter((b: Boolean) => b)
        .size == 0,
      "Targets are not OHE of a categorical variable"
    )
    val outputIdx  = argmax(predictions(*, ::))
    val targetIdx  = argmax(targets(*, ::))
    val numClasses = predictions.cols
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
    val correct = (0 to outputs.rows - 1)
      .map { (idx: Int) =>
        outputs(idx, ::) == targets(idx, ::)
      }
      .filter(_ == true)
      .size
    (correct.toDouble / outputs.rows) * 100
  }

}

object Model extends Model {}
