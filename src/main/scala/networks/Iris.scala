package networks

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._
import java.io.File
import scala.io.Source
import com.github.tototoshi.csv._
import scala.util.Random

import networks._
import MultiLayerPerceptron._

object IrisMLP extends App {

  def preprocess(fileName: String): (BDM[Double], BDM[Double]) = {
    val data = CSVReader.open(Source.fromResource(fileName)).all()

    val xsData = data map { (r: List[String]) =>
      DenseVector(r: _*).slice(0, 4)
    }
    val xs = DenseMatrix(xsData: _*).mapValues(_.toDouble)

    val ysData = data
      .map { (r: List[String]) => DenseVector(r: _*) }
      .map { v => v(-1) }
      .map {
        case "Iris-setosa"     => Array(1.0, 0.0, 0.0)
        case "Iris-versicolor" => Array(0.0, 1.0, 0.0)
        case "Iris-virginica"  => Array(0.0, 0.0, 1.0)
      }
    val ys = DenseMatrix(ysData: _*)

    (xs, ys)
  }

  def trainTestSplit(
      xs: BDM[Double],
      ys: BDM[Double],
      testRatio: Double
  ): (BDM[Double], BDM[Double], BDM[Double], BDM[Double]) = {
    require(xs.rows == ys.rows)
    val idx = Random.shuffle(1 to xs.rows - 1)
    val (testIdx, trainIdx) = idx.splitAt((xs.rows * testRatio).toInt)
    (
      xs(trainIdx, ::).toDenseMatrix,
      xs(testIdx, ::).toDenseMatrix,
      ys(trainIdx, ::).toDenseMatrix,
      ys(testIdx, ::).toDenseMatrix
    )
  }

  def confMatrix(outputs: BDM[Double], targets: BDM[Double]): BDM[Int] = {
    require(outputs.rows == targets.rows)
    require(outputs.cols == targets.cols)
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

  // xs: 150 x 4
  // ys: 150 x 3 (OHE)
  val (xs, ys) = preprocess("iris.csv")
  val (xsTrain, xsTest, ysTrain, ysTest) = trainTestSplit(xs, ys, 0.3)

  val net = mlp(Seq(4, 4, 3), eta = 0.5, beta = 2)

  val trained = Network.train(net, xsTrain, ysTrain, 5000)

  val predictions = trained.activate(xsTest)

  val m = DenseMatrix.horzcat(predictions, ysTest)
  m(*, ::) map { case x =>
    println(x.slice(0, 3))
    println(x.slice(3, 6))
    println("\n")
  }
  val cm = confMatrix(predictions, ysTest)
  println(cm.toString(100, 100))
}
