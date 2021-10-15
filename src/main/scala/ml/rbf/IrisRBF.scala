package ml.rbf

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._
import java.io.File
import scala.io.Source
import com.github.tototoshi.csv._
import scala.util.Random

import data.IrisDataSource
import ml.networks.Perceptron.{perceptron, train}

object IrisRBF extends App {

  val irisData = IrisDataSource("iris.csv")
  val (trainDS, testDS) = irisData.trainTestSplit(testRatio = 0.4)

  val rbfNet = new RBFNetwork {
    val nIn = 4
    val nOut = 3
    val rbfLayer = Sample(0.4, 4, 42)
    val outputLayer = perceptron(4, 3, 0.3)
  }

  val trained = RBFNetwork.train(rbfNet, trainDS.xs, trainDS.ys, 5000)

  val predictions = trained.predict(testDS.xs)

  println(predictions.rows)
  println(predictions.cols)
  val m = DenseMatrix.horzcat(predictions, testDS.ys)
  m(*, ::) map { case x =>
    println(x.slice(0, 3))
    println(x.slice(3, 6))
    println("\n")
  }
  // val cm = confMatrix(predictions, ysTest)
  // println(cm.toString(100, 100))
}
