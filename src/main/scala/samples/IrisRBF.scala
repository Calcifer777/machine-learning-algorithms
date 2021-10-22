package samples

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._

import data.IrisDataSource
import ml.networks._
import RBFNetwork.train

object IrisRBF extends App {

  val irisData          = IrisDataSource("iris.csv")
  val (trainDS, testDS) = irisData.trainTestSplit(testRatio = 0.4)

  val rbfNet = new RBFNetwork {
    val nIn        = 4
    val nOut       = 3
    val rbfLayer   = SamplingLayer(0.4, 5, 42)
    val perceptron = Perceptron.perceptron(5, 3, 0.3)
  }

  val trainedRBF = train(rbfNet, trainDS.xs, trainDS.ys, 120)

  val predictions = trainedRBF.predict(testDS.xs)

  val cm        = trainedRBF.confMatrix(predictions, testDS.ys)
  val precision = trainedRBF.precision(predictions, testDS.ys)
  println(cm.toString(100, 100))
  println(f"Precision: ${precision}%.0f%%")
}
