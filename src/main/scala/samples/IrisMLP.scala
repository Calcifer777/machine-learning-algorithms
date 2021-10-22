package samples

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._

import data.IrisDataSource
import ml.networks._
import MultiLayerPerceptron.mlp
import Network.train

object IrisMLP extends App {

  val irisData          = IrisDataSource("iris.csv")
  val (trainDS, testDS) = irisData.trainTestSplit(testRatio = 0.5)
  val targets           = trainDS.ys

  val nn = mlp(Seq(4, 5, 3), 0.3, 0.5)

  val trained = train(nn, trainDS.xs, trainDS.ys, 2500)

  val predictions = trained.predictWithTrace(testDS.xs)
  val predictionsNorm = predictions(1)(*, ::).map { v =>
    val am = argmax(v)
    val v2 = DenseVector.fill[Double](v.length)(0.0)
    v2(argmax(v)) = 1.0
    v2
  }

  val cm = trained.confMatrix(predictionsNorm, testDS.ys)
  println("\n" + cm.toString(100, 100) + "\n")
  val precision = trained.precision(predictionsNorm, testDS.ys)
  println(f"Precision: ${precision}%.0f%%")

}
