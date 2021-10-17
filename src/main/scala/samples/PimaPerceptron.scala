package samples

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._

import data.PimaDataSource
import ml.networks.Network.train
import ml.networks.Perceptron.perceptron

object PimaPerceptron extends App {
  // Load dataset
  val pimaData = PimaDataSource("pima.csv")
  val (trainDS, testDS) = pimaData.trainTestSplit(testRatio = 0.4)
  // Setup and train perceptron
  val pcn = perceptron(9, 1, 0.5)
  val trained = train(pcn, trainDS.xs, trainDS.ys, 2500)
  // Make predictions
  val predictions = trained.predict(testDS.xs)
  val precision = trained.precision(predictions, testDS.ys)
  println(f"Precision: ${precision}%.0f%%")

}
