package rbf

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.linalg._
import breeze.numerics._
import scala.util.Random
import scala.math
import networks.{Perceptron, Network}
import Perceptron._

trait RBFNetwork {

  /** 
   *  Inputs:                 SampleSize x nInputs
   *  Weights Hidden Layer:   nInputs x nHidden
   *  Hidden Layer Nodes:     SampleSize x nHidden
   *  Hidden Layer Weights:   SampleSize x nHidden
   * 
   */

  val nIn: Int
  val nOut: Int
  val rbfLayer: RBFLayer
  val outputLayer: Network

  // RBF layer (with bias) must be compatible with outputLayer
  // require(rbfLayer.nHidden+1 == outputLayer.inputSize)

}

object RBFNetwork {

  def train(
      net: RBFNetwork,
      inputs: BDM[Double],
      targets: BDM[Double],
      epochs: Int
  ): RBFNetwork = {
    val trainedRbf = net.rbfLayer.train(inputs)
    val trainedOutput = Network.train(net.outputLayer, inputs, targets, epochs)
    new RBFNetwork {
      val nIn = 10
      val nOut = 10
      val rbfLayer = trainedRbf
      val outputLayer = trainedOutput
    }

  }

}
