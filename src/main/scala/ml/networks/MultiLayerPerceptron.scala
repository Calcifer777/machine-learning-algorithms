package ml.networks

import breeze.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.numerics.round
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import com.typesafe.scalalogging.LazyLogging
import scala.annotation
import scala.math.exp

/** 
 * TODO: eta is a training hyperparameter, should not be tied to the network
 */
case class MultiLayerPerceptron(
    weights: Seq[BDM[Double]],
    eta: Double = 1.0, // learning rate
    beta: Double = 1.0 // sigmoid coefficient
) extends Network[MultiLayerPerceptron] {

  require(weights.size == 2)

  def makeBias(size: Int): BDM[Double] = DenseMatrix.fill(size, 1)(-1)

  def addBias(m: BDM[Double]): BDM[Double] =
    DenseMatrix.horzcat(m, makeBias(m.rows))

  /** * TODO: generalize into case class */
  def activationFunction(x: Double): Double = 1.0 / (1.0 + exp(-beta * x))

  def predict(inputs: BDM[Double]): BDM[Double] = {
    val hidden = addBias(addBias(inputs) * weights(0).map(activationFunction))
    val outputs = (hidden * weights(1)).map(activationFunction)
    outputs
  }

  def predictWithTrace(inputs: BDM[Double]): Seq[BDM[Double]] = {
    val hidden_outputs = addBias(inputs) * weights(0)
    // println("HIDDEN OUTPUTS")
    // println(hidden_outputs.toString(200, 200))

    // println(activationFunction(0.42))
    val hidden = addBias(hidden_outputs.map(activationFunction))
    // println("HIDDEN WITH BIAS")
    // println(hidden.toString(200, 200))

    val outputs = (hidden * weights(1)).map(activationFunction)
    // println("OUTPUTS")
    // println(outputs.toString(200, 200))
    Seq(hidden, outputs)
  }

  def trainIteration(
      inputs: BDM[Double],
      targets: BDM[Double]
  ): MultiLayerPerceptron = {

    // println("START W0")
    // println(weights(0).toString(100, 100))
    // println("START W1")
    // println(weights(1).toString(100, 100))
    val activations = predictWithTrace(inputs)
    val hidden = activations(0)
    val outputs = activations(1)
    // println("HIDDEN")
    // println(hidden.toString(200, 200))
    // println("OUTPUTS")
    // println(outputs.toString(200, 200))

    val delta_out = (outputs - targets) *:* (beta * outputs *:* (1.0 - outputs))
    // println("DELTA_OUT")
    // println(delta_out.toString(200, 200))

    val delta_hid =
      hidden *:* (beta * (1.0 - hidden)) *:* (delta_out * weights(1).t)
    // println("DELTA_HIDDEN")
    // println(delta_hid.toString(200, 200))
    val delta_hid_adj = delta_hid(::, 0 to weights(0).cols - 1)
    val updateWOutput = eta * (hidden.t * delta_out)
    val updateWHidden = eta * (addBias(inputs).t * delta_hid_adj)
    val w_output = weights(1) - updateWOutput
    val w_hidden = weights(0) - updateWHidden
    // println("W HIDDEN")
    // println(w_hidden.toString(200, 200))
    // println("W OUTPUT")
    // println(w_output.toString(200, 200))
    // println("UDPATE W HIDDEN")
    // println(updateWHidden.toString(200, 200))
    // println("UDPATE W OUTPUT")
    // println(updateWOutput.toString(200, 200))

    MultiLayerPerceptron(Seq(w_hidden, w_output), eta, beta)

  }

}

object MultiLayerPerceptron {

  val uniform01 = breeze.stats.distributions.Uniform(-0.1, 0.1)

  def makeWeights(rows: Int, cols: Int): BDM[Double] =
    DenseMatrix.rand(rows, cols, uniform01)

  /**
   * TODO: move to tensors
   * TODO: use bias only if specified
   */
  def mlp(
      dimensions: Seq[Int],
      eta: Double,
      beta: Double
  ): MultiLayerPerceptron = {
    val weigthsDims = dimensions zip dimensions.tail
    // val weights = Seq(
    //   makeWeights(weigthsDims(0)._1 + 1, weigthsDims(0)._2),
    //   makeWeights(weigthsDims(1)._1 + 1, weigthsDims(1)._2)
    // )
    val w0 = DenseMatrix(
      Array(
        0.009762700178756134,
        0.029178823186104597,
        0.058345007336642496,
        -0.08257413909384051,
        0.09572366915228378
      ),
      Array(
        0.043037873042421765,
        -0.012482558794216517,
        0.0057789846658025945,
        -0.09595632145332833,
        0.05983171403679474
      ),
      Array(
        0.020552676492767347,
        0.07835460024092486,
        0.013608912804081413,
        0.06652396904574381,
        -0.007704128263957874
      ),
      Array(
        0.008976637861882825,
        0.0927325525064954,
        0.08511932741427053,
        0.055631348908992934,
        0.05610583626890042
      ),
      Array(
        -0.015269039203036752,
        -0.0233116970131841,
        -0.08579278785766316,
        0.07400242945577876,
        -0.07634511375238705
      )
    )
    val w1 = DenseMatrix(
      Array(0.02798420364266571, 0.05484673898493697, 0.023386798460748717),
      Array(-0.07132934171744224, -0.008769934724121242, 0.08874961704790138),
      Array(0.088933783050226, 0.013686788485104365, 0.03636405900105827),
      Array(0.004369664507828969, -0.09624203942972126, -0.02809842046182319),
      Array(-0.01706761278629179, 0.023527098148092215, -0.012593609385643662),
      Array(-0.04708887879153392, 0.02241914338565283, 0.03952623933339222)
    )
    val weights = Seq(w0, w1)
    println("W0")
    println(w0.toString(200, 200))
    println("W1")
    println(w1.toString(200, 200))
    MultiLayerPerceptron(
      weights,
      eta,
      beta
    )
  }
}
