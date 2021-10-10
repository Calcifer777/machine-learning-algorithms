package networks

import scala.annotation
import com.typesafe.scalalogging.LazyLogging
import breeze.linalg._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import spire.math.Algebraic.Expr.Mul
import scala.math.exp
import breeze.numerics.round

case class MultiLayerPerceptron(
    weights: Seq[MD],
    eta: Double,
    beta: Double
) extends Network {

  require(weights.size == 2)

  def makeBias(size: Int): MD = DenseMatrix.fill(size, 1)(-1)

  def addBias(m: MD): MD = DenseMatrix.horzcat(m, makeBias(m.rows))

  /** * TODO: generalize into case class */
  def activationFunction(x: Double): Double = 1.0 / (1.0 + exp(-beta * x))

  def act2(x: Double, actType: String = "sigmoid"): Double = {
    if (actType == "sigmoid") 1 / (1 + exp(-beta * x))
    else if (actType == "relu") if (x < 0) 0 else x
    else throw new RuntimeException(s"activation function type $actType not supported ")
  }

  // assume sigmoid activation for outputs
  def activate(input: MD): MD = {
    val h = (addBias(input) * weights(0)).map(activationFunction)
    (addBias(h) * weights(1)).map(activationFunction)
  }

  def activateWithTrace(input: MD): Seq[MD] = {
    val h = addBias((addBias(input) * weights(0)).map(activationFunction))
    val o = (h * weights(1)).map(activationFunction)   
    Seq(h, o)
  }

  def trainIteration(inputs: MD, targets: MD): MultiLayerPerceptron = {
    // println("TARGETS")
    // println(targets.toString(100, 100))

    val h = addBias((addBias(inputs) * weights(0)).map(activationFunction))
    val o = (h * weights(1)).map(activationFunction)   
    // println("OUTPUTS")
    // println(o.toString(100, 100))

    val d_output = beta * (o - targets) *:* o *:* (1.0 - o)
    // println("D_OUTPUT")
    // println(d_output.toString(100, 100))
    // deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
    val d_hidden = beta * h *:* (1.0-h) *:* (d_output * weights(1).t)          
    // deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
    // println("HIDDEN")
    // println(h.toString(100, 100))
    // println("D_HIDDEN")
    // println(d_hidden.toString(100, 100))
    
    val w_output = weights(1) - eta * (h.t * d_output)
    // println("NEW W2")
    // println(w_output.toString(100, 100))
    // updatew2 =               eta * (np.dot(np.transpose(self.hidden),deltao))
    // println("OLD W1")
    // println(weights(0).toString(100, 100))
    val w_hidden = weights(0) - eta * ( addBias(inputs).t * d_hidden(::, 0 to weights(0).cols-1))
    // println("NEW W1")
    // println(w_hidden.toString(100, 100))
    // updatew1 =               eta*(np.dot(np.transpose(inputs),deltah[:,:-1]))
    // println("TEST")
    // println(addBias(inputs).t)
    // println("D_HIDDEN")
    // println(d_hidden.toString(100, 100))
    // println("D_HIDDEN MOD")
    // println(d_hidden(::, 0 to weights(0).cols-1))
    // println("\n\n\n")

    MultiLayerPerceptron(
      Seq(w_hidden, w_output),
      eta,
      beta
    )
  }

}

object MultiLayerPerceptron {

  val uniform01 = breeze.stats.distributions.Uniform(-0.1, 0.1)

  def makeWeights(rows: Int, cols: Int): MD =
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
    val weights = Seq(
      makeWeights(weigthsDims(0)._1+1, weigthsDims(0)._2),
      makeWeights(weigthsDims(1)._1+1, weigthsDims(1)._2)
    )
    MultiLayerPerceptron(
      weights,
      eta,
      beta
    )
  }
}
