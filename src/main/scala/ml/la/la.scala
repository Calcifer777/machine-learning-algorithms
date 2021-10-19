package ml.la

import breeze.linalg._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, sum}
import breeze.numerics._

object LA {

  def cov(
      bdm: DenseMatrix[Double],
      sample: Boolean = true
  ): DenseMatrix[Double] = {
    // set col means to zero
    val n = (bdm.rows - (if (sample) 1 else 0)).toDouble
    val means = (sum(bdm(::, *)) / bdm.rows.toDouble).t
    val centered = bdm(*, ::) - means
    centered.t * centered / n
  }

  def lda(
      data: BDM[Double],
      classes: BDM[Double],
      numComponents: Int = 2
  ): Seq[BDM[Double]] = {
    val cv = cov(data, true)
    val obs = data.rows
    val nClasses = classes.cols
    val covMat = cov(data)

    // Data prep
    val c2 = DenseMatrix(
      (1 to classes.cols)
        .map((i: Int) => Array.fill(obs)(i.toDouble)): _*
    ).t
    val flatClasses = c2 *:* classes
    // Sequences of BDM by class
    val s = flatClasses(::, 1).findAll(_ == 2.0).toSeq
    val classIdx = (0 to nClasses - 1)
      .map { (i: Int) => flatClasses(::, i).findAll(_ == i.toDouble + 1).toSeq }
    val classData = classIdx.map { (idx: Seq[Int]) =>
      data(idx, ::).toDenseMatrix
    }

    // LDA
    val means = classData map { m => sum(m(::, *)) }
    val priors = classData map { m => m.rows / data.rows }
    val classCov = classData map { m => cov(m) }
    val sW = classCov
      .map { m => m * (m.rows.toDouble / data.rows) }
      .reduce { (m1, m2) => m1 + m2 }
    val sB = covMat - sW
    val eigs = eig(pinv(sW) * sB)
    val (eigVal, eigVec) = (eigs.eigenvalues, eigs.eigenvectors)

    // Sort eigenvalues by size and take the corresponding eigenvectors
    val componentsIdx = argsort(eigVal).reverse.slice(0, numComponents)
    val componentsData = data * eigVec(::, componentsIdx).toDenseMatrix
    val components = classIdx.map { (idx: Seq[Int]) =>
      componentsData(idx, ::).toDenseMatrix
    }
    components
  }

  def pca(
      data: BDM[Double],
      numCmp: Int = 2,
      normalize: Boolean = true
  ): (BDM[Double], BDM[Double]) = {
    val colMeans = (sum(data(::, *)) / data.rows.toDouble).t
    val centered = data(*, ::) - colMeans
    val covMat = (centered.t * centered) / data.rows.toDouble
    val eigs = eig(covMat)
    val (eVals, eVecs) = (eigs.eigenvalues, eigs.eigenvectors)
    val cmpIdx = argsort(eVals).reverse.slice(0, numCmp)
    val norms = norm(eVecs(::, *))
    val eVecsNorm1: BDM[Double] = eVecs(*, ::) / norms.t
    val eVecsNorm2 = eVecsNorm1(*, ::) * sqrt(eVals)
    val eVecsNorm3 = DenseMatrix(eVecsNorm2.toArray.map(_.toArray): _*)
    val eVecsNorm4 = eVecsNorm3(::, cmpIdx).toDenseMatrix

    val newX = data * eVecsNorm4
    println(newX.rows)
    println(newX.cols)
    println(eVecsNorm4.rows)
    println(eVecsNorm4.cols)
    val newY = (newX * eVecsNorm4.t)
    val newY2 = newY(*, ::) + colMeans
    (newX, newY2)
  }
}
