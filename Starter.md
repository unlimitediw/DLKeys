### This is a learning note for lightGBM

* boosting -> gradient boosting: It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function.

* Adavantages:
  * Faster training speed and higher effiency
  * Lower memory usage
  * Better accuracy
  * Support of parallel and GPU learning
  * Capable of handling large-scale data
  


## review

> Boosting method

* Weak learners -> Strong learner
* You need to first define a class of weak learners

#
    Input: lossF, alpha(hyperpara), (x,y), A(pre-defined weak learners)
    H0 = 0
    for t = 0:T-1:
      ri = deri(l/H)
      h = argmin(sum(ri*h(xi)) (by taylor, l(H+apha*h) = l(H) + alpha * deri(l/H) * h(selected)
      if sum(ri*h(xi)) < 0:
        H = H + alpha * h
      else:
        return H

* For gradient boost:
  * Sum(h^2(x)) = 1, rescale h with iteration
  * transform argmin(sum(rh)) to argmax(sum(h+r)^2)
  * Easy to solve now with gradient method
  * care about T

* For AdaBoost:
  * Compute gradient r = deri(l/H) = -y * e^(-y * H) (for convenience w = (1/Z) * e^(-y * H), Z = sum(e^(-y * H))
  * argmin(sum(rh)) can derivated to argmin(w), so we only need to reduce weight classification error.
  * alpha = 0.5 * ln((1-e)/e)
  * w = w * e ^ (-alpha * h * y) (h is what we select that can reduce training error)
  * Z = Z * 2 * (root(e * (1-e)))
  * care about e<0.5

