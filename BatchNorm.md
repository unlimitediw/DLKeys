### Batch Normalization
> Why
* Nomalization benefits input layer, it is also appliable to hidden layer.
* It allows each layer of a network to learn by itself a little bit more independently of other layers.

> Effect
* We can use higher learning rates because there's no activation result too high or low.
* It reduces overfitting due to the regularization effects (noises). But should be also used with dropout.

> How 
* Two para(trainable): gamma and beta(mean) for standard deviation.
  * gamma may fixed.
  * beta = mini-batch mean
