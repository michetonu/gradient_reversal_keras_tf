# Gradient Reversal Layer for Keras
Keras implementation of a gradient inversion layer for the Tensorflow backend, following the paper [Domain-Adversarial Training of Neural Networks](http://jmlr.org/papers/volume17/15-239/15-239.pdf).
Modified the Theano version by Pumpikano found [here](https://github.com/pumpikano/tf-dann), expanding on the work done by VanushVaswani found [here](https://github.com/fchollet/keras/pull/4031).

The layer can be placed in a Functional model such as:

```
Flip = flipGradientTF.GradientReversal(hp_lambda)
dann_in = Flip(previous_layer_output)
dann_out = Dense(2)(dann_in)
```

where `hp_lambda` is the constant which multiplies the flipped gradient.


An example of a model where this is implemented (with full code) can be found here: https://github.com/michetonu/DA-RNN_manoeuver_anticipation

