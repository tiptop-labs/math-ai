| dataset | description                                  |
| ------- |:--------------------------------------------:|
| mult999 | multiplications, suitable for (NNs and) RNNs |

| model              | description    | type                | accuracy |
| ------------------ |:--------------:|:-------------------:|:--------:|
| mult999_first_nn0  | infer 1st char | NN (simple)         |     100% |
| mult999_second_nn1 | infer 2nd char | NN (1 hidden layer) |          |

requirements: Python 3.5, TensorFlow 1.5 (tf-nightly)
