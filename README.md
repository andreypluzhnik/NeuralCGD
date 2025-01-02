# Neural CGD

This is my implementation of the [Deep Conjugate Direction Method]( https://arxiv.org/abs/2205.10763) by Kaneda et al, 2023. It uses a neural network to act as a preconditioner 
in the Conjugate Gradient Descent Method. 

## Setup
Provided in `src/` are the training, evaluation and data creation codes while`models/` contains a ready to use model. 
Training data is generated from matrices found in fluid flow problems that help solve for pressure of the fluid. 
Running `src/create_dataset.py` creates a training dataset, adding `-t` as a command line argument marks the data for testing to be used by `src/eval.py`.
A new model can be trained up with `train.py`.

## Results
The method holds up quite well for matrices of size 512x512 — it outperforms classical CGD up to a tested tolerance of `1e-8`
as long as the test matrix is not singnificantly different from training one. In the context of fluid flow, different matrices arise from
different internal boundaries; when the test domain is about `~20-30%` the resulting matrices differ enough from the training matrix
that the model lags behind classical methods.

![alt text](https://github.com/andreypluzhnik/NeuralCGD/blob/master/plots/random_it_plot.png)

In the plot the solid lines correspond to the results the classical and neural preconditioners on randomized domains, 
whereas individual points are for specific domains. 
