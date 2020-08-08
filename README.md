[![DOI](https://zenodo.org/badge/286022491.svg)](https://zenodo.org/badge/latestdoi/286022491)

# SK-MachineLearning
A python code to predict SK parameters using Machine Learning

 A method is proposed to construct tight-binding (TB) models for solids using machine learning (ML) techniques. The approach is based on the LCAO method in combination with Slater-Koster (SK) integrals which are used to obtain optimal SK parameters. Lattice constant is used to generate training examples to construct a linear ML model. We successfully used this method to find a TB model for BiTeCl where spin-orbit coupling plays an essential role in its topological behavior.

The SK parameters are non-linearly connected to the lattice constant which is connected to the dimensionless scale factor s. In fact they are exponential functions of the scale factor and therefore, we can linearize them by using the logarithm function. Here we assume that each response is a linear function of the predictors with an additive error. A linear model proposes a map between inputs and outputs that have to be fitted or learned from the input data. It should be emphasized that the inputs can be generated by a nonlinear map from some original space by which we are able to use linear regression to fit nonlinear functions. To construct our ML model we use the scikit-learn python library. The most common method to estimate the parameters is to minimize the least-squares error between the predicted values by the model and the actual target outputs for each data point.
