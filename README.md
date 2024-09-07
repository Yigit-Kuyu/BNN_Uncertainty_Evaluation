# BNN_Uncertainty_Evaluation

### Running the Code

1. **Generate and Save Data:**

   First, run `BNN.py` to generate and save the necessary data. This script will create noisy observations from a sinusoidal function and save them in a file named `data.npz`.

   ```bash
   python BNN.py
   ```

   To speed up the process in subsequent runs, set the `first_run` variable to `0` in the `run_mcmc` function.

2. **Compare Inference Methods:**

   Next, run `CompareMethods.py` to compare the different inference methods and obtain the results.

   ```bash
   python CompareMethods.py
   ```

## Inference Methods

### 1. Variational Inference (VI)

Variational Inference approximates the posterior distribution by optimizing a family of distributions to minimize the Kullback-Leibler (KL) divergence from the true posterior. In this code, we use the `AutoNormal` guide from Pyro for automatic variational inference.

### 2. Markov Chain Monte Carlo (MCMC)

MCMC methods sample from the posterior distribution by constructing a Markov chain that has the desired distribution as its equilibrium distribution. We use the No-U-Turn Sampler (NUTS), a variant of Hamiltonian Monte Carlo (HMC), for efficient sampling.

### 3. Bayes by Backpropagation (BBB)

Bayes by Backpropagation uses the reparameterization trick to perform variational inference by backpropagating through stochastic nodes. This method allows for efficient gradient-based optimization of the variational parameters.

## Metrics

### 1. Mean Squared Error (MSE)

MSE measures the average squared difference between the true values and the predicted values. It is a common metric for regression tasks.

### 2. Area Under the Receiver Operating Characteristic Curve (AUROC)

AUROC measures the ability of the model to distinguish between classes. In this context, it is used to evaluate the uncertainty estimates by treating the standard deviation of predictions as a measure of uncertainty.

### 3. Expected Calibration Error (ECE)

ECE measures the difference between predicted confidence and actual accuracy. It is used to evaluate how well the predicted uncertainties are calibrated.

## Results

The results for each inference method, including the time taken, MSE, AUROC, and ECE, are printed to the console and plotted as graphs showing the predictions with uncertainty.

## Example Output
