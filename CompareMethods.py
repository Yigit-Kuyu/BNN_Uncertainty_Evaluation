from BNN import run_vi, run_mcmc, run_bayes_by_backprop
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score
import time
import numpy as np
from BNN import predict_with_uncertainty
import torch

def expected_calibration_error(y_true, y_pred, y_std, n_bins=10):
    # Create bin boundaries for calibration
    bin_boundaries = np.linspace(0, 1, n_bins + 1)  # Generate evenly spaced bin boundaries
    bin_lowers = bin_boundaries[:-1]  # Lower boundaries of the bins
    bin_uppers = bin_boundaries[1:]  # Upper boundaries of the bins
    ece = 0.0  # Initialize ECE to zero

    # Iterate through each bin to calculate ECE
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine which predictions fall within the current bin
        in_bin = (y_std > bin_lower) & (y_std <= bin_upper)
        prop_in_bin = in_bin.mean()  # Proportion of predictions in the current bin

        # If there are predictions in the bin, calculate accuracy and average confidence
        if prop_in_bin > 0:
            accuracy_in_bin = np.abs(y_true[in_bin] - y_pred[in_bin]).mean()  # Mean accuracy in the bin
            avg_confidence_in_bin = y_std[in_bin].mean()  # Mean confidence in the bin
            # Update ECE based on the difference between average confidence and accuracy
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece  

if __name__ == "__main__":
    data = np.load('data.npz')
    x_obs = data['x_obs']
    y_obs = data['y_obs']
    x_true = data['x_true']
    y_true = data['y_true']
    x_test = torch.from_numpy(x_true).float().unsqueeze(1)

    methods = {
        "VI": run_vi,
        "MCMC": run_mcmc,
        "Bayes by Backprop": run_bayes_by_backprop
    }

    results = {}
    for method_name, method in methods.items():
        start_time = time.time()
        if method_name == "Bayes by Backprop":
            samples, model = method(x_obs, y_obs, num_iterations=1000, num_samples=50, seed=42)
            samples = {key: value.squeeze(1) for key, value in samples.items()}
        elif method_name == "VI":
            samples, model, elbo_losses = method(x_obs, y_obs, num_samples=50, seed=42)
            samples= {key: value.squeeze(1) for key, value in samples.items()}
        else:
            samples, model = method(x_obs, y_obs, num_samples=50, seed=42)
        end_time = time.time()
        mean_pred, std_pred = predict_with_uncertainty(model, samples, x_test)
        mse = mean_squared_error(y_true, mean_pred.detach().numpy())
        auroc = roc_auc_score((y_true > np.median(y_true)).astype(int), std_pred.detach().numpy())
        ece = expected_calibration_error(y_true, mean_pred.detach().numpy(), std_pred.detach().numpy())
        results[method_name] = {
            "time": end_time - start_time,
            "mse": mse,
            "auroc": auroc,
            "ece": ece
        }

    
    for method_name, result in results.items():
        print("###############################################################")
        print(f"Method: {method_name}")
        print(f"Time: {result['time']} seconds")
        print(f"MSE: {result['mse']}")
        print(f"AUROC: {result['auroc']}")
        print(f"ECE: {result['ece']}")
        print("###############################################################")

    # Plot predictions with uncertainty for each method
    for method_name, method in methods.items():
        if method_name == "Bayes by Backprop":
            samples, model = method(x_obs, y_obs, num_iterations=1000, num_samples=50, seed=42)
            samples = {key: value.squeeze(1) for key, value in samples.items()}
        elif method_name == "VI":
            samples, model, elbo_losses = method(x_obs, y_obs, num_samples=50, seed=42)
            samples= {key: value.squeeze(1) for key, value in samples.items()}
        else:
            samples, model = method(x_obs, y_obs, num_samples=50, seed=42)
        
        mean_pred, std_pred = predict_with_uncertainty(model, samples, x_test)
        plt.figure(figsize=(12, 6))
        plt.plot(x_true, y_true, 'g-', linewidth=2, label='True function')
        plt.plot(x_obs, y_obs, 'ko', markersize=4, label='Observations')
        plt.plot(x_test, mean_pred, 'b-', linewidth=2, label='Mean prediction')
        plt.fill_between(x_test.squeeze(), mean_pred - 2*std_pred, mean_pred + 2*std_pred, alpha=0.3, color='blue', label='95% Credible Interval')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'BNN Predictions with Uncertainty ({method_name})')
        plt.show()



