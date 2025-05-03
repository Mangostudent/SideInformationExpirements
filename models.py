import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

class RegularizedLogisticRegressionModel:
    """
    A regularized logistic regression model trained in two steps:
    1. Train an intermediate model to predict Z from X.
    2. Train the final model to predict Y using X and the predicted Z,
       with regularization applied only to the weight of the predicted Z feature.
    """
    def __init__(self, reg, sample_xz, sample_xy):
        """
        Initialize the model.

        Args:
            reg (float): Regularization strength for the Z_pred weight. Must be non-negative.
            sample_xz (tuple): A tuple (X_xz, Z) for training the intermediate X -> Z model.
            sample_xy (tuple): A tuple (X_xy, Y) for training the final (X, Z_pred) -> Y model.
        """
        assert reg >= 0, "Regularization value must be non-negative"
        self.reg = reg
        self.sample_xz = sample_xz  # tuple (X_xz, Z)
        self.sample_xy = sample_xy  # tuple (X_xy, Y)
        self.u = None  # weights for X→Z model
        self.w = None  # weights for final model
        self.intermediate_model = None

    def fit(self):
        """
        Fits the intermediate and final logistic regression models.
        Stores the intermediate model in self.intermediate_model,
        its weights in self.u, and the final model weights in self.w.

        Returns:
            self: The fitted model instance.
        """
        X_xz, Z = self.sample_xz
        X_xy, Y = self.sample_xy

        # Step 1: Fit intermediate logistic regression model (X -> Z)
        clf = LogisticRegression(fit_intercept=True, solver='lbfgs')
        # Ensure Z is flattened for scikit-learn compatibility
        clf.fit(X_xz, Z.ravel()) 
        u_c = clf.intercept_[0]
        u_x = clf.coef_[0]
        self.u = (u_c, u_x)
        self.intermediate_model = clf

        # Step 2: Predict Z for the second dataset (X_xy) using the intermediate model
        # Use predict, not predict_proba, to get class labels directly (+1/-1 assumed based on later usage)
        Z_pred = clf.predict(X_xy) 

        # Step 3: Prepare augmented features [1, X, Z_pred] for the final model
        n_samples = X_xy.shape[0]
        # Ensure X_xy is 2D
        if X_xy.ndim == 1:
            X_xy = X_xy.reshape(-1, 1)
        X_aug = np.hstack([np.ones((n_samples, 1)), X_xy, Z_pred.reshape(-1, 1)])

        # Step 4: Initialize weights for the final model (w_intercept, w_x, w_z_pred)
        n_features = X_aug.shape[1]
        w0 = np.zeros(n_features)

        # Step 5: Define the loss function for the final model
        # Logistic loss + regularization on the weight for Z_pred (last weight)
        def loss_fn(w):
            logits = X_aug @ w
            # Use log1p for numerical stability: log(1 + exp(-x))
            logistic_loss = np.mean(np.log1p(np.exp(-Y * logits)))  
            # Regularization term applied only to the last weight (w_z_pred)
            reg_term = self.reg * (w[-1] ** 2)  
            return logistic_loss + reg_term

        # Step 6: Minimize the loss function using BFGS optimizer
        res = minimize(loss_fn, w0, method='BFGS')
        if not res.success:
            print(f"Warning: Optimization for final model did not converge. Message: {res.message}")
        self.w = res.x
        return self # Return self for potential chaining

    def predict(self, X_test):
        """
        Predicts Y using the fitted model. Requires the intermediate model
        to predict Z first.

        Args:
            X_test (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted labels (+1 or -1).
        """
        if self.intermediate_model is None or self.w is None:
            raise RuntimeError("Model must be fitted before prediction.")
        
        # Ensure X_test is 2D
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        # Predict Z using the intermediate model
        Z_pred_test = self.intermediate_model.predict(X_test)

        # Prepare augmented features [1, X_test, Z_pred_test]
        n_samples = X_test.shape[0]
        X_aug_test = np.hstack([np.ones((n_samples, 1)), X_test, Z_pred_test.reshape(-1, 1)])

        # Calculate logits using the final model weights
        logits = X_aug_test @ self.w
        
        # Return predictions based on the sign of the logits
        return np.where(logits >= 0, 1, -1)


class RegularizedLogisticRegressionModelFullInformation:
    """
    A standard regularized logistic regression model trained using full information (X, Y, Z).
    Regularization is applied only to the weight of the Z feature.
    """
    def __init__(self, reg, sample):
        """
        Initialize the model.

        Args:
            reg (float): Regularization strength for the Z weight. Must be non-negative.
            sample (tuple): A tuple (X, Y, Z) containing the full dataset.
        """
        assert reg >= 0, "Regularization value must be non-negative"
        self.reg = reg
        self.X, self.Y, self.Z = sample
        self.w = None

    def fit(self):
        """
        Fits the logistic regression model using the full dataset (X, Y, Z).
        Regularization is applied to the weight corresponding to Z.

        Returns:
            np.ndarray: The learned weights w.
        """
        n_samples = self.X.shape[0]
        # Ensure X and Z are 2D
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1, 1)
        if self.Z.ndim == 1:
            self.Z = self.Z.reshape(-1, 1)
            
        # Prepare augmented features [1, X, Z]
        X_aug = np.hstack([np.ones((n_samples, 1)), self.X, self.Z])
        n_features = X_aug.shape[1]
        w0 = np.zeros(n_features) # Initial weights

        # Define the loss function: Logistic loss + regularization on Z weight
        def loss_fn(w):
            logits = X_aug @ w
            logistic_loss = np.mean(np.log1p(np.exp(-self.Y * logits)))
            # Regularization term applied only to the last weight (w_z)
            reg_term = self.reg * (w[-1] ** 2)
            return logistic_loss + reg_term

        # Minimize the loss function
        res = minimize(loss_fn, w0, method='BFGS')
        if not res.success:
            print(f"Warning: Optimization for full info model did not converge. Message: {res.message}")
        self.w = res.x
        return self.w # Return the learned weights

# Example usage demonstrating class functionality
if __name__ == "__main__":
    np.random.seed(42)
    # Generate synthetic data (ensure consistent shapes)
    X_xz = np.random.randn(100, 1) # Single feature X
    true_u = np.array([-1.0]) # Weight for X
    Z_prob = 1 / (1 + np.exp(-(X_xz @ true_u + 0.2)))
    Z = np.where(np.random.rand(100) < Z_prob.flatten(), 1, -1) # Binary Z (+1/-1)

    X_xy = np.random.randn(80, 1) # Single feature X
    true_w_xy = np.array([1.0]) # Weight for X in Y prediction
    Y_prob = 1 / (1 + np.exp(-(X_xy @ true_w_xy + 0.1)))
    Y = np.where(np.random.rand(80) < Y_prob.flatten(), 1, -1) # Binary Y (+1/-1)

    reg = 0.1
    # Instantiate and fit the two-stage model
    model = RegularizedLogisticRegressionModel(reg, (X_xz, Z), (X_xy, Y))
    model.fit() # Fit the model (returns self, weights stored in model.w and model.u)
    print("Learned weights w (two-stage):", model.w)
    print("Intermediate model weights u (X->Z):", model.u)

    # Test prediction with the two-stage model
    preds = model.predict(X_xy)
    print("Predictions (two-stage):", preds[:10])
    print("True labels Y:", Y[:10])
    accuracy = np.mean(preds == Y)
    print(f"Accuracy (two-stage): {accuracy:.2f}")

    # Test RegularizedLogisticRegressionModelFullInformation
    # Need Z corresponding to X_xy for full information model
    # For demonstration, let's predict Z for X_xy using the true intermediate model (or the fitted one)
    Z_xy_true_prob = 1 / (1 + np.exp(-(X_xy @ true_u + 0.2)))
    Z_xy = np.where(np.random.rand(80) < Z_xy_true_prob.flatten(), 1, -1)
    
    sample_full = (X_xy, Y, Z_xy) # Use X_xy, Y, and corresponding Z_xy
    model_full_info = RegularizedLogisticRegressionModelFullInformation(reg, sample_full)
    w_full_info = model_full_info.fit()
    print("\nLearned weights w (full information):", w_full_info)