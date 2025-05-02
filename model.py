import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

class RegularizedLogisticRegressionModel:
    def __init__(self, reg, sample_xz, sample_xy):
        assert reg >= 0, "Regularization value must be non-negative"
        self.reg = reg
        self.sample_xz = sample_xz  # tuple (X_xz, Z)
        self.sample_xy = sample_xy  # tuple (X_xy, Y)
        self.u = None  # weights for X→Z model
        self.w = None  # weights for final model
        self.intermediate_model = None

    def fit(self):
        X_xz, Z = self.sample_xz
        X_xy, Y = self.sample_xy

        # Step 1: Fit logistic regression to predict Z from X
        clf = LogisticRegression(fit_intercept=True, solver='lbfgs')
        clf.fit(X_xz, Z)
        u_c = clf.intercept_[0]
        u_x = clf.coef_[0]
        self.u = (u_c, u_x)
        self.intermediate_model = clf

        # Step 2: Use learned model to predict Z for X in (X, Y) data
        Z_pred_proba = clf.predict_proba(X_xy)[:, 1]  # probability of Z=1
        Z_pred = np.where(Z_pred_proba >= 0.5, 1, -1)  # convert to +1/-1

        # Step 3: Prepare features for final model: [1, X, Z_pred]
        n_samples = X_xy.shape[0]
        X_aug = np.hstack([np.ones((n_samples, 1)), X_xy, Z_pred.reshape(-1, 1)])

        # Step 4: Initialize weights (w_c, w_x, w_z) to zero
        n_features = X_aug.shape[1]
        w0 = np.zeros(n_features)

        # Step 5: Define logistic loss + reg*|w_z|^2
        def loss_fn(w):
            logits = X_aug @ w
            logistic_loss = np.mean(np.log1p(np.exp(-Y * logits)))  
            reg_term = self.reg * (w[-1] ** 2)  
            return logistic_loss + reg_term

        # Step 6: Minimize the loss
        res = minimize(loss_fn, w0, method='BFGS')
        self.w = res.x
        return {
            'model': self,
            'intermediate_model': clf,
            'w': self.w,
            'u': (u_c, u_x)
        }

    def predict(self, X, Z):
        if self.w is None:
            raise ValueError("Model must be fit before prediction")
        """
        Predict Y from X and Z using the learned model.
        X: shape (n_samples, n_features)
        Z: shape (n_samples,) or (n_samples, 1)
        Returns: +1/-1 labels
        """
        n_samples = X.shape[0]
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        X_aug = np.hstack([np.ones((n_samples, 1)), X, Z])
        logits = X_aug @ self.w
        proba = 1 / (1 + np.exp(-logits))
        return np.where(proba >= 0.5, 1, -1)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    # Generate synthetic data
    X_xz = np.random.randn(100, 2)
    true_u = np.array([0.5, -1.0])
    Z = (np.random.rand(100) < 1 / (1 + np.exp(-(X_xz @ true_u + 0.2)))).astype(int)

    X_xy = np.random.randn(80, 2)
    Y = (np.random.rand(80) < 1 / (1 + np.exp(-(X_xy @ np.array([1.0, -0.5]) + 0.1)))).astype(int)

    reg = 0.1
    model = RegularizedLogisticRegressionModel(reg, (X_xz, Z), (X_xy, Y))
    results = model.fit()
    print("Learned weights w:", results['w'])
    print("Intermediate model weights u:", results['u'])

    # Test prediction
    Z_pred_test = model.intermediate_model.predict_proba(X_xy)[:, 1]
    Z_pred_test = np.where(Z_pred_test >= 0.5, 1, -1)
    preds = model.predict(X_xy, Z_pred_test)
    print("Predictions:", preds[:10])
    print("True labels:", Y[:10])