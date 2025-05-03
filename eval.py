import numpy as np
from data import JointDistribution
from training import trained_models, benchmark_models  # assumes dictionaries are created at import
from param import t, sample_run, k_values, reg_values, min_reg
risk = {}
upperbound = {}

for k in k_values:
    print(f"Evaluating for k={k}...")
    
    for reg in reg_values:
        #print(f"  Evaluating for reg={reg}...")
        diffs = []
        expr = []
        for _ in range(sample_run):
            # Sample data once per sample run
            dist = JointDistribution(k)
            X, Y, Z = dist.sample(size=t)
            XZ = np.column_stack([X, Z])

            # Lower Benchmark model (X, Z) -> Y
            lower_bench_model = benchmark_models[(k, reg_values[0])][0]
            lower_bench_logits = lower_bench_model.intercept_ + lower_bench_model.coef_[0, 0] * X.flatten() + lower_bench_model.coef_[0, 1] * Z.flatten()
            Rwstar = np.mean(np.log(1 + np.exp(-Y * lower_bench_logits)))

            # Upper Benchmark model X -> Y
            upper_bench_model = benchmark_models[(k, reg_values[0])][1]
            upper_bench_logits = upper_bench_model.intercept_ + upper_bench_model.coef_[0, 0] * X.flatten()
            Lvstar = np.mean(np.log(1 + np.exp(-Y * upper_bench_logits)))

            # Trained model X -> Z, (X,(X -> Z)) -> Y
            trained_model = trained_models[(k, reg)]
            n_samples = X.shape[0]
            X_aug = np.hstack([np.ones((n_samples, 1)), X.reshape(-1,1), Z.reshape(-1, 1)])
            logits = X_aug @ trained_model.w  
            trained_loss = np.mean(np.log(1 + np.exp(-Y * logits)))

            # Assuming comparison with lower benchmark model
            diffs.append(trained_loss - Rwstar)

            # Trained intermediate model X -> Z
            intermediate_trained_model = trained_models[(k, reg)].intermediate_model
            Z_pred = intermediate_trained_model.predict(X.reshape(-1, 1))
            Rustar = np.mean(Z_pred != Z.flatten())

            #w^*, w_reg^*, w_{phi^*,reg}^*
            wstar = np.hstack([lower_bench_model.intercept_, lower_bench_model.coef_.flatten()])
            wphistarstar = trained_models[(k, min_reg)].w
            wregstar = benchmark_models[(k, reg)][-1].w

            expr.append(min(reg*(wstar[-1])**2, Lvstar - Rwstar + reg*(wphistarstar[-1])**2) + min(np.log(2)/reg, max(np.linalg.norm(wregstar), np.linalg.norm(wstar)))*Rustar)

        risk[(k, reg)] = np.mean(diffs)
        upperbound[(k, reg)] = np.mean(expr)


