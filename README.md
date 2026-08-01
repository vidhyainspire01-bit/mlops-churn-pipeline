# mlops-churn-pipeline
Production-grade MLOps pipeline: Databricks Asset Bundles, Unity Catalog, MLflow aliases, and Champion/Challenger promotion


databricks bundle validate --target dev
databricks bundle deploy --target dev

databricks bundle run churn_training_job --target dev

## Rebuild wheel
rm -rf dist/ build/ src/*.egg-info
python -m build --wheel

# Deploy + run
databricks bundle deploy --target dev
databricks bundle run churn_training_job --target dev


How to test it locally 

install:
pip install -e .
pip list | grep churn
local AUC TESTING
python3 << 'EOF'
from churn_model.features.synthetic_data import generate_churn_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

df = generate_churn_dataset(n_samples=10000, random_state=42)
df = pd.get_dummies(df, columns=["segment", "contract_type"], drop_first=True)
X = df.drop(columns=["customer_id", "churned"])
y = df["churned"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="logloss")
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba)
print(f"Local AUC: {auc:.4f}")
print(f"Churn rate: {y.mean():.2%}")
EOF


Phase -2:
Run unit test locally 
pip install pytest pytest-cov


when we have a challenger to champion promotion ,keep changes in dev.yaml and run the below steps
rm -rf dist/ build/ src/*.egg-info
python -m build --wheel
databricks bundle deploy --target dev
databricks bundle run churn_training_job --target dev
databricks bundle run churn_promotion_job --target dev

Phase 3: CI/CD	- GitHub Actions + OIDC federation + PR quality gates
Phase 4: Serving -	Databricks Model Serving endpoint (simplified)
Phase 5: Drift Monitoring - PSI drift job writing to Delta
Phase 6: Docs + Demo - README, architecture diagram, demo script

Phase3:
create microsoft etra id to connect github action to deploy the azure resource securely for that we need to connect github organization name and id

# Get GitHub user/org ID (works for personal accounts too)
curl -s https://api.github.com/users/vidhyainspire01-bit | grep '"id"' | head -1
eg:  "id": 223743781

# Get repository ID
curl -s https://api.github.com/repos/vidhyainspire01-bit/mlops-churn-pipeline | grep '"id"' | head -1
eg: 1317610814
Now we have 3 federated credentials: main, develop, pull_request. Each maps to a different scenario in the CI/CD flow.
Step 3.3 — Get the Databricks Service Principal ID # CI test
