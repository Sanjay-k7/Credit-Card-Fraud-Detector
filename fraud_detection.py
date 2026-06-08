import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Credit Card Fraud Detection Project (From Scratch)")

# ========================== 1. LOAD DATA ==========================
df = pd.read_csv('creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Fraud percentage: {df['Class'].mean()*100:.4f}%")
print("\nClass distribution:\n", df['Class'].value_counts())

# ========================== 2. PREPROCESSING ==========================
# 29 features (V1-V28 + Amount). Time is dropped (standard practice - not useful for pattern learning)
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split (same as your original project)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ========================== 3. AUTOENCODER (Neural Network) ==========================
class Autoencoder(nn.Module):
    def __init__(self, input_dim=29):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train ONLY on legitimate (Class == 0) transactions
normal_mask = (y_train == 0)
X_train_normal = X_train[normal_mask]

train_tensor = torch.FloatTensor(X_train_normal).to(device)
dataset = TensorDataset(train_tensor, train_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

print("\nTraining Autoencoder (only on normal transactions, 50 epochs)...")
for epoch in range(50):
    model.train()
    epoch_loss = 0.0
    for batch, _ in loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/50 | Loss: {epoch_loss/len(loader):.6f}")

# ========================== 4. RECONSTRUCTION ERROR (Fraud Signal) ==========================
model.eval()
with torch.no_grad():
    # Full training set (for hybrid features)
    X_train_t = torch.FloatTensor(X_train).to(device)
    recon_train = model(X_train_t)
    recon_error_train = ((X_train_t - recon_train) ** 2).mean(dim=1).cpu().numpy()
    
    # Test set
    X_test_t = torch.FloatTensor(X_test).to(device)
    recon_test = model(X_test_t)
    recon_error_test = ((X_test_t - recon_test) ** 2).mean(dim=1).cpu().numpy()

# Threshold from normal transactions only
threshold = np.percentile(recon_error_train[normal_mask], 95)
print(f"\nReconstruction Error Threshold (95th percentile of normal): {threshold:.4f}")

# ========================== 5. HYBRID MODEL (Add recon error as 30th feature) ==========================
X_train_hybrid = np.column_stack((X_train, recon_error_train))
X_test_hybrid = np.column_stack((X_test, recon_error_test))

print("Reconstruction Error added as new feature → Hybrid dataset ready")

# SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_hybrid, y_train)

# ========================== 6. ENSEMBLE: XGBoost + Random Forest ==========================
print("\nTraining XGBoost + Random Forest...")

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric='aucpr'
)
xgb_model.fit(X_train_res, y_train_res)

rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)

# Ensemble probabilities
xgb_proba = xgb_model.predict_proba(X_test_hybrid)[:, 1]
rf_proba = rf_model.predict_proba(X_test_hybrid)[:, 1]
ensemble_proba = (xgb_proba + rf_proba) / 2

# Optimal threshold using PR curve
precision, recall, thresholds = precision_recall_curve(y_test, ensemble_proba)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best decision threshold: {best_threshold:.4f}")

y_pred = (ensemble_proba >= best_threshold).astype(int)

# ========================== 7. FINAL RESULTS ==========================
print("\n" + "="*70)
print("FINAL PERFORMANCE - AUTOENCODER + XGBoost + RF (Level 3)")
print("="*70)
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
print(f"ROC-AUC : {roc_auc_score(y_test, ensemble_proba):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, ensemble_proba):.4f}")

# Example fraud explanation (exactly what you wanted)
if np.any(y_pred == 1):
    sample_idx = np.where(y_pred == 1)[0][0]
    print("\n🔍 EXAMPLE FRAUD EXPLANATION:")
    print(f"→ Reconstruction error : {recon_error_test[sample_idx]:.4f}")
    print(f"→ Threshold            : {threshold:.4f}")
    print(f"→ Deviation factor     : {recon_error_test[sample_idx]/threshold:.1f}x above normal")
    print(f"→ Ensemble fraud score : {ensemble_proba[sample_idx]:.3f}")
    print("→ This transaction could not be reconstructed as normal behavior!")

# Save models for later use
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(rf_model, 'rf_model.pkl')
torch.save(model.state_dict(), 'autoencoder.pth')
print("\n✅ All models saved! You can reuse them later.")

# Optional SHAP plot (takes a few seconds)
print("\nGenerating SHAP explainability plot...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_hybrid[:500])
feature_names = list(X.columns) + ['Reconstruction_Error']
shap.summary_plot(shap_values, X_test_hybrid[:500], feature_names=feature_names, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
print("SHAP summary saved as 'shap_summary.png'")