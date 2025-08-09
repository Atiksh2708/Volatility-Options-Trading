import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

df=pd.read_parquet(r"C:\Users\Lenovo\Volatility-Options-Trading\data\AAPL_20y_returns.parquet")

# drop early NaNs
df = df.dropna()

# 3. Fit GARCH(1,1) on returns (in percent is often used)
returns_pct = df['log_return'] * 100.0
garch = arch_model(returns_pct, p=1, q=0, mean='Constant', dist='normal')
garch_res = garch.fit(disp='off')
# conditional volatility (in percent) -> back to decimal
garch_cond_vol = pd.Series(garch_res.conditional_volatility / 100.0, index=returns_pct.index)

# align & add to df
df = df.assign(garch_vol=garch_cond_vol)
df = df.dropna(subset=['garch_vol', 'rolling_vol'])   # ensure no mismatches

# ---------------------------
# 4. Prepare sequences for LSTM


df['garch_vol_annual'] = df['garch_vol'] * np.sqrt(252)

# Feature matrix and target (predict next-day realized_vol)
features = df[['rolling_vol', 'garch_vol_annual']].copy()
target = df['rolling_vol'].shift(-1)   # predict next day
# drop last NaN in target
features = features.iloc[:-1]
target = target.iloc[:-1]

# scaling features (fit scaler on training later; but for simplicity scale whole dataset here)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# create sequences of length L
L = 100   # lag/window length from paper
def create_sequences(X, y, seq_len=L):
    Xs, ys, idxs = [], [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:(i + seq_len)])
        ys.append(y[i + seq_len])
        idxs.append(features.index[i + seq_len])  # index aligned to the prediction day
    return np.array(Xs), np.array(ys), pd.DatetimeIndex(idxs)

X, y, seq_index = create_sequences(X_scaled, target.values, seq_len=L)
print("Sequence shape:", X.shape, "Target shape:", y.shape)

train_frac = 0.8
split = int(len(X) * train_frac)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
idx_train, idx_val = seq_index[:split], seq_index[split:]

print("Train samples:", len(X_train), "Val samples:", len(X_val))


# 6. Build the LSTM model (2 layers: 32 and 16 units)

tf.keras.backend.clear_session()
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(16, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# callbacks
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# 7. Train

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)
# 8. Predict & Evaluate

y_pred_val = model.predict(X_val).ravel()

# Put into Series with proper index
y_val_s = pd.Series(y_val, index=idx_val)
y_pred_s = pd.Series(y_pred_val, index=idx_val)

# Metrics
mse = mean_squared_error(y_val_s, y_pred_s)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val_s, y_pred_s)
mape = np.mean(np.abs((y_val_s - y_pred_s) / y_val_s))  # avoid zeros in target; handle if zeros appear

print("Validation MSE:", mse)
print("Validation RMSE:", rmse)
print("Validation MAE:", mae)
print("Validation MAPE:", mape)

# 9. Plot results (one example)

plt.figure(figsize=(12,5))
plt.plot(y_val_s.index, y_val_s, label='Actual Realized Vol (val)', alpha=0.8)
plt.plot(y_pred_s.index, y_pred_s, label='Predicted Vol (hybrid LSTM)', alpha=0.8)
plt.legend()
plt.title("Hybrid GARCH + LSTM: Predicted vs Actual (Validation)")
plt.ylabel("Annualized Volatility")
plt.show()

# Plot training loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.yscale('log')
plt.legend()
plt.title("Training/Validation Loss (MSE)")
plt.show()



