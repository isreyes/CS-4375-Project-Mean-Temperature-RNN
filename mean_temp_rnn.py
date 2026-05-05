import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# load data
df = pd.read_csv('DailyDelhiClimateTrain.csv')

# filter out unrealistic pressure values
df = df[(df['meanpressure'] > 900) & (df['meanpressure'] < 1100)]

df = df.drop(columns=['date'])

# normalize data to [0,1] range
# Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# creation of sequences
# using 30 days to predict next day temperature
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])
    return np.array(X), np.array(y)

SEQ_LENGTH = 30
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# train/test split 80/20
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# RNN implementation
class RNN:
    # input, output, hidden layer sizes and learning rate
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        scale = 0.01 # initialize weights
        
        # hidden layer weights
        self.Wx = np.random.randn(hidden_size, input_size) * scale
        self.Wh = np.random.randn(hidden_size, hidden_size) * scale
        self.bh = np.zeros((hidden_size, 1))

        # output layer weights
        self.Wy = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))

    # forward pass
    def forward(self, X):
        seq_length = X.shape[0]
        h = np.zeros((self.hidden_size, 1)) # initial hidden state
        self.inputs, self.hiddens = [], [h]

        for t in range(seq_length):
            xt = X[t].reshape(-1, 1) 
            h = np.tanh(self.Wx @ xt + self.Wh @ h + self.bh)
            self.inputs.append(xt)
            self.hiddens.append(h)
        y_pred = self.Wy @ h + self.by
        return y_pred.flatten()[0]
    
    # backward pass
    def backward(self, y_pred, y_true):
        error = y_pred - y_true

        # output layer gradients
        dWy = error * self.hiddens[-1].T
        dby = error

        # backpropagation through time (BPTT)
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        dh_next = self.Wy.T * error
        
        for t in reversed(range(len(self.inputs))):
            dh = dh_next * (1  -self.hiddens[t+1] ** 2)
            dbh += dh
            dWx += dh @ self.inputs[t].T
            dWh += dh @ self.hiddens[t].T
            dh_next = self.Wh.T @ dh

        # gradient clipping
        # prevents exploding gradients
        dWx = np.clip(dWx, -1, 1)
        dWh = np.clip(dWh, -1, 1)
        dbh = np.clip(dbh, -1, 1)
        dWy = np.clip(dWy, -1, 1)
        dby = np.clip(dby, -1, 1)

        # updating weights
        self.Wx -= self.learning_rate * dWx
        self.Wh -= self.learning_rate * dWh
        self.bh -= self.learning_rate * dbh
        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

    # training
    def train(self, X_train, y_train, epochs=50):
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            for j in range(len(X_train)):
                y_pred = self.forward(X_train[j])
                loss = (y_pred - y_train[j]) ** 2
                total_loss += loss
                self.backward(y_pred, y_train[j])
            avg_loss = total_loss / len(X_train)
            losses.append(avg_loss)

            if (epoch % 10 == 0):
                print (f"Epoch: {epoch }, Loss: {avg_loss:.4f}")

        return losses
        
    # prediction
    def predict(self, X_test):
        predictions = []
        for j in range(len(X_test)):
            y_pred = self.forward(X_test[j])
            predictions.append(y_pred)
        return np.array(predictions)



# expirement 1
#rnn = RNN(input_size=4, hidden_size=32, output_size=1, learning_rate=0.001)
#expirement 2
#rnn = RNN(input_size=4, hidden_size=64, output_size=1, learning_rate=0.001)
#expirement 3
rnn = RNN(input_size=4, hidden_size=64, output_size=1, learning_rate=0.005)

# expirement 1
#losses = rnn.train(X_train, y_train, epochs=50)
# expirement 2
#losses = rnn.train(X_train, y_train, epochs=100)
# expirement 3
losses = rnn.train(X_train, y_train, epochs=200)

predictions = rnn.predict(X_test)

# calculate train and test rmse
train_predictions = rnn.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
train_rmse = np.sqrt(train_mse)
print(f"Train RMSE: {train_rmse:.4f}")

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.4f}")



# plot loss
plt.figure(figsize=(12, 4))
plt.plot(losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()

# plot predictions vs actual
plt.figure(figsize=(12, 4))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Predicted vs Actual Temperature')
plt.xlabel('Day')
plt.ylabel('Normalized Temperature')
plt.legend()
plt.show()

print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print(f"Input Shape: {X_train.shape}")

# rmse to celsius
celsius = rmse * (df['meanpressure'].max() - df['meanpressure'].min())
print(f"Test RMSE in Celsius: {celsius:.2f} °C")
