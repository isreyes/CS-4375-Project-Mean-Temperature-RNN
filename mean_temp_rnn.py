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
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(df)

# creation of sequences
# using 30 days to predict next day temperature
def build_sequences(data, sequence):
    inputs, targets = [], []
    for start in range(len(data) - sequence):
        end = start + sequence
        inputs.append(data[start:end])
        targets.append(data[end][0])
    return np.array(inputs), np.array(targets)

SEQUENCE = 30
X, y = build_sequences(normalized_data, SEQUENCE)

# train/test split 80/20
split_data = int(len(X) * 0.8)
training_X = X[:split_data]
testing_X = X[split_data:]
training_y = y[:split_data]
testing_y = y[split_data:]

# RNN implementation
class RNN:
    # input, output, hidden layer sizes and learning rate
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        scale = 0.01 # initialize weights
        
        # hidden layer weights
        self.W_input = np.random.randn(hidden_size, input_size) * scale
        self.W_hidden = np.random.randn(hidden_size, hidden_size) * scale
        self.b_hidden = np.zeros((hidden_size, 1))

        # output layer weights
        self.W_output = np.random.randn(output_size, hidden_size) * scale
        self.b_output = np.zeros((output_size, 1))

    # forward pass
    def forward(self, X):
        seq_length = X.shape[0]
        current_hidden = np.zeros((self.hidden_size, 1))
        self.input_cache = []
        self.hidden_state = [current_hidden]

        for time_step in range(seq_length):
            current_input = X[time_step].reshape(-1, 1) 
            current_hidden = np.tanh(self.W_input @ current_input + self.W_hidden @ current_hidden + self.b_hidden)
            self.input_cache.append(current_input)
            self.hidden_state.append(current_hidden)

        # final output prediction
        y_pred = self.W_output @ current_hidden + self.b_output
        return y_pred.flatten()[0]
    
    # backward pass
    def backward(self, y_pred, y_true):
        error = y_pred - y_true

        # output layer gradients
        W_grad_output = error * self.hidden_state[-1].T
        b_grad_output = error

        # backpropagation through time (BPTT)
        W_grad_input = np.zeros(self.W_input.shape)
        W_grad_hidden = np.zeros(self.W_hidden.shape)
        b_grad_hidden = np.zeros(self.b_hidden.shape)
        hidden_grad_next = self.W_output.T * error

        # backpropagate through time steps
        for time_step in reversed(range(len(self.input_cache))):
            # tanh derivative
            dh = hidden_grad_next * (1  -self.hidden_state[time_step+1] ** 2)
            b_grad_hidden += dh
            W_grad_input += dh @ self.input_cache[time_step].T
            W_grad_hidden += dh @ self.hidden_state[time_step].T
            hidden_grad_next = self.W_hidden.T @ dh

        # gradient clipping
        # prevents exploding gradients
        W_grad_input = np.clip(W_grad_input, -1, 1)
        W_grad_hidden = np.clip(W_grad_hidden, -1, 1)
        b_grad_hidden = np.clip(b_grad_hidden, -1, 1)
        W_grad_output = np.clip(W_grad_output, -1, 1)
        b_grad_output = np.clip(b_grad_output, -1, 1)

        # updating weights
        self.W_input -= self.learning_rate * W_grad_input
        self.W_hidden -= self.learning_rate * W_grad_hidden
        self.b_hidden -= self.learning_rate * b_grad_hidden
        self.W_output -= self.learning_rate * W_grad_output
        self.b_output -= self.learning_rate * b_grad_output

    # training
    def train(self, X_train, y_train, epochs=50):
        epoch_losses = []

        for epoch in range(epochs):
            cumulative_loss = 0
            for j in range(len(X_train)):
                y_pred = self.forward(X_train[j]) # forward pass
                sample_loss = (y_pred - y_train[j]) ** 2
                cumulative_loss += sample_loss
                self.backward(y_pred, y_train[j]) # backward pass
            avg_epoch_loss = cumulative_loss / len(X_train)
            epoch_losses.append(avg_epoch_loss)

            if (epoch % 10 == 0):
                print (f"Epoch: {epoch }, Loss: {avg_epoch_loss:.4f}")

        return epoch_losses

    # prediction
    def predict(self, X_test):
        all_predictions = []
        for j in range(len(X_test)):
            y_pred = self.forward(X_test[j])
            all_predictions.append(y_pred)
        return np.array(all_predictions)



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
training_losses = rnn.train(training_X, training_y, epochs=200)

test_predictions = rnn.predict(testing_X)

# calculate train and test rmse
train_predictions = rnn.predict(training_X)
train_mse = mean_squared_error(training_y, train_predictions)
train_rmse = np.sqrt(train_mse)
print(f"Train RMSE: {train_rmse:.4f}")

test_mse = mean_squared_error(testing_y, test_predictions)
test_rmse = np.sqrt(test_mse)
print(f"Test RMSE: {test_rmse:.4f}")

# rmse to celsius
celsius = test_rmse * (df['meantemp'].max() - df['meantemp'].min())
print(f"Test RMSE in Celsius: {celsius:.2f} °C")

# plot loss
plt.figure(figsize=(12, 4))
plt.plot(training_losses)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()

# plot predictions vs actual
plt.figure(figsize=(12, 4))
plt.plot(testing_y, label='Actual')
plt.plot(test_predictions, label='Predicted')
plt.title('Predicted vs Actual Temperature')
plt.xlabel('Day')
plt.ylabel('Normalized Temperature')
plt.legend()
plt.show()

print(f"Training Samples: {len(training_X)}")
print(f"Test Samples: {len(testing_X)}")
print(f"Input Shape: {training_X.shape}")