# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often
struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships 
by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear
layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model

<img width="703" height="592" alt="image" src="https://github.com/user-attachments/assets/6bd3857f-3a98-4f65-bd9c-e1a1504eb59f" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: V M KAVIYA

### Register Number: 212224040154

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv("/content/dataset.csv")
X = dataset1[['Input']].values
y = dataset1[['Output']].values

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Name:V M KAVIYA
# Register Number:212224040154
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)

# Name:
# Register Number:
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()


        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information

<img width="180" height="263" alt="image" src="https://github.com/user-attachments/assets/08737c58-57e8-4193-8890-84f582f9103b" />


### OUTPUT
#### i.)
<img width="362" height="452" alt="image" src="https://github.com/user-attachments/assets/70173a19-42d0-4404-bbd9-25f36cb446ed" />

#### ii.)
<img width="427" height="232" alt="image" src="https://github.com/user-attachments/assets/3a08b616-9721-4440-a831-a2d1d8a22ca1" />

#### iii.)
<img width="725" height="31" alt="image" src="https://github.com/user-attachments/assets/6a967706-32f3-4deb-b81b-e998a87b9b13" />



### Training Loss Vs Iteration Plot


<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/6ab6c60f-2627-4852-b806-0f0bc4fabf24" />


### New Sample Data Prediction


<img width="676" height="32" alt="image" src="https://github.com/user-attachments/assets/6bb5887a-8e01-40e0-894e-7d8cd739a831" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
