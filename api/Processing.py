import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Veri yükleme
data = pd.read_csv('Csvler/Turkey.csv')

# 'entity' ve 'code' kolonlarını düşürme
data = data.drop(columns=['Entity', 'Code'])

# Yıl sıralaması
data = data.sort_values(by='Year')

# Numpy array'e dönüştürme
years = data['Year'].values.reshape(-1, 1)
population = data['Population'].values.reshape(-1, 1)

# Veriyi ölçeklendirme
scaler_year = MinMaxScaler()
scaler_population = MinMaxScaler()

years_scaled = scaler_year.fit_transform(years)
population_scaled = scaler_population.fit_transform(population)

# PyTorch tensörlerine dönüştürme
years_tensor = torch.tensor(years_scaled, dtype=torch.float32)
population_tensor = torch.tensor(population_scaled, dtype=torch.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(years_tensor, population_tensor, test_size=0.2, random_state=42)

# LSTM modeli tanımlama
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 2000
learning_rate = 0.01

model = LSTM(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Modeli eğitme
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train.view(-1, 1, 1))
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Modeli test etme ve tahmin yapma
model.eval()
predicted = model(X_test.view(-1, 1, 1)).detach().numpy()
predicted = scaler_population.inverse_transform(predicted)

# Gerçek ve tahmin edilen değerleri görselleştirme
y_test_actual = scaler_population.inverse_transform(y_test.detach().numpy())

plt.plot(years, population, label='Gerçek Nüfus')
plt.scatter(X_test.view(-1).detach().numpy(), predicted, color='red', label='Tahmin Edilen Nüfus')
plt.xlabel('Yıl')
plt.ylabel('Nüfus')
plt.legend()
plt.show()

# Son yıldan 10 yıl sonrası için tahmin yapma
last_year = data['Year'].max()
future_years = np.array([[last_year + i] for i in range(1, 11)])
future_years_scaled = scaler_year.transform(future_years)
future_years_tensor = torch.tensor(future_years_scaled, dtype=torch.float32)

model.eval()
future_predictions = model(future_years_tensor.view(-1, 1, 1)).detach().numpy()
future_predictions = scaler_population.inverse_transform(future_predictions)

for year, pred in zip(future_years.flatten(), future_predictions.flatten()):
    print(f'Yıl: {year}, Tahmin Edilen Nüfus: {pred:.0f}')
