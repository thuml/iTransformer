from SNGP import SNGP
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np



class NeuralNetworkSNGP(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNetworkSNGP, self).__init__()
    # Define hidden layers (replace with your desired architecture)
    self.fc1 = nn.utils.spectral_norm(nn.Linear(input_size, hidden_size))
    self.relu = nn.ReLU()
    self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size))
    # SNGPModule layer with relevant parameters
    self.sngp = SNGP(hidden_size, num_classes, 0.1, 1.0, 1.0, 1.0, 5, torch.device("cpu"))

  def forward(self, x):
    # Pass through hidden layers
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    # Pass to SNGPModule layer for prediction
    logits = self.sngp(x)
    return logits




import pandas as pd

data = {
    'evse_id': ['0', '1', '2', '3', '4'],
    'connection_time': [
        '2023-06-01 00:05:00',
        '2023-06-01 00:30:00',
        '2023-06-01 00:45:00',
        '2023-06-01 01:10:00',
        '2023-06-01 01:35:00'
    ],
    'disconnection_time': [
        '2023-06-01 00:20:00',
        '2023-06-01 00:45:00',
        '2023-06-01 01:00:00',
        '2023-06-01 01:30:00',
        '2023-06-01 01:50:00'
    ]
}

df = pd.DataFrame(data)
df['connection_time'] = pd.to_datetime(df['connection_time'])
df['disconnection_time'] = pd.to_datetime(df['disconnection_time'])

start_time = df['connection_time'].min().floor('15T')
end_time = df['disconnection_time'].max().ceil('15T')
time_intervals = pd.date_range(start=start_time, end=end_time, freq='15T')

evse_ids = df['evse_id'].unique()
time_index = pd.MultiIndex.from_product([evse_ids, time_intervals], names=['evse_id', 'time_interval'])

occupancy_df = pd.DataFrame(index=time_index).reset_index()

def determine_occupancy(evse_id, interval_start, interval_end):
    is_occupied = ((df['evse_id'] == evse_id) &
                   (df['connection_time'] < interval_end) &
                   (df['disconnection_time'] > interval_start)).any()
    return is_occupied

occupancy_df['occupancy'] = occupancy_df.apply(
    lambda row: determine_occupancy(row['evse_id'], row['time_interval'], row['time_interval'] + pd.Timedelta('15 minutes')),
    axis=1
)

occupancy_df['site_id'] = occupancy_df['evse_id'].str[0]

occupancy_df['hour'] = occupancy_df['time_interval'].dt.hour
occupancy_df['day'] = occupancy_df['time_interval'].dt.day
occupancy_df['month'] = occupancy_df['time_interval'].dt.month
occupancy_df['year'] = occupancy_df['time_interval'].dt.year

pivot_df = occupancy_df.pivot_table(index=['time_interval', 'hour', 'day', 'month', 'year'], columns='evse_id', values='occupancy').reset_index()
pivot_df.columns.name = None
pivot_df = pivot_df.fillna(False)
print(pivot_df)

num_of_targets= 8
num_of_inputs = 4
def generate_data(num_examples):

    # Synthetische Daten erstellen
    data = []
    for _ in range(num_examples):
        hours = np.random.randint(0, 24, size=4)  # Stunden (0-23)
        days = np.random.randint(1, 32, size=4)   # Tage (1-31)
        months = np.random.randint(1, 13, size=4) # Monate (1-12)
        years = np.full(4, 2023)                  # Jahr (fest auf 2023 gesetzt)

        example = [[hour, day, month, year] for hour, day, month, year in zip(hours, days, months, years)]
        data.append(example)

    print("Generated synthetic data:")
    for example in data:
        print(example)
    return data


data = generate_data(10)

data = torch.tensor(data, dtype=torch.float32)
targets = torch.randint(0, num_of_targets, size=(data.size(0),))


# TensorDataset und DataLoader erstellen
dataset = TensorDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# Define the model with appropriate input size
model = NeuralNetworkSNGP(num_of_inputs, 64, num_of_targets)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
  for i, (timeseries, labels) in enumerate(dataloader):
    optimizer.zero_grad()
    logits = model(timeseries)

    labels = F.one_hot(labels, num_classes=num_of_targets)
    labels = labels.view(labels.size(0), num_of_targets)  # Reshape to [batch_size, num_classes]
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    # Print training progress (optional)
    if i % 50 == 0:
      print(f"Epoch: {epoch+1}, Step: {i}, Loss: {loss.item():.4f}")

new_data = generate_data(2)
new_data = torch.tensor(new_data, dtype=torch.float32)


prediction = model(new_data)
sigmoid_probs = torch.sigmoid(prediction)  

print(sigmoid_probs)
predicted_class = sigmoid_probs.argmax(dim=1)
print(f"Predicted class for new data:\n {predicted_class}")


