import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.mlls import VariationalELBO

# Beispiel Zeitreihendaten erstellen
n_samples = 100
time_steps = 10
n_features = 1

# Dummy-Daten: Sinuswelle
x = torch.linspace(0, 8 * torch.pi, n_samples).unsqueeze(-1)
y = torch.sin(x) + 0.1 * torch.randn_like(x)

# Dataset erstellen
dataset = TensorDataset(x[:-time_steps], y[time_steps:])
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# SNGPModule-Output Layer
class SNGPLayer(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super(SNGPLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=n_features)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Modell zusammenstellen
class SNGPModel(nn.Module):
    def __init__(self, inducing_points):
        super(SNGPModel, self).__init__()
        self.nn = SimpleNN()
        self.gp = SNGPLayer(inducing_points)

    def forward(self, x):
        features = self.nn(x)
        return self.gp(features)


# Inducing Points (müssen innerhalb der Trainingsdaten liegen)
inducing_points = x[:20].clone()

# Modell initialisieren
model = SNGPModel(inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Optimizer und Verlustfunktion
optimizer = optim.Adam(model.parameters(), lr=0.01)
mll = VariationalELBO(likelihood, model.gp, num_data=len(y[time_steps:]))

# Training
model.train()
likelihood.train()

num_epochs = 100

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = -mll(output, batch_y).sum()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluation
model.eval()
likelihood.eval()

test_x = torch.linspace(8 * torch.pi, 10 * torch.pi, n_samples).unsqueeze(-1)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(model(test_x))

# Ergebnisse anzeigen
import matplotlib.pyplot as plt

mean = preds.mean
lower, upper = preds.confidence_region()

plt.plot(test_x.numpy(), mean.numpy(), 'b')
plt.fill_between(test_x.numpy().squeeze(), lower.numpy(), upper.numpy(), alpha=0.5)
plt.show()
