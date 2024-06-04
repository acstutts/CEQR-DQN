import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn.functional as F
from utils import gamma_cal_loss
from quantilelosses import loss_evi

output_size = 1
num_quantiles = 50
evi_coeff = 1e-8

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(1, 50),  # 2 input features
            nn.PReLU(),
            nn.Linear(50, 50),
            nn.PReLU(),
            nn.Linear(50, output_size) # 1 output value
        )

        self.NIG = nn.Sequential(
            nn.Linear(1, 400),  # 2 input features
            nn.PReLU(),
            nn.Linear(400, 400),
            nn.PReLU(),
            nn.Linear(400, 4*output_size*2) # 1 output value
        )

        init_modules = [self.output, self.NIG]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def evi_split(self, out):
        mu, logv, logalpha, logbeta = torch.split(out, output_size*2, dim=-1)

        v = F.softplus(logv)
        alpha = F.softplus(logalpha) + 1
        beta = F.softplus(logbeta)
        return torch.concat([mu, v, alpha, beta], axis=-1)

    def forward(self, x):
        out = self.output(x)
        G_evi = self.NIG(x)
        G_evi = self.evi_split(G_evi)

        return out, G_evi

model = SimpleNN()

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output1, output2 = model(data)
        gamma, v, alpha, beta = torch.split(output2, output_size*2, dim=-1)
        gamma = gamma.view(data.shape[0], 1, 2).squeeze()
        v = v.view(data.shape[0], 1, 2).squeeze()
        alpha = alpha.view(data.shape[0], 1, 2).squeeze()
        beta = beta.view(data.shape[0], 1, 2).squeeze()
        mse_loss = criterion(output1, target)
        loss_evidence = loss_evi(target, gamma, v, alpha, beta, evi_coeff)
        loss_gamma_cal = gamma_cal_loss(gamma,target,0.5,device)
        loss = mse_loss + loss_evidence + loss_gamma_cal
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output1, output2 = model(data)
            gamma, v, alpha, beta = torch.split(output2, output_size*2, dim=-1)
            gamma = gamma.view(data.shape[0], 1, 2).squeeze()
            v = v.view(data.shape[0], 1, 2).squeeze()
            alpha = alpha.view(data.shape[0], 1, 2).squeeze()
            beta = beta.view(data.shape[0], 1, 2).squeeze()
            mse_loss = criterion(output1, target)
            loss_evidence = loss_evi(target, gamma, v, alpha, beta, evi_coeff)
            loss_gamma_cal = gamma_cal_loss(gamma,target,0.5,device)
            loss = mse_loss + loss_evidence + loss_gamma_cal
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

# Set random seed for reproducibility
set_seeds(3)

# Function definition
def complex_nonlinear_function(x):
    return np.sin(3 * x) * np.cos(2 * x) + 0.5 * np.exp(-x**2) + x**2 - 0.1 * x

# Generate dataset
num_samples = 10000
x = np.random.uniform(-3, 3, num_samples)

# Apply the function to each (x, y) pair
f_x = complex_nonlinear_function(x)

# Adding random noise
noise = np.random.normal(loc=0, scale= 1.5 * np.exp(-0.4 * np.abs(x)))
f_x_noisy = f_x + noise

# Create a DataFrame
df = pd.DataFrame({
    'x': x,
    'f_x_noisy': f_x_noisy
})

df_sorted = df.sort_values(by='x')

print(df.head())

#----------------- train model

# Convert the DataFrame to PyTorch tensors
X = torch.tensor(df[['x']].values, dtype=torch.float32)
y = torch.tensor(df['f_x_noisy'].values, dtype=torch.float32).view(-1, 1)

# Split the dataset into training and testing
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(TensorDataset(X,y), [train_size, test_size])

# Create Data Loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss = test(model, test_loader, criterion, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}")

# ------------- evaluate model statistics on test set split from training set

model.eval()  # Set the model to evaluation mode
predictions = []
actuals = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output1,outpu2 = model(data)
        predictions.extend(output1.view(-1).tolist())
        actuals.extend(target.view(-1).tolist())

# Calculate metrics
mse = mean_squared_error(actuals, predictions)
r2 = r2_score(actuals, predictions)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(actuals, predictions, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted Values (MSE: {mse:.2f}, R2: {r2:.2f})')
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r')  # Diagonal line
plt.show()

#--------------------- test model on fresh set

# Generate dataset
num_samples = 1000
x = np.random.uniform(-6, 6, num_samples)

# Apply the function to each (x, y) pair
f_x = complex_nonlinear_function(x)

# Adding random noise
# Generate y values with Gaussian noise (higher around 0, lower away from 0)
noise = np.random.normal(loc=0, scale= 1.5 * np.exp(-0.4 * np.abs(x)))
f_x_noisy = f_x + noise

# Create a DataFrame
df = pd.DataFrame({
    'x': x,
    'f_x_noisy': f_x_noisy
})

df_sorted = df.sort_values(by='x')

data_tensor = torch.tensor(df_sorted[['x']].values, dtype=torch.float32).to(device)

# Use the model to predict values
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predicted_values, output2 = model(data_tensor)
    predicted_values = predicted_values.cpu().numpy()

gamma, v, alpha, beta = torch.split(output2, output_size*2, dim=-1)
gamma = gamma.cpu().numpy()
v = v.cpu().numpy()
alpha = alpha.cpu().numpy()
beta = beta.cpu().numpy()

# Flatten the predicted values for plotting
predicted_values = predicted_values.flatten()

x_test = df_sorted['x']
y_test = df_sorted['f_x_noisy']

var = np.sqrt((beta /(v*(alpha - 1))))

xplot = np.arange(len(x_test))

avelength1 = abs(np.round(np.mean(gamma[:,1] - gamma[:,0]),4))
count = np.sum((y_test > np.minimum(gamma[:,0], gamma[:,1])) & (y_test < np.maximum(gamma[:,0], gamma[:,1])))

bin_edges = np.linspace(df_sorted['x'].min(), df_sorted['x'].max(), 100)  # Adjust the number of bins as needed
binned = pd.cut(df_sorted['x'], bins=bin_edges, include_lowest=True)
quantiles = df_sorted.groupby(binned)['f_x_noisy'].quantile([0.05, 0.95]).unstack()
# Calculate midpoints of the bins
midpoints = [(interval.left + interval.right) / 2 for interval in quantiles.index.categories]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Helvetica']
plt.rcParams.update({'font.size': 16})
fig = plt.figure(figsize=(6, 6))

x_groundtruth = plt.scatter([x_test],[y_test],c='k',marker='o',s=2,label='Ground Truth')
upperquantile_plot = plt.plot(x_test,gamma[:,1],'b',linewidth=3,label='Upper Quantile (95%)')
lowerquantile_plot = plt.plot(x_test,gamma[:,0],'r',linewidth=3,label='Lower Quantile (5%)')
shade1= plt.axvspan(min(x_test), -3, color='gray', alpha=0.5,label='OOD')
fill = plt.fill_between(x_test,gamma[:,0],gamma[:,1],color='green',alpha=0.4,label='Average Al. Unc.: '+str(avelength1)+'\nMarginal Coverage: '+str(round((count/1000)*100,2))+'%')
xline1 = plt.axvline(-3,color='gray')
shade2=plt.axvspan(3, max(x_test), color='gray', alpha=0.5)
xline2 = plt.axvline(3,color='gray')
ep2 =plt.fill_between(x_test, gamma[:,1]+var[:,1], gamma[:,1]-var[:,1], color='blue', alpha=0.1, label='Upper Quantile Ep. Unc.')
ep1 =plt.fill_between(x_test, gamma[:,0]+var[:,0], gamma[:,0]-var[:,0], color='red', alpha=0.1,label='Lower Quantile Ep. Unc.')

plt.ylim(-5, 20)
plt.xlim(-6,6)
plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.075), ncol=2,prop={'size':12})
plt.tight_layout()  # Adjusts the subplots to fit into the figure area.
plt.savefig('./results/uncertainty_example.png',dpi=1200,bbox_inches='tight')
plt.show()
