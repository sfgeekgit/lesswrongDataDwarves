import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# WIP. the code runs, but results aren't great.
# ###!! Autocomplete of this comment said the following: The model is overfitting the training data.
# so... that's like, one AI's opinion....


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{device=}")

seed_value = 777

# Set seeds to ensure reproducibility
torch.manual_seed(seed_value)



# Load the dataset
df = pd.read_csv('dwarves_formated.csv')

# Drop the columns not used for training
df = df.drop(['ID', 'FortSurvived'], axis=1)

# Separate features and target variable
X = df.drop('FortValue', axis=1)
y = df['FortValue']

# Convert boolean columns to integers
bool_cols = [col for col in X.columns if X[col].dtype == 'bool']
X[bool_cols] = X[bool_cols].astype(int)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).view(-1, 1).to(device)

#X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
#y_train_tensor = torch.tensor(y_train.values, dtype=torch.float)
#X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float)
#y_test_tensor = torch.tensor(y_test.values, dtype=torch.float)



# Create TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor.view(-1, 1))
test_dataset = TensorDataset(X_test_tensor, y_test_tensor.view(-1, 1))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        # input size is 19 for this dataset
        #print("Create Model. input_size: ", input_size)
        super(MLP, self).__init__()
        hidden = 32
        dropout = 0.2
        self.layers = nn.Sequential(
        nn.Linear(input_size, hidden),            
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        #  add more layers maybe?
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize the model, loss function, and optimizer
##model = MLP(X_train_tensor.shape[1])
model = MLP(X_train_tensor.shape[1]).to(device)


criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 1000
for epoch in range(epochs):
    model.train()

    for data, targets in train_loader:

        data, targets = data.to(device), targets.to(device) 


        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Add code here to evaluate the model on the test set and make predictions
