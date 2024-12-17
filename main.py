import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# Part1: Load data
url = "https://drive.google.com/uc?id=1WwId5tysI_-rgYaco70qWHroWse-wwU2"
df = pd.read_csv(url)

# ---------------------------------------------------------------------------
# Part2: data pre-processing
print(df.shape)     # Check shape of data set
print(df.columns)   # Check columns of data set
null = df.isnull().sum()    # Check missing value (Null data) : no null data
print(null)
print('\n')

# Step2-1: Feature Selection
all_features = ['Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority',
                'Order Date', 'Order ID', 'Ship Date', 'Units Sold', 'Unit Price',
                'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit']

categorical_features = df.select_dtypes(include=[object])
numerical_features = df.select_dtypes(include=[np.number])
print(categorical_features.columns)
print(numerical_features.columns)
print("\n")


# Step2-2: Check Correlation Matrix for numerical some features
corr_matrix = df[['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue']].corr()

# Step2-3: Plot heat map to checked Correlation
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title('Heat map of Correlation Matrix')
plt.show()

# ---------------------------------------------------------------------------
# Part3: Defines Parameters
feature = ['Unit Price', 'Unit Cost']
X = df[feature].values.reshape(-1, len(feature))
y = df['Total Revenue'].values.reshape(-1, 1)

# Step3-1: Scale X values For exploration
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Fit the scaler to y
# y_scaled = y
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y).flatten()

fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')
axes = [ax1, ax2, ax3]
for ax in axes:
    ax.scatter(xs=X_scaled[:,0], ys=X_scaled[:,1], zs=y_scaled, c='b', alpha=0.5)
    ax.set_xlabel('Unit Price', fontsize=12)
    ax.set_ylabel('Unit Cost', fontsize=12)
    ax.set_zlabel('Total Revenue', fontsize=12)
    ax.locator_params(nbins=6, axis='x')
    ax.locator_params(nbins=5, axis='y')
    ax.locator_params(nbins=5, axis='z')
ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=130)
ax3.view_init(elev=60, azim=165)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.5)
plt.show()

# ---------------------------------------------------------------------------
# Part4: Create Model Pipeline
model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Spilt data set for training 90% and test 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train Linear Regression Model
model.fit(X_train, y_train)

# Part5: Prediction
y_pred = model.predict(X_test)

unit_price_pred = 1000
unit_cost_pred = 800
m_predict = model.predict([[unit_price_pred,unit_cost_pred]])  # Predict

print(f'If Features Unit Price = {unit_price_pred:,.2f} and Unit Cost = {unit_cost_pred:,.2f} '
      f'then predicted Revenue = {m_predict[0][0]:,.2f}')
print("\n")


# Part6: Evaluation of Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
r2_model = model.score(X_test, y_test)
n = len(y_test)
n_features = 2
adjusted_r2 =  1 - ((1 - r2) * (n - 1) / (n - n_features - 1))

print(f'Mean Squared Error: {mse:,.4f}')
print(f'R-squared: {r2:.4f}')
print(f'R-squared: {r2_model:.4f}')
print(f'adjusted_r2: {adjusted_r2}')

# ---------------------------------------------------------------------------
# Part7: Model Visualization
fig = plt.figure(figsize=(20,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs=X_scaled[:,0], ys=X_scaled[:,1], zs=y_scaled, c='b', alpha=0.5)
ax.set_xlabel('Unit Price', fontsize=12)
ax.set_ylabel('Unit Cost', fontsize=12)
ax.set_zlabel('Total Revenue', fontsize=12)
ax.locator_params(nbins=6, axis='x')
ax.locator_params(nbins=6, axis='y')
ax.locator_params(nbins=7, axis='z')
ax.view_init(elev=28, azim=120)

X_surf = np.linspace(X_scaled[:,0].min(),X_scaled[:,0].max(),50)
y_surf = np.linspace(X_scaled[:,1].min(), X_scaled[:,1].max(), 50)
x_surf, y_surf = np.meshgrid(X_surf, y_surf)

# Inverse transform the grid points
grid_points = scaler_X.inverse_transform(np.c_[x_surf.ravel(), y_surf.ravel()])

# Predicting the output using the trained model
z_surf = model.predict(grid_points).reshape(x_surf.shape)

# Scaling the predicted values
z_surf_scaled = scaler_y.transform(z_surf.reshape(-1, 1)).reshape(z_surf.shape)

# Plotting the surface
ax.plot_surface(x_surf, y_surf, z_surf_scaled, rstride=10, cstride=10, color='r', alpha=0.4, edgecolor='none')

fig.tight_layout()
plt.show()