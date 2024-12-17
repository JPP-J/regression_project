import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols

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
# plt.show()

# ---------------------------------------------------------------------------
# Part3: Defines Parameters
feature = ['Unit Price', 'Unit Cost']
X = df[feature].values.reshape(-1, len(feature))
X = sm.add_constant(X)  # Add intercept
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
# plt.show()


# Option+++++
f1_min = df['Unit Price'].min()
f1_max = df['Unit Price'].max()
f2_min = df['Unit Cost'].min()
f2_max = df['Unit Cost'].max()
t_min =  df['Total Revenue'].min()
t_max = df['Total Revenue'].max()
print(f1_min, f1_max, f2_min, f2_max)
print(f'{t_min:,.2f}, {t_max:,.2f}')

# ---------------------------------------------------------------------------
# Part4: Create Model

# Spilt data set for training 90% and test 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Add a constant and use statsmodels to compute p-values
X_sm = sm.add_constant(X_train)  # Add intercept
stat_mod = sm.OLS(y_train, X_sm).fit()

# Display summary (includes p-values)
print(stat_mod.summary())

# Fit a model with interaction and polynomial terms
# inter_poly_model = ols('y ~ X1 + X2 + I(X1**2) + X1:X2', data=df).fit()

# Clean up column names to avoid spaces
df.columns = df.columns.str.replace(" ", "_")

# Now fit the model using the cleaned column names check interation and non-linear
# inter_poly_model = ols('Total_Revenue ~ Unit_Price + Unit_Cost + Unit_Price:Unit_Cost', data=df).fit()
inter_poly_model = ols('Total_Revenue ~ Unit_Price + Unit_Cost + I(Unit_Price**2) + I(Unit_Cost**2) + Unit_Price:Unit_Cost', data=df).fit()

# Display summary
print(f'\ninter_poly_model:')
print(inter_poly_model.summary())

# ============================================= END ======================================================



