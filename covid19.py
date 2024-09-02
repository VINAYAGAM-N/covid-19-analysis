import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('covid_dataset.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df['confirmed'] = imputer.fit_transform(df[['confirmed']])
df['recovered'] = imputer.fit_transform(df[['recovered']])
df['critical'] = imputer.fit_transform(df[['critical']])
df['deaths'] = imputer.fit_transform(df[['deaths']])

# Convert categorical variables to numeric
label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country'])
df['code'] = label_encoder.fit_transform(df['code'])

# Drop columns that won't be used
drop_columns = ['lastChange', 'lastUpdate']
drop_columns = [col for col in drop_columns if col in df.columns]
df.drop(columns=drop_columns, inplace=True)

# Define features and target
X = df.drop('confirmed', axis=1)
y = df['confirmed']

# Convert target variable to a binary outcome for classification
y = (y > y.median()).astype(int)  # Convert to binary based on median value

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature selection using RFE
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector_rfe = RFE(clf_rf, n_features_to_select=5)
selector_rfe.fit(X_train, y_train)
X_train_rfe = selector_rfe.transform(X_train)
X_test_rfe = selector_rfe.transform(X_test)
selected_features_rfe = X.columns[selector_rfe.support_]

# Feature selection using CFS (SelectKBest with ANOVA F-test)
selector_cfs = SelectKBest(f_classif, k=5)
selector_cfs.fit(X_train, y_train)
X_train_cfs = selector_cfs.transform(X_train)
X_test_cfs = selector_cfs.transform(X_test)
selected_features_cfs = X.columns[selector_cfs.get_support()]

# Classification using Logistic Regression before feature selection
clf_lr = LogisticRegression(max_iter=2000, random_state=42)
clf_lr.fit(X_train, y_train)
accuracy_lr = clf_lr.score(X_test, y_test)

# Classification using Random Forest before feature selection
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
accuracy_rf = clf_rf.score(X_test, y_test)

# Logistic Regression after RFE
clf_lr.fit(X_train_rfe, y_train)
accuracy_lr_rfe = clf_lr.score(X_test_rfe, y_test)

# Random Forest after RFE
clf_rf.fit(X_train_rfe, y_train)
accuracy_rf_rfe = clf_rf.score(X_test_rfe, y_test)

# Logistic Regression after CFS
clf_lr.fit(X_train_cfs, y_train)
accuracy_lr_cfs = clf_lr.score(X_test_cfs, y_test)

# Random Forest after CFS
clf_rf.fit(X_train_cfs, y_train)
accuracy_rf_cfs = clf_rf.score(X_test_cfs, y_test)

# Plotting
# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of COVID-19 Metrics')
plt.show()

# 2. Pair Plot
sns.pairplot(df[['confirmed', 'recovered', 'critical', 'deaths']])
plt.title('Pair Plot of COVID-19 Metrics')
plt.show()

# 3. Feature Importance from RFE
# Get feature importances for RFE-selected features
clf_rf.fit(X_train, y_train)  # Ensure clf_rf is trained to get feature importances
importances_rfe = clf_rf.feature_importances_
rfe_importances = importances_rfe[selector_rfe.support_]

plt.figure(figsize=(10, 6))
sns.barplot(x=selected_features_rfe, y=rfe_importances)
plt.title('Feature Importance after RFE')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

# 4. Feature Importance from CFS
# Get feature importances for CFS-selected features
importances_cfs = importances_rfe[selector_cfs.get_support()]

plt.figure(figsize=(10, 6))
sns.barplot(x=selected_features_cfs, y=importances_cfs)
plt.title('Feature Importance after CFS')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

# 5. Classification Performance Comparison
models = ['Logistic Regression', 'Random Forest']
before_fs = [accuracy_lr, accuracy_rf]
after_rfe = [accuracy_lr_rfe, accuracy_rf_rfe]
after_cfs = [accuracy_lr_cfs, accuracy_rf_cfs]

x = range(len(models))

plt.figure(figsize=(12, 6))
plt.bar(x, before_fs, width=0.2, label='Before Feature Selection', align='center')
plt.bar([i + 0.2 for i in x], after_rfe, width=0.2, label='After RFE', align='center')
plt.bar([i + 0.4 for i in x], after_cfs, width=0.2, label='After CFS', align='center')
plt.xticks([i + 0.2 for i in x], models)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Performance')
plt.legend()
plt.show()

# Print results
results = {
    'Model': ['Logistic Regression', 'Random Forest'],
    'Before Feature Selection': [accuracy_lr, accuracy_rf],
    'After RFE': [accuracy_lr_rfe, accuracy_rf_rfe],
    'After CFS': [accuracy_lr_cfs, accuracy_rf_cfs]
}

results_df = pd.DataFrame(results)
print(results_df)
