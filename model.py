import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('C:/Users/phili/PycharmProjects/Cancer/bc.csv')
df['diagnosis'] = df['diagnosis'].astype('category')

X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis'].cat.codes  # Converts 'M' to 1 and 'B' to 0 automatically

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'cancer_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
