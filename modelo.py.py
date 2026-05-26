import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Carregar a base de dados
df = pd.read_csv("obesity.csv")

# 1. Tratamento de Dados (Feature Engineering)
# Arredondamento obrigatório de variáveis com decimais
cols_to_round = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
for col in cols_to_round:
    df[col] = df[col].round().astype(int)

# Separação entre features (X) e target (y)
X = df.drop('Obesity_level', axis=1)
y = df['Obesity_level']

# Codificação de variáveis categóricas
label_encoders = {}
categorical_cols = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Codificação da variável alvo
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# 2. Divisão de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Treinamento do Modelo Preditivo
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Avaliação
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Assertividade do Modelo: {accuracy * 100:.2f}%")

# 5. Exportação do modelo e encoders para o deploy
joblib.dump(model, 'modelo_obesidade.pkl')
joblib.dump(label_encoders, 'encoders.pkl')
joblib.dump(le_target, 'target_encoder.pkl')