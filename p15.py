import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns
import sklearn

sns.set(font_scale=1.3)
np.set_printoptions(
    precision=6, 
    suppress=True,
    edgeitems=10,
    linewidth=10000,
    formatter=dict(float=lambda x: f'{x:.2f}')
)
np.random.seed(42)
print(sklearn.__version__)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.arange(-5, 5, 0.1)
y = sigmoid(X)
plt.figure(figsize=(8, 6))
plt.plot(X, y)
plt.title('Funkcja Sigmoid')
plt.show()

from sklearn.datasets import load_breast_cancer

raw_data = load_breast_cancer()
print(raw_data.keys())
print(raw_data.DESCR)

all_data = raw_data.copy()
data = all_data['data']
target = all_data['target']

print(f'rozmiar data: {data.shape}')
print(f'rozmiar target: {target.shape}')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print(y_pred[:30])
y_prob = log_reg.predict_proba(X_test)
print(y_prob[:30])

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

def plot_confusion_matrix(cm):
    cm = cm[::-1]
    cm = pd.DataFrame(
        cm,
        columns=['pred_0', 'pred_1'],
        index=['true_1', 'true_0']
    )
    fig = ff.create_annotated_heatmap(
        z=cm.values,
        x=list(cm.columns),
        y=list(cm.index),
        colorscale='ice',
        showscale=True,
        reversescale=True
    )
    fig.update_layout(
        width=500,
        height=500,
        title='Confusion Matrix',
        font_size=16
    )
    fig.show()

plot_confusion_matrix(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))