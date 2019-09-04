import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model # Linear Regression


data = {
    'luas': np.arange(100, 300, 20),
    'harga': [500,665,720,795,885,1200,1500,1600,1775,2000]
}
df = pd.DataFrame(data)
# print(df)

# Plotting
# plt.scatter(df['luas'], df['harga'])
# plt.show()

# Bikin Model ML metode linear regression
model = linear_model.LinearRegression()

# Training model dengan data yang kita punya
# model.fit(dataindependent 2D, dependen)
model.fit(df[['luas']], df['harga'])

m = model.coef_
c = model.intercept_
# print(m)
# print(c)

# prediksi
# print(model.predict([[ 100 ]])) # 398.63636364
# print(model.predict([[ 3000 ]])) # 25091.57575758
# print(model.predict(df[['luas']]))

plt.style.use('ggplot')
plt.plot(
    df['luas'], df['harga'], 'ro',
    df['luas'], model.predict(df[['luas']]), 'g-'
)
plt.grid(True)
plt.xlabel('Luas (m2)')
plt.ylabel('Harga (Rp)')
plt.legend(['Data', 'Best Fit Line'])

plt.show()


