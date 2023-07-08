import matplotlib.pyplot as plt
import pandas as pd

gens=list(range(1,101)) #For evolutionary strategies
iterations=list(range(1,4000)) #For DQN

#For DQN
df=pd.read_csv("RL/delivery_plot.csv")
df=df.rename(columns={'delivery run - score': 'score'})
df['score'] = df['score'].astype(float)

df.plot(x='Step',y='score',linewidth=0.2)
plt.legend()
plt.show()