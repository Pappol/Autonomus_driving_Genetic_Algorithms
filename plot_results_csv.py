import matplotlib.pyplot as plt
import pandas as pd
import wandb

gens=list(range(1,101)) #For evolutionary strategies
iterations=list(range(1,4000)) #For DQN

#For DQN
df=pd.read_csv("RL/delivery_plot.csv")
df=df.rename(columns={'delivery run - score': 'score'})
df['score'] = df['score'].astype(float)
df.plot(x='Step',y='score',linewidth=0.2)
plt.grid(axis="y")

#For CMA_ES
#df=df.rename(columns={'delivery run - score': 'score'})ù
api = wandb.Api()
run3 = api.run("/jacopodona/highway_CMA/runs/pw5opfhv") #for cma es 3
run1 = api.run("/jacopodona/highway_CMA/runs/nnqio6qf") #for cma es 1
df3=run3.history()
df1=run1.history()
#
df3.plot(x="Generation",y=["Best Fitness","Median Fitness"])
plt.grid(axis="y")
df1.plot(x="Generation",y=["Best Fitness","Median Fitness"])
plt.grid(axis="y")
plt.legend()
plt.show()