import argparse
import matplotlib.pyplot as plt
import pandas as pd
import wandb

def plot_dqn():
    df = pd.read_csv("RL/delivery_plot.csv")
    df = df.rename(columns={'delivery run - score': 'score'})
    df['score'] = df['score'].astype(float)
    df.plot(x='Step', y='score', linewidth=0.2)
    plt.grid(axis="y")
    plt.show()

def plot_cma_es():
    api = wandb.Api()
    run3 = api.run("/jacopodona/highway_CMA/runs/pw5opfhv")  # for cma es 3
    run1 = api.run("/jacopodona/highway_CMA/runs/nnqio6qf")  # for cma es 1
    df3 = run3.history()
    df1 = run1.history()

    df3.plot(x="Generation", y=["Best Fitness", "Median Fitness"])
    plt.grid(axis="y")
    df1.plot(x="Generation", y=["Best Fitness", "Median Fitness"])
    plt.grid(axis="y")
    plt.legend()
    plt.show()

def plot_neat():
    api = wandb.Api()
    run10 = api.run("/pappol/neat-testing/runs/uv6wtuy9")
    run5 = api.run("/pappol/neat-testing/runs/9xaqx4a1")
    run3 = api.run("/pappol/neat-testing/runs/mklzsq8s")

    df10 = run10.history()
    df5 = run5.history()
    df3 = run3.history()

    plt.plot(df10["_step"], df10["mean_fitness"], label="10 envs")
    plt.plot(df5["_step"], df5["mean_fitness"], label="5 envs")
    plt.plot(df3["_step"], df3["mean_fitness"], label="3 envs")

    # Change labels and title
    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    
    plt.grid(axis="y")
    plt.legend()
    plt.show()

    plt.plot(df10["_step"], df10["std_fitness"], label="10 envs")
    plt.plot(df5["_step"], df5["std_fitness"], label="5 envs")
    plt.plot(df3["_step"], df3["std_fitness"], label="3 envs")

    # Change labels and title
    plt.xlabel("Generation")
    plt.ylabel("Std Fitness")


    plt.grid(axis="y")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plotting script for DQN, CMA-ES, and NEAT")
    parser.add_argument("algorithm", choices=["dqn", "cma-es", "neat"], help="Choose which algorithm to plot")

    args = parser.parse_args()

    if args.algorithm == "dqn":
        plot_dqn()
    elif args.algorithm == "cma-es":
        plot_cma_es()
    elif args.algorithm == "neat":
        plot_neat()

if __name__ == "__main__":
    main()
