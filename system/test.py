import wandb
api = wandb.Api()
runs = api.runs("Gerid/FedDCA")

for run in runs:
    history = run.history()
    filename = f"{run.name}.csv"
    history.to_csv(filename, index=False)