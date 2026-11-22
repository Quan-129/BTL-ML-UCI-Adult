from modules.splitting import load_saved_splits
from modules.training import run_cv, refit_and_validate, plot_experiments
from modules.experiment_report import update_experiment_report

if __name__ == "__main__":
    frames, meta = load_saved_splits()
    run_cv(frames["train"], meta["train"])
    refit_and_validate(frames["train"], meta["train"], frames["validation"], meta["validation"])
    plot_experiments()
    update_experiment_report()
