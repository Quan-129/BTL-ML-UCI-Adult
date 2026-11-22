from modules.splitting import load_saved_splits
from modules.evaluation import freeze_selection, evaluate_frozen, analyze_errors, audit_test_profile_overlap, plot_evaluation
from modules.test_report import update_test_report

if __name__ == "__main__":
    frames, meta = load_saved_splits()
    freeze_selection()
    evaluate_frozen(frames["test"], meta["test"])
    analyze_errors(frames["test"], meta["test"])
    audit_test_profile_overlap(frames["train"], frames["validation"], frames["test"])
    plot_evaluation()
    update_test_report()
