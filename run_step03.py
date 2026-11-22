from pathlib import Path
from modules.data_loader import download_adult, load_adult
from modules.splitting import load_or_create_split, export_splits, plot_split_balance, update_latex

if __name__ == "__main__":
    download_adult("data")
    development, test = load_adult("data")
    tr, va, groups, manifest, origin = load_or_create_split(development, "results/step_03/split_indices.json")
    export_splits(development, test, tr, va, groups, manifest, split_source=origin)
    plot_split_balance()
    update_latex()
    print("Đã xuất các tập và cập nhật nguồn LaTeX. Notebook tạo thêm ZIP báo cáo.")
