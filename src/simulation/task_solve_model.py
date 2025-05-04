import json
from pathlib import Path
from typing import Annotated, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from pytask import Product
from src.model.growth_model import StochasticGrowthModel


def task_solve_baseline_model(
    result_file: Annotated[Path, Product] = Path("results/data/baseline_model.json"),
    figure_file: Annotated[Path, Product] = Path("results/figures/baseline_policy.png"),
):
    """
    基本的な成長モデルを解き、結果を保存するタスク
    
    Parameters
    ----------
    result_file : Path
        結果を保存するJSONファイルのパス
    figure_file : Path
        政策関数のプロットを保存するファイルのパス
    """
    # モデルの初期化と解法
    model = StochasticGrowthModel(
        alpha=0.3,
        beta=0.95,
        delta=0.1,
    )
    value_function, policy = model.solve(method='value_iteration')
    
    # 結果の保存
    results = {
        "parameters": {
            "alpha": model.alpha,
            "beta": model.beta,
            "delta": model.delta,
            "sigma": model.sigma,
        },
        "capital_grid": model.capital_grid.tolist(),
        "value_function": value_function.tolist(),
        "policy_indices": policy.tolist(),
        "policy_values": [model.capital_grid[i].item() for i in policy],
    }
    
    # 出力ディレクトリの作成
    result_file.parent.mkdir(parents=True, exist_ok=True)
    figure_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 結果をJSONで保存
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # 政策関数のプロット
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(model.capital_grid, [model.capital_grid[i] for i in policy], 'b-', lw=2, alpha=0.7, label='政策関数')
    ax.plot(model.capital_grid, model.capital_grid, 'k--', lw=1, alpha=0.5, label='45度線')
    
    # 定常状態の資本ストックを見つける（45度線と政策関数の交点）
    steady_state_idx = np.argmin(np.abs(model.capital_grid - np.array([model.capital_grid[i] for i in policy])))
    steady_state_k = model.capital_grid[steady_state_idx]
    ax.plot([steady_state_k], [steady_state_k], 'ro', ms=10, label=f'定常状態 k*={steady_state_k:.2f}')
    
    ax.set_xlabel('現在の資本ストック k')
    ax.set_ylabel('次期の資本ストック k\'')
    ax.set_title('最適政策関数')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"基本モデルの解法が完了しました。結果は {result_file} に保存されました。")
    print(f"政策関数のプロットは {figure_file} に保存されました。")
    
    return results


