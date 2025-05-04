import numpy as np
import pytest
from src.model.growth_model import StochasticGrowthModel


# 異なるグリッドサイズでの収束テスト
@pytest.mark.parametrize("n_states", [20, 50, 100])
def test_grid_size_convergence(n_states):
    """グリッドサイズを変えても定常状態が大きく変わらないことをテスト"""
    model = StochasticGrowthModel(
        alpha=0.3,
        beta=0.95,
        delta=0.1,
        n_states=n_states
    )
    
    # モデルを解く
    v, policy = model.solve(method='value_iteration')
    
    # 定常状態を見つける
    policy_values = np.array([model.capital_grid[i] for i in policy])
    diffs = np.abs(policy_values - model.capital_grid)
    steady_state_idx = np.argmin(diffs)
    steady_state_k = model.capital_grid[steady_state_idx]
    
    # 理論的な定常状態
    theoretical_ss = (model.alpha * model.beta / 
                     (1 - model.beta * (1 - model.delta))) ** (1 / (1 - model.alpha))
    
    # グリッドサイズが変わっても理論値との相対誤差が一定以下であることを確認
    relative_error = np.abs(steady_state_k - theoretical_ss) / theoretical_ss
    assert relative_error < 0.1, \
        f"グリッドサイズ {n_states} での定常状態 {steady_state_k} の相対誤差が大きすぎます: {relative_error:.2%}"


# 値関数反復法と政策反復法の結果が一致することをテスト
def test_methods_agreement():
    """値関数反復法と政策反復法が同じ結果を生成することをテスト"""
    model_vi = StochasticGrowthModel(alpha=0.3, beta=0.95, delta=0.1)
    v_vi, policy_vi = model_vi.solve(method='value_iteration')
    
    model_pi = StochasticGrowthModel(alpha=0.3, beta=0.95, delta=0.1)
    v_pi, policy_pi = model_pi.solve(method='policy_iteration')
    
    # 値関数が近いことをテスト
    assert np.allclose(v_vi, v_pi, rtol=1e-3), \
        "値関数反復法と政策反復法で異なる値関数が得られました"
    
    # 政策が一致することをテスト
    assert np.array_equal(policy_vi, policy_pi), \
        "値関数反復法と政策反復法で異なる政策が得られました"


# 数値精度の違いによる結果への影響をテスト
@pytest.mark.parametrize("tolerance", [1e-3, 1e-6, 1e-9])
def test_numerical_precision(tolerance):
    """異なる収束許容誤差でも結果が大きく変わらないことをテスト"""
    model = StochasticGrowthModel(
        alpha=0.3,
        beta=0.95,
        delta=0.1,
        tolerance=tolerance
    )
    
    # モデルを解く
    v, policy = model.solve(method='value_iteration')
    
    # 定常状態を見つける
    policy_values = np.array([model.capital_grid[i] for i in policy])
    diffs = np.abs(policy_values - model.capital_grid)
    steady_state_idx = np.argmin(diffs)
    steady_state_k = model.capital_grid[steady_state_idx]
    
    # 基準となる高精度な解を取得
    reference_model = StochasticGrowthModel(
        alpha=0.3,
        beta=0.95,
        delta=0.1,
        tolerance=1e-10,
        max_iter=2000
    )
    ref_v, ref_policy = reference_model.solve(method='value_iteration')
    
    ref_policy_values = np.array([reference_model.capital_grid[i] for i in ref_policy])
    ref_diffs = np.abs(ref_policy_values - reference_model.capital_grid)
    ref_steady_state_idx = np.argmin(ref_diffs)
    ref_steady_state_k = reference_model.capital_grid[ref_steady_state_idx]
    
    # 許容誤差を変えても定常状態が大きく変わらないことを確認
    assert np.abs(steady_state_k - ref_steady_state_k) < 0.1, \
        f"許容誤差 {tolerance} での定常状態 {steady_state_k} が基準値 {ref_steady_state_k} と大きく異なります"
