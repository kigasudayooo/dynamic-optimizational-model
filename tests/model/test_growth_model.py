import numpy as np
import pytest
from src.model.growth_model import StochasticGrowthModel


# 基本的なモデルのセットアップ
@pytest.fixture
def basic_model():
    return StochasticGrowthModel(
        alpha=0.3,
        beta=0.95,
        delta=0.1,
        k_min=0.1,
        k_max=10.0,
        n_states=50
    )


# モデルのパラメータが正しく設定されているかテスト
def test_model_parameters(basic_model):
    """モデルのパラメータが正しく設定されているかテスト"""
    assert basic_model.alpha == 0.3
    assert basic_model.beta == 0.95
    assert basic_model.delta == 0.1
    assert len(basic_model.capital_grid) == 50
    assert basic_model.capital_grid[0] == 0.1
    assert basic_model.capital_grid[-1] == 10.0


# 生産関数の性質をテスト
def test_production_function(basic_model):
    """コブ・ダグラス生産関数の性質をテスト"""
    # 正の限界生産性
    k_values = [0.5, 1.0, 2.0, 5.0]
    for k in k_values:
        k_plus = k * 1.01  # 少し大きい値
        y = basic_model._production_function(k)
        y_plus = basic_model._production_function(k_plus)
        assert y_plus > y, f"生産関数は単調増加であるべき: f({k_plus}) <= f({k})"
    
    # 規模に関する収穫逓減
    for k in k_values:
        y1 = basic_model._production_function(k)
        y2 = basic_model._production_function(2 * k)
        assert y2 < 2 * y1, f"生産関数は規模に関して収穫逓減であるべき: f({2*k}) >= 2*f({k})"
    
    # 原点での性質
    assert basic_model._production_function(0) == 0, "f(0) = 0 であるべき"


# 定常状態の性質をテスト
def test_steady_state_properties(basic_model):
    """定常状態の性質をテスト"""
    # モデルを解く
    v, policy = basic_model.solve(method='value_iteration')
    
    # 政策関数の配列を取得
    policy_values = np.array([basic_model.capital_grid[i] for i in policy])
    
    # 定常状態を見つける（k' = k となる点）
    diffs = np.abs(policy_values - basic_model.capital_grid)
    steady_state_idx = np.argmin(diffs)
    steady_state_k = basic_model.capital_grid[steady_state_idx]
    
    # 定常状態の資本が正であることを確認
    assert steady_state_k > 0, "定常状態の資本ストックは正であるべき"
    
    # 理論的な定常状態と比較
    # コブ・ダグラス生産関数の場合、定常状態は以下の式で近似できる
    # k* = (alpha * beta / (1 - beta * (1 - delta)))^(1/(1-alpha))
    theoretical_ss = (basic_model.alpha * basic_model.beta / 
                     (1 - basic_model.beta * (1 - basic_model.delta))) ** (1 / (1 - basic_model.alpha))
    
    # 数値解と理論解が近いことを確認（グリッドの粗さを考慮）
    assert np.abs(steady_state_k - theoretical_ss) < 0.5, \
        f"定常状態の数値解 {steady_state_k} が理論解 {theoretical_ss} と大きく異なります"


# パラメータ変化に対する単調性をテスト
@pytest.mark.parametrize("beta", [0.9, 0.95, 0.99])
def test_monotonicity_with_beta(beta):
    """割引因子が大きいほど定常状態の資本も大きくなることをテスト"""
    model = StochasticGrowthModel(alpha=0.3, beta=beta, delta=0.1)
    v, policy = model.solve(method='value_iteration')
    
    policy_values = np.array([model.capital_grid[i] for i in policy])
    diffs = np.abs(policy_values - model.capital_grid)
    steady_state_idx = np.argmin(diffs)
    steady_state_k = model.capital_grid[steady_state_idx]
    
    # 理論的な定常状態
    theoretical_ss = (model.alpha * model.beta / 
                     (1 - model.beta * (1 - model.delta))) ** (1 / (1 - model.alpha))
    
    # より大きなbetaに対して定常状態も大きくなることを確認
    if beta > 0.9:
        prev_model = StochasticGrowthModel(alpha=0.3, beta=beta-0.05, delta=0.1)
        prev_v, prev_policy = prev_model.solve(method='value_iteration')
        
        prev_policy_values = np.array([prev_model.capital_grid[i] for i in prev_policy])
        prev_diffs = np.abs(prev_policy_values - prev_model.capital_grid)
        prev_steady_state_idx = np.argmin(prev_diffs)
        prev_steady_state_k = prev_model.capital_grid[prev_steady_state_idx]
        
        assert steady_state_k > prev_steady_state_k, \
            f"β={beta}の定常状態資本 {steady_state_k} は β={beta-0.05}の定常状態資本 {prev_steady_state_k} よりも大きいべき"
