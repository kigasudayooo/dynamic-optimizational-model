import numpy as np
import pytest
from src.model.bellman import DynamicProgram


# 簡単なテスト用モデルを作成
def create_test_model():
    """単純な決定論的モデルを作成"""
    # 状態空間
    state_grid = np.array([1.0, 2.0, 3.0])
    
    # 行動空間
    action_grid = np.array([0.5, 1.0, 1.5])
    
    # 報酬関数（単純に状態と行動の和）
    def reward_func(s, a):
        return s + a
    
    # 遷移関数（決定論的）
    def transition_func(s_next, s, a):
        # a == s_nextの場合のみ確率1
        return 1.0 if np.isclose(s_next, a) else 0.0
    
    return DynamicProgram(
        state_grid=state_grid,
        action_grid=action_grid,
        reward_func=reward_func,
        transition_func=transition_func,
        discount_factor=0.9,
        tolerance=1e-6,
        max_iter=100,
    )


# ベルマンオペレータのテスト
def test_bellman_operator():
    """ベルマンオペレータが正しく最大化を行っているかテスト"""
    model = create_test_model()
    
    # 初期値関数
    v = np.array([1.0, 2.0, 3.0])
    
    # ベルマンオペレータの適用
    Tv = model.bellman_operator(v)
    
    # 各状態での最適な行動を手動で計算
    expected_values = []
    for i, s in enumerate(model.state_grid):
        best_value = -np.inf
        for a in model.action_grid:
            r = model.reward_func(s, a)
            expected_future = 0
            for j, s_next in enumerate(model.state_grid):
                prob = model.transition_func(s_next, s, a)
                expected_future += prob * v[j]
            value = r + model.discount_factor * expected_future
            best_value = max(best_value, value)
        expected_values.append(best_value)
    
    expected_Tv = np.array(expected_values)
    
    # 結果が期待値と一致するか検証
    assert np.allclose(Tv, expected_Tv), \
        f"ベルマンオペレータの結果: {Tv}, 期待値: {expected_Tv}"


# 値関数反復法の収束をテスト
def test_value_iteration_convergence():
    """値関数反復法が収束するかテスト"""
    model = create_test_model()
    
    # 値関数反復法を実行
    v, policy = model.value_iteration()
    
    # 収束した値関数を使ってベルマンオペレータを再度適用
    Tv = model.bellman_operator(v)
    
    # 収束条件: T[v] ≈ v
    assert np.allclose(Tv, v, atol=model.tolerance), \
        f"値関数反復法が収束していません: |T[v] - v| = {np.max(np.abs(Tv - v))}"


# 政策反復法の収束をテスト
def test_policy_iteration_convergence():
    """政策反復法が収束するかテスト"""
    model = create_test_model()
    
    # 政策反復法を実行
    v, policy = model.policy_iteration()
    
    # 収束した政策を評価
    v_eval = model._policy_evaluation(policy)
    
    # 政策改善
    new_policy = model._policy_improvement(v_eval)
    
    # 政策が変化しないことを確認（収束条件）
    assert np.array_equal(policy, new_policy), \
        "政策反復法が収束していません: 政策がまだ改善可能です"
