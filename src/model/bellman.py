import numpy as np
from typing import Callable, Optional, Tuple


class DynamicProgram:
    """ベルマン方程式を解くための基本クラス"""
    
    def __init__(
        self,
        state_grid: np.ndarray,
        action_grid: np.ndarray,
        reward_func: Callable,
        transition_func: Callable,
        discount_factor: float = 0.95,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        """
        Parameters
        ----------
        state_grid : np.ndarray
            状態空間のグリッド
        action_grid : np.ndarray
            行動空間のグリッド
        reward_func : Callable
            報酬関数 R(s, a)
        transition_func : Callable
            遷移関数 P(s'|s, a)
        discount_factor : float, default=0.95
            割引因子
        tolerance : float, default=1e-6
            値関数反復の収束許容誤差
        max_iter : int, default=1000
            値関数反復の最大反復回数
        """
        self.state_grid = state_grid
        self.action_grid = action_grid
        self.reward_func = reward_func
        self.transition_func = transition_func
        self.discount_factor = discount_factor
        self.tolerance = tolerance
        self.max_iter = max_iter
        
        # 初期化
        self.value_function = np.zeros_like(state_grid)
        self.policy = np.zeros_like(state_grid, dtype=int)
        
    def bellman_operator(self, v: np.ndarray) -> np.ndarray:
        """ベルマンオペレータ T[v](s) = max_a { R(s,a) + γ Σ_s' P(s'|s,a) v(s') }"""
        n_states = len(self.state_grid)
        Tv = np.zeros(n_states)
        
        for i, s in enumerate(self.state_grid):
            values = []
            for a in self.action_grid:
                # 即時報酬
                r = self.reward_func(s, a)
                
                # 将来価値の期待値
                expected_future_value = 0
                for j, s_next in enumerate(self.state_grid):
                    prob = self.transition_func(s_next, s, a)
                    expected_future_value += prob * v[j]
                
                # 割引現在価値
                values.append(r + self.discount_factor * expected_future_value)
            
            # 最大化
            Tv[i] = max(values)
            # 最適政策の更新
            self.policy[i] = np.argmax(values)
            
        return Tv
    
    def value_iteration(self) -> Tuple[np.ndarray, np.ndarray]:
        """値関数反復法でベルマン方程式を解く"""
        v = np.zeros_like(self.state_grid)
        
        for i in range(self.max_iter):
            v_new = self.bellman_operator(v)
            
            # 収束判定
            error = np.max(np.abs(v_new - v))
            if error < self.tolerance:
                print(f"Value iteration converged in {i+1} iterations.")
                break
                
            v = v_new
            
        if i == self.max_iter - 1:
            print(f"Warning: Value iteration did not converge within {self.max_iter} iterations.")
            
        self.value_function = v
        return self.value_function, self.policy
    
    def policy_iteration(self) -> Tuple[np.ndarray, np.ndarray]:
        """政策反復法でベルマン方程式を解く"""
        v = np.zeros_like(self.state_grid)
        policy = np.zeros_like(self.state_grid, dtype=int)
        
        for i in range(self.max_iter):
            # 政策評価（政策が与えられた下での値関数の計算）
            v = self._policy_evaluation(policy)
            
            # 政策改善
            new_policy = self._policy_improvement(v)
            
            # 政策が収束したかチェック
            if np.array_equal(new_policy, policy):
                print(f"Policy iteration converged in {i+1} iterations.")
                break
                
            policy = new_policy
            
        if i == self.max_iter - 1:
            print(f"Warning: Policy iteration did not converge within {self.max_iter} iterations.")
            
        self.value_function = v
        self.policy = policy
        return self.value_function, self.policy
    
    def _policy_evaluation(self, policy: np.ndarray) -> np.ndarray:
        """与えられた政策の下での値関数を計算"""
        v = np.zeros_like(self.state_grid)
        
        for _ in range(self.max_iter):
            v_new = np.zeros_like(v)
            
            for i, s in enumerate(self.state_grid):
                a = self.action_grid[policy[i]]
                
                # 即時報酬
                r = self.reward_func(s, a)
                
                # 将来価値の期待値
                expected_future_value = 0
                for j, s_next in enumerate(self.state_grid):
                    prob = self.transition_func(s_next, s, a)
                    expected_future_value += prob * v[j]
                
                # 割引現在価値
                v_new[i] = r + self.discount_factor * expected_future_value
            
            # 収束判定
            error = np.max(np.abs(v_new - v))
            if error < self.tolerance:
                break
                
            v = v_new
            
        return v
    
    def _policy_improvement(self, v: np.ndarray) -> np.ndarray:
        """値関数に基づく政策改善"""
        new_policy = np.zeros_like(self.state_grid, dtype=int)
        
        for i, s in enumerate(self.state_grid):
            values = []
            for a_idx, a in enumerate(self.action_grid):
                # 即時報酬
                r = self.reward_func(s, a)
                
                # 将来価値の期待値
                expected_future_value = 0
                for j, s_next in enumerate(self.state_grid):
                    prob = self.transition_func(s_next, s, a)
                    expected_future_value += prob * v[j]
                
                # 割引現在価値
                values.append(r + self.discount_factor * expected_future_value)
            
            # 最大化行動を選択
            new_policy[i] = np.argmax(values)
            
        return new_policy
