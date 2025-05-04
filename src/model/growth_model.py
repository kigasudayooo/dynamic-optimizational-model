import numpy as np
from src.model.bellman import DynamicProgram


class StochasticGrowthModel(DynamicProgram):
    """確率的成長モデル"""
    
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.95,
        delta: float = 0.1,
        sigma: float = 0.02,
        k_min: float = 0.1,
        k_max: float = 10.0,
        n_states: int = 100,
        n_shocks: int = 7,
    ):
        """
        Parameters
        ----------
        alpha : float, default=0.3
            資本分配率
        beta : float, default=0.95
            割引因子
        delta : float, default=0.1
            資本減耗率
        sigma : float, default=0.02
            生産性ショックの標準偏差
        k_min : float, default=0.1
            資本グリッドの最小値
        k_max : float, default=10.0
            資本グリッドの最大値
        n_states : int, default=100
            資本グリッドの数
        n_shocks : int, default=7
            生産性ショックの数
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.sigma = sigma
        
        # 状態空間（資本ストック）
        self.capital_grid = np.linspace(k_min, k_max, n_states)
        
        # 行動空間（次期資本ストック）
        self.next_capital_grid = np.copy(self.capital_grid)
        
        # 生産性ショック
        self.shocks, self.shock_probs = self._create_shock_grid(n_shocks)
        
        # DynamicProgramの初期化
        super().__init__(
            state_grid=self.capital_grid,
            action_grid=self.next_capital_grid,
            reward_func=self._reward_function,
            transition_func=self._transition_function,
            discount_factor=beta,
        )
    
    def _create_shock_grid(self, n_shocks: int) -> tuple:
        """生産性ショックのグリッドを作成"""
        from scipy.stats import norm
        
        # 正規分布の離散近似
        shock_min = -3 * self.sigma
        shock_max = 3 * self.sigma
        shocks = np.linspace(shock_min, shock_max, n_shocks)
        
        # 各ショックの確率
        x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), n_shocks)
        shock_probs = norm.pdf(x)
        shock_probs = shock_probs / shock_probs.sum()  # 正規化
        
        return np.exp(shocks), shock_probs
    
    def _production_function(self, k: float, z: float = 1.0) -> float:
        """コブ・ダグラス生産関数 Y = zK^α"""
        return z * k ** self.alpha
    
    def _reward_function(self, k: float, k_next: float, z: float = 1.0) -> float:
        """効用関数 u(c) = log(c)、ここでc = zk^α + (1-δ)k - k_next"""
        c = self._production_function(k, z) + (1 - self.delta) * k - k_next
        if c <= 0:
            return -np.inf  # 消費が負の場合は実行不可能
        return np.log(c)
    
    def _transition_function(self, k_next: float, k: float, k_prime: float) -> float:
        """遷移確率
        
        この場合、k_next（次期の資本）は状態変数、k（現在の資本）と
        k_prime（選択する次期資本）は現在の状態と行動を表す
        """
        # 確率的ショックのない決定論的なモデルでは、
        # k_prime（選択した次期資本）とk_next（実現する次期資本）が一致する場合のみ確率1
        return 1.0 if np.isclose(k_next, k_prime) else 0.0
    
    def solve(self, method: str = 'value_iteration') -> tuple:
        """モデルを解く"""
        if method == 'value_iteration':
            return self.value_iteration()
        elif method == 'policy_iteration':
            return self.policy_iteration()
        else:
            raise ValueError("Method must be either 'value_iteration' or 'policy_iteration'")
