"""
============================================================
 SCENARIO 1 — Mechatronic Game Theory in Cyber-Physical
              Logistics Systems
 Ho Chi Minh City University of Technology (HCMUT)
 Mechatronics — K25
============================================================

WHAT THIS SIMULATION DOES
--------------------------
1.  Builds a directed warehouse graph G = (V, E) with
    weighted edges (traversal cost in seconds).
2.  Runs two competing AGV fleets — AlphaFleet and
    BetaFleet — each governed by an autonomous AI agent.
3.  Applies M/M/c queueing theory every decision cycle
    to compute real network wait times.
4.  Uses Mixed Strategy Nash Equilibrium (MSNE) from
    zero-sum game theory to choose routes.
5.  Updates policy gradients at 10 Hz (every 100 ms).
6.  Plots four result charts at the end.

HOW TO RUN
----------
    python scenario1_simulation.py

DEPENDENCIES
------------
    pip install networkx matplotlib numpy
"""

import math, random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ─────────────────────────────────────────────────────────
#  SECTION 1 — M/M/c Queueing Theory
# ─────────────────────────────────────────────────────────

class MMCQueue:
    """
    M/M/c queueing model for the shared Wi-Fi 6E network.

    Notation
    --------
    λ (lambda_) : packet arrival rate  [packets / second]
    μ (mu)      : service rate / channel [packets / second]
    c           : number of parallel channels
    ρ           : utilisation ratio = λ / (c × μ)

    Stability condition: ρ < 1.
    As ρ → 1, wait time W → ∞  (exponential blowup).
    """

    def __init__(self, lambda_: float, mu: float, c: int):
        self.lambda_ = lambda_
        self.mu      = mu
        self.c       = c

    # ── Step 1 ── utilisation ────────────────────────────
    def rho(self) -> float:
        """ρ = λ / (c × μ)"""
        return self.lambda_ / (self.c * self.mu)

    # ── Step 2 ── P₀ via Erlang C formula ───────────────
    def P0(self) -> float:
        """
        P₀ = 1 / [ Σ_{n=0}^{c-1} (cρ)^n/n!  +  (cρ)^c/(c!(1−ρ)) ]
        Probability that all channels are idle.
        """
        rho   = self.rho()
        if rho >= 1:
            return 0.0
        c_rho = self.c * rho
        sigma = sum((c_rho**n) / math.factorial(n) for n in range(self.c))
        tail  = (c_rho**self.c) / (math.factorial(self.c) * (1 - rho))
        return 1.0 / (sigma + tail)

    # ── Step 3 ── Erlang C probability ──────────────────
    def erlang_C(self) -> float:
        """
        C(c,ρ) = [(cρ)^c / (c!(1−ρ))] × P₀
        Probability an arriving packet must wait.
        """
        rho   = self.rho()
        if rho >= 1:
            return 1.0
        c_rho = self.c * rho
        tail  = (c_rho**self.c) / (math.factorial(self.c) * (1 - rho))
        return tail * self.P0()

    # ── Step 4 ── Average queue length ──────────────────
    def Lq(self) -> float:
        """Lq = C(c,ρ) × ρ / (1 − ρ)"""
        rho = self.rho()
        if rho >= 1:
            return float('inf')
        return self.erlang_C() * rho / (1 - rho)

    # ── Step 5 ── Average wait time (Little's Law) ───────
    def W(self) -> float:
        """W = Lq / λ  —  seconds a packet waits in queue."""
        lq = self.Lq()
        return float('inf') if lq == float('inf') else lq / self.lambda_

    def summary(self) -> dict:
        return dict(
            lambda_=self.lambda_, mu=self.mu, c=self.c,
            rho=round(self.rho(),   4),
            P0 =round(self.P0(),    4),
            C  =round(self.erlang_C(), 4),
            Lq =round(self.Lq(),    4),
            W  =round(self.W(),     6),
        )


# ─────────────────────────────────────────────────────────
#  SECTION 2 — Zero-Sum Game Solver (MSNE)
# ─────────────────────────────────────────────────────────

class ZeroSumGame:
    """
    2×2 zero-sum game solved via Mixed Strategy Nash Equilibrium.

    A[i,j] = AlphaFleet's payoff when Alpha plays row i
              and Beta plays column j.  Beta's payoff = −A[i,j].
    """

    def __init__(self, A: np.ndarray):
        assert A.shape == (2, 2)
        self.A = A.astype(float)

    # ── Step 1 ── saddle point (pure NE?) ────────────────
    def saddle_point(self) -> Optional[Tuple[int, int]]:
        """maximin == minimax  →  pure strategy NE exists."""
        maximin = self.A.min(axis=1).max()
        minimax = self.A.max(axis=0).min()
        if maximin == minimax:
            r = int(np.argmax(self.A.min(axis=1)))
            c = int(np.argmin(self.A.max(axis=0)))
            return (r, c)
        return None

    # ── Step 2 ── Alpha's optimal mix p* ─────────────────
    def alpha_p_star(self) -> float:
        """
        Make Beta indifferent:
          p·A[0,0]+(1-p)·A[1,0] = p·A[0,1]+(1-p)·A[1,1]
          p* = (A[1,1]-A[1,0]) / (A[0,0]-A[0,1]-A[1,0]+A[1,1])
        """
        a = self.A
        d = a[0,0] - a[0,1] - a[1,0] + a[1,1]
        p = 0.5 if d == 0 else (a[1,1] - a[1,0]) / d
        return float(np.clip(p, 0.0, 1.0))

    # ── Step 3 ── Beta's optimal mix q* ──────────────────
    def beta_q_star(self) -> float:
        """
        Make Alpha indifferent:
          q·A[0,0]+(1-q)·A[0,1] = q·A[1,0]+(1-q)·A[1,1]
          q* = (A[1,1]-A[0,1]) / (A[0,0]-A[0,1]-A[1,0]+A[1,1])
        """
        a = self.A
        d = a[0,0] - a[0,1] - a[1,0] + a[1,1]
        q = 0.5 if d == 0 else (a[1,1] - a[0,1]) / d
        return float(np.clip(q, 0.0, 1.0))

    # ── Step 4 ── Game value V ────────────────────────────
    def game_value(self) -> float:
        """V = (A[0,0]·A[1,1] - A[0,1]·A[1,0]) / denominator"""
        a = self.A
        d = a[0,0] - a[0,1] - a[1,0] + a[1,1]
        if d == 0:
            return float((a[0,0] + a[1,1]) / 2)
        return float((a[0,0]*a[1,1] - a[0,1]*a[1,0]) / d)

    def solve(self) -> dict:
        p = self.alpha_p_star()
        q = self.beta_q_star()
        return dict(
            saddle_point  = self.saddle_point(),
            alpha_p_star  = round(p,   4),
            beta_q_star   = round(q,   4),
            game_value_V  = round(self.game_value(), 4),
        )


# ─────────────────────────────────────────────────────────
#  SECTION 3 — Warehouse Graph G = (V, E)
# ─────────────────────────────────────────────────────────

def build_warehouse_graph(rows: int = 5, cols: int = 8,
                           seed: int = 42) -> nx.DiGraph:
    """
    Directed grid graph of the warehouse.
    Edge weight = traversal cost in seconds (randomised to
    model congestion zones realistically).
    """
    rng = random.Random(seed)
    G   = nx.DiGraph()
    for r in range(rows):
        for c in range(cols):
            G.add_node((r, c), pos=(c, rows - 1 - r))

    for r in range(rows):
        for c in range(cols):
            base = rng.uniform(1.5, 4.0)
            if c + 1 < cols:
                G.add_edge((r,c), (r,c+1), weight=round(base,      2))
                G.add_edge((r,c+1),(r,c),  weight=round(base*1.1,  2))
            if r + 1 < rows:
                G.add_edge((r,c), (r+1,c), weight=round(base*0.9,  2))
                G.add_edge((r+1,c),(r,c),  weight=round(base*1.0,  2))
    return G


# ─────────────────────────────────────────────────────────
#  SECTION 4 — AGV and Fleet Classes
# ─────────────────────────────────────────────────────────

@dataclass
class AGV:
    """Single Automated Guided Vehicle."""
    agv_id    : str
    fleet_name: str
    position  : tuple
    path      : List[tuple] = field(default_factory=list)
    total_dist: float = 0.0
    deliveries: int   = 0
    battery   : float = 100.0

    def move_one_step(self, G: nx.DiGraph) -> float:
        """Advance one edge along current path. Returns cost."""
        if len(self.path) < 2:
            return 0.0
        src, dst   = self.path[0], self.path[1]
        cost       = G[src][dst]['weight']
        self.total_dist += cost
        self.battery     = max(0.0, self.battery - cost * 0.05)
        self.position    = dst
        self.path.pop(0)
        if len(self.path) == 1:
            self.deliveries += 1
        return cost


class Fleet:
    """
    Autonomous AGV fleet with policy-gradient learning.
    Decision frequency: 10 Hz (100 ms per cycle).
    """

    def __init__(self, name, color, n_agvs, G, nodes, seed=0):
        rng        = random.Random(seed)
        self.name  = name
        self.color = color
        self.G     = G
        self.nodes = nodes
        self.policy = np.array([0.5, 0.5])   # [P(Route A), P(Route B)]

        self.agvs: List[AGV] = [
            AGV(agv_id=f"{name[0]}{i+1:02d}",
                fleet_name=name,
                position=rng.choice(nodes))
            for i in range(n_agvs)
        ]

        self.throughput_log: List[int]   = []
        self.wait_log      : List[float] = []
        self.policy_log    : List[float] = []

    def _two_routes(self, src, dst):
        """
        Fast O(E log V) dual-route computation using Dijkstra.
        Route A : shortest path by weight.
        Route B : shortest path after tripling A's edge costs
                  → forces a structurally different path.
        """
        try:
            pathA = nx.shortest_path(self.G, src, dst, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, None

        G2 = self.G.copy()
        for u, v in zip(pathA[:-1], pathA[1:]):
            G2[u][v]['weight'] *= 3.0
        try:
            pathB = nx.shortest_path(G2, src, dst, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pathB = pathA
        return pathA, pathB

    def assign_routes(self, p_star: float):
        """Sample from mixed strategy to pick Route A or B."""
        for agv in self.agvs:
            dst = random.choice([n for n in self.nodes
                                 if n != agv.position])
            pathA, pathB = self._two_routes(agv.position, dst)
            if pathA is None:
                continue
            agv.path = pathA if random.random() < p_star else pathB

    def update_policy(self, reward: float, lr: float = 0.05):
        """Policy gradient ascent: shift weight toward rewarding route."""
        self.policy += np.array([reward, -reward]) * lr
        self.policy  = np.clip(self.policy, 0.01, 0.99)
        self.policy /= self.policy.sum()

    def step(self, p_star: float, wait_time: float) -> int:
        """One 100 ms cycle: assign → move → count deliveries."""
        self.assign_routes(p_star)
        deliveries = 0
        for agv in self.agvs:
            agv.move_one_step(self.G)
            if agv.deliveries:
                deliveries    += agv.deliveries
                agv.deliveries = 0
        self.throughput_log.append(deliveries)
        self.wait_log.append(wait_time)
        self.policy_log.append(self.policy[0])
        return deliveries


# ─────────────────────────────────────────────────────────
#  SECTION 5 — Main Simulation
# ─────────────────────────────────────────────────────────

class WarehouseSimulation:
    """Full simulation: queue → game → fleets → policy update."""

    def __init__(self, n_cycles=200, n_agvs=5,
                 lambda_base=40.0, mu=25.0, channels=2,
                 rows=5, cols=8, seed=42):

        random.seed(seed);  np.random.seed(seed)
        self.n_cycles     = n_cycles
        self.lambda_base  = lambda_base
        self.mu           = mu
        self.channels     = channels

        print("Building warehouse graph G = (V, E) …")
        self.G     = build_warehouse_graph(rows, cols, seed)
        self.nodes = list(self.G.nodes)
        print(f"  |V| = {len(self.nodes)},  |E| = {len(self.G.edges)}")

        self.alpha = Fleet("AlphaFleet","#185FA5",n_agvs,
                           self.G,self.nodes, seed=1)
        self.beta  = Fleet("BetaFleet", "#A32D2D",n_agvs,
                           self.G,self.nodes, seed=2)

        self.rho_log: List[float] = []
        self.W_log  : List[float] = []
        self.V_log  : List[float] = []
        self.ap_log : List[float] = []
        self.bq_log : List[float] = []
        self.t_log  : List[float] = []

    def _payoff_matrix(self, rho: float) -> np.ndarray:
        """
        Dynamic 2×2 payoff matrix — congestion scales penalties.
        Higher ρ makes shared-route choices more costly.
        """
        pen = rho * 5.0
        return np.array([
            [ 3.0 - pen*0.5,  -1.0 + pen*0.2],
            [-2.0 + pen*0.3,   4.0 - pen*0.4],
        ])

    def run(self):
        print(f"\nRunning {self.n_cycles} cycles at 10 Hz …")
        for cyc in range(self.n_cycles):

            # ── 1. Dynamic arrival rate (sinusoidal + noise) ──
            lam = max(1.0, self.lambda_base * (
                1 + 0.2 * math.sin(cyc * 0.1)
                  + random.uniform(-0.1, 0.1)))

            # ── 2. M/M/c → ρ, W ──────────────────────────────
            q   = MMCQueue(lam, self.mu, self.channels)
            rho = q.rho()
            W   = min(q.W(), 10.0)       # cap for numerical stability

            # ── 3. Payoff matrix → MSNE ───────────────────────
            game = ZeroSumGame(self._payoff_matrix(rho))
            p    = game.alpha_p_star()
            qv   = game.beta_q_star()
            V    = game.game_value()

            # ── 4. BetaFleet has one fewer effective channel ───
            W_b = min(MMCQueue(lam, self.mu,
                               max(1, self.channels-1)).W(), 10.0)

            # ── 5. Step fleets ────────────────────────────────
            ra = self.alpha.step(p,  W)
            rb = self.beta.step( qv, W_b)

            # ── 6. Policy gradient update ─────────────────────
            self.alpha.update_policy(ra - W  * 0.1)
            self.beta.update_policy(-(rb - W_b * 0.1))

            self.rho_log.append(rho);  self.W_log.append(W)
            self.V_log.append(V);      self.ap_log.append(p)
            self.bq_log.append(qv);    self.t_log.append(cyc * 0.1)

            if (cyc + 1) % 50 == 0:
                print(f"  [{cyc+1:>3}/{self.n_cycles}] "
                      f"ρ={rho:.3f}  W={W:.4f}s  "
                      f"V={V:+.3f}  p*={p:.3f}  q*={qv:.3f}")

        print("\nSimulation complete.")
        self._summary()

    def _summary(self):
        at = sum(self.alpha.throughput_log)
        bt = sum(self.beta.throughput_log)
        print("\n" + "="*52)
        print("  RESULTS SUMMARY")
        print("="*52)
        print(f"  AlphaFleet deliveries : {at}")
        print(f"  BetaFleet  deliveries : {bt}")
        print(f"  Alpha advantage       : {at - bt:+d}")
        print(f"  Avg ρ                 : {np.mean(self.rho_log):.4f}")
        print(f"  Avg W                 : {np.mean(self.W_log):.6f} s")
        print(f"  Avg game value V      : {np.mean(self.V_log):+.4f}")
        print(f"  Final p*  (Alpha)     : {self.ap_log[-1]:.4f}")
        print(f"  Final q*  (Beta)      : {self.bq_log[-1]:.4f}")
        print("="*52)

    # ── Plotting ──────────────────────────────────────────
    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(
            "Scenario 1 — Mechatronic Game Theory in "
            "Cyber-Physical Logistics Systems\n"
            "AlphaFleet (Algorithm A)  vs  BetaFleet (Algorithm B)",
            fontsize=12, fontweight='bold', y=0.98)
        t = self.t_log

        # Plot 1 — utilisation ρ
        ax = axes[0,0]
        ax.plot(t, self.rho_log, '#185FA5', lw=1.5, label='ρ')
        ax.axhline(1.0, color='#E24B4A', lw=1.2, ls='--',
                   label='ρ=1 (instability)')
        ax.fill_between(t, self.rho_log, 1.0,
                        where=[r > 0.85 for r in self.rho_log],
                        alpha=0.15, color='#E24B4A', label='Danger zone')
        ax.set(title='Network Utilisation ρ  (M/M/c)',
               xlabel='Time (s)', ylabel='ρ = λ/(c·μ)', ylim=(0,1.3))
        ax.legend(fontsize=8);  ax.grid(alpha=0.3)

        # Plot 2 — wait times
        ax = axes[0,1]
        ax.plot(t, self.alpha.wait_log, '#185FA5', lw=1.5,
                label='AlphaFleet W')
        ax.plot(t, self.beta.wait_log,  '#A32D2D', lw=1.5, ls='--',
                label='BetaFleet W')
        ax.fill_between(t, self.alpha.wait_log, self.beta.wait_log,
                        alpha=0.08, color='#888888')
        ax.set(title='Network Wait Time W per Fleet',
               xlabel='Time (s)', ylabel='Wait time (s)')
        ax.legend(fontsize=8);  ax.grid(alpha=0.3)

        # Plot 3 — cumulative deliveries
        ax = axes[1,0]
        ac = np.cumsum(self.alpha.throughput_log)
        bc = np.cumsum(self.beta.throughput_log)
        ax.plot(t, ac, '#185FA5', lw=2, label=f'AlphaFleet ({ac[-1]})')
        ax.plot(t, bc, '#A32D2D', lw=2, ls='--',
                label=f'BetaFleet ({bc[-1]})')
        ax.fill_between(t, ac, bc, alpha=0.08, color='#185FA5')
        ax.set(title='Cumulative Deliveries',
               xlabel='Time (s)', ylabel='Count')
        ax.legend(fontsize=8);  ax.grid(alpha=0.3)

        # Plot 4 — MSNE convergence
        ax = axes[1,1]
        ax.plot(t, self.ap_log, '#185FA5', lw=1.8,
                label="Alpha p*  P(Route A)")
        ax.plot(t, self.bq_log, '#A32D2D', lw=1.8, ls='--',
                label="Beta  q*  P(Route A)")
        ax.plot(t, self.V_log,  '#1D9E75', lw=1.2, alpha=0.7,
                label='Game value V')
        ax.axhline(0, color='#888888', lw=0.7, ls=':')
        ax.set(title='MSNE Strategy Convergence',
               xlabel='Time (s)', ylabel='Probability / Value')
        ax.legend(fontsize=8);  ax.grid(alpha=0.3)

        plt.tight_layout(rect=[0,0,1,0.95])
        out = "/mnt/user-data/outputs/scenario1_results.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Results plot saved → {out}")

    def plot_graph(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title(
            "Warehouse Graph  G = (V, E)\n"
            "Nodes = intersections  ·  Edges = aisles"
            "  (width ∝ traversal cost)",
            fontsize=12, fontweight='bold')

        pos     = nx.get_node_attributes(self.G, 'pos')
        weights = [self.G[u][v]['weight'] for u,v in self.G.edges]
        mn, mx  = min(weights), max(weights)
        widths  = [0.5 + 2.0*(w-mn)/(mx-mn) for w in weights]

        nx.draw_networkx_edges(self.G, pos, ax=ax, width=widths,
            alpha=0.45, edge_color='#AAAAAA', arrows=True,
            arrowsize=7, connectionstyle='arc3,rad=0.08')
        nx.draw_networkx_nodes(self.G, pos, ax=ax,
            node_color='#B5D4F4', node_size=160, alpha=0.9)

        a_pos = list({a.position for a in self.alpha.agvs})
        b_pos = list({b.position for b in self.beta.agvs})
        nx.draw_networkx_nodes(self.G, pos, ax=ax,
            nodelist=a_pos, node_color='#185FA5', node_size=260)
        nx.draw_networkx_nodes(self.G, pos, ax=ax,
            nodelist=b_pos, node_color='#A32D2D', node_size=260)
        nx.draw_networkx_labels(self.G, pos, ax=ax,
            font_size=6, font_color='#333333')

        ax.legend(handles=[
            mpatches.Patch(color='#185FA5', label='AlphaFleet AGV'),
            mpatches.Patch(color='#A32D2D', label='BetaFleet AGV'),
            mpatches.Patch(color='#B5D4F4', label='Empty node'),
        ], loc='upper right', fontsize=9)
        ax.axis('off')

        out = "/mnt/user-data/outputs/scenario1_graph.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Graph saved          → {out}")


# ─────────────────────────────────────────────────────────
#  SECTION 6 — Standalone Demos
# ─────────────────────────────────────────────────────────

def demo_mmc():
    print("\n" + "─"*52)
    print("  M/M/c QUEUE DEMO")
    print("─"*52)
    for lam, mu, c in [(40,25,2),(60,25,2),(60,25,3)]:
        q = MMCQueue(lam, mu, c)
        print(f"\n  λ={lam}, μ={mu}, c={c}")
        for k,v in q.summary().items():
            print(f"    {k:<10} = {v}")


def demo_game():
    print("\n" + "─"*52)
    print("  ZERO-SUM GAME DEMO")
    print("─"*52)
    A = np.array([[3,-1],[-2,4]], dtype=float)
    print(f"\n  Payoff matrix A:\n{A}")
    sol = ZeroSumGame(A).solve()
    for k,v in sol.items():
        print(f"    {k:<20} = {v}")


# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_mmc()
    demo_game()

    sim = WarehouseSimulation(
        n_cycles    = 200,
        n_agvs      = 5,
        lambda_base = 40.0,
        mu          = 25.0,
        channels    = 2,
        rows        = 5,
        cols        = 8,
        seed        = 42,
    )
    sim.plot_graph()
    sim.run()
    sim.plot_results()
