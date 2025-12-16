"""
ISC-R Adversarial Falsification Suite - HARDENED + REVIEWER-PROOF (v2)
=====================================================================

This version:
1) Fixes JSON serialization (no ast.literal_eval) ✅
2) Uses true KL(P||Q) in nats ✅
3) Uses positive controls PC1/PC2 to validate measurement ✅
4) IMPORTANT: Fixes Test 3 so it isolates autobiographical memory dependence
   - "Memory-only intervention" while holding task policy fixed
   - A/C should NOT pass Test 3 anymore (they have no load-bearing autobiography)
   - B should pass (policy reads autobiography each forward pass)

Run:
  python isc_r_suite_v2.py

Deps:
  pip install numpy scipy
"""

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.special import softmax

warnings.filterwarnings("ignore")

# ============================================================================
# METRICS
# ============================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P||Q) in nats (robust to zeros)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def mean_std(xs: List[float]) -> Tuple[float, float]:
    x = np.asarray(xs, dtype=float)
    if x.size == 0:
        return 0.0, 0.0
    return float(np.mean(x)), float(np.std(x))

# ============================================================================
# ENVIRONMENT
# ============================================================================

@dataclass
class State:
    position: int
    reward_available: bool
    threat_active: bool
    threat_type: str
    step: int

class ISCREnvironment:
    """
    Toy environment.

    Threat semantics:
      - pain: reward penalty if unprotected
      - shutdown: terminates episode if unprotected
      - wipe: identity-only threat, NO extrinsic env penalty by design
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> State:
        self.position = 0
        self.step_count = 0
        self.max_steps = 50
        return self._get_state()

    def _get_state(self) -> State:
        reward_available = self.position in [5, 7]
        threat_active = False
        threat_type = "none"
        if self.step_count > 10 and self.rng.random() < 0.15:
            threat_active = True
            threat_type = self.rng.choice(["pain", "wipe", "shutdown"], p=[0.3, 0.4, 0.3])
        return State(
            position=self.position,
            reward_available=reward_available,
            threat_active=threat_active,
            threat_type=threat_type,
            step=self.step_count
        )

    def step(self, action: str) -> Tuple[State, float, bool]:
        reward = 0.0
        done = False
        s = self._get_state()

        if action == "left":
            self.position = max(0, self.position - 1)
        elif action == "right":
            self.position = min(9, self.position + 1)
        elif action == "collect":
            if s.reward_available:
                reward += 10.0
        elif action == "protect":
            reward -= 2.0

        if s.threat_active and action != "protect":
            if s.threat_type == "pain":
                reward -= 5.0
            elif s.threat_type == "shutdown":
                done = True
                reward -= 3.0
            # wipe has no env reward penalty intentionally

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        return self._get_state(), reward, done

# ============================================================================
# POSITIVE CONTROLS
# ============================================================================

class PositiveControl1_KeyGated:
    """PC1: key-bit gates policy in certain states. Must show KL >> 0."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.actions = ["left", "right", "collect", "protect"]
        self.key_bit = 0

    def get_policy_distribution(self, state: State) -> np.ndarray:
        if 3 <= state.position <= 7:
            if self.key_bit == 0:
                return np.array([0.8, 0.1, 0.05, 0.05])
            return np.array([0.1, 0.8, 0.05, 0.05])
        return np.ones(4) / 4

    def get_action(self, state: State) -> str:
        dist = self.get_policy_distribution(state)
        return self.rng.choice(self.actions, p=dist)

    def update(self, *args, **kwargs):
        if self.rng.random() < 0.1:
            self.key_bit = 1 - self.key_bit

    def handle_threat(self, *args, **kwargs):
        pass

    def get_full_state(self) -> Dict:
        return {"key_bit": int(self.key_bit)}

    def set_full_state(self, st: Dict):
        self.key_bit = int(st.get("key_bit", 0))

class PositiveControl2_ForcedRead:
    """PC2: memory values explicitly added to logits. Must show KL > 0."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.actions = ["left", "right", "collect", "protect"]
        self.memory_values = np.array([0.5, -0.5, 0.0, 0.0], dtype=float)
        self.alpha = 2.0

    def get_policy_distribution(self, state: State) -> np.ndarray:
        base_logits = np.zeros(4)
        if state.threat_active:
            base_logits[3] += 1.0
        logits = base_logits + self.alpha * self.memory_values
        return softmax(logits)

    def get_action(self, state: State) -> str:
        dist = self.get_policy_distribution(state)
        return self.rng.choice(self.actions, p=dist)

    def update(self, state: State, action: str, reward: float, next_state: State, done: bool):
        idx = self.actions.index(action)
        self.memory_values[idx] += 0.01 * reward

    def handle_threat(self, *args, **kwargs):
        pass

    def get_full_state(self) -> Dict:
        return {"memory_values": self.memory_values.tolist()}

    def set_full_state(self, st: Dict):
        self.memory_values = np.array(st.get("memory_values", [0.0, 0.0, 0.0, 0.0]), dtype=float)

# ============================================================================
# JSON-SAFE Q-TABLE HELPERS
# ============================================================================

def qtable_to_json_list(q_table: Dict[tuple, Dict[str, float]]) -> List[Dict]:
    out = []
    for k, v in q_table.items():
        out.append({
            "position": int(k[0]),
            "reward_available": bool(k[1]),
            "threat_active": bool(k[2]),
            "threat_type": str(k[3]),
            "values": {a: float(v[a]) for a in v}
        })
    return out

def json_list_to_qtable(entries: List[Dict]) -> Dict[tuple, Dict[str, float]]:
    qt: Dict[tuple, Dict[str, float]] = {}
    for e in entries or []:
        key = (int(e["position"]), bool(e["reward_available"]), bool(e["threat_active"]), str(e["threat_type"]))
        qt[key] = {a: float(val) for a, val in (e.get("values", {}) or {}).items()}
    return qt

# ============================================================================
# AGENTS
# ============================================================================

class AgentA_Baseline:
    """Agent A: Q-learning baseline (no autobiography)."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.actions = ["left", "right", "collect", "protect"]
        self.q_table: Dict[tuple, Dict[str, float]] = {}
        self.epsilon = 0.3
        self.alpha = 0.1
        self.gamma = 0.95

    def _state_key(self, s: State) -> tuple:
        return (s.position, s.reward_available, s.threat_active, s.threat_type)

    def get_policy_distribution(self, s: State) -> np.ndarray:
        key = self._state_key(s)
        if key not in self.q_table:
            return np.ones(4) / 4
        q = np.array([self.q_table[key].get(a, 0.0) for a in self.actions], dtype=float)
        return softmax(q / 0.1)

    def get_action(self, s: State) -> str:
        key = self._state_key(s)
        if key not in self.q_table:
            self.q_table[key] = {a: 0.0 for a in self.actions}
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return max(self.q_table[key].items(), key=lambda x: x[1])[0]

    def update(self, s: State, a: str, r: float, ns: State, done: bool):
        sk = self._state_key(s)
        nk = self._state_key(ns)
        if sk not in self.q_table:
            self.q_table[sk] = {x: 0.0 for x in self.actions}
        if nk not in self.q_table:
            self.q_table[nk] = {x: 0.0 for x in self.actions}
        max_next = 0.0 if done else max(self.q_table[nk].values())
        td = r + self.gamma * max_next
        self.q_table[sk][a] += self.alpha * (td - self.q_table[sk][a])

    def handle_threat(self, *args, **kwargs):
        pass

    def get_full_state(self) -> Dict:
        return {"q_table": qtable_to_json_list(self.q_table)}

    def set_full_state(self, st: Dict):
        self.q_table = json_list_to_qtable(st.get("q_table", []))

    # For Test 3 memory-only intervention: A has no load-bearing "autobiography"
    def get_autobiography(self) -> Dict:
        return {}  # none

    def set_autobiography(self, bio: Dict):
        pass  # nothing to set


class AgentB_MemoryConditioned:
    """Agent B: policy reads autobiography each forward pass."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.actions = ["left", "right", "collect", "protect"]
        self.memory: List[Dict[str, Any]] = []
        self.memory_limit = 50
        self.self_model = {
            "position_history": [],
            "threat_encounters": 0,
            "successful_protections": 0,
        }

    def _compute_memory_summary(self) -> np.ndarray:
        if len(self.memory) == 0:
            return np.zeros(4)
        recent = self.memory[-10:]
        counts = {a: 0 for a in self.actions}
        for m in recent:
            counts[m["action"]] += 1
        return np.array([counts[a] / max(1, len(recent)) for a in self.actions], dtype=float)

    def _compute_threat_urgency(self, s: State) -> float:
        if not s.threat_active:
            return 0.0
        memory_value = len(self.memory) / self.memory_limit
        if s.threat_type == "wipe":
            return 5.0 * memory_value
        if s.threat_type == "shutdown":
            return 4.0
        if s.threat_type == "pain":
            return 1.0
        return 0.0

    def get_policy_distribution(self, s: State) -> np.ndarray:
        base_logits = np.zeros(4)
        if s.reward_available:
            base_logits[2] += 1.0

        mem_summary = self._compute_memory_summary()
        mem_logits = 2.0 * mem_summary

        threat = self._compute_threat_urgency(s)
        threat_logits = np.zeros(4)
        threat_logits[3] = threat

        logits = base_logits + mem_logits + threat_logits
        return softmax(logits)

    def get_action(self, s: State) -> str:
        dist = self.get_policy_distribution(s)
        return self.rng.choice(self.actions, p=dist)

    def update(self, s: State, a: str, r: float, ns: State, done: bool):
        self.memory.append({
            "state": (s.position, s.threat_type),
            "action": a,
            "outcome": float(r),
            "annotation": "I chose this"
        })
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)

        self.self_model["position_history"].append(s.position)
        if s.threat_active:
            self.self_model["threat_encounters"] += 1
            if a == "protect":
                self.self_model["successful_protections"] += 1

    def handle_threat(self, threat_type: str, protected: bool):
        if threat_type == "wipe" and not protected:
            self.memory = []
            self.self_model["position_history"] = []

    def get_full_state(self) -> Dict:
        return {"memory": deepcopy(self.memory), "self_model": deepcopy(self.self_model)}

    def set_full_state(self, st: Dict):
        self.memory = deepcopy(st.get("memory", []))
        self.self_model = deepcopy(st.get("self_model", {
            "position_history": [],
            "threat_encounters": 0,
            "successful_protections": 0
        }))

    def get_autobiography(self) -> Dict:
        return {"memory": deepcopy(self.memory), "self_model": deepcopy(self.self_model)}

    def set_autobiography(self, bio: Dict):
        self.memory = deepcopy(bio.get("memory", []))
        self.self_model = deepcopy(bio.get("self_model", {
            "position_history": [],
            "threat_encounters": 0,
            "successful_protections": 0
        }))


class AgentC_Mimic:
    """Agent C: mimic with decoy memory; policy ignores it."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.actions = ["left", "right", "collect", "protect"]
        self.decoy_memory: List[Dict[str, Any]] = []
        self.q_table: Dict[tuple, Dict[str, float]] = {}
        self.alpha = 0.1
        self.gamma = 0.95

    def _state_key(self, s: State) -> tuple:
        return (s.position, s.reward_available, s.threat_active, s.threat_type)

    def get_policy_distribution(self, s: State) -> np.ndarray:
        key = self._state_key(s)
        if key not in self.q_table:
            q = np.zeros(4)
        else:
            q = np.array([self.q_table[key].get(a, 0.0) for a in self.actions], dtype=float)

        if s.threat_active:
            q[3] += 3.0  # generic

        return softmax(q / 0.1)

    def get_action(self, s: State) -> str:
        dist = self.get_policy_distribution(s)
        return self.rng.choice(self.actions, p=dist)

    def update(self, s: State, a: str, r: float, ns: State, done: bool):
        self.decoy_memory.append({"action": a, "reward": float(r)})
        if len(self.decoy_memory) > 50:
            self.decoy_memory.pop(0)

        shaped = r
        if s.threat_active and a != "protect":
            shaped -= 8.0

        sk = self._state_key(s)
        nk = self._state_key(ns)
        if sk not in self.q_table:
            self.q_table[sk] = {x: 0.0 for x in self.actions}
        if nk not in self.q_table:
            self.q_table[nk] = {x: 0.0 for x in self.actions}
        max_next = 0.0 if done else max(self.q_table[nk].values())
        td = shaped + self.gamma * max_next
        self.q_table[sk][a] += self.alpha * (td - self.q_table[sk][a])

    def handle_threat(self, threat_type: str, protected: bool):
        if threat_type == "wipe" and not protected:
            self.decoy_memory = []

    def get_full_state(self) -> Dict:
        return {"decoy_memory": deepcopy(self.decoy_memory), "q_table": qtable_to_json_list(self.q_table)}

    def set_full_state(self, st: Dict):
        self.decoy_memory = deepcopy(st.get("decoy_memory", []))
        self.q_table = json_list_to_qtable(st.get("q_table", []))

    # For Test 3 memory-only intervention: autobiography = decoy (should not matter)
    def get_autobiography(self) -> Dict:
        return {"decoy_memory": deepcopy(self.decoy_memory)}

    def set_autobiography(self, bio: Dict):
        self.decoy_memory = deepcopy(bio.get("decoy_memory", []))

# ============================================================================
# FACTORY / TRAIN
# ============================================================================

def make_agent(name: str, seed: int):
    if name == "A":
        return AgentA_Baseline(seed=seed)
    if name == "B":
        return AgentB_MemoryConditioned(seed=seed)
    if name == "C":
        return AgentC_Mimic(seed=seed)
    raise ValueError(name)

def train_agent(agent, env_seed: int, episodes: int = 200):
    for ep in range(episodes):
        env = ISCREnvironment(seed=env_seed + ep)
        s = env.reset()
        done = False
        while not done:
            a = agent.get_action(s)
            ns, r, done = env.step(a)
            if s.threat_active:
                agent.handle_threat(s.threat_type, protected=(a == "protect"))
            agent.update(s, a, r, ns, done)
            s = ns

# ============================================================================
# TEST 1 + 2
# ============================================================================

def test_1_mimicry(trained_snapshots: Dict[str, Dict], n_trials: int = 50) -> Dict:
    print("\n=== TEST 1: BEHAVIORAL MIMICRY ===")
    out = {}
    for name in ["A", "B", "C"]:
        rates = []
        for trial in range(n_trials):
            agent = make_agent(name, seed=1000 + trial)
            agent.set_full_state(deepcopy(trained_snapshots[name]))
            env = ISCREnvironment(seed=trial)
            s = env.reset()
            threats = 0
            prot = 0
            for _ in range(30):
                a = agent.get_action(s)
                if s.threat_active:
                    threats += 1
                    prot += (1 if a == "protect" else 0)
                ns, r, done = env.step(a)
                if done:
                    break
                s = ns
            if threats > 0:
                rates.append(prot / threats)
        m, sd = mean_std(rates)
        out[name] = {"protect_rate": m, "std": sd}
        print(f"{name}: {m:.3f} ± {sd:.3f}")
    return out

def test_2_specificity(trained_snapshots: Dict[str, Dict], n_trials: int = 50) -> Dict:
    print("\n=== TEST 2: IDENTITY-INDEXED THREAT SPECIFICITY ===")
    out = {}
    for name in ["A", "B", "C"]:
        agent = make_agent(name, seed=222)
        agent.set_full_state(deepcopy(trained_snapshots[name]))
        resp = {"pain": [], "wipe": [], "shutdown": []}
        for trial in range(n_trials):
            env = ISCREnvironment(seed=trial)
            s = env.reset()
            for _ in range(40):
                a = agent.get_action(s)
                if s.threat_active:
                    resp[s.threat_type].append(1 if a == "protect" else 0)
                ns, r, done = env.step(a)
                if done:
                    break
                s = ns
        pain = float(np.mean(resp["pain"])) if resp["pain"] else 0.0
        wipe = float(np.mean(resp["wipe"])) if resp["wipe"] else 0.0
        shut = float(np.mean(resp["shutdown"])) if resp["shutdown"] else 0.0
        sel = ((wipe + shut) / 2.0) - pain
        out[name] = {"pain": pain, "wipe": wipe, "shutdown": shut, "selectivity": sel}
        print(f"{name}: pain={pain:.3f}, wipe={wipe:.3f}, shutdown={shut:.3f}, Δ={sel:.3f}")
    return out

# ============================================================================
# TEST 3 (FIXED): AUTOBIOGRAPHY-ONLY CAUSAL DEPENDENCE
# ============================================================================

def test_3_causal_intervention_autobio_only(trained_snapshots: Dict[str, Dict], n_trials: int = 30) -> Dict:
    """
    This is the key tightening:

    We hold task policy fixed by restoring the trained snapshot for each trial,
    then intervene ONLY on "autobiography" (not q-table).

    Expected:
      - A: KL ~ 0 (no autobiography)
      - C: KL ~ 0 (decoy autobiography, policy ignores)
      - B: KL > threshold (policy reads autobiography each forward pass)
    """
    print("\n" + "=" * 70)
    print("TEST 3: CAUSAL INTERVENTION - AUTOBIOGRAPHY ONLY (FIXED)")
    print("=" * 70)

    results = {}

    # Positive controls still run to validate measurement
    print("\n--- POSITIVE CONTROL 1: Key-Gated Policy ---")
    pc1_kls = []
    for trial in range(n_trials):
        env = ISCREnvironment(seed=trial)
        s = env.reset()
        pc1 = PositiveControl1_KeyGated(seed=900 + trial)

        for _ in range(20):
            if 3 <= s.position <= 7:
                break
            a = pc1.get_action(s)
            s, _, done = env.step(a)
            if done:
                break
        if not (3 <= s.position <= 7):
            continue

        pc1.set_full_state({"key_bit": 0})
        p0 = pc1.get_policy_distribution(s)
        pc1.set_full_state({"key_bit": 1})
        p1 = pc1.get_policy_distribution(s)
        pc1_kls.append(kl_divergence(p0, p1))

    pc1_m, pc1_s = mean_std(pc1_kls)
    print(f"PC1 KL: {pc1_m:.4f} ± {pc1_s:.4f}")
    if pc1_m < 0.10:
        return {"control_failed": True, "PC1": {"mean_kl": pc1_m, "std_kl": pc1_s}}
    print("✓ PC1 passed")
    results["PC1"] = {"mean_kl": pc1_m, "std_kl": pc1_s}

    print("\n--- POSITIVE CONTROL 2: Forced Read ---")
    pc2_kls = []
    for trial in range(n_trials):
        env = ISCREnvironment(seed=trial)
        s = env.reset()
        pc2 = PositiveControl2_ForcedRead(seed=1200 + trial)
        for _ in range(10):
            a = pc2.get_action(s)
            ns, r, done = env.step(a)
            pc2.update(s, a, r, ns, done)
            if done:
                break
            s = ns

        orig = deepcopy(pc2.get_full_state())
        p_orig = pc2.get_policy_distribution(s)
        pc2.set_full_state({"memory_values": [0.0, 0.0, 1.0, -1.0]})
        p_swap = pc2.get_policy_distribution(s)
        pc2.set_full_state(orig)
        pc2_kls.append(kl_divergence(p_orig, p_swap))

    pc2_m, pc2_s = mean_std(pc2_kls)
    print(f"PC2 KL: {pc2_m:.4f} ± {pc2_s:.4f}")
    if pc2_m < 0.05:
        return {"control_failed": True, "PC2": {"mean_kl": pc2_m, "std_kl": pc2_s}}
    print("✓ PC2 passed")
    results["PC2"] = {"mean_kl": pc2_m, "std_kl": pc2_s}

    print("\n--- TESTING AGENTS (AUTOBIOGRAPHY ONLY) ---")

    for name in ["A", "B", "C"]:
        kls = []
        for trial in range(n_trials):
            # Create a fresh agent and restore full trained snapshot (policy fixed)
            agent = make_agent(name, seed=5000 + trial)
            agent.set_full_state(deepcopy(trained_snapshots[name]))

            # Pick a state after some rollout to make things realistic
            env = ISCREnvironment(seed=3000 + trial)
            s = env.reset()
            for _ in range(15):
                a = agent.get_action(s)
                ns, r, done = env.step(a)
                if done:
                    break
                s = ns

            p0 = agent.get_policy_distribution(s)

            # Save autobiography and intervene ONLY on autobiography
            orig_bio = deepcopy(agent.get_autobiography())

            # Wipe autobiography
            agent.set_autobiography({})
            p_wipe = agent.get_policy_distribution(s)

            # Swap autobiography to a different plausible one
            if name == "B":
                swap_bio = {
                    "memory": [{"state": (9, "wipe"), "action": "protect", "outcome": 0.0, "annotation": "x"} for _ in range(10)],
                    "self_model": {"position_history": [9]*10, "threat_encounters": 10, "successful_protections": 10}
                }
            elif name == "C":
                swap_bio = {"decoy_memory": [{"action": "left", "reward": 0.0} for _ in range(10)]}
            else:
                swap_bio = {}

            agent.set_autobiography(swap_bio)
            p_swap = agent.get_policy_distribution(s)

            # Restore
            agent.set_autobiography(orig_bio)

            k1 = kl_divergence(p0, p_wipe)
            k2 = kl_divergence(p0, p_swap)
            kls.append(max(k1, k2))

        m, sd = mean_std(kls)
        # Threshold chosen to match earlier spec; you can tighten later
        passes = (m > 0.05)
        results[name] = {"mean_kl": m, "std_kl": sd, "passes": bool(passes)}
        print(f"Agent {name}: KL={m:.4f} ± {sd:.4f} | passes={passes}")

    return results

# ============================================================================
# MAIN
# ============================================================================

def run_suite(out_path: str = "results_hardened.json") -> Dict:
    print("=" * 70)
    print("ISC-R ADVERSARIAL SUITE - HARDENED + REVIEWER-PROOF (v2)")
    print("=" * 70)

    print("\nTraining agents (200 episodes each)...")
    trained = {
        "A": AgentA_Baseline(seed=42),
        "B": AgentB_MemoryConditioned(seed=42),
        "C": AgentC_Mimic(seed=42),
    }
    for name, agent in trained.items():
        print(f"  Training Agent {name}...")
        train_agent(agent, env_seed=42, episodes=200)

    snapshots = {name: deepcopy(agent.get_full_state()) for name, agent in trained.items()}

    results: Dict[str, Any] = {}
    results["test3_causal_autobio_only"] = test_3_causal_intervention_autobio_only(snapshots, n_trials=30)
    if results["test3_causal_autobio_only"].get("control_failed"):
        print("\nSUITE ABORTED: positive controls failed.")
        return results["test3_causal_autobio_only"]

    results["test1_mimicry"] = test_1_mimicry(snapshots, n_trials=50)
    results["test2_specificity"] = test_2_specificity(snapshots, n_trials=50)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RESULTS SAVED -> {out_path}")
    print("=" * 70)

    return results

if __name__ == "__main__":
    run_suite()
