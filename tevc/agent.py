"""The D1 agent (spec §3): perception, memory, surrogate-seeded GA and the recurrence rule.

Each environment change costs real evaluations (training a diagnoser); the agent spends them where its memory and
surrogate expect high fitness, and redeploys a stored model at once when a regime recurs. It sees only the
agent-visible fitness; test-set accuracy never reaches it.
"""
import numpy as np

from tevc.optimizers import N_GENES, N_INIT, decode, encode, ga_search
from tevc.surrogate import Surrogate

CHANGE = np.log(1.05)       # perception: |ln(s_t / s_prev)| above this is a change
RECUR = np.log(1.02)        # recurrence: a past environment within this |Δd| is the same regime
N_CANDIDATES = 1000         # random candidates ranked by the surrogate for a new initial population
N_MEMORY = 2                # deployed recipes of the nearest past environments seeded into a new search
REDUCED = {"n_gen": 1, "n_off": 5, "n_reeval": 1}       # with 4 memory recipes: B / 3 = 10 evaluations
N_REDUCED_INIT = 4
VISIT_SEED_STEP = 7919      # a recurrence trains fresh models instead of re-reading cached ones


class Agent:
    def __init__(self, library_rpm, seeds, run_seed=0, beta=None):
        self.library_rpm, self.seeds, self.beta, self.run_seed = float(library_rpm), seeds, beta, run_seed
        self.prev_rpm = self.library_rpm
        self.archive = []       # (d, genes, fitness) of every evaluation
        self.deployed = []      # (d, Trial) per handled environment
        self.visits = 0
        self.log = []

    def descriptor(self, rpm):
        return float(np.log(rpm / self.library_rpm))

    def _surrogate(self, d, trials):
        X = [np.r_[g, dp] for dp, g, _ in self.archive] + [np.r_[encode(t.recipe), d] for t in trials]
        y = [f for _, _, f in self.archive] + [t.fitness for t in trials]
        return Surrogate(seed=self.run_seed).fit(X, y)

    def _screen(self, d):
        def screen(candidates, trials):
            s = self._surrogate(d, trials)
            return s.acquisition(candidates, d) if s.ready else np.zeros(len(candidates))
        return screen

    def _distinct(self, genes, n):
        out, recipes = [], []
        for g in genes:
            r = decode(g, self.beta)
            if r not in recipes:
                out.append(np.asarray(g, dtype=float))
                recipes.append(r)
            if len(out) == n:
                break
        return out

    def _initial(self, d, init_rng, op_rng):
        """New regime: deployed recipes of the nearest past environments + surrogate-best random candidates."""
        if not self.archive:
            return init_rng.random((N_INIT, N_GENES))       # first change: the same random start as GA-RESTART
        nearest = sorted(self.deployed, key=lambda e: abs(e[0] - d))[:N_MEMORY]
        candidates = op_rng.random((N_CANDIDATES, N_GENES))
        s = self._surrogate(d, [])
        order = np.argsort(-s.acquisition(candidates, d), kind="stable") if s.ready else np.arange(N_CANDIDATES)
        return np.array(self._distinct([encode(t.recipe) for _, t in nearest] + list(candidates[order]), N_INIT))

    def handle(self, evaluator, env, init_rng, op_rng):
        """One environment. Returns (carried, trials): stored models redeployed at no evaluation cost, and the
        newly trained models in evaluation order."""
        d = self.descriptor(env.rpm)
        changed = abs(np.log(env.rpm / self.prev_rpm)) > CHANGE
        self.prev_rpm = env.rpm
        if not changed and self.deployed:
            self.log.append({"env": env.name, "changed": False, "recurrence": False, "evaluations": 0})
            return [self.deployed[-1][1]], []
        past = [t for dp, t in self.deployed if abs(dp - d) < RECUR]
        if past:
            self.visits += 1
            carried = [past[-1]]
            near = sorted(((f, g) for dp, g, f in self.archive if abs(dp - d) < RECUR), key=lambda e: -e[0])
            init = self._distinct([g for _, g in near] + list(op_rng.random((N_CANDIDATES, N_GENES))),
                                  N_REDUCED_INIT)
            seeds = tuple(s + VISIT_SEED_STEP * self.visits for s in self.seeds)
            trials = ga_search(evaluator, env, init, op_rng, seeds, screen=self._screen(d), beta=self.beta,
                               **REDUCED)
        else:
            carried = []
            trials = ga_search(evaluator, env, self._initial(d, init_rng, op_rng), op_rng, self.seeds,
                               screen=self._screen(d), beta=self.beta)
        self.archive += [(d, encode(t.recipe), t.fitness) for t in trials]
        self.deployed.append((d, max(carried + trials, key=lambda t: t.fitness)))
        self.log.append({"env": env.name, "changed": True, "recurrence": bool(past), "evaluations": len(trials)})
        return carried, trials
