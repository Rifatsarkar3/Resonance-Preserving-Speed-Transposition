"""The D1 agent (spec §3), v2: perception, memory, a GP world model and evolutionary expected-improvement search.

Each environment change costs real evaluations (training a diagnoser). The agent perceives a change from the speed,
recalls what it learnt at the nearest past speeds, models fitness over recipe genes ⊕ d = ln(s / s_lib) from every
past evaluation, and spends the budget on expected-improvement picks among uniform and evolved candidate recipes.
When a regime recurs it redeploys the stored model at once and keeps improving on it with the full budget. With an
empty memory it starts from the RPST recipe of earlier work (ANCHORS["H2_LIKE"]). It sees only the agent-visible
fitness; test-set accuracy never reaches it.
v2 (2026-10-10, development data only) replaced v1's random-forest-screened GA and its 10-evaluation search after a
recurrence, which lost to WARM-TRANSFER on PU (git history keeps v1). Every method spends B evaluations per change.
"""
import numpy as np

from tevc.optimizers import B, N_GENES, N_INIT, decode, ei_search, encode, fit_gp, ga_search
from tevc.recipe import ANCHORS

CHANGE = np.log(1.05)       # perception: |ln(s_t / s_prev)| above this is a change
RECUR = np.log(1.02)        # recurrence: a past environment within this |Δd| is the same regime
PRIOR = ANCHORS["H2_LIKE"]  # the RPST recipe of earlier work: the first evaluation when memory is empty
N_CANDIDATES = 1000         # random candidates ranked by the world model for a new regime's start
N_MEMORY = 2                # deployed recipes of the nearest past environments seeded into a new regime
N_MODEL = 2                 # + the world model's best guesses at the new speed
N_RECALL = 2                # a recurring regime's best recipes, retrained first with the visit's fresh seed
VISIT_SEED_STEP = 7919      # a recurrence trains fresh models instead of re-reading cached ones
NOSURR_GA = {"n_gen": 3, "n_off": 6, "n_reeval": 0}     # ablation without world model: 12 + 18 = B evaluations


class Agent:
    """memory / surrogate / evolve / prior switch the agent's components off for the ablations (spec §5)."""

    def __init__(self, library_rpm, seeds, run_seed=0, beta=None, memory=True, surrogate=True, evolve=True,
                 prior=True):
        self.library_rpm, self.seeds, self.beta, self.run_seed = float(library_rpm), seeds, beta, run_seed
        self.memory, self.use_surrogate, self.evolve, self.prior = memory, surrogate, evolve, prior
        self.prev_rpm = self.library_rpm
        self.archive = []       # (d, genes, fitness) of every evaluation
        self.deployed = []      # (d, Trial) per handled environment
        self.visits = 0
        self.log = []

    def descriptor(self, rpm):
        return float(np.log(rpm / self.library_rpm))

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

    def _model_best(self, d, rng):
        """Random candidates, best first by the world model's posterior mean at d (unranked without it)."""
        candidates = np.array([encode(decode(c, self.beta)) for c in rng.random((N_CANDIDATES, N_GENES))])
        if not self.use_surrogate:
            return list(candidates)
        gp = fit_gp([np.r_[g, dp] for dp, g, _ in self.archive], [f for _, _, f in self.archive], self.run_seed)
        mean = gp.predict(np.column_stack([candidates, np.full(len(candidates), d)]))
        return list(candidates[np.argsort(-mean, kind="stable")])

    def _start(self, d, past, init_rng, op_rng):
        if past:                # recurrence: the regime's best recipes first
            regime = sorted(((f, g) for dp, g, f in self.archive if abs(dp - d) < RECUR), key=lambda e: -e[0])
            return self._distinct([g for _, g in regime], N_RECALL)
        if self.archive:        # new regime: nearest deployed recipes, then the world model's best guesses
            nearest = sorted(self.deployed, key=lambda e: abs(e[0] - d))[:N_MEMORY]
            return self._distinct([encode(t.recipe) for _, t in nearest] + self._model_best(d, op_rng),
                                  N_MEMORY + N_MODEL)
        start = list(init_rng.random((N_INIT, N_GENES)))   # empty memory: the start shared with GA-RESTART
        return self._distinct(([encode(PRIOR)] if self.prior else []) + start, N_INIT)

    def _search(self, evaluator, env, d, init, rng, seed, carried):
        if not self.use_surrogate:      # ablation: memory-seeded GA without a world model
            init = self._distinct(list(init) + list(rng.random((N_CANDIDATES, N_GENES))), N_INIT)
            return ga_search(evaluator, env, init, rng, (seed, seed), beta=self.beta, **NOSURR_GA)
        return ei_search(evaluator, env, init, rng, seed, d, archive=self.archive,
                         elites=[(g, f) for dp, g, f in self.archive if abs(dp - d) < RECUR],
                         incumbent=max((t.fitness for t in carried), default=-np.inf), beta=self.beta,
                         evolve=self.evolve, budget=B)

    def handle(self, evaluator, env, init_rng, op_rng):
        """One environment. Returns (carried, trials): stored models redeployed at no evaluation cost, and the
        newly trained models in evaluation order."""
        d = self.descriptor(env.rpm)
        changed = abs(np.log(env.rpm / self.prev_rpm)) > CHANGE
        self.prev_rpm = env.rpm
        if not self.memory:
            self.archive, self.deployed = [], []        # ablation: every change starts from scratch
        if not changed and self.deployed:
            self.log.append({"env": env.name, "changed": False, "recurrence": False, "evaluations": 0})
            return [self.deployed[-1][1]], []
        past = [t for dp, t in self.deployed if abs(dp - d) < RECUR]
        seed = self.seeds[0]
        if past:
            self.visits += 1
            seed += VISIT_SEED_STEP * self.visits
        carried = [past[-1]] if past else []
        trials = self._search(evaluator, env, d, self._start(d, past, init_rng, op_rng), op_rng, seed, carried)
        self.archive += [(d, encode(t.recipe), t.fitness) for t in trials]
        self.deployed.append((d, max(carried + trials, key=lambda t: t.fitness)))
        self.log.append({"env": env.name, "changed": True, "recurrence": bool(past), "evaluations": len(trials)})
        return carried, trials
