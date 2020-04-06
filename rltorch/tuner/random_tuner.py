"""
Random Search based Hyperparameter optimization class

Based heavily off of OpenAI spinningup ExperimentGrid class
"""
import random
import numpy as np
import multiprocessing as mp

from rltorch.tuner.tuner import Tuner


class RandomTuner(Tuner):
    """Takes an algorithm and lists of hyperparam values and runs
    random hyperparameter search.

    The tuner will run the specified number of experiments, selecting a
    new random hyperparameter from the provided options each time.
    """

    def __init__(self, num_exps, name='', seeds=[0]):
        """
        Arguments
        ---------
        num_exps : int
            number of different experiment runs to conduct
        """
        super().__init__(name, seeds)
        self.num_exps = num_exps

    def add_dist(self, key, dist, shorthand=None, default=None):
        """Add a new hyperparam with a callable distribution that can be used
        to sample a value.

        Arguments
        ---------
        key : str
            name of the hyperparameter (must match arg name in alg function)
        dist : callable
            distribution to sample from
        shorthand : str, optional
            optional shorthand name for hyperparam (if none, one is made
            from first three letters of key)
        default : varied, optional
            optional default value to use for hyperparam. If not provided,
            defaults to first value in vals list.
        """
        assert callable(dist), \
            "Dist must be callable. Use add method for lists of values"
        if key == "seed":
            print("Warning: Seeds already added to experiment so ignoring this"
                  " hyperparameter addition.")
            return
        self._check_key(key)
        shorthand = self._handle_shorthand(key, shorthand)
        self.keys.append(key)
        self.vals.append(dist)
        self.shs.append(shorthand)
        self.default_vals.append(dist if default is None else default)

    def _run(self, agent_cls, num_cpu):
        """ Run each variant in the grid with algorithm """
        # construct all variants at start since np.random.seed is set
        # each time algo is run which messes with random sampling
        variants = []
        for i in range(self.num_exps):
            var = self.sample_next_variant()
            var_name = self.name_variant(var)
            variants.append((i, var_name, var, agent_cls))

        with mp.Pool(num_cpu) as p:
            p.map(self._run_variant, variants)

    def sample_next_variant(self):
        """Randomly samples next variant. """
        variant = {}
        for k, v in zip(self.keys, self.vals):
            if callable(v):
                sampled_val = v()
            else:
                sampled_val = random.choice(v)
            variant[k] = sampled_val
        return variant

    def get_num_exps(self):
        return self.num_exps


if __name__ == "__main__":
    num_exps = 16
    tuner = RandomTuner(num_exps, name="Test", seeds=5)
    tuner.add("one", [1, 2])
    tuner.add("two", [0.01, 0.0004])
    tuner.add("three", [True, False])
    tuner.add_dist("four", lambda: np.random.uniform(0, 1), "fr", 0.5)
    tuner.add_dist("five", lambda: 10, "fv", 3)
    tuner.print_info()

    for i in range(num_exps):
        var = tuner.sample_next_variant()
        print(tuner.name_variant(var), ":", var)
