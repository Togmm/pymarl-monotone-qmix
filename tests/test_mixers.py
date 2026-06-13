import os
import sys
import unittest
from types import SimpleNamespace

import torch as th
import torch.nn as nn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from modules.mixers import REGISTRY
from learners.q_learner import QLearner


class _DummyMAC(nn.Module):
    def __init__(self):
        super(_DummyMAC, self).__init__()
        self.weight = nn.Parameter(th.zeros(1))


class _DummyConsoleLogger(object):
    def info(self, message):
        pass


class _DummyLogger(object):
    console_logger = _DummyConsoleLogger()


class MixerTest(unittest.TestCase):
    def _args(self):
        return SimpleNamespace(
            n_agents=3,
            state_shape=10,
            mixing_embed_dim=16,
            hypernet_layers=2,
            hypernet_embed=16,
            amco_state_embed_dim=8,
            amco_mono_hidden_dim=16,
            amco_mono_depth=4,
            smm_state_embed_dim=8,
            smm_K=3,
            smm_h_K=4,
            smnn_state_embed_dim=8,
            smnn_exp_unit_size=[8, 8],
            smnn_relu_unit_size=[6, 6],
            smnn_conf_unit_size=[6, 6],
            lr=0.0005,
            optim_alpha=0.99,
            optim_eps=0.00001,
            learner_log_interval=2000,
        )

    def test_q_learner_constructs_each_registered_mixer(self):
        for name in ("vdn", "qmix", "amco", "smm", "smnn"):
            args = self._args()
            args.mixer = name
            learner = QLearner(_DummyMAC(), {}, _DummyLogger(), args)

            self.assertEqual(
                learner.mixer.__class__, learner.target_mixer.__class__, name
            )

    def test_registered_mixers_forward_and_backward(self):
        for name in ("vdn", "qmix", "amco", "smm", "smnn"):
            mixer = REGISTRY[name](self._args())
            agent_qs = th.randn(2, 4, 3, requires_grad=True)
            states = th.randn(2, 4, 10)

            output = mixer(agent_qs, states)
            self.assertEqual(tuple(output.shape), (2, 4, 1), name)

            output.sum().backward()
            self.assertIsNotNone(agent_qs.grad, name)

    def test_partial_monotone_mixers_are_monotone_in_agent_qs(self):
        for name in ("amco", "smm", "smnn"):
            mixer = REGISTRY[name](self._args())
            agent_qs = th.randn(2, 4, 3, requires_grad=True)
            states = th.randn(2, 4, 10)

            mixer(agent_qs, states).sum().backward()
            min_gradient = agent_qs.grad.min().item()
            self.assertGreaterEqual(min_gradient, -1e-7, name)


if __name__ == "__main__":
    unittest.main()
