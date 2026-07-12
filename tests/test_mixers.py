import os
import sys
import unittest
from types import SimpleNamespace

import torch as th
import torch.nn as nn
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from modules.mixers import REGISTRY
from modules.mixers.smm_monotone import SMMSmoothMonotonicNN
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

    def test_amco_rejects_non_monotone_main_activation(self):
        args = self._args()
        args.amco_mono_activation = "silu"

        with self.assertRaises(ValueError):
            REGISTRY["amco"](args)

    def test_mmm2_mixer_parameter_counts_are_matched(self):
        counts = {}
        for name in ("qmix", "amco", "smm", "smnn"):
            config_path = os.path.join(
                ROOT, "src", "config", "algs", "{}.yaml".format(name)
            )
            with open(config_path) as config_file:
                config = yaml.safe_load(config_file)

            config.update(n_agents=10, state_shape=322)
            mixer = REGISTRY[name](SimpleNamespace(**config))
            counts[name] = sum(p.numel() for p in mixer.parameters())

        qmix_count = counts["qmix"]
        for name in ("amco", "smm", "smnn"):
            relative_difference = abs(counts[name] - qmix_count) / qmix_count
            self.assertLess(relative_difference, 0.01, name)

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

    def test_grouped_hll_is_monotone_in_all_original_agent_qs(self):
        args = self._args()
        args.n_agents = 24
        args.env_args = {"map_name": "bane_vs_bane"}
        args.hll_q_groups_by_map = {"bane_vs_bane": 8}
        args.hll_lattice_size_by_map = {"bane_vs_bane": 2}
        args.hll_max_vertices = 4096
        mixer = REGISTRY["hll"](args)

        lower_qs = th.randn(2, 3, args.n_agents)
        higher_qs = lower_qs + th.rand_like(lower_qs)
        states = th.randn(2, 3, args.state_shape)

        lower_output = mixer(lower_qs, states)
        higher_output = mixer(higher_qs, states)
        self.assertTrue(th.all(higher_output >= lower_output - 1e-7))

    def test_state_value_residual_does_not_change_agent_q_gradients(self):
        for name in ("amco", "smm", "smnn"):
            mixer = REGISTRY[name](self._args())
            agent_qs = th.randn(2, 4, 3, requires_grad=True)
            states = th.randn(2, 4, 10)

            output = mixer(agent_qs, states)
            grad = th.autograd.grad(output.sum(), agent_qs)[0]

            self.assertGreaterEqual(grad.min().item(), -1e-7, name)
            self.assertTrue(any(p.requires_grad for p in mixer.state_value.parameters()))

    def test_smm_temperature_matches_scaled_logsumexp(self):
        smm = SMMSmoothMonotonicNN(n=1, K=1, h_K=2, beta=0.0)
        with th.no_grad():
            smm.z[0].zero_()
            smm.t[0].copy_(th.tensor([0.0, 1.0]))
            smm.beta.fill_(0.0)

        output = smm(th.zeros(1, 1))
        expected = th.logsumexp(th.tensor([0.0, 1.0]), dim=0)

        self.assertTrue(th.allclose(output.squeeze(), expected))

    def test_smnn_core_is_centered_at_zero(self):
        mixer = REGISTRY["smnn"](self._args())
        zero_agent_qs = th.zeros(1, 3)
        zero_states = th.zeros(1, 10)

        core_output = mixer.smnn(
            zero_agent_qs, mixer.state_encoder(zero_states)
        )

        self.assertTrue(th.allclose(core_output, th.zeros_like(core_output)))


if __name__ == "__main__":
    unittest.main()
