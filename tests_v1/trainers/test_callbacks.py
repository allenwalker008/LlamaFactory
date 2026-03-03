"""Unit tests for the callback system.

These tests exercise the callback primitives (TrainerState, TrainerCallback,
CallbackHandler) without importing heavy dependencies (torch, deepspeed, …).
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal stubs so we can import from the callbacks package without torch
# ---------------------------------------------------------------------------

@dataclass
class _FakeArgs:
    logging_steps: int = 1


# ---------------------------------------------------------------------------
# Import the actual callback classes
# ---------------------------------------------------------------------------

import sys, types  # noqa: E402

# Stub out heavy modules before importing callbacks
for mod_name in (
    "torch", "torch.nn", "torch.nn.functional",
    "llamafactory.v1.accelerator.interface",
    "llamafactory.v1.accelerator.helper",
    "llamafactory.v1.utils",
    "llamafactory.v1.utils.logging",
    "llamafactory.v1.utils.helper",
    "llamafactory.v1.utils.types",
):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Provide a minimal logging stub
_log_mod = sys.modules["llamafactory.v1.utils.logging"]
_logger_stub = MagicMock()
_log_mod.get_logger = lambda *a, **kw: _logger_stub  # type: ignore

from llamafactory.v1.utils.callbacks.trainer_callback import (  # noqa: E402
    CallbackHandler,
    TrainerCallback,
    TrainerState,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class RecordingCallback(TrainerCallback):
    """Records every hook call for assertion."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.logged: list[dict[str, Any]] = []

    def on_train_begin(self, args, state, **kw):
        self.calls.append("on_train_begin")

    def on_train_end(self, args, state, **kw):
        self.calls.append("on_train_end")

    def on_epoch_begin(self, args, state, **kw):
        self.calls.append("on_epoch_begin")

    def on_epoch_end(self, args, state, **kw):
        self.calls.append("on_epoch_end")

    def on_step_begin(self, args, state, **kw):
        self.calls.append("on_step_begin")

    def on_step_end(self, args, state, **kw):
        self.calls.append("on_step_end")

    def on_log(self, args, state, logs, **kw):
        self.calls.append("on_log")
        self.logged.append(dict(logs))

    def on_save(self, args, state, **kw):
        self.calls.append("on_save")


def _simulate_steps(handler, args, state, num_steps=3):
    """Simulate a training loop (trainer manages scheduling)."""
    handler.on_train_begin(args, state)
    handler.on_epoch_begin(args, state)
    for step in range(1, num_steps + 1):
        state.global_step = step
        state.loss = 1.0 / step
        handler.on_step_begin(args, state)
        handler.on_step_end(args, state)

        # Trainer-managed scheduling
        if step % args.logging_steps == 0:
            logs = {"step": step, "loss": state.loss}
            handler.on_log(args, state, logs)

    handler.on_epoch_end(args, state)
    handler.on_train_end(args, state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCallbackSystem(unittest.TestCase):
    """Core tests for the simplified callback system."""

    def test_hook_order(self):
        """on_train_begin → on_epoch_begin → steps → on_epoch_end → on_train_end."""
        rec = RecordingCallback()
        handler = CallbackHandler([rec])
        state = TrainerState(num_training_steps=2)
        args = _FakeArgs(logging_steps=1)

        _simulate_steps(handler, args, state, num_steps=2)

        expected = [
            "on_train_begin",
            "on_epoch_begin",
            "on_step_begin", "on_step_end", "on_log",
            "on_step_begin", "on_step_end", "on_log",
            "on_epoch_end",
            "on_train_end",
        ]
        self.assertEqual(rec.calls, expected)

    def test_multi_callback(self):
        """Multiple callbacks all receive the same hooks."""
        rec1, rec2 = RecordingCallback(), RecordingCallback()
        handler = CallbackHandler([rec1, rec2])
        state = TrainerState(num_training_steps=1)
        args = _FakeArgs(logging_steps=1)

        _simulate_steps(handler, args, state, num_steps=1)
        self.assertEqual(rec1.calls, rec2.calls)

    def test_log_history(self):
        """LoggingCallback appends to state.log_history (simulated)."""
        state = TrainerState(num_training_steps=3)

        class HistoryCallback(TrainerCallback):
            def on_log(self, args, state, logs, **kw):
                state.log_history.append(dict(logs))

        handler = CallbackHandler([HistoryCallback()])
        args = _FakeArgs(logging_steps=1)
        _simulate_steps(handler, args, state, num_steps=3)

        self.assertEqual(len(state.log_history), 3)
        self.assertEqual(state.log_history[0]["step"], 1)
        self.assertEqual(state.log_history[2]["step"], 3)

    def test_logging_steps_frequency(self):
        """on_log is called only at multiples of logging_steps."""
        rec = RecordingCallback()
        handler = CallbackHandler([rec])
        state = TrainerState(num_training_steps=6)
        args = _FakeArgs(logging_steps=3)

        _simulate_steps(handler, args, state, num_steps=6)
        # Steps 3 and 6 should trigger on_log
        log_calls = [c for c in rec.calls if c == "on_log"]
        self.assertEqual(len(log_calls), 2)


    def test_add_callback_dynamic(self):
        """add_callback() appends after construction."""
        handler = CallbackHandler([])
        rec = RecordingCallback()
        handler.add_callback(rec)

        state = TrainerState(num_training_steps=1)
        args = _FakeArgs(logging_steps=1)
        _simulate_steps(handler, args, state, num_steps=1)

        self.assertIn("on_train_begin", rec.calls)

    def test_kwargs_forwarded(self):
        """kwargs from _call (model, optimizer, …) are forwarded to callbacks."""
        received = {}

        class KwargsCapture(TrainerCallback):
            def on_step_end(self, args, state, **kwargs):
                received.update(kwargs)

        fake_trainer = MagicMock()
        fake_trainer.model = "fake_model"
        fake_trainer.optimizer = "fake_optim"
        fake_trainer.lr_scheduler = "fake_sched"
        fake_trainer.train_batch_generator = "fake_loader"

        handler = CallbackHandler([KwargsCapture()], trainer=fake_trainer)
        state = TrainerState(num_training_steps=1)
        args = _FakeArgs()
        handler.on_step_end(args, state)

        self.assertEqual(received["model"], "fake_model")
        self.assertEqual(received["optimizer"], "fake_optim")
        self.assertEqual(received["lr_scheduler"], "fake_sched")
        self.assertEqual(received["train_dataloader"], "fake_loader")

    def test_no_control_needed(self):
        """Callbacks are pure observers — no TrainerControl involved."""
        # Simply verify TrainerControl doesn't exist in the module
        import llamafactory.v1.utils.callbacks.trainer_callback as mod
        self.assertFalse(hasattr(mod, "TrainerControl"))


if __name__ == "__main__":
    unittest.main()
