"""Scheduler tests with the CPU MockBackend (OOM splitting, failures, timeouts, resume, online batching)."""

from __future__ import annotations

from PIL import Image

from vle.batching.scheduler import BatchScheduler, SchedulerConfig
from vle.inference.base import InferenceRequest
from vle.inference.mock_backend import MockBackend


def _reqs(n):
    img = Image.new("RGB", (32, 32))
    return [InferenceRequest(prompt_id="p", text="t", images=[img], image_ids=[f"i{k}"], request_id=f"r{k}") for k in range(n)]


def test_static_all_ok():
    s = BatchScheduler(MockBackend(), SchedulerConfig(policy="static", batch_size=4))
    st = s.run(_reqs(10))
    assert st.n_ok == 10 and st.batch_sizes == [4, 4, 2]
    assert len(st.request_latencies) == 10


def test_independent_batch_one():
    st = BatchScheduler(MockBackend(), SchedulerConfig(policy="independent", batch_size=8)).run(_reqs(5))
    assert st.batch_sizes == [1] * 5


def test_oom_split():
    s = BatchScheduler(MockBackend(max_batch=3), SchedulerConfig(policy="static", batch_size=8))
    st = s.run(_reqs(8))
    assert st.n_ok == 8 and st.n_oom_events >= 1 and max(st.batch_sizes) <= 3


def test_failure_isolated_by_bisection():
    s = BatchScheduler(MockBackend(fail_ids=["i3"]), SchedulerConfig(policy="static", batch_size=8, max_retries=1))
    st = s.run(_reqs(8))
    assert st.n_ok == 7 and st.n_failed == 1 and st.n_retries == 1


def test_timeout():
    s = BatchScheduler(MockBackend(hang_ids=["i1"]), SchedulerConfig(policy="static", batch_size=1, timeout_s=0.5))
    st = s.run(_reqs(3))
    assert st.n_timeout == 1 and st.n_ok == 2


def test_resume_checkpoint(tmp_path):
    ck = tmp_path / "out.jsonl"
    s = BatchScheduler(MockBackend(), SchedulerConfig(policy="static", batch_size=4, checkpoint_path=str(ck)))
    s.run(_reqs(6))
    s.close()
    s2 = BatchScheduler(MockBackend(), SchedulerConfig(policy="static", batch_size=4, checkpoint_path=str(ck)))
    st = s2.run(_reqs(9))
    s2.close()
    assert st.n_requests == 3 and st.n_ok == 3


def test_adaptive_grows_and_respects_oom():
    s = BatchScheduler(MockBackend(base_latency_s=0.02, per_item_s=0.0005, max_batch=16),
                       SchedulerConfig(policy="adaptive", batch_size=1, max_batch_size=64))
    st = s.run(_reqs(120))
    assert st.n_ok == 120
    assert max(st.batch_sizes) <= 16 and max(st.batch_sizes) >= 8


def test_online_dynamic():
    s = BatchScheduler(MockBackend(base_latency_s=0.01), SchedulerConfig(policy="dynamic", batch_size=8, max_wait_s=0.05))
    st = s.run_online(_reqs(40), arrival_rate_hz=200, seed=0)
    assert st.n_ok == 40 and max(st.batch_sizes) > 1
    assert all(l >= 0 for l in st.request_latencies)
