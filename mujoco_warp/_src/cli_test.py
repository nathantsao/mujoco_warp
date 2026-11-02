# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CLI warm-up, graph/eager tracing, and backend-independent control replay tests."""

import copy
from unittest import mock

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp import testspeed
from mujoco_warp import viewer
from mujoco_warp._src import cli
from mujoco_warp._src import warp_util


@wp.kernel
def _increment_time(time_out: wp.array[float]):
  worldid = wp.tid()
  time_out[worldid] += 1.0


def _fixture(nworld):
  return test_data.fixture(
    xml="""
    <mujoco>
      <option timestep="0.002"><flag contact="disable"/></option>
      <size nuserdata="2"/>
      <worldbody>
        <camera name="camera" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
        <body pos="0 0 1">
          <joint name="hinge" axis="0 1 0" damping="0.2"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 -0.5" size="0.05" mass="1"/>
        </body>
      </worldbody>
      <actuator><motor joint="hinge"/></actuator>
    </mujoco>
    """,
    nworld=nworld,
    nconmax=4,
    njmax=8,
    overrides={"opt.graph_conditional": False},
  )


class CliTest(parameterized.TestCase):
  @parameterized.product(capture=(False, True), nworld=(1, 2))
  def test_unroll(self, capture, nworld):
    """Compilation warm-up never advances the measured initial state."""
    _, _, m, d = _fixture(nworld)
    d.time.assign(np.arange(nworld, dtype=np.float32))
    initial = d.time.numpy().copy()

    def increment(m, d):
      del m
      wp.launch(_increment_time, dim=d.nworld, outputs=[d.time])

    with (
      flagsaver.as_parsed(device=str(wp.get_device()), nstep="3", event_trace="false"),
      mock.patch.object(warp_util, "graph_capture_supported", return_value=capture),
    ):
      cli.unroll(increment, m, d, None)

    np.testing.assert_array_equal(d.time.numpy(), initial + 3)

  @parameterized.parameters(1, 2)
  def test_eager_warmup_restores_nested_state(self, nworld):
    mjm, _, m, d = _fixture(nworld)
    rc = mjw.create_render_context(mjm, nworld=nworld, cam_res=(2, 2), use_textures=False)
    arrays = (d.time, d.qacc_warmstart, d.tree_asleep, d.contact.dist, d.efc.force, rc.rgb_data)
    for index, array in enumerate(arrays):
      array.fill_(index + 1)
    initial = [array.numpy().copy() for array in arrays]

    def mutate(model, data, render_context):
      del model
      for array in (data.time, data.qacc_warmstart, data.tree_asleep, data.contact.dist, data.efc.force):
        array.zero_()
      render_context.rgb_data.zero_()

    with (
      flagsaver.as_parsed(device=str(wp.get_device()), nstep="0", event_trace="false"),
      mock.patch.object(warp_util, "graph_capture_supported", return_value=False),
    ):
      cli.unroll(mutate, m, d, rc)
    for array, expected in zip(arrays, initial):
      np.testing.assert_array_equal(array.numpy(), expected)

  def test_eager_warmup_restores_state_on_error(self):
    _, _, m, d = _fixture(1)
    initial = d.qpos.numpy().copy()

    def fail(model, data):
      del model
      data.qpos.fill_(13)
      raise RuntimeError("warm-up failed")

    with (
      flagsaver.as_parsed(device=str(wp.get_device()), nstep="3", event_trace="false"),
      mock.patch.object(warp_util, "graph_capture_supported", return_value=False),
      self.assertRaisesRegex(RuntimeError, "warm-up failed"),
    ):
      cli.unroll(fail, m, d, None)
    np.testing.assert_array_equal(d.qpos.numpy(), initial)

  @parameterized.parameters(False, True)
  def test_unroll_trace_samples(self, capture):
    if wp.get_device().is_cpu:
      self.skipTest("Event timing requires a GPU")
    _, _, m, d = _fixture(1)
    samples = []

    def increment(model, data):
      del model
      wp.launch(_increment_time, dim=data.nworld, outputs=[data.time])

    def callback(index, trace, latency):
      self.assertEqual(index, len(samples))
      self.assertGreaterEqual(latency, 0)
      samples.append(trace["step"][0])

    with (
      flagsaver.as_parsed(device=str(wp.get_device()), nstep="3", event_trace="true"),
      mock.patch.object(warp_util, "graph_capture_supported", return_value=capture),
    ):
      cli.unroll(warp_util.event_scope(increment, name="step"), m, d, None, callback)
    self.assertEqual([len(sample) for sample in samples], [1, 1, 1])
    self.assertTrue(np.all(np.isfinite(samples)))
    np.testing.assert_array_equal(d.time.numpy(), [3])

  @parameterized.product(capture=(False, True), nworld=(1, 2), integrator=(mjw.IntegratorType.EULER, mjw.IntegratorType.RK4))
  def test_unroll_control_replay_matches_mujoco(self, capture, nworld, integrator):
    mjm, mjd, _, _ = _fixture(nworld)
    mjm.opt.integrator = integrator
    m = mjw.put_model(mjm)
    m.opt.graph_conditional = False
    d = mjw.put_data(mjm, mjd, nworld=nworld, nconmax=4, njmax=8)
    references = [copy.copy(mjd) for _ in range(nworld)]
    for world, reference in enumerate(references):
      reference.qpos[:] = 0.3 + world * 0.2
      reference.qvel[:] = 0.1 - world * 0.3
    d.qpos.assign(np.stack([reference.qpos for reference in references]))
    d.qvel.assign(np.stack([reference.qvel for reference in references]))
    d.qacc.fill_(wp.inf)
    ctrls = [np.array([value], dtype=np.float32) for value in (0.15, -0.2, 0.4)]
    for ctrl in ctrls:
      for reference in references:
        reference.ctrl[:] = ctrl
        mujoco.mj_step(mjm, reference)

    with (
      flagsaver.as_parsed(device=str(wp.get_device()), nstep="3", event_trace="false", noise_std="0", noise_rate="0"),
      mock.patch.object(warp_util, "graph_capture_supported", return_value=capture),
    ):
      cli.unroll(mjw.step, m, d, None, ctrls=ctrls)
    np.testing.assert_allclose(d.time.numpy(), [reference.time for reference in references], rtol=1e-5, atol=1e-5)
    for name in ("qpos", "qvel", "qacc"):
      expected = np.stack([getattr(reference, name) for reference in references])
      np.testing.assert_allclose(getattr(d, name).numpy(), expected, rtol=1e-5, atol=1e-5, err_msg=name)
    if nworld == 2:
      self.assertFalse(np.array_equal(d.qpos.numpy()[0], d.qpos.numpy()[1]))

  @parameterized.parameters(False, True)
  def test_viewer_capture_policy(self, capture):
    mjm, _, m, d = test_data.fixture("pendula.xml", overrides={"opt.graph_conditional": False})
    initial = d.time.numpy().copy()
    with mock.patch.object(warp_util, "graph_capture_supported", return_value=capture):
      graph = viewer._compile_step(m, d)
    if capture and not wp.get_device().is_cpu:
      self.assertIsNotNone(graph)
      wp.capture_launch(graph)
      np.testing.assert_allclose(d.time.numpy(), initial + mjm.opt.timestep, rtol=1e-5, atol=1e-8)
    else:
      self.assertIsNone(graph)
      np.testing.assert_array_equal(d.time.numpy(), initial)

  @parameterized.product(capture=(False, True), nworld=(1, 2))
  def test_render_benchmark_capture_policy(self, capture, nworld):
    """Render callbacks advance physics eagerly when graph capture is unavailable."""
    if wp.get_device().is_cpu:
      self.skipTest("Render benchmark requires a GPU")
    mjm, _, m, d = test_data.fixture("pendula.xml", nworld=nworld, overrides={"opt.graph_conditional": False})
    initial = d.time.numpy().copy()
    rc = mock.sentinel.render_context

    def unroll(fn, model, data, render_context, callback, ctrls):
      self.assertIs(render_context, rc)
      self.assertIsNone(ctrls)
      for i in range(3):
        fn(model, data, render_context)
        callback(i, {}, 0.001)
      return 0.0

    with (
      flagsaver.as_parsed(
        device=str(wp.get_device()),
        function="render",
        format="json",
        nworld=str(nworld),
        nstep="3",
        event_trace="false",
        clear_warp_cache="false",
        overflow_behavior="error",
      ),
      mock.patch.object(cli, "load_model", return_value=mjm),
      mock.patch.object(cli, "init_structs", return_value=(m, d, rc, None)),
      mock.patch.object(cli, "unroll", side_effect=unroll),
      mock.patch.object(warp_util, "graph_capture_supported", return_value=capture),
      mock.patch.object(wp, "ScopedCapture", wraps=wp.ScopedCapture) as scoped_capture,
      mock.patch.object(wp, "capture_launch", wraps=wp.capture_launch) as capture_launch,
      mock.patch.object(mjw, "step", wraps=mjw.step) as step,
      mock.patch.object(mjw, "refit_bvh") as refit_bvh,
      mock.patch.object(mjw, "render") as render,
      mock.patch("builtins.print"),
    ):
      testspeed._main(["testspeed", "pendula.xml"])

    self.assertEqual(scoped_capture.call_count, int(capture))
    self.assertEqual(step.call_count, 1 if capture else 3)
    self.assertEqual(capture_launch.call_count, 3 if capture else 0)
    self.assertEqual(refit_bvh.call_args_list, [mock.call(m, d, rc)] * 3)
    self.assertEqual(render.call_args_list, [mock.call(m, d, rc)] * 3)
    np.testing.assert_allclose(d.time.numpy(), initial + 3 * mjm.opt.timestep, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
  absltest.main()
