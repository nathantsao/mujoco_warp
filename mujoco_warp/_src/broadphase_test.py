# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for broadphase functions."""

import copy

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

import mujoco_warp as mjw
from mujoco_warp import BroadphaseFilter
from mujoco_warp import BroadphaseType
from mujoco_warp import DisableBit
from mujoco_warp import test_data
from mujoco_warp._src import collision_driver


def broadphase_caller(m, d):
  """Run broadphase and return the CollisionContext with results."""
  ctx = collision_driver.create_collision_context(d.naconmax)
  d.ncollision.zero_()
  if m.opt.broadphase == BroadphaseType.NXN:
    collision_driver.nxn_broadphase(m, d, ctx)
  else:
    collision_driver.sap_broadphase(m, d, ctx)
  return ctx


class BroadphaseTest(parameterized.TestCase):
  # filter combinations
  plane_sphere = BroadphaseFilter.PLANE | BroadphaseFilter.SPHERE
  plane_aabb = BroadphaseFilter.PLANE | BroadphaseFilter.AABB
  plane_obb = BroadphaseFilter.PLANE | BroadphaseFilter.OBB
  plane_sphere_aabb = plane_sphere | BroadphaseFilter.AABB
  plane_sphere_obb = plane_sphere | BroadphaseFilter.OBB
  plane_sphere_aabb_obb = plane_sphere_aabb | BroadphaseFilter.OBB

  @parameterized.parameters(1, 2)
  def test_sap_tile_graph_capture(self, nworld):
    """Captured tiled SAP preserves collision pairs across heterogeneous worlds."""
    mjm, mjd, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <worldbody>
          <body pos="-.15 0 0">
            <freejoint/>
            <geom type="sphere" size=".2"/>
          </body>
          <body pos=".15 0 0">
            <freejoint/>
            <geom type="sphere" size=".2"/>
          </body>
          <body pos=".8 0 0">
            <freejoint/>
            <geom type="sphere" size=".2"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      overrides={"opt.broadphase": BroadphaseType.SAP_TILE},
      nworld=nworld,
    )
    references = [copy.copy(mjd) for _ in range(nworld)]
    if nworld == 2:
      references[1].qpos[14] = 0.0
    with wp.ScopedCapture() as capture:
      ctx = broadphase_caller(m, d)

    def collision_pairs(context):
      count = int(d.ncollision.numpy()[0])
      pairs = context.collision_pair.numpy()[:count]
      worldids = context.collision_worldid.numpy()[:count]
      return [(int(world), *sorted(pair)) for world, pair in zip(worldids, pairs, strict=True)]

    for _ in range(3):
      expected = []
      for world, reference in enumerate(references):
        reference.qpos[1] += 0.005
        mujoco.mj_forward(mjm, reference)
        expected.extend((world, *sorted(pair)) for pair in reference.contact.geom)
      d.geom_xpos.assign(np.stack([reference.geom_xpos for reference in references]))
      d.geom_xmat.assign(np.stack([reference.geom_xmat for reference in references]))
      d.ncollision.fill_(-1)
      ctx.collision_pair.fill_(-1)
      ctx.collision_worldid.fill_(-1)
      wp.capture_launch(capture.graph)
      captured = collision_pairs(ctx)
      self.assertCountEqual(captured, expected)
      eager = collision_pairs(broadphase_caller(m, d))
      self.assertCountEqual(captured, eager)
      self.assertEqual(len(captured), 1 if nworld == 1 else 4)
      if nworld == 2:
        self.assertEqual(sum(pair[0] == 0 for pair in captured), 1)
        self.assertEqual(sum(pair[0] == 1 for pair in captured), 3)

  @parameterized.product(
    broadphase=list(BroadphaseType),
    filter=[plane_sphere, plane_aabb, plane_obb, plane_sphere_aabb, plane_sphere_obb, plane_sphere_aabb_obb],
  )
  def test_broadphase(self, broadphase, filter):
    """Tests collision broadphase algorithms."""
    _XML = """
      <mujoco>
        <worldbody>
          <body>
            <freejoint/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body>
            <freejoint/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body>
            <freejoint/>
            <geom type="capsule" size="0.1 0.1"/>
          </body>
          <body>
            <freejoint/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body>
            <freejoint/>
            <!-- self collision -->
            <geom type="sphere" size="0.1"/>
            <geom type="sphere" size="0.1"/>
            <!-- parent-child self collision -->
            <body>
              <geom type="sphere" size="0.1"/>
              <joint type="hinge"/>
            </body>
          </body>
        </worldbody>
        <keyframe>
          <key qpos='0 0 0 1 0 0 0
                    1 0 0 1 0 0 0
                    2 0 0 1 0 0 0
                    3 0 0 1 0 0 0
                    4 0 0 1 0 0 0
                    0'/>
          <key qpos='0 0 0 1 0 0 0
                    .05 0 0 1 0 0 0
                    2 0 0 1 0 0 0
                    3 0 0 1 0 0 0
                    4 0 0 1 0 0 0
                    0'/>
          <key qpos='0 0 0 1 0 0 0
                    .01 0 0 1 0 0 0
                    .02 0 0 1 0 0 0
                    3 0 0 1 0 0 0
                    4 0 0 1 0 0 0
                    0'/>
          <key qpos='0 0 0 1 0 0 0
                    1 0 0 1 0 0 0
                    2 0 0 1 0 0 0
                    2 0 0 1 0 0 0
                    4 0 0 1 0 0 0
                    0'/>
        </keyframe>
      </mujoco>
    """

    # one world and zero collisions
    mjm, _, m, d0 = test_data.fixture(xml=_XML, keyframe=0)

    m.opt.broadphase = broadphase
    m.opt.broadphase_filter = filter

    ctx0 = broadphase_caller(m, d0)
    np.testing.assert_allclose(d0.ncollision.numpy()[0], 0)

    # one world and one collision
    _, mjd1, _, d1 = test_data.fixture(xml=_XML, keyframe=1)
    ctx1 = broadphase_caller(m, d1)

    np.testing.assert_allclose(d1.ncollision.numpy()[0], 1)
    np.testing.assert_allclose(ctx1.collision_pair.numpy()[0][0], 0)
    np.testing.assert_allclose(ctx1.collision_pair.numpy()[0][1], 1)

    # one world and three collisions
    _, mjd2, _, d2 = test_data.fixture(xml=_XML, keyframe=2)
    ctx2 = broadphase_caller(m, d2)

    ncollision = d2.ncollision.numpy()[0]
    np.testing.assert_allclose(ncollision, 3)

    collision_pairs = [[0, 1], [0, 2], [1, 2]]
    for i in range(ncollision):
      self.assertTrue([ctx2.collision_pair.numpy()[i][0], ctx2.collision_pair.numpy()[i][1]] in collision_pairs)

    # two worlds and four collisions
    d3 = mjw.make_data(mjm, nworld=2, nconmax=512, njmax=512)
    d3.geom_xpos = wp.array(
      np.vstack([np.expand_dims(mjd1.geom_xpos, axis=0), np.expand_dims(mjd2.geom_xpos, axis=0)]),
      dtype=wp.vec3,
    )
    d3.geom_xmat = wp.array(
      np.vstack([np.expand_dims(mjd1.geom_xmat, axis=0), np.expand_dims(mjd2.geom_xmat, axis=0)]),
      dtype=wp.mat33,
    )
    ctx3 = broadphase_caller(m, d3)

    ncollision = d3.ncollision.numpy()[0]
    np.testing.assert_allclose(ncollision, 4)

    collision_pairs = ctx3.collision_pair.numpy()[:ncollision]
    worldids = ctx3.collision_worldid.numpy()[:ncollision]
    self.assertCountEqual(
      [(int(worldid), int(pair[0]), int(pair[1])) for worldid, pair in zip(worldids, collision_pairs, strict=True)],
      [(0, 0, 1), (1, 0, 1), (1, 0, 2), (1, 1, 2)],
    )

    # one world and zero collisions: contype and conaffinity incompatibility
    mjm4, _, m4, d4 = test_data.fixture(xml=_XML, keyframe=1)
    mjm4.geom_contype[:3] = 0
    m4 = mjw.put_model(mjm4)

    ctx4 = broadphase_caller(m4, d4)
    np.testing.assert_allclose(d4.ncollision.numpy()[0], 0)

    # one world and one collision: geomtype ordering
    _, _, _, d5 = test_data.fixture(xml=_XML, keyframe=3)
    ctx5 = broadphase_caller(m, d5)
    np.testing.assert_allclose(d5.ncollision.numpy()[0], 1)
    np.testing.assert_allclose(ctx5.collision_pair.numpy()[0][0], 3)
    np.testing.assert_allclose(ctx5.collision_pair.numpy()[0][1], 2)

  @parameterized.parameters(
    (0, 0, 0),
    (0, 0.011, 1),
    (0.011, 0, 1),
    (0.00999, 0, 0),
    (0, 0.00999, 0),
    (0.00999, 0.00999, 1),
  )
  def test_broadphase_margin(self, margin1, margin2, ncollision):
    _MJCF = f"""
      <mujoco>
        <worldbody>
          <body>
            <geom type="sphere" size=".1" margin="{margin1}"/>
            <joint type="slide" axis="1 0 0"/>
          </body>
          <body>
            <geom type="sphere" size=".1" margin="{margin2}"/>
            <joint type="slide" axis="1 0 0"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos="0 .21"/>
        </keyframe>
      </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=_MJCF, keyframe=0)
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], ncollision)

  @parameterized.parameters((0, 0), (DisableBit.FILTERPARENT, 1))
  def test_broadphase_filterparent(self, disablebit, expected_collisions):
    _MJCF = """
      <mujoco>
        <worldbody>
          <body>
            <geom type="sphere" size=".1"/>
            <joint type="slide"/>
            <body>
              <geom type="sphere" size=".1"/>
              <joint type="slide"/>
            </body>
          </body>
        </worldbody>
        <keyframe>
          <key qpos="0 0"/>
        </keyframe>
      </mujoco>
    """
    _, _, m, d = test_data.fixture(xml=_MJCF, keyframe=0, overrides={"opt.disableflags": disablebit})

    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], expected_collisions)

  def test_broadphase_filter(self):
    plane = BroadphaseFilter.PLANE
    sphere = BroadphaseFilter.SPHERE
    aabb = BroadphaseFilter.AABB
    obb = BroadphaseFilter.OBB
    plane_sphere = plane | sphere
    plane_aabb = plane | aabb
    plane_obb = plane | obb

    _PLANE_CAPSULE_CAPSULE = """
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <light type="directional" pos="0 0 1"/>
          <geom name="floor" size="10 10 .001" type="plane"/>
          <body>
            <geom type="capsule" size=".05 .1" rgba="0 1 0 1"/>
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide" axis="0 0 1"/>
            <joint type="hinge" axis="0 1 0"/>
          </body>
          <body>
            <geom type="capsule" size=".05 .1" rgba="1 0 0 1"/>
            <joint type="slide" axis="1 0 0"/>
            <joint type="slide" axis="0 0 1"/>
            <joint type="hinge" axis="0 1 0"/>
          </body>
        </worldbody>
        <keyframe>
          <key qpos="-.5 .25 0 .5 .25 0"/>
          <key qpos="-.5 .075 1.57 .5 .25 0"/>
          <key qpos="-.075 .25 0 .075 .25 0"/>
          <key qpos="0 .25 .7853 0 .45 .7853"/>
        </keyframe>
      </mujoco>
    """

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=0)
    m.opt.broadphase_filter = plane_sphere
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 0)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=0)
    m.opt.broadphase_filter = plane_aabb
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 0)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=0)
    m.opt.broadphase_filter = plane_obb
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 0)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=1)
    m.opt.broadphase_filter = plane_sphere
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 1)

    # note: collision_driver._plane_filter checks bounding sphere
    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=1)
    m.opt.broadphase_filter = plane
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 2)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=1)
    m.opt.broadphase_filter = plane_sphere
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 1)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=1)
    m.opt.broadphase_filter = plane_obb
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 1)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=2)
    m.opt.broadphase_filter = plane_sphere
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 1)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=2)
    m.opt.broadphase_filter = plane_aabb
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 0)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=2)
    m.opt.broadphase_filter = plane_obb
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 0)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=3)
    m.opt.broadphase_filter = plane_sphere
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 1)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=3)
    m.opt.broadphase_filter = plane_aabb
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 1)

    _, _, m, d = test_data.fixture(xml=_PLANE_CAPSULE_CAPSULE, keyframe=3)
    m.opt.broadphase_filter = plane_obb
    ctx = broadphase_caller(m, d)
    self.assertEqual(d.ncollision.numpy()[0], 0)

  @parameterized.product(broadphase=list(BroadphaseType), nworld=[1, 2])
  def test_broadphase_missing_mesh(self, broadphase, nworld):
    """A mesh geom whose batched geom_dataid is -1 takes no pair, even below a plane."""
    _, _, m, d = test_data.fixture(
      xml="""
      <mujoco>
        <asset>
          <mesh name="tetrahedron" vertex="0 0 0  .1 0 0  0 .1 0  0 0 .1"/>
        </asset>
        <worldbody>
          <geom type="plane" size="1 1 .01"/>
          <body pos="0 0 -1">
            <freejoint/>
            <geom type="mesh" mesh="tetrahedron"/>
          </body>
        </worldbody>
      </mujoco>
      """,
      nworld=nworld,
    )
    m.opt.broadphase = broadphase

    if nworld == 2:
      qpos = d.qpos.numpy()
      qpos[1, 2] = -2.0
      d.qpos.assign(qpos)
      mjw.kinematics(m, d)

      dataid = np.tile(m.geom_dataid.numpy(), (2, 1))
      dataid[0, 1] = -1  # absent in world 0, present in world 1
      m.geom_dataid = wp.array(dataid, dtype=int)

    ctx = collision_driver.create_collision_context(d.naconmax)
    ctx.collision_pair.fill_(wp.vec2i(-1, -1))
    ctx.collision_worldid.fill_(-1)
    d.ncollision.zero_()

    if m.opt.broadphase == BroadphaseType.NXN:
      collision_driver.nxn_broadphase(m, d, ctx)
    else:
      collision_driver.sap_broadphase(m, d, ctx)

    self.assertEqual(d.ncollision.numpy()[0], 1)
    self.assertEqual(ctx.collision_worldid.numpy()[0], 1 if nworld == 2 else 0)
    np.testing.assert_array_equal(ctx.collision_pair.numpy()[0], [0, 1])


if __name__ == "__main__":
  wp.init()
  absltest.main()
