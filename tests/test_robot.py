"""MJCF normalization must preserve the pinned dynamics and independent resets."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'adapters/python'))
from robot_scene import make_scene
from adapter import Mujoco, mjcf
from report import geometry, count_unit


def test_robot_keyframe_controls_stand_up_and_reset_two_independent_environments():
    if not (ROOT / '.physmark/sources/menagerie/unitree_go1/go1.xml').exists():
        pytest.skip('fetch the pinned Menagerie first')
    scene = make_scene(ROOT)
    assert scene['size'] == 13 and scene['robot']['ctrl'] == [0, .9, -1.8]*4
    sim = Mujoco(dict(scene=scene, backend='cpu', worlds=2), mjcf(scene))
    initial = np.array(sim.poses())
    sim.advance(480)
    final = np.array(sim.poses()).reshape(2, 13, 7)
    assert np.all(final[:,0,2] > .22)
    assert np.allclose(final[0], final[1])
    assert sim.data[0] is not sim.data[1]
    sim.reset()
    np.testing.assert_allclose(sim.poses(), initial)
    assert sim.awake() == 26
    assert count_unit('robot_go1') == 'robots'
    for parts in scene['robot']['render_parts']:
        faces = geometry({'parts':parts})
        assert np.isfinite(np.array(faces)).all()


def test_robot_scene_rejects_multiple_robots_in_one_world():
    with pytest.raises(ValueError, match='size=1'):
        make_scene(ROOT, size=2)
