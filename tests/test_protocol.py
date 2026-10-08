import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'adapters/python'))
sys.path.insert(0,str(ROOT/'tools'))
from adapter import mjcf, Mujoco
from report import summarize, leaderboard


def scene(name='mixed_pile',size=6):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)/'scene.json'
        subprocess.run([str(ROOT/'target/release/physmark'),'--root',str(ROOT),'scene',name,'--size',str(size),'--output',str(out)],check=True)
        return json.loads(out.read_text())


def test_mjcf_preserves_mass_shapes_and_hinge_anchor():
    import mujoco
    import numpy as np
    for name in ['mixed_pile','pendulums']:
        s=scene(name);xml=mjcf(s);model=mujoco.MjModel.from_xml_string(xml)
        assert not model.opt.enableflags & int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        for b in s['bodies']:
            idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f"body_{b['id']}")
            if b['dynamic']:
                assert model.body_mass[idx]==b['mass']
                np.testing.assert_allclose(model.body_inertia[idx],b['inertia'])
        d=mujoco.MjData(model);mujoco.mj_forward(model,d)
        for j in s['hinges']:
            idx=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,f"joint_{j['body']}")
            np.testing.assert_allclose(d.xanchor[idx],j['anchor'],atol=1e-12)


def test_reset_replays_from_initial_state():
    import numpy as np
    s=scene('box_pile',4)
    job=dict(scene=s,backend='cpu',worlds=2)
    sim=Mujoco(job,mjcf(s));initial=np.array(sim.poses())
    sim.advance(80);final=np.array(sim.poses());assert not np.allclose(final,initial)
    sim.reset();np.testing.assert_allclose(sim.poses(),initial)
    sim.advance(80);np.testing.assert_allclose(sim.poses(),final)


def test_failed_trials_cannot_improve_rank():
    records=[]
    for engine in ['mujoco','rapier']:
        for size in [4,8]:
            for repeat in [0,1]:
                records.append(dict(engine=engine,backend='cpu',scenario='box_pile',size=size,worlds=1,status='passed',ms_per_step=2 if engine=='mujoco' else 1))
    ranks=leaderboard(summarize(records),['mujoco','rapier']);assert ranks[0]['engine']=='rapier';assert ranks[0]['score']==2
    records[-1]['status']='error'
    ranks=leaderboard(summarize(records),['mujoco','rapier']);assert next(r for r in ranks if r['engine']=='rapier')['score'] is None
    incomplete=[r for r in records if not (r['engine']=='rapier' and r['size']==8)]
    ranks=leaderboard(summarize(incomplete),['mujoco','rapier']);assert next(r for r in ranks if r['engine']=='rapier')['score'] is None


def test_missing_and_duplicate_trials_withhold_scores():
    suite=dict(engines=['mujoco','rapier'],scenarios=['box_pile'],sizes=[4,8],worlds=[1],seeds=[42],repeats=2)
    records=[dict(engine='mujoco',backend='cpu',scenario='box_pile',size=4,worlds=1,status='passed',ms_per_step=1,seed=42,repeat=0)]*2
    rows=summarize(records,suite)
    assert len(rows)==4
    assert not any(r['eligible'] for r in rows)
    assert all(r['score'] is None for r in leaderboard(rows,suite['engines']))
