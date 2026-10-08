#!/usr/bin/env python3
"""Source build helper invoked by the Rust CLI; all artifacts stay in .physmark.

Dependencies are version-locked separately from the upstream source commits.
MuJoCo's native library AND Python extension are compiled from the same commit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.physmark'
SOURCES = CACHE / 'sources'
BUILD = CACHE / 'build'
PY = CACHE / 'envs/python/bin/python'


def run(*args, env=None):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, env=env, check=True)


def install(*args, env=None):
    run('uv', 'pip', 'install', '--python', PY, *args, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engines', default='all')
    args = parser.parse_args()
    engines = {'rapier','nexus','genesis','newton','mujoco','mjx'} if args.engines == 'all' else set(args.engines.split(','))
    if engines - {'rapier','nexus','genesis','newton','mujoco','mjx'}:
        parser.error('unknown engine')
    BUILD.mkdir(parents=True, exist_ok=True)
    if not PY.exists():
        run('uv', 'venv', '--python', '3.12', PY.parent.parent)
    if (ROOT / 'requirements.lock').exists():
        install('-r', ROOT / 'requirements.lock')
    else:
        install('numpy', 'matplotlib', 'Pillow', 'pytest', 'setuptools', 'wheel', 'pybind11', 'cmake', 'ninja')
    if 'rapier' in engines:
        run('cargo','build','--release','--locked','--manifest-path','adapters/rapier/Cargo.toml')
    if 'nexus' in engines:
        backend = os.environ.get('PHYSMARK_NEXUS_FEATURES', 'metal' if platform.system() == 'Darwin' else 'webgpu')
        run('cargo','build','--release','--locked','--manifest-path','adapters/nexus/Cargo.toml','--no-default-features','--features',backend)
    signature = hashlib.sha256((ROOT/'engines.lock').read_bytes()+(ROOT/'requirements.lock').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    cache_file = BUILD/'python-source-stamps.json'
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    needed = engines & {'mujoco','mjx','newton','genesis'}
    if needed and any(cache.get(e) != signature for e in needed):
        src = SOURCES / 'mujoco'
        native, prefix = BUILD / 'mujoco', BUILD / 'mujoco-install'
        run('cmake','-S',src,'-B',native,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_INSTALL_PREFIX={prefix}','-DMUJOCO_BUILD_EXAMPLES=OFF','-DMUJOCO_BUILD_SIMULATE=OFF','-DMUJOCO_BUILD_TESTS=OFF','-DMUJOCO_USE_FILAMENT=OFF')
        run('cmake','--build',native,'-j',str(min(os.cpu_count() or 4,8)))
        run('cmake','--install',native)
        # Replicate upstream make_sdist.py's layout, without modifying the checkout.
        stage = BUILD / 'mujoco-python-source'
        if stage.exists():
            shutil.rmtree(stage)
        shutil.copytree(src/'python',stage,ignore=shutil.ignore_patterns('build','dist','*.egg-info','__pycache__'))
        for name in ['cmake','simulate']:
            shutil.copytree(src/name,stage/'mujoco'/name,dirs_exist_ok=True)
        shutil.copy2(src/'LICENSE',stage/'LICENSE')
        env = dict(os.environ, PYTHONPATH=str(src/'python/mujoco'))
        for script, dest in [('generate_enum_traits.py','enum_traits.h'),('generate_function_traits.py','function_traits.h'),('generate_spec_bindings.py','specs.cc.inc')]:
            with (stage/'mujoco'/dest).open('w') as f:
                subprocess.run([str(PY),str(src/'python/mujoco/codegen'/script)],env=env,stdout=f,check=True)
        env = dict(os.environ, MUJOCO_PATH=str(prefix), MUJOCO_PLUGIN_PATH=str(prefix/'lib'), CMAKE_BUILD_PARALLEL_LEVEL='8')
        install('--no-build-isolation','--no-deps','--reinstall-package','mujoco',stage,env=env)
        for engine, package in [('mjx',src/'mjx'),('newton',SOURCES/'mujoco-warp'),('newton',SOURCES/'newton'),('genesis',SOURCES/'genesis')]:
            if engine in engines:
                install('--no-deps','--reinstall',package)
        for engine in needed:
            cache[engine] = signature
        cache_file.write_text(json.dumps(cache))
    run('cargo','build','--release','--locked')


if __name__ == '__main__':
    main()
