#!/usr/bin/env python3
"""Offline reports and engine-neutral replay rendering, never called during timing."""
import argparse
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

COLORS = dict(rapier='#00b8a9', nexus='#ee9f34', genesis='#e86582', newton='#9467df', mujoco='#579df2', mjx='#b9c44a')


def summarize(records, suite=None):
    groups = {}
    for r in records:
        key = (r['engine'], r['backend'], r['scenario'], r['size'], r['worlds'])
        groups.setdefault(key, []).append(r)
    if suite:
        import itertools
        for engine, scenario, size, worlds in itertools.product(suite['engines'],suite['scenarios'],suite['sizes'],suite['worlds']):
            if not any(key[0]==engine and key[2:]==(scenario,size,worlds) for key in groups):
                groups[(engine,'unavailable',scenario,size,worlds)]=[]
    rows = []
    for (engine, backend, scenario, size, worlds), runs in sorted(groups.items()):
        good = [r for r in runs if r['status'] == 'passed']
        values = [r['ms_per_step'] for r in good]
        expected=len(suite['seeds'])*suite['repeats'] if suite else len(runs)
        complete=len(good)==expected and expected>0
        if suite:
            complete=complete and {(r['seed'],r['repeat']) for r in good}=={(seed,repeat) for seed in suite['seeds'] for repeat in range(suite['repeats'])}
        rows.append(dict(engine=engine, backend=backend, scenario=scenario, size=size, worlds=worlds, passed=len(good), total=expected, eligible=complete, median_ms=float(np.median(values)) if values else None, p25_ms=float(np.percentile(values,25)) if values else None, p75_ms=float(np.percentile(values,75)) if values else None, min_ms=min(values) if values else None, max_ms=max(values) if values else None))
    return rows


def leaderboard(rows, engines):
    cases = {(r['scenario'],r['size'],r['worlds']) for r in rows}
    baseline = {(r['scenario'],r['size'],r['worlds']):r for r in rows if r['engine']=='mujoco' and r['eligible']}
    leaders=[]
    for engine in engines:
        subset=[r for r in rows if r['engine']==engine]
        complete=len(subset)==len(cases) and all(r['eligible'] for r in subset)
        ratios=[baseline[(r['scenario'],r['size'],r['worlds'])]['median_ms']/r['median_ms'] for r in subset if r['eligible'] and (r['scenario'],r['size'],r['worlds']) in baseline]
        score=math.exp(sum(map(math.log,ratios))/len(ratios)) if complete and len(ratios)==len(cases) and ratios else None
        leaders.append(dict(engine=engine,backend=subset[0]['backend'] if subset else '?',passed=sum(r['passed'] for r in subset),total=sum(r['total'] for r in subset),complete=complete,score=score))
    return sorted(leaders,key=lambda r:-(r['score'] or 0))


def curves(rows, out):
    files=[]
    for scenario, worlds in sorted({(r['scenario'],r['worlds']) for r in rows}):
        fig, ax=plt.subplots(figsize=(7.5,4.5),layout='constrained')
        for engine in COLORS:
            rs=sorted([r for r in rows if r['engine']==engine and r['scenario']==scenario and r['worlds']==worlds and r['eligible']],key=lambda r:r['size'])
            if not rs:
                continue
            x=[r['size'] for r in rs];y=[r['median_ms'] for r in rs]
            ax.plot(x,y,'o-',label=f"{engine} / {rs[0]['backend']}",color=COLORS[engine])
            ax.fill_between(x,[r['p25_ms'] for r in rs],[r['p75_ms'] for r in rs],color=COLORS[engine],alpha=.15)
        ax.set(xscale='log',yscale='log',xlabel='Dynamic bodies per world',ylabel='Synchronized wall time / step (ms)',title=f'{scenario.replace("_"," ")} · {worlds} world(s)')
        ax.grid(True,alpha=.2);ax.legend(fontsize=8)
        filename=f'curves/{scenario}-w{worlds}.svg';fig.savefig(out/filename);plt.close(fig);files.append(filename)
    return files


def rotation(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def geometry(body):
    size=body['size']
    if body['shape']=='box':
        v=np.array([[x,y,z] for x in [-size[0],size[0]] for y in [-size[1],size[1]] for z in [-size[2],size[2]]])
        return [v[ids] for ids in [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]]
    r=size[0];h=size[2] if body['shape']=='capsule' else 0
    lat=np.linspace(-np.pi/2,np.pi/2,9);lon=np.linspace(0,2*np.pi,13)
    v=np.array([[[r*np.cos(a)*np.cos(b),r*np.cos(a)*np.sin(b),r*np.sin(a)+(h if a>=0 else -h)] for b in lon] for a in lat])
    return [np.array([v[i,j],v[i+1,j],v[i+1,j+1],v[i,j+1]]) for i in range(8) for j in range(12)]


def render(record,out,no_video=False):
    capture=out/record['capture'];data=json.loads(capture.read_text())
    job=json.loads((out/record['directory']/'job.json').read_text());scene=job['scene'];frames=data['frames']
    if not frames:
        raise ValueError('capture contains no frames')
    bodies=[b for b in scene['bodies'] if b['dynamic']]
    if any(len(f['poses'])!=len(bodies) for f in frames):
        raise ValueError('capture pose count mismatch')
    positions=np.array([f['poses'] for f in frames])[:,:,:3]
    if not np.isfinite(positions).all():
        raise ValueError('capture contains non-finite positions')
    lo=positions.min(axis=(0,1))-1.;hi=positions.max(axis=(0,1))+1.
    if scene['scenario']!='free_fall':lo[2]=-.3
    center=(hi+lo)/2;span=max(hi-lo);lo=center-span/2;hi=center+span/2
    fig=plt.figure(figsize=(7.2,5.4),dpi=100,facecolor='#101923');ax=fig.add_subplot(111,projection='3d',facecolor='#101923')
    fig.subplots_adjust(0,0,1,.92);ax.set_axis_off();ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]));ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=-62)
    if scene['scenario']!='free_fall':
        floor=np.array([[lo[0]-span,lo[1]-span,0],[hi[0]+span,lo[1]-span,0],[hi[0]+span,hi[1]+span,0],[lo[0]-span,hi[1]+span,0]])
        ax.add_collection3d(Poly3DCollection([floor],facecolors='#263746',edgecolors='#486172',alpha=.7))
    for b in scene['bodies']:
        if not b['dynamic'] and b['position'][2]>0:
            faces=[v@rotation(b['rotation']).T+np.array(b['position']) for v in geometry(b)]
            ax.add_collection3d(Poly3DCollection(faces,facecolors='#64778a',edgecolors='#a3afbb',linewidths=.4))
    meshes=[geometry(b) for b in bodies]
    collection=Poly3DCollection([],facecolors=COLORS[record['engine']],edgecolors='#102a34',linewidths=.35,alpha=.98)
    ax.add_collection3d(collection)
    caption=fig.text(.04,.965,'',color='white',size=11,va='top')
    def draw(frame):
        faces=[]
        for mesh,pose in zip(meshes,frame['poses']):
            R=rotation(pose[3:]);p=np.array(pose[:3]);faces.extend(v@R.T+p for v in mesh)
        collection.set_verts(faces)
        caption.set_text(f"{record['engine'].upper()} / {record['backend']}   •   {scene['scenario'].replace('_',' ')}   •   n={len(bodies)}\nWorld 0   ·   t={frame['step']*scene['dt']:.3f} s   ·   untimed replay")
        fig.canvas.draw()
    media=out/record['directory'];draw(frames[-1]);fig.savefig(media/'simulation.png',facecolor=fig.get_facecolor())
    if not no_video and shutil.which('ffmpeg'):
        width,height=fig.canvas.get_width_height();duration=frames[-1]['step']*scene['dt'];fps=30
        times=np.array([f['step']*scene['dt'] for f in frames]);indices=np.searchsorted(times,np.linspace(0,duration,max(2,int(duration*fps)+1))).clip(max=len(frames)-1)
        cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s',f'{width}x{height}','-pix_fmt','rgba','-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(media/'simulation.mp4')]
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for i in indices:
                draw(frames[i]);process.stdin.write(fig.canvas.buffer_rgba())
        finally:
            process.stdin.close()
            status=process.wait()
        if status:
            raise RuntimeError(f'ffmpeg exited {status}')
    plt.close(fig)


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhysMark — __TITLE__</title>
<style>:root{color-scheme:dark;--bg:#0d141b;--panel:#17222e;--line:#304252;--text:#eaf1f7;--muted:#a5b8c7;--accent:#41dcc3}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:38px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:28px}.brand{font-size:30px;font-weight:750;letter-spacing:-1px}small,.muted{color:var(--muted)}h1{font-size:42px;letter-spacing:-1.5px;margin:30px 0 8px}h2{font-size:22px;margin-top:38px}p{max-width:970px}a{color:var(--accent)}.stats,.leaders,.gallery,.charts{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:22px}.number{font-size:30px;font-weight:700}.leaders .score{font-size:34px;font-weight:750;color:var(--accent)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:11px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:500}select,input{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;padding:10px;font:inherit;margin:0 8px 12px 0}.scroll{overflow:auto}.passed{color:#52dfb1}.error,.invalid{color:#ff8d95}.charts{grid-template-columns:repeat(auto-fit,minmax(400px,1fr))}.charts img{width:100%;background:white;border-radius:8px}.gallery{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.gallery video,.gallery img{width:100%;border-radius:8px}.gallery .card{padding:12px}.tag{display:inline-block;border:1px solid var(--line);padding:3px 9px;border-radius:16px;font-size:12px}details{margin-top:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);font-size:12px}footer{margin-top:50px;border-top:1px solid var(--line);padding-top:22px;color:var(--muted)}@media(max-width:650px){main{padding:20px}h1{font-size:32px}.charts{grid-template-columns:1fr}header{display:block}}</style>
<main><header><div class="brand">physmark<span style="color:var(--accent)">.</span></div><div class="muted">Physics, measured. <span class="tag">LOCAL BENCHMARK</span></div></header><h1>__TITLE__</h1><p class="muted">__HARDWARE__ · __DATE__<br>Synchronized physics stepping. Rendering and compilation excluded. Sleeping disabled.</p><div class="stats" id="stats"></div><h2>Leaderboard</h2><p class="muted">Geometric mean speed relative to native MuJoCo across every case in this suite (higher is faster). A score requires every requested trial and every baseline case to pass. These are baseline sanity checks, not matched-accuracy results. CPU and GPU entries are labelled separately.</p><div class="leaders" id="leaders"></div><h2>Performance curves</h2><p class="muted">Median wall time per simulation step across all worlds; shaded bands show the 25th–75th percentiles across repeats and seeds. Incomplete cases are omitted.</p><div class="charts">__CURVES__</div><h2>Trials</h2><select id="engine"><option value="">All engines</option></select><select id="scenario"><option value="">All scenarios</option></select><select id="status"><option value="">All outcomes</option><option>passed</option><option>invalid</option><option>error</option></select><div class="scroll"><table><thead><tr><th>Engine / backend</th><th>Scenario</th><th>Bodies × worlds</th><th>Seed / repeat</th><th>ms / step</th><th>World steps / s</th><th>Outcome</th><th>Evidence</th></tr></thead><tbody id="trials"></tbody></table></div><h2>Simulation gallery</h2><p class="muted">Independent untimed replay, rendered by the same renderer and camera rules. Batched cases show world 0. Videos play at simulated time.</p><div class="gallery">__GALLERY__</div><details><summary>Methodology, source pins and machine metadata</summary><pre id="metadata"></pre></details><footer><a href="results.json">Raw trials</a> · <a href="summary.json">Summary statistics</a> · <a href="metadata.json">Provenance</a> · PhysMark protocol v1<br>No single score establishes physical accuracy. Inspect solver settings, validation errors, coverage and simulations alongside timing.</footer></main>
<script>const DATA=__DATA__;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>x==null?'—':Number(x).toLocaleString(undefined,{maximumSignificantDigits:4});
document.querySelector('#stats').innerHTML=[['Engines',DATA.leaders.length],['Trials',DATA.records.length],['Passed',DATA.records.filter(r=>r.status==='passed').length],['Cases',new Set(DATA.rows.map(r=>[r.scenario,r.size,r.worlds].join('/'))).size]].map(([label,n])=>`<div class="card"><small>${label}</small><div class="number">${n}</div></div>`).join('');
document.querySelector('#leaders').innerHTML=DATA.leaders.map((r,i)=>`<div class="card"><small>${r.score==null?'UNRANKED':'#'+(i+1)} · ${esc(r.backend)}</small><h3>${esc(r.engine)}</h3><div class="score">${r.score==null?'—':fmt(r.score)+'×'}</div><small>${r.passed}/${r.total} trials passed${r.score==null?' · incomplete coverage or missing baseline':''}</small></div>`).join('');
for(const field of ['engine','scenario']){const el=document.querySelector('#'+field);for(const v of [...new Set(DATA.records.map(r=>r[field]))].sort()){const opt=document.createElement('option');opt.textContent=v;el.append(opt)}}
function draw(){document.querySelector('#trials').innerHTML=DATA.records.filter(r=>['engine','scenario','status'].every(k=>!document.querySelector('#'+k).value||r[k]===document.querySelector('#'+k).value)).map(r=>`<tr><td>${esc(r.engine)} <small>${esc(r.backend)}</small></td><td>${esc(r.scenario)}</td><td>${r.size} × ${r.worlds}</td><td>${r.seed} / ${r.repeat}</td><td>${fmt(r.ms_per_step)}</td><td>${fmt(r.world_steps_per_second)}</td><td class="${r.status}" title="${esc(r.error||r.validation?.reasons.join('; ')||'')}">${r.status}</td><td><a href="${r.directory}/job.json">job</a> · <a href="${r.directory}/adapter.log">log</a>${r.status!=='error'?` · <a href="${r.directory}/result.json">result</a>`:''}</td></tr>`).join('')}
for(const el of document.querySelectorAll('select'))el.addEventListener('change',draw);draw();document.querySelector('#metadata').textContent=JSON.stringify(DATA.metadata,null,2);</script></html>'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--no-video',action='store_true');args=parser.parse_args();out=args.directory.resolve()
    records=json.loads((out/'results.json').read_text());metadata=json.loads((out/'metadata.json').read_text());rows=summarize(records,metadata['suite']);leaders=leaderboard(rows,metadata['suite']['engines'])
    (out/'summary.json').write_text(json.dumps(dict(cases=rows,leaderboard=leaders),indent=2));(out/'curves').mkdir(exist_ok=True);files=curves(rows,out)
    gallery=[];failures=[]
    for r in records:
        if 'capture' not in r:continue
        try:
            if not (out/r['directory']/'simulation.png').exists() or (not args.no_video and not (out/r['directory']/'simulation.mp4').exists()):render(r,out,args.no_video)
            base=r['directory'];title=f"{r['engine']} / {r['backend']} · {r['scenario']} · {r['size']} × {r['worlds']}"
            media=f'<video controls loop muted preload="none" poster="{base}/simulation.png" src="{base}/simulation.mp4"></video>' if (out/base/'simulation.mp4').exists() else f'<img loading="lazy" src="{base}/simulation.png" alt="{html.escape(title)}">'
            gallery.append(f'<div class="card">{media}<small>{html.escape(title)}</small><br><a href="{base}/capture.json">Trajectory</a> · <a href="{base}/simulation.png">PNG</a></div>')
        except Exception as e:
            failures.append(dict(id=r['id'],error=str(e)));print(f"Render failed for {r['id']}: {e}",file=sys.stderr)
    (out/'render-status.json').write_text(json.dumps(dict(failures=failures,ffmpeg=shutil.which('ffmpeg')),indent=2))
    import datetime
    data=json.dumps(dict(records=records,metadata=metadata,rows=rows,leaders=leaders),allow_nan=False).replace('<','\\u003c')
    page=HTML.replace('__TITLE__',html.escape(metadata['suite']['name'])).replace('__HARDWARE__',html.escape(metadata['hardware']['cpu'] or metadata['hardware']['uname'])).replace('__DATE__',datetime.datetime.fromtimestamp(metadata['created_unix'],datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')).replace('__CURVES__',''.join(f'<a href="{f}"><img src="{f}" alt="Performance curve"></a>' for f in files)).replace('__GALLERY__',''.join(gallery) or '<p class="muted">No successful captures in this run.</p>').replace('__DATA__',data)
    (out/'index.html').write_text(page)
    print(f'Report: {out / "index.html"}')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
