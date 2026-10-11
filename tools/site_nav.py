"""Navigation shared by the round landing page and the suite reports: platform selector and axis toggle.

A published site holds one self-contained round directory per platform plus `platforms.json`:

    site/platforms.json            {"platforms": [{"id": "linux-…", "label": "Linux · …"}, …]}
    site/<platform id>/platform.json
    site/<platform id>/index.html  round landing page
    site/<platform id>/<suite>/index.html

Pages find the site root and their platform from their own URL, so each platform directory can be
uploaded on its own; only `platforms.json` lists them. Standard library only.
"""
import html
import re

# Colours of every page, dark by default; the head script picks the visitor's theme before rendering.
PALETTE = """:root{color-scheme:dark;--bg:#0d141b;--bar:#0d141bf2;--panel:#17222e;--line:#304252;--text:#eaf1f7;--muted:#a5b8c7;--accent:#41dcc3;--good:#52dfb1;--bad:#ff8d95}:root[data-theme=light]{color-scheme:light;--bg:#f5f7f9;--bar:#f5f7f9f2;--panel:#fff;--line:#d5dee6;--text:#14202b;--muted:#566877;--accent:#0b8572;--good:#11845a;--bad:#c22f3c}:root[data-theme=light] img.logo-dark,:root:not([data-theme=light]) img.logo-light{display:none!important}"""

THEME_HEAD = f"""<style>{PALETTE}</style><script>(()=>{{let t=null;try{{t=localStorage.getItem('physmark-theme')}}catch(e){{}}if(t!=='light'&&t!=='dark')t=matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';document.documentElement.dataset.theme=t}})()</script>"""

CSS = """.pm-bar{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;padding:10px 0;margin-bottom:6px;background:var(--bar);border-bottom:1px solid var(--line);font-size:14px;color:var(--muted)}.pm-bar label,.pm-toggle{display:flex;align-items:center;gap:8px}.pm-bar select{margin:0;padding:6px 8px;background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:6px;font:inherit}.pm-bar select:disabled{opacity:1}.pm-toggle button{background:var(--panel);color:var(--muted);border:1px solid var(--line);padding:6px 12px;font:inherit;cursor:pointer}.pm-toggle button:first-of-type{border-radius:6px 0 0 6px}.pm-toggle button:last-of-type{border-radius:0 6px 6px 0;margin-left:-1px}.pm-toggle button[aria-pressed=true]{background:var(--accent);color:var(--bg);border-color:var(--accent)}.pm-bar a{color:var(--accent);text-decoration:none}.pm-theme{margin-left:auto}"""

JS = r"""addEventListener('DOMContentLoaded',()=>{const store={get(k){try{return localStorage.getItem(k)}catch(e){return null}},set(k,v){try{localStorage.setItem(k,v)}catch(e){}}};
const themes=[...document.querySelectorAll('.pm-theme button')];
function setTheme(t){document.documentElement.dataset.theme=t;for(const b of themes)b.setAttribute('aria-pressed',String(b.dataset.theme===t))}
for(const b of themes)b.addEventListener('click',()=>{store.set('physmark-theme',b.dataset.theme);setTheme(b.dataset.theme)});setTheme(document.documentElement.dataset.theme||'dark');
const buttons=[...document.querySelectorAll('.pm-axes button')];
function setScale(s){s=s==='log'?'log':'linear';for(const b of buttons)b.setAttribute('aria-pressed',String(b.dataset.scale===s));for(const img of document.querySelectorAll('img[data-linear]')){img.src=img.dataset[s];const a=img.closest('a');if(a)a.href=img.dataset[s]}}
if(buttons.length){for(const b of buttons)b.addEventListener('click',()=>{store.set('physmark-axes',b.dataset.scale);setScale(b.dataset.scale)});setScale(store.get('physmark-axes'))}
const sel=document.querySelector('#pm-platform');if(!sel)return;
const depth=Number(sel.dataset.depth),parts=location.pathname.split('/'),here=parts[parts.length-2-depth],page=parts.slice(parts.length-1-depth).join('/'),root='../'.repeat(depth+1);
(async()=>{let list;try{const r=await fetch(root+'platforms.json',{cache:'no-cache'});if(!r.ok)return;list=(await r.json()).platforms}catch(e){return}
if(!Array.isArray(list)||!list.some(p=>p.id===here))return;
const published=await Promise.all(list.map(p=>p.id===here||fetch(root+encodeURIComponent(p.id)+'/platform.json',{cache:'no-cache'}).then(r=>r.ok,()=>false)));
list=list.filter((p,i)=>published[i]);sel.replaceChildren(...list.map(p=>new Option(p.label,p.id,false,p.id===here)));sel.disabled=list.length<2;store.set('physmark-platform',here)})();
sel.addEventListener('change',async()=>{const base=root+encodeURIComponent(sel.value)+'/';let target=base+page;try{if(!(await fetch(target,{method:'HEAD',cache:'no-cache'})).ok)target=base}catch(e){target=base}store.set('physmark-platform',sel.value);location.href=target+location.hash})});"""


def bar(depth, label, axes=False, home=False):
    """The sticky toolbar; pages also need THEME_HEAD in their head. depth: directory levels between the page and its platform directory, or
    None for a page outside any platform (no selector). label: this platform's fallback label."""
    items = []
    if home:
        items.append('<a href="../index.html">← All suites</a>')
    if depth is not None:
        items.append(
            f'<label>Platform <select id="pm-platform" data-depth="{depth}" disabled '
            f'title="Other platforms appear once this round is published with physmark site">'
            f"<option>{html.escape(label)}</option></select></label>"
        )
    if axes:
        items.append(
            '<span class="pm-toggle pm-axes" role="group" aria-label="Chart axes">Axes '
            '<button type="button" data-scale="linear" aria-pressed="true">Linear</button>'
            '<button type="button" data-scale="log" aria-pressed="false">Logarithmic</button></span>'
        )
    items.append(
        '<span class="pm-toggle pm-theme" role="group" aria-label="Colour theme">'
        '<button type="button" data-theme="light" aria-pressed="false">Light</button>'
        '<button type="button" data-theme="dark" aria-pressed="true">Dark</button></span>'
    )
    return f'<style>{CSS}</style><nav class="pm-bar">{"".join(items)}</nav><script>{JS}</script>'


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "platform"


def default_platform(plan):
    """(id, label) describing the machine a round ran on, from its round-plan.json."""
    caps = plan["capabilities"]
    system = {"Darwin": "macOS"}.get(caps["system"], caps["system"])
    if caps.get("nvidia_gpus"):
        device = caps["nvidia_gpus"][0].split(",")[0].strip()
    else:
        device = caps["cpu"] if caps.get("metal") or caps["cpu"] not in ("", caps["machine"]) else caps["machine"]
    label = f"{system} · {device}"
    return slug(label), label
