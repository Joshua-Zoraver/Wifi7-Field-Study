import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { predict as rfPredict } from './predictor.js';

/* ═══════════════════════════════════════════════════════════════════════════
   DEV FLAGS
   ───────────────────────────────────────────────────────────────────────────
   Toggles for unfinished or in-progress features. Default-off so a published
   build is clean; flip to `true` (or use the URL override below) to surface.

   URL override: append `?preview=<flag>` to the URL, e.g.
     https://yoursite.com/?preview=data
   That sets DEV_FLAGS[<flag>] to true at runtime for the current session only.
═══════════════════════════════════════════════════════════════════════════ */

const DEV_FLAGS = {
  // Show the "Data" page in the top nav. Currently hidden because RSSI
  // collection is in progress and we don't want partial data visible on the
  // public site. Flip to `true` when the CSV is final.
  showDataPage: false,
};

// Apply URL-based overrides once at module load.
if (typeof window !== 'undefined') {
  try {
    const params = new URLSearchParams(window.location.search);
    const preview = params.get('preview');
    if (preview) {
      for (const key of preview.split(',')) {
        const flag = `show${key.charAt(0).toUpperCase()}${key.slice(1)}Page`;
        if (flag in DEV_FLAGS) DEV_FLAGS[flag] = true;
      }
      // Also support the literal flag name, in case I forget the convention.
      const direct = params.get('flag');
      if (direct && direct in DEV_FLAGS) DEV_FLAGS[direct] = true;
    }
  } catch { /* ignore — non-browser env, malformed URL, etc. */ }
}

/* ═══════════════════════════════════════════════════════════════════════════
   PREDICTOR INTERFACE
   ───────────────────────────────────────────────────────────────────────────
   Throughput prediction is delegated to a trained Random Forest exported
   from the model-development notebook. The model takes the deployment
   feature set (band, mode, distance, wall count, dominant material) and
   returns {throughput, uncertainty, lower, upper} in Mbps.

   Honest CV performance from the notebook: R² = 0.65 ± 0.20, MAE = 96 ± 25
   Mbps under 5-fold GroupKFold (leave-configurations-out). The uncertainty
   value returned here is tree-prediction spread — useful as a relative
   indicator of model confidence, but not a calibrated prediction interval.
   See report Section 4.6.

   The geometry-to-features bridge (`featurize` below) is unchanged; only
   the predict() body has been swapped for the RF call.
═══════════════════════════════════════════════════════════════════════════ */

const Predictor = {
  predict({ band, mode, distance, wallCount, dominantMaterial }) {
    return rfPredict({ band, mode, distance, wallCount, dominantMaterial });
  },
};

/* ═══════════════════════════════════════════════════════════════════════════
   GEOMETRY → FEATURES
   Bridge between the editor's geometric world and the predictor's feature
   space. Casts the router→pin segment against all walls, returns count and
   the dominant (worst-attenuating) material.
═══════════════════════════════════════════════════════════════════════════ */

function segmentsIntersect(p1, p2, p3, p4) {
  // Returns intersection point or null.
  const d1x = p2.x - p1.x, d1y = p2.y - p1.y;
  const d2x = p4.x - p3.x, d2y = p4.y - p3.y;
  const denom = d1x * d2y - d1y * d2x;
  if (Math.abs(denom) < 1e-9) return null;
  const t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;
  const u = ((p3.x - p1.x) * d1y - (p3.y - p1.y) * d1x) / denom;
  if (t < 0 || t > 1 || u < 0 || u > 1) return null;
  return { x: p1.x + t * d1x, y: p1.y + t * d1y };
}

function featurize(router, pin, walls) {
  const dx = (pin.x - router.x) / SCALE;
  const dy = (pin.y - router.y) / SCALE;
  const distance = Math.sqrt(dx * dx + dy * dy);
  let wallCount = 0;
  const materials = {};
  for (const w of walls) {
    if (segmentsIntersect(router, pin, w.start, w.end)) {
      wallCount++;
      materials[w.material] = (materials[w.material] || 0) + 1;
    }
  }
  // Dominant = worst attenuator present (Steel > Wood > Glass).
  const order = ['Steel', 'Wood', 'Glass'];
  let dominantMaterial = 'None';
  for (const m of order) if (materials[m]) { dominantMaterial = m; break; }
  return { distance, wallCount, dominantMaterial };
}

/* ═══════════════════════════════════════════════════════════════════════════
   CONSTANTS
═══════════════════════════════════════════════════════════════════════════ */

const SCALE = 50; // pixels per metre at zoom = 1
const SNAP_PIXELS = 12;
const ANGLE_SNAP_DEGREES = 8;
const MATERIALS = ['Glass', 'Wood', 'Steel'];
const MATERIAL_COLORS = {
  Glass: '#7dd3c0',
  Wood:  '#c89968',
  Steel: '#8a92a3',
};
const BANDS = ['2.4GHz', '5GHz', '6GHz'];
const MODES = ['Standard', 'MLO'];

const MODE_DEFAULT = 'default';
const MODE_WALL    = 'wall';
const MODE_ROUTER  = 'router';
const MODE_PIN     = 'pin';
const MODE_DELETE  = 'delete';

/* ═══════════════════════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════════════════════ */

function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function snapToAngle(start, end) {
  const dx = end.x - start.x, dy = end.y - start.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  if (length < 1) return end;
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
  // Snap to nearest 90° if close
  for (const target of [0, 90, 180, -90, -180]) {
    if (Math.abs(angle - target) < ANGLE_SNAP_DEGREES) {
      const rad = target * Math.PI / 180;
      return {
        x: start.x + Math.cos(rad) * length,
        y: start.y + Math.sin(rad) * length,
      };
    }
  }
  return end;
}

function snapToEndpoints(point, walls) {
  let nearest = null, nearestDist = SNAP_PIXELS;
  for (const w of walls) {
    for (const ep of [w.start, w.end]) {
      const d = dist(point, ep);
      if (d < nearestDist) { nearestDist = d; nearest = ep; }
    }
  }
  return nearest || point;
}

function uid() { return Math.random().toString(36).slice(2, 9); }

/* ═══════════════════════════════════════════════════════════════════════════
   APP
═══════════════════════════════════════════════════════════════════════════ */

export default function App() {
  const [page, setPage] = useState('tool'); // 'tool' | 'data' | 'about'

  // If the data page is hidden by the dev flag, force any 'data' state back
  // to 'tool' so the page can't be reached even if URL/state says otherwise.
  const effectivePage = (page === 'data' && !DEV_FLAGS.showDataPage) ? 'tool' : page;

  return (
    <div className="app">
      <Style />
      <Header page={effectivePage} setPage={setPage} />
      {effectivePage === 'tool' && <Tool />}
      {effectivePage === 'data' && <Dataset />}
      {effectivePage === 'about' && <About />}
      <Footer />
    </div>
  );
}

function Header({ page, setPage }) {
  // Build the nav based on dev flags so hidden pages don't appear in the UI.
  const pages = ['tool'];
  if (DEV_FLAGS.showDataPage) pages.push('data');
  pages.push('about');

  // Where the field-study site lives relative to this tool. In production the
  // tool is built into /tool/ alongside the field study at the repo root, so
  // '../' resolves correctly. In dev (`npm run dev`) there's nothing at the
  // parent path, so the link will 404 — that's fine, you're not really
  // clicking it during tool development.
  const FIELD_STUDY_URL = import.meta.env?.DEV ? '/' : '../';

  return (
    <header className="header">
      <div className="brand">
        <span className="brand-mark">▚</span>
        <span className="brand-text">
          <span className="brand-title">SIGNAL</span>
          <span className="brand-sub">wifi 7 propagation studio</span>
        </span>
      </div>
      <nav className="nav">
        <a
          className="nav-link nav-link-back"
          href={FIELD_STUDY_URL}
          title="Back to the field-study site"
        >← field study</a>
        {pages.map(p => (
          <button
            key={p}
            className={`nav-link ${page === p ? 'active' : ''}`}
            onClick={() => setPage(p)}
          >{p}</button>
        ))}
      </nav>
    </header>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <span>FYP · WiFi 7 throughput visualisation</span>
      <span className="footer-mono">predictor v1.0 · random forest, 300 trees</span>
    </footer>
  );
}

/* ─── TOOL ─────────────────────────────────────────────────────────────── */

function Tool() {
  // Editor state
  const [walls, setWalls] = useState([]);
  const [router, setRouter] = useState(null);
  const [pins, setPins] = useState([]);
  const [activePinId, setActivePinId] = useState(null);
  const [mode, setMode] = useState(MODE_DEFAULT);
  const [material, setMaterial] = useState('Wood');
  const [band, setBand] = useState('5GHz');
  const [opMode, setOpMode] = useState('Standard');
  const [selected, setSelected] = useState(null);
  const [showHelp, setShowHelp] = useState(true);

  // Wall placement state
  const [pendingStart, setPendingStart] = useState(null);
  const [cursor, setCursor] = useState(null);

  // Camera
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);

  // Undo stack — each entry is a function that reverts the last action
  const [undoStack, setUndoStack] = useState([]);
  const pushUndo = useCallback((fn) => {
    setUndoStack(s => [...s, fn].slice(-50));
  }, []);

  /* Hotkeys */
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      if (e.key === '1') setMode(MODE_DEFAULT);
      else if (e.key === '2') setMode(MODE_ROUTER);
      else if (e.key === '3') setMode(MODE_PIN);
      else if (e.key === '4' || e.key.toLowerCase() === 'w') setMode(MODE_WALL);
      else if (e.key.toLowerCase() === 'd') setMode(MODE_DELETE);
      else if (e.key === 'Escape') { setMode(MODE_DEFAULT); setPendingStart(null); }
      else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        setUndoStack(s => {
          if (s.length === 0) return s;
          const last = s[s.length - 1];
          last();
          return s.slice(0, -1);
        });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const reset = () => {
    if (!confirm('Reset everything? This cannot be undone.')) return;
    setWalls([]); setRouter(null); setPins([]); setActivePinId(null);
    setUndoStack([]); setSelected(null); setPendingStart(null);
  };

  /* Geometry → features → prediction (memoized per pin) */
  const predictions = useMemo(() => {
    if (!router) return {};
    const out = {};
    for (const pin of pins) {
      const features = featurize(router, pin, walls);
      out[pin.id] = {
        ...features,
        ...Predictor.predict({ band, mode: opMode, ...features }),
      };
    }
    return out;
  }, [router, pins, walls, band, opMode]);

  return (
    <main className="tool">
      <Sidebar
        mode={mode} setMode={setMode}
        material={material} setMaterial={setMaterial}
        band={band} setBand={setBand}
        opMode={opMode} setOpMode={setOpMode}
        reset={reset}
        canUndo={undoStack.length > 0}
        undo={() => {
          if (undoStack.length === 0) return;
          const last = undoStack[undoStack.length - 1];
          last();
          setUndoStack(s => s.slice(0, -1));
        }}
        showHelp={showHelp} setShowHelp={setShowHelp}
        wallCount={walls.length} pinCount={pins.length} hasRouter={!!router}
      />

      <div className="canvas-region">
        <Canvas
          walls={walls} setWalls={setWalls}
          router={router} setRouter={setRouter}
          pins={pins} setPins={setPins}
          activePinId={activePinId} setActivePinId={setActivePinId}
          mode={mode} material={material}
          pendingStart={pendingStart} setPendingStart={setPendingStart}
          cursor={cursor} setCursor={setCursor}
          pan={pan} setPan={setPan}
          zoom={zoom} setZoom={setZoom}
          selected={selected} setSelected={setSelected}
          predictions={predictions}
          pushUndo={pushUndo}
        />
        <ModeBadge mode={mode} />
        {showHelp && <HelpPanel close={() => setShowHelp(false)} />}
      </div>

      <RightPanel
        selected={selected} walls={walls} router={router} pins={pins}
        predictions={predictions}
        activePinId={activePinId} setActivePinId={setActivePinId}
        band={band} opMode={opMode}
      />
    </main>
  );
}

/* ─── SIDEBAR ──────────────────────────────────────────────────────────── */

function Sidebar({
  mode, setMode, material, setMaterial, band, setBand, opMode, setOpMode,
  reset, canUndo, undo, showHelp, setShowHelp,
  wallCount, pinCount, hasRouter,
}) {
  return (
    <aside className="sidebar">
      <Section title="Mode">
        <div className="mode-grid">
          <ModeButton k="1" label="Inspect"  active={mode === MODE_DEFAULT} onClick={() => setMode(MODE_DEFAULT)} />
          <ModeButton k="W" label="Wall"     active={mode === MODE_WALL}    onClick={() => setMode(MODE_WALL)} />
          <ModeButton k="2" label="Router"   active={mode === MODE_ROUTER}  onClick={() => setMode(MODE_ROUTER)} />
          <ModeButton k="3" label="Pin"      active={mode === MODE_PIN}     onClick={() => setMode(MODE_PIN)} />
          <ModeButton k="D" label="Delete"   active={mode === MODE_DELETE}  onClick={() => setMode(MODE_DELETE)} />
        </div>
      </Section>

      <Section title="Wall material">
        <div className="material-grid">
          {MATERIALS.map(m => (
            <button
              key={m}
              className={`material-chip ${material === m ? 'active' : ''}`}
              onClick={() => setMaterial(m)}
            >
              <span className="material-swatch" style={{ background: MATERIAL_COLORS[m] }} />
              {m}
            </button>
          ))}
        </div>
      </Section>

      <Section title="Radio config">
        <Field label="Band">
          <select className="select" value={band} onChange={e => setBand(e.target.value)}>
            {BANDS.map(b => <option key={b}>{b}</option>)}
          </select>
        </Field>
        <Field label="Mode">
          <select className="select" value={opMode} onChange={e => setOpMode(e.target.value)}>
            {MODES.map(m => <option key={m}>{m}</option>)}
          </select>
        </Field>
      </Section>

      <Section title="Actions">
        <button className="btn" onClick={undo} disabled={!canUndo}>↶ Undo</button>
        <button className="btn btn-danger" onClick={reset}>⟲ Reset</button>
        <button className="btn btn-ghost" onClick={() => setShowHelp(s => !s)}>
          {showHelp ? '× Hide help' : '? Show help'}
        </button>
      </Section>

      <Section title="Scene">
        <div className="stat-row"><span>walls</span><span className="num">{wallCount}</span></div>
        <div className="stat-row"><span>router</span><span className="num">{hasRouter ? '01' : '00'}</span></div>
        <div className="stat-row"><span>pins</span><span className="num">{String(pinCount).padStart(2, '0')}</span></div>
      </Section>
    </aside>
  );
}

function Section({ title, children }) {
  return (
    <div className="section">
      <h3 className="section-title">{title}</h3>
      <div className="section-body">{children}</div>
    </div>
  );
}

function ModeButton({ k, label, active, onClick }) {
  return (
    <button className={`mode-btn ${active ? 'active' : ''}`} onClick={onClick}>
      <span className="mode-key">{k}</span>
      <span className="mode-label">{label}</span>
    </button>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

/* ─── CANVAS ───────────────────────────────────────────────────────────── */

function Canvas({
  walls, setWalls, router, setRouter, pins, setPins,
  activePinId, setActivePinId, mode, material,
  pendingStart, setPendingStart, cursor, setCursor,
  pan, setPan, zoom, setZoom, selected, setSelected,
  predictions, pushUndo,
}) {
  const svgRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [dragOrigin, setDragOrigin] = useState(null);
  const [size, setSize] = useState({ w: 0, h: 0 });

  // Track canvas dimensions so the SVG transform can use explicit pixel
  // centring rather than percentage-based translate (which is invalid in
  // SVG transform attributes and rendered inconsistently across browsers).
  useEffect(() => {
    if (!svgRef.current) return;
    const update = () => {
      const r = svgRef.current.getBoundingClientRect();
      setSize({ w: r.width, h: r.height });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(svgRef.current);
    return () => ro.disconnect();
  }, []);

  const screenToWorld = useCallback((sx, sy) => {
    const rect = svgRef.current.getBoundingClientRect();
    const x = (sx - rect.left - rect.width / 2) / zoom - pan.x;
    const y = (sy - rect.top - rect.height / 2) / zoom - pan.y;
    return { x, y };
  }, [pan, zoom]);

  const handleMove = (e) => {
    let p = screenToWorld(e.clientX, e.clientY);
    p = snapToEndpoints(p, walls);
    if (mode === MODE_WALL && pendingStart) {
      p = snapToAngle(pendingStart, p);
    }
    setCursor(p);

    if (dragging && mode === MODE_DEFAULT) {
      const dx = (e.clientX - dragOrigin.sx) / zoom;
      const dy = (e.clientY - dragOrigin.sy) / zoom;
      setPan({ x: dragOrigin.px + dx, y: dragOrigin.py + dy });
    }
  };

  const handleDown = (e) => {
    if (e.button !== 0) return;
    let p = screenToWorld(e.clientX, e.clientY);
    p = snapToEndpoints(p, walls);

    if (mode === MODE_DEFAULT) {
      // Hit-test for selection first
      const hit = hitTest(p, walls, router, pins);
      if (hit) {
        setSelected(hit);
        if (hit.type === 'pin') setActivePinId(hit.id);
      } else {
        setSelected(null);
        setDragging(true);
        setDragOrigin({ sx: e.clientX, sy: e.clientY, px: pan.x, py: pan.y });
      }
    } else if (mode === MODE_WALL) {
      if (!pendingStart) {
        setPendingStart(p);
      } else {
        const end = snapToAngle(pendingStart, p);
        if (dist(pendingStart, end) > 5) {
          const newWall = { id: uid(), start: pendingStart, end, material };
          setWalls(ws => [...ws, newWall]);
          pushUndo(() => setWalls(ws => ws.filter(w => w.id !== newWall.id)));
        }
        setPendingStart(null);
      }
    } else if (mode === MODE_ROUTER) {
      const prev = router;
      const next = { x: p.x, y: p.y };
      setRouter(next);
      pushUndo(() => setRouter(prev));
    } else if (mode === MODE_PIN) {
      const newPin = { id: uid(), x: p.x, y: p.y };
      setPins(ps => [...ps, newPin]);
      setActivePinId(newPin.id);
      pushUndo(() => setPins(ps => ps.filter(pp => pp.id !== newPin.id)));
    } else if (mode === MODE_DELETE) {
      const hit = hitTest(p, walls, router, pins);
      if (hit?.type === 'wall') {
        const w = walls.find(x => x.id === hit.id);
        setWalls(ws => ws.filter(x => x.id !== hit.id));
        pushUndo(() => setWalls(ws => [...ws, w]));
      } else if (hit?.type === 'pin') {
        const pin = pins.find(x => x.id === hit.id);
        setPins(ps => ps.filter(x => x.id !== hit.id));
        if (activePinId === hit.id) setActivePinId(null);
        pushUndo(() => setPins(ps => [...ps, pin]));
      } else if (hit?.type === 'router') {
        const prev = router;
        setRouter(null);
        pushUndo(() => setRouter(prev));
      }
    }
  };

  const handleUp = () => { setDragging(false); setDragOrigin(null); };

  const handleWheel = (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    setZoom(z => Math.min(3, Math.max(0.3, z * factor)));
  };

  // Render via SVG (artifact-friendly, no Three.js needed for 2D)
  return (
    <div
      className={`canvas mode-${mode}`}
      onMouseMove={handleMove}
      onMouseDown={handleDown}
      onMouseUp={handleUp}
      onMouseLeave={handleUp}
      onWheel={handleWheel}
    >
      <svg ref={svgRef} className="canvas-svg">
        <defs>
          <pattern id="grid" width={SCALE} height={SCALE} patternUnits="userSpaceOnUse">
            <path d={`M ${SCALE} 0 L 0 0 0 ${SCALE}`} fill="none" stroke="#1f2530" strokeWidth="0.5" />
          </pattern>
          <pattern id="grid-major" width={SCALE * 5} height={SCALE * 5} patternUnits="userSpaceOnUse">
            <path d={`M ${SCALE * 5} 0 L 0 0 0 ${SCALE * 5}`} fill="none" stroke="#2a3140" strokeWidth="1" />
          </pattern>
        </defs>
        <g transform={`translate(${size.w / 2} ${size.h / 2}) scale(${zoom}) translate(${pan.x} ${pan.y})`}>
          <rect x="-5000" y="-5000" width="10000" height="10000" fill="url(#grid)" />
          <rect x="-5000" y="-5000" width="10000" height="10000" fill="url(#grid-major)" />
          <line x1="-5000" y1="0" x2="5000" y2="0" stroke="#2e3548" strokeWidth="1" />
          <line x1="0" y1="-5000" x2="0" y2="5000" stroke="#2e3548" strokeWidth="1" />

          {/* Walls */}
          {walls.map(w => (
            <WallShape
              key={w.id} wall={w}
              selected={selected?.type === 'wall' && selected.id === w.id}
            />
          ))}

          {/* Pending wall preview */}
          {mode === MODE_WALL && pendingStart && cursor && (
            <PendingWall start={pendingStart} end={cursor} material={material} />
          )}

          {/* Router → active pin line */}
          {router && activePinId && pins.find(p => p.id === activePinId) && (
            <RouterPinLine
              router={router}
              pin={pins.find(p => p.id === activePinId)}
              prediction={predictions[activePinId]}
            />
          )}

          {/* Pins */}
          {pins.map(p => (
            <PinShape
              key={p.id} pin={p}
              active={p.id === activePinId}
              prediction={predictions[p.id]}
            />
          ))}

          {/* Router */}
          {router && <RouterShape router={router} selected={selected?.type === 'router'} />}

          {/* Cursor crosshair when placing */}
          {mode !== MODE_DEFAULT && cursor && (
            <CursorMarker p={cursor} mode={mode} />
          )}
        </g>
      </svg>
    </div>
  );
}

function hitTest(p, walls, router, pins) {
  // Pins first (small targets, prioritise)
  for (const pin of pins) {
    if (dist(p, pin) < 14) return { type: 'pin', id: pin.id };
  }
  if (router && dist(p, router) < 18) return { type: 'router' };
  for (const w of walls) {
    if (pointToSegmentDistance(p, w.start, w.end) < 8) return { type: 'wall', id: w.id };
  }
  return null;
}

function pointToSegmentDistance(p, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-6) return dist(p, a);
  let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
  t = Math.max(0, Math.min(1, t));
  return dist(p, { x: a.x + t * dx, y: a.y + t * dy });
}

function WallShape({ wall, selected }) {
  const color = MATERIAL_COLORS[wall.material];
  return (
    <g>
      <line
        x1={wall.start.x} y1={wall.start.y}
        x2={wall.end.x} y2={wall.end.y}
        stroke={color} strokeWidth={selected ? 7 : 5}
        strokeLinecap="round"
        opacity={selected ? 1 : 0.85}
      />
      {selected && <WallLabel wall={wall} />}
    </g>
  );
}

function WallLabel({ wall }) {
  const length = dist(wall.start, wall.end) / SCALE;
  const mx = (wall.start.x + wall.end.x) / 2;
  const my = (wall.start.y + wall.end.y) / 2;
  return (
    <g transform={`translate(${mx}, ${my - 18})`}>
      <rect x="-46" y="-12" width="92" height="22" rx="3" fill="#0e1220" stroke="#3ed3b5" strokeWidth="1" />
      <text textAnchor="middle" y="3" fill="#e8ecf3" fontFamily="ui-monospace, monospace" fontSize="11">
        {length.toFixed(2)}m · {wall.material}
      </text>
    </g>
  );
}

function PendingWall({ start, end, material }) {
  const color = MATERIAL_COLORS[material];
  const length = dist(start, end) / SCALE;
  const mx = (start.x + end.x) / 2;
  const my = (start.y + end.y) / 2;
  return (
    <g>
      <line x1={start.x} y1={start.y} x2={end.x} y2={end.y}
        stroke={color} strokeWidth="4" strokeDasharray="6 4" opacity="0.7" />
      <circle cx={start.x} cy={start.y} r="4" fill={color} />
      <circle cx={end.x} cy={end.y} r="4" fill={color} />
      <g transform={`translate(${mx}, ${my - 18})`}>
        <rect x="-30" y="-11" width="60" height="20" rx="3" fill="#0e1220" stroke={color} strokeWidth="1" />
        <text textAnchor="middle" y="3" fill="#e8ecf3" fontFamily="ui-monospace, monospace" fontSize="11">
          {length.toFixed(2)}m
        </text>
      </g>
    </g>
  );
}

function RouterShape({ router, selected }) {
  return (
    <g transform={`translate(${router.x}, ${router.y})`}>
      <circle r="22" fill="#3ed3b5" opacity="0.08" />
      <circle r="14" fill="#3ed3b5" opacity="0.16" />
      <circle r="9" fill="#3ed3b5" stroke="#0e1220" strokeWidth="2" />
      <text textAnchor="middle" y="3" fill="#0e1220" fontSize="9" fontWeight="700"
        fontFamily="ui-monospace, monospace">R</text>
      {selected && <circle r="20" fill="none" stroke="#3ed3b5" strokeWidth="1.5" strokeDasharray="3 3" />}
    </g>
  );
}

function PinShape({ pin, active, prediction }) {
  const color = active ? '#f5b945' : '#5c6376';
  return (
    <g transform={`translate(${pin.x}, ${pin.y})`}>
      <circle r="6" fill={color} stroke="#0e1220" strokeWidth="2" />
      {active && prediction && (
        <g transform="translate(0, -22)">
          <rect x="-44" y="-14" width="88" height="22" rx="3" fill="#0e1220" stroke="#f5b945" strokeWidth="1" />
          <text textAnchor="middle" y="2" fill="#e8ecf3" fontFamily="ui-monospace, monospace" fontSize="11">
            {prediction.throughput} Mbps
          </text>
        </g>
      )}
    </g>
  );
}

function RouterPinLine({ router, pin, prediction }) {
  const length = prediction?.distance ?? (dist(router, pin) / SCALE);
  const mx = (router.x + pin.x) / 2;
  const my = (router.y + pin.y) / 2;
  return (
    <g>
      <line x1={router.x} y1={router.y} x2={pin.x} y2={pin.y}
        stroke="#f5b945" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.7" />
      <g transform={`translate(${mx}, ${my})`}>
        <rect x="-32" y="-10" width="64" height="18" rx="3" fill="#0e1220" stroke="#f5b945" strokeWidth="1" />
        <text textAnchor="middle" y="3" fill="#e8ecf3" fontFamily="ui-monospace, monospace" fontSize="10">
          {length.toFixed(2)}m
        </text>
      </g>
    </g>
  );
}

function CursorMarker({ p, mode }) {
  const colors = {
    [MODE_WALL]:   '#7dd3c0',
    [MODE_ROUTER]: '#3ed3b5',
    [MODE_PIN]:    '#f5b945',
    [MODE_DELETE]: '#e76e6e',
  };
  const c = colors[mode] || '#e8ecf3';
  return (
    <g transform={`translate(${p.x}, ${p.y})`}>
      <circle r="3" fill={c} />
      <line x1="-12" y1="0" x2="-6" y2="0" stroke={c} strokeWidth="1" />
      <line x1="6"  y1="0" x2="12" y2="0" stroke={c} strokeWidth="1" />
      <line x1="0" y1="-12" x2="0" y2="-6" stroke={c} strokeWidth="1" />
      <line x1="0" y1="6"   x2="0" y2="12" stroke={c} strokeWidth="1" />
    </g>
  );
}

function ModeBadge({ mode }) {
  const labels = {
    [MODE_DEFAULT]: 'inspect / pan',
    [MODE_WALL]:    'placing wall — click start, click end',
    [MODE_ROUTER]:  'placing router — click to set',
    [MODE_PIN]:     'placing pin — click to add',
    [MODE_DELETE]:  'delete — click an object',
  };
  return <div className={`mode-badge mode-badge-${mode}`}>{labels[mode]}</div>;
}

/* ─── HELP ─────────────────────────────────────────────────────────────── */

function HelpPanel({ close }) {
  return (
    <div className="help">
      <div className="help-head">
        <span>How to use</span>
        <button className="help-close" onClick={close}>×</button>
      </div>
      <ol className="help-list">
        <li><b>Build walls.</b> Press <kbd>W</kbd>, choose a material, click a start point and click again for the end. Walls snap to existing endpoints and to 90°.</li>
        <li><b>Place the router.</b> Press <kbd>2</kbd>, click anywhere. Only one router is allowed.</li>
        <li><b>Drop pins to test.</b> Press <kbd>3</kbd> and click locations where you want predicted throughput. The active pin shows a dotted line and distance to the router.</li>
        <li><b>Inspect.</b> Press <kbd>1</kbd> to drag the canvas, or click any wall/router/pin to see its details.</li>
        <li><b>Delete.</b> Press <kbd>D</kbd> and click an object. Or press <kbd>Esc</kbd> to leave any mode.</li>
        <li><b>Undo</b> with <kbd>⌘Z</kbd> / <kbd>Ctrl+Z</kbd>.</li>
      </ol>
      <p className="help-note">
        Predicted throughput comes from a Random Forest trained on the empirical field study (300 trees, leave-configurations-out CV: R² = 0.65 ± 0.20, MAE = 96 Mbps). The uncertainty value is tree-spread, not a calibrated interval — read it as relative model confidence, not a hard error bar.
      </p>
    </div>
  );
}

/* ─── RIGHT PANEL ──────────────────────────────────────────────────────── */

function RightPanel({ selected, walls, router, pins, predictions, activePinId, setActivePinId, band, opMode }) {
  return (
    <aside className="right">
      <Section title="Selection">
        {!selected && <div className="empty">Nothing selected. In inspect mode, click any object.</div>}
        {selected?.type === 'wall' && (() => {
          const w = walls.find(x => x.id === selected.id);
          if (!w) return null;
          return (
            <div>
              <KV k="type" v="wall" />
              <KV k="material" v={w.material} />
              <KV k="length" v={`${(dist(w.start, w.end) / SCALE).toFixed(2)} m`} />
            </div>
          );
        })()}
        {selected?.type === 'router' && router && (
          <div>
            <KV k="type" v="router" />
            <KV k="position" v={`${(router.x / SCALE).toFixed(2)}, ${(router.y / SCALE).toFixed(2)} m`} />
            <KV k="band" v={band} />
            <KV k="mode" v={opMode} />
          </div>
        )}
        {selected?.type === 'pin' && (() => {
          const pin = pins.find(p => p.id === selected.id);
          const pred = predictions[selected.id];
          if (!pin) return null;
          return (
            <div>
              <KV k="type" v="pin" />
              <KV k="position" v={`${(pin.x / SCALE).toFixed(2)}, ${(pin.y / SCALE).toFixed(2)} m`} />
              {pred && <>
                <hr className="hr" />
                <KV k="distance"   v={`${pred.distance.toFixed(2)} m`} />
                <KV k="walls"      v={String(pred.wallCount)} />
                <KV k="dominant"   v={pred.dominantMaterial} />
                <hr className="hr" />
                <KV k="throughput" v={`${pred.throughput} Mbps`} highlight />
                <KV k="±"          v={`${pred.uncertainty} Mbps`} />
                <KV k="range"      v={`${pred.lower} – ${pred.upper}`} />
              </>}
            </div>
          );
        })()}
      </Section>

      <Section title="All pins">
        {pins.length === 0 && <div className="empty">No pins yet. Press 3 to place one.</div>}
        {pins.map((p, i) => {
          const pred = predictions[p.id];
          return (
            <button key={p.id}
              className={`pin-row ${activePinId === p.id ? 'active' : ''}`}
              onClick={() => setActivePinId(p.id)}
            >
              <span className="pin-row-i">P{String(i + 1).padStart(2, '0')}</span>
              <span className="pin-row-d">{pred ? `${pred.distance.toFixed(1)}m` : '—'}</span>
              <span className="pin-row-t">{pred ? `${pred.throughput} Mbps` : (router ? '—' : 'no router')}</span>
            </button>
          );
        })}
      </Section>
    </aside>
  );
}

function KV({ k, v, highlight }) {
  return (
    <div className={`kv ${highlight ? 'kv-hl' : ''}`}>
      <span className="kv-k">{k}</span>
      <span className="kv-v">{v}</span>
    </div>
  );
}

/* ─── DATASET PAGE ─────────────────────────────────────────────────────── */

// Lightweight CSV parser. Sufficient for the project's CSV format (no quoted
// commas, no embedded newlines). For the deployed Vite build, drop the file
// into `public/Wifi7Tests.csv` and this loader will pick it up automatically.
function parseCSV(text) {
  const lines = text.replace(/\r/g, '').split('\n').filter(l => l.length > 0);
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const cells = line.split(',');
    const obj = {};
    headers.forEach((h, i) => {
      const v = cells[i] != null ? cells[i].trim() : '';
      if (v === '') { obj[h] = null; return; }
      const n = Number(v);
      obj[h] = Number.isFinite(n) && v !== '' ? n : v;
    });
    return obj;
  });
}

function Dataset() {
  const [rows, setRows] = useState(null); // null = loading, [] = no data, array = loaded
  const [source, setSource] = useState('loading'); // 'loading' | 'csv' | 'sample'
  const [sortKey, setSortKey] = useState('Throughput_Mbps');
  const [sortDir, setSortDir] = useState('desc');

  // Fetch the real dataset from /Wifi7Tests.csv if available (i.e. when running
  // under a Vite build with the file in `public/`). Fall back to the in-memory
  // sample if the fetch fails or the file isn't present.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('Wifi7Tests.csv');
        if (!res.ok) throw new Error(`status ${res.status}`);
        const text = await res.text();
        const parsed = parseCSV(text);
        if (!cancelled) {
          setRows(parsed);
          setSource('csv');
        }
      } catch (err) {
        if (!cancelled) {
          setRows(generateSampleRows());
          setSource('sample');
        }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // NOTE: The fetched CSV includes PrimaryRSSI and SecondaryRSSI columns in
  // memory (visible via the React DevTools state inspector), but they are
  // deliberately EXCLUDED from the displayed columns below. Coverage is
  // currently partial and we don't want incomplete data shown publicly yet.
  // To surface RSSI when collection is complete: add 'PrimaryRSSI' and
  // 'SecondaryRSSI' to the `displayCols` array below.
  const displayCols = source === 'csv'
    ? ['Environment', 'Mode', 'PrimaryBand', 'Distance', 'Material', 'PartitionAmount', 'Throughput_Mbps']
    : ['Environment', 'Mode', 'Band', 'Distance', 'Material', 'Walls', 'Throughput'];

  const sorted = useMemo(() => {
    if (!rows) return [];
    const r = [...rows];
    r.sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av;
      }
      return sortDir === 'asc'
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return r;
  }, [rows, sortKey, sortDir]);

  const toggleSort = (k) => {
    if (sortKey === k) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(k); setSortDir('desc'); }
  };

  const formatCell = (val, col) => {
    if (val == null) return '—';
    if (typeof val === 'number') {
      if (col === 'Distance') return val.toFixed(2);
      if (col === 'Throughput_Mbps' || col === 'Throughput') return val.toFixed(1);
      return String(val);
    }
    return String(val);
  };

  return (
    <main className="data-page">
      {!DEV_FLAGS.showDataPage && (
        <div className="preview-banner">
          <span className="preview-banner-tag">PREVIEW</span>
          <span>This page is currently hidden from the public site. RSSI collection is still in progress; once complete, set <code>DEV_FLAGS.showDataPage = true</code> in <code>SignalApp.jsx</code> to publish.</span>
        </div>
      )}
      <div className="data-head">
        <h2>Empirical dataset</h2>
        <p>
          Throughput measurements from the field study.{' '}
          {source === 'loading' && <span>Loading…</span>}
          {source === 'csv' && <>
            Showing <b>{rows.length}</b> measurements across {countUnique(rows, ['Environment','Mode','PrimaryBand','Distance','Material','PartitionAmount'])} unique configurations.
            {' '}
            <RSSICoverageNote rows={rows} />
          </>}
          {source === 'sample' && <>
            Showing a representative sample of <b>{rows.length}</b> rows. The full <code>Wifi7Tests.csv</code> file was not loaded; this fallback uses per-band/per-material means with realistic variance.
          </>}
        </p>
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>{displayCols.map(c => (
              <th key={c} onClick={() => toggleSort(c)} className={sortKey === c ? 'sorted' : ''}>
                {prettyHeader(c)}{sortKey === c && (sortDir === 'asc' ? ' ↑' : ' ↓')}
              </th>
            ))}</tr>
          </thead>
          <tbody>
            {sorted.slice(0, 500).map((r, i) => (
              <tr key={i}>
                {displayCols.map(c => {
                  const isNum = typeof r[c] === 'number';
                  const isStrong = c === 'Throughput_Mbps' || c === 'Throughput';
                  return (
                    <td key={c} className={`${isNum ? 'num' : ''} ${isStrong ? 'strong' : ''}`}>
                      {formatCell(r[c], c)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        {sorted.length > 500 && (
          <div className="muted" style={{ padding: '12px 14px', borderTop: '1px solid var(--line)' }}>
            Showing first 500 of {sorted.length} rows. Sort to surface different subsets.
          </div>
        )}
      </div>
      <div className="data-foot">
        <p className="muted">
          {source === 'csv'
            ? <>Live data loaded from <code>public/Wifi7Tests.csv</code>. Some columns are present in the source file but not displayed here while collection is in progress.</>
            : <>Drop <code>Wifi7Tests.csv</code> into the <code>public/</code> folder of the Vite project to load the real dataset.</>
          }
        </p>
      </div>
    </main>
  );
}

function prettyHeader(col) {
  const map = {
    PrimaryBand: 'Band',
    PartitionAmount: 'Walls',
    Throughput_Mbps: 'Throughput',
  };
  return map[col] || col;
}

function countUnique(rows, keys) {
  const set = new Set();
  for (const r of rows) set.add(keys.map(k => r[k]).join('|'));
  return set.size;
}

// Reports how much of the RSSI data is populated, since collection is in
// progress. Only renders when the underlying CSV has the RSSI columns.
function RSSICoverageNote({ rows }) {
  if (!rows || rows.length === 0) return null;
  if (!('PrimaryRSSI' in rows[0])) return null;
  const p = rows.filter(r => r.PrimaryRSSI != null).length;
  const s = rows.filter(r => r.SecondaryRSSI != null).length;
  const pct = Math.round((p / rows.length) * 100);
  return (
    <span className="rssi-note">
      RSSI coverage: <b>{p}</b>/{rows.length} primary ({pct}%), <b>{s}</b> secondary.
    </span>
  );
}

function generateSampleRows() {
  // Per-band/material means roughly aligned with report Section 3.
  const cells = [
    { env: 'Home-LOS',  mode: 'Standard', band: '5GHz',   mat: 'None',  walls: 0, dist: 1.5,  mean: 669, sd: 35 },
    { env: 'Home-LOS',  mode: 'Standard', band: '6GHz',   mat: 'None',  walls: 0, dist: 1.5,  mean: 529, sd: 30 },
    { env: 'Home-LOS',  mode: 'Standard', band: '2.4GHz', mat: 'None',  walls: 0, dist: 1.5,  mean: 104, sd: 12 },
    { env: 'Home-LOS',  mode: 'MLO',      band: '5GHz',   mat: 'None',  walls: 0, dist: 1.5,  mean: 247, sd: 26 },
    { env: 'Uni-LOS',   mode: 'Standard', band: '6GHz',   mat: 'None',  walls: 0, dist: 1.5,  mean: 677, sd: 32 },
    { env: 'Uni-LOS',   mode: 'MLO',      band: '5GHz',   mat: 'None',  walls: 0, dist: 1.5,  mean: 198, sd: 24 },
    { env: 'Home-Wall', mode: 'Standard', band: '6GHz',   mat: 'Glass', walls: 1, dist: 1.5,  mean: 620, sd: 38 },
    { env: 'Home-Wall', mode: 'Standard', band: '6GHz',   mat: 'Wood',  walls: 1, dist: 1.5,  mean: 592, sd: 41 },
    { env: 'Home-Wall', mode: 'Standard', band: '6GHz',   mat: 'Steel', walls: 1, dist: 1.5,  mean:  80, sd: 14 },
    { env: 'Home-Wall', mode: 'Standard', band: '5GHz',   mat: 'Glass', walls: 1, dist: 1.5,  mean: 596, sd: 36 },
    { env: 'Home-Wall', mode: 'Standard', band: '5GHz',   mat: 'Wood',  walls: 1, dist: 1.5,  mean: 541, sd: 42 },
    { env: 'Home-Wall', mode: 'Standard', band: '5GHz',   mat: 'Steel', walls: 1, dist: 1.5,  mean: 188, sd: 28 },
    { env: 'Home-Wall', mode: 'Standard', band: '2.4GHz', mat: 'Glass', walls: 1, dist: 1.5,  mean:  76, sd: 10 },
    { env: 'Home-Wall', mode: 'Standard', band: '2.4GHz', mat: 'Wood',  walls: 1, dist: 1.5,  mean:  66, sd:  9 },
    { env: 'Home-Wall', mode: 'Standard', band: '2.4GHz', mat: 'Steel', walls: 1, dist: 1.5,  mean:  35, sd:  7 },
    { env: 'Home-Wall', mode: 'MLO',      band: '5GHz',   mat: 'Glass', walls: 1, dist: 1.5,  mean: 489, sd: 22 },
    { env: 'Uni-Wall',  mode: 'MLO',      band: '5GHz',   mat: 'Steel', walls: 1, dist: 4.5,  mean:  21, sd:  6 },
    { env: 'Uni-Wall',  mode: 'Standard', band: '5GHz',   mat: 'Steel', walls: 1, dist: 4.5,  mean: 188, sd: 24 },
    { env: 'Home-LOS',  mode: 'Standard', band: '5GHz',   mat: 'None',  walls: 0, dist: 6.0,  mean: 740, sd: 48 },
    { env: 'Home-LOS',  mode: 'Standard', band: '5GHz',   mat: 'None',  walls: 0, dist: 12.0, mean: 800, sd: 55 },
  ];
  const rows = [];
  for (const c of cells) {
    for (let i = 0; i < 4; i++) {
      // Box-Muller for ~normal noise
      const u1 = Math.random(), u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1 || 1e-9)) * Math.cos(2 * Math.PI * u2);
      rows.push({
        Environment: c.env, Mode: c.mode, Band: c.band,
        Distance: c.dist, Material: c.mat, Walls: c.walls,
        Throughput: Math.max(1, c.mean + z * c.sd),
      });
    }
  }
  return rows;
}

/* ─── ABOUT PAGE ───────────────────────────────────────────────────────── */

function About() {
  return (
    <main className="about">
      <h2>About this tool</h2>
      <p>SIGNAL is a 2D environment editor and Wi-Fi 7 throughput predictor built as part of a Final Year Project on real-world WiFi 7 performance. The companion study collected 779 measurements across two sites, three bands, and four wall configurations.</p>

      <h3>How prediction works</h3>
      <p>Place walls and a router. For every pin you drop, the tool computes the line-of-sight from router to pin, ray-casts it against the walls, and feeds the resulting feature vector — distance, wall count, dominant wall material — into the throughput predictor.</p>

      <h3>The predictor (current state)</h3>
      <p>The tool uses a Random Forest model trained on the empirical dataset of 779 throughput measurements across 72 unique configurations. Under 5-fold leave-configurations-out cross-validation, the model achieves R² = 0.65 ± 0.20 with mean absolute error of 96 Mbps. The model is exported from <code>Wifi7_Model_Development.ipynb</code> as a standalone JavaScript module (<code>predictor.js</code>) containing 300 decision trees, so prediction runs entirely in the browser with no server call.</p>

      <h3>Roadmap</h3>
      <ul>
        <li>3D view with cut-away based on viewing angle</li>
        <li>Multi-story support: copy current layout to new storey, edit per-floor</li>
        <li>Calibrated prediction intervals via conformal wrapping</li>
        <li>Heat-map overlay (predicted throughput across the whole floor, not just at pins)</li>
      </ul>

      <h3>Hotkeys</h3>
      <p><kbd>1</kbd> inspect · <kbd>2</kbd> router · <kbd>3</kbd> pin · <kbd>W</kbd> wall · <kbd>D</kbd> delete · <kbd>Esc</kbd> exit mode · <kbd>⌘Z</kbd>/<kbd>Ctrl+Z</kbd> undo</p>
    </main>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   STYLE
═══════════════════════════════════════════════════════════════════════════ */

function Style() {
  return <style>{`
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
      --bg:        #0a0d14;
      --panel:    #0e1220;
      --panel-2:  #131829;
      --line:     #1f2538;
      --line-2:   #2a3148;
      --text:     #e8ecf3;
      --text-dim: #8c93a6;
      --text-mute:#5c6376;
      --accent:   #3ed3b5;
      --warn:     #f5b945;
      --danger:   #e76e6e;
      --glass:    #7dd3c0;
      --wood:     #c89968;
      --steel:    #8a92a3;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body, #root { height: 100%; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: 'IBM Plex Sans', system-ui, sans-serif;
      font-size: 14px;
      line-height: 1.5;
      overflow: hidden;
    }

    .app { display: grid; grid-template-rows: 56px 1fr 28px; height: 100vh; }

    /* Header */
    .header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 20px; border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
      width: 28px; height: 28px;
      display: grid; place-items: center;
      color: var(--accent); font-size: 18px;
      border: 1px solid var(--line-2); border-radius: 3px;
    }
    .brand-text { display: flex; flex-direction: column; line-height: 1.1; }
    .brand-title {
      font-family: 'IBM Plex Mono', monospace;
      font-weight: 600; letter-spacing: 0.18em; font-size: 13px;
    }
    .brand-sub {
      color: var(--text-mute); font-size: 10px; letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .nav { display: flex; gap: 4px; }
    .nav-link {
      background: transparent; color: var(--text-dim);
      border: 1px solid transparent; padding: 6px 14px;
      font-family: 'IBM Plex Mono', monospace; font-size: 11px;
      letter-spacing: 0.12em; text-transform: uppercase; cursor: pointer;
      border-radius: 3px;
    }
    .nav-link:hover { color: var(--text); }
    .nav-link.active {
      color: var(--accent); border-color: var(--line-2);
      background: var(--panel-2);
    }
    /* Back-link to the field-study site — visually a sibling of the page
       nav, but signals "leaves the SPA" via subtle styling differences:
       slightly dimmer until hover, and a right margin separating it from
       the in-app page tabs. Behaves as a real <a>, so middle-click and
       cmd-click for new tab work as expected. */
    .nav-link-back {
      text-decoration: none;
      color: var(--text-mute);
      margin-right: 8px;
      padding-right: 14px;
      border-right: 1px solid var(--line);
      border-radius: 0;
    }
    .nav-link-back:hover {
      color: var(--accent);
    }

    /* Footer */
    .footer {
      display: flex; justify-content: space-between; align-items: center;
      padding: 0 20px; border-top: 1px solid var(--line);
      background: var(--panel); color: var(--text-mute); font-size: 11px;
    }
    .footer-mono { font-family: 'IBM Plex Mono', monospace; }

    /* Tool layout */
    .tool {
      display: grid; grid-template-columns: 240px 1fr 280px;
      height: 100%; min-height: 0;
    }
    .sidebar, .right {
      background: var(--panel); border-right: 1px solid var(--line);
      overflow-y: auto;
    }
    .right { border-right: none; border-left: 1px solid var(--line); }

    /* Sections */
    .section { border-bottom: 1px solid var(--line); padding: 14px 16px; }
    .section-title {
      font-family: 'IBM Plex Mono', monospace; font-size: 10px;
      font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase;
      color: var(--text-mute); margin-bottom: 10px;
    }
    .section-body { display: flex; flex-direction: column; gap: 8px; }

    /* Mode buttons */
    .mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
    .mode-btn {
      background: var(--panel-2); border: 1px solid var(--line-2);
      color: var(--text-dim); padding: 8px 10px; cursor: pointer;
      display: flex; align-items: center; gap: 8px; border-radius: 3px;
      transition: background 0.1s, border-color 0.1s;
    }
    .mode-btn:hover { background: #161b2e; color: var(--text); }
    .mode-btn.active {
      border-color: var(--accent); color: var(--accent); background: #142623;
    }
    .mode-key {
      font-family: 'IBM Plex Mono', monospace; font-size: 10px;
      width: 18px; height: 18px; display: grid; place-items: center;
      background: #1a2034; border-radius: 2px; color: var(--text-mute);
    }
    .mode-btn.active .mode-key { background: #1d3a35; color: var(--accent); }
    .mode-label {
      font-size: 12px; font-family: 'IBM Plex Mono', monospace;
      letter-spacing: 0.04em;
    }

    /* Material chips */
    .material-grid { display: flex; flex-direction: column; gap: 4px; }
    .material-chip {
      background: var(--panel-2); border: 1px solid var(--line-2);
      color: var(--text-dim); padding: 7px 10px; cursor: pointer;
      display: flex; align-items: center; gap: 10px; border-radius: 3px;
      font-family: 'IBM Plex Mono', monospace; font-size: 12px; text-align: left;
    }
    .material-chip:hover { background: #161b2e; color: var(--text); }
    .material-chip.active {
      border-color: var(--accent); color: var(--text);
    }
    .material-swatch { width: 14px; height: 14px; border-radius: 2px; }

    /* Fields, selects */
    .field { display: flex; flex-direction: column; gap: 4px; }
    .field-label {
      font-family: 'IBM Plex Mono', monospace; font-size: 10px;
      color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.12em;
    }
    .select {
      background: var(--panel-2); color: var(--text);
      border: 1px solid var(--line-2); padding: 7px 8px; border-radius: 3px;
      font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    }
    .select:focus { outline: none; border-color: var(--accent); }

    /* Buttons */
    .btn {
      background: var(--panel-2); color: var(--text-dim);
      border: 1px solid var(--line-2); padding: 8px 12px; cursor: pointer;
      font-family: 'IBM Plex Mono', monospace; font-size: 11px;
      letter-spacing: 0.06em; border-radius: 3px; text-align: left;
    }
    .btn:hover:not(:disabled) { background: #161b2e; color: var(--text); }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-danger { color: #d68888; }
    .btn-danger:hover { background: #2a1818; color: var(--danger); border-color: #4a2424; }
    .btn-ghost { color: var(--text-mute); }

    /* Stat rows */
    .stat-row {
      display: flex; justify-content: space-between;
      font-family: 'IBM Plex Mono', monospace; font-size: 11px;
      color: var(--text-dim); padding: 3px 0;
    }
    .stat-row .num { color: var(--accent); }

    /* Canvas */
    .canvas-region { position: relative; overflow: hidden; background: #0c1019; }
    .canvas {
      width: 100%; height: 100%; cursor: crosshair;
      user-select: none;
    }
    .canvas.mode-default { cursor: grab; }
    .canvas.mode-default:active { cursor: grabbing; }
    .canvas-svg { width: 100%; height: 100%; display: block; }

    .mode-badge {
      position: absolute; top: 16px; left: 16px;
      background: rgba(14, 18, 32, 0.92);
      border: 1px solid var(--line-2);
      padding: 6px 12px; border-radius: 3px;
      font-family: 'IBM Plex Mono', monospace; font-size: 11px;
      color: var(--text-dim); letter-spacing: 0.04em;
      backdrop-filter: blur(8px);
    }
    .mode-badge-wall    { border-color: #2d4540; color: var(--glass); }
    .mode-badge-router  { border-color: #1d3a35; color: var(--accent); }
    .mode-badge-pin     { border-color: #3a2f1c; color: var(--warn); }
    .mode-badge-delete  { border-color: #3a2424; color: var(--danger); }

    /* Help */
    .help {
      position: absolute; right: 16px; bottom: 16px; width: 320px;
      background: rgba(14, 18, 32, 0.96); border: 1px solid var(--line-2);
      border-radius: 4px; padding: 14px 16px;
      backdrop-filter: blur(8px);
    }
    .help-head {
      display: flex; justify-content: space-between; align-items: center;
      font-family: 'IBM Plex Mono', monospace; font-size: 10px;
      letter-spacing: 0.16em; text-transform: uppercase;
      color: var(--text-mute); margin-bottom: 10px;
    }
    .help-close {
      background: none; border: none; color: var(--text-dim);
      cursor: pointer; font-size: 18px; line-height: 1; padding: 0 4px;
    }
    .help-list { padding-left: 18px; font-size: 12px; color: var(--text-dim); }
    .help-list li { margin-bottom: 6px; }
    .help-list b { color: var(--text); font-weight: 500; }
    .help-note {
      font-size: 10px; color: var(--text-mute); margin-top: 10px;
      padding-top: 10px; border-top: 1px solid var(--line);
      font-style: italic;
    }
    kbd {
      font-family: 'IBM Plex Mono', monospace; font-size: 10px;
      background: var(--panel-2); border: 1px solid var(--line-2);
      padding: 1px 5px; border-radius: 2px; color: var(--accent);
    }

    /* Right panel: KV table */
    .kv {
      display: flex; justify-content: space-between;
      padding: 4px 0; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    }
    .kv-k { color: var(--text-mute); text-transform: uppercase; letter-spacing: 0.06em; font-size: 10px; }
    .kv-v { color: var(--text); }
    .kv-hl .kv-v { color: var(--accent); font-size: 14px; font-weight: 600; }
    .empty {
      font-size: 11px; color: var(--text-mute); padding: 8px 0;
      font-style: italic;
    }
    .hr { border: none; border-top: 1px solid var(--line); margin: 6px 0; }

    .pin-row {
      display: grid; grid-template-columns: 40px 50px 1fr;
      gap: 8px; align-items: center;
      background: transparent; border: 1px solid transparent;
      border-radius: 3px; padding: 6px 8px;
      font-family: 'IBM Plex Mono', monospace; font-size: 11px;
      color: var(--text-dim); cursor: pointer; text-align: left;
    }
    .pin-row:hover { background: var(--panel-2); color: var(--text); }
    .pin-row.active {
      background: var(--panel-2); border-color: var(--warn);
      color: var(--text);
    }
    .pin-row-i { color: var(--text-mute); }
    .pin-row.active .pin-row-i { color: var(--warn); }
    .pin-row-d { color: var(--text-dim); }
    .pin-row-t { text-align: right; color: var(--accent); }

    /* Dataset page */
    .data-page { padding: 32px 48px; overflow-y: auto; }

    /* Preview banner — shown only when this page is reached via dev override */
    .preview-banner {
      display: flex; align-items: center; gap: 12px;
      background: #2a1f08; border: 1px solid #5a4416;
      color: #f5b945; padding: 10px 14px; border-radius: 3px;
      margin-bottom: 20px; font-size: 12px; line-height: 1.5;
    }
    .preview-banner code {
      background: rgba(0,0,0,0.4); padding: 1px 5px; border-radius: 2px;
      font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #f5b945;
    }
    .preview-banner-tag {
      font-family: 'IBM Plex Mono', monospace; font-weight: 600;
      letter-spacing: 0.16em; font-size: 10px;
      background: #f5b945; color: #0e1220; padding: 3px 8px; border-radius: 2px;
      flex-shrink: 0;
    }

    .rssi-note {
      display: inline-block; margin-left: 8px; padding: 2px 8px;
      background: var(--panel-2); border: 1px solid var(--line-2);
      border-radius: 2px; font-family: 'IBM Plex Mono', monospace;
      font-size: 10px; color: var(--text-dim);
    }
    .rssi-note b { color: var(--accent); }
    .data-head h2 {
      font-family: 'IBM Plex Mono', monospace; font-weight: 600;
      letter-spacing: 0.04em; font-size: 22px; margin-bottom: 8px;
    }
    .data-head p { color: var(--text-dim); margin-bottom: 24px; max-width: 70ch; }
    .table-wrap {
      border: 1px solid var(--line); border-radius: 3px; overflow-x: auto;
      max-width: 100%;
    }
    .data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .data-table th {
      background: var(--panel); text-align: left; padding: 10px 14px;
      font-family: 'IBM Plex Mono', monospace; font-weight: 500;
      letter-spacing: 0.06em; text-transform: uppercase; font-size: 10px;
      color: var(--text-mute); cursor: pointer; user-select: none;
      border-bottom: 1px solid var(--line);
    }
    .data-table th:hover { color: var(--text); }
    .data-table th.sorted { color: var(--accent); }
    .data-table td {
      padding: 8px 14px; border-bottom: 1px solid var(--line);
      color: var(--text-dim); font-family: 'IBM Plex Mono', monospace;
    }
    .data-table tr:hover td { background: var(--panel-2); color: var(--text); }
    .data-table td.num { text-align: right; }
    .data-table td.strong { color: var(--accent); font-weight: 600; }
    .data-foot { margin-top: 16px; }
    .muted { color: var(--text-mute); font-size: 11px; max-width: 70ch; }
    .muted code {
      background: var(--panel-2); padding: 1px 5px; border-radius: 2px;
      font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    }

    /* About page */
    .about { padding: 32px 48px; max-width: 70ch; overflow-y: auto; }
    .about h2 {
      font-family: 'IBM Plex Mono', monospace; font-weight: 600;
      letter-spacing: 0.04em; font-size: 22px; margin-bottom: 16px;
    }
    .about h3 {
      font-family: 'IBM Plex Mono', monospace; font-size: 11px;
      letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent);
      margin: 24px 0 10px;
    }
    .about p { color: var(--text-dim); margin-bottom: 12px; }
    .about ul { color: var(--text-dim); padding-left: 20px; }
    .about li { margin-bottom: 6px; }
    .about code {
      background: var(--panel-2); padding: 1px 5px; border-radius: 2px;
      font-family: 'IBM Plex Mono', monospace; font-size: 11px;
      color: var(--accent);
    }
  `}</style>;
}
