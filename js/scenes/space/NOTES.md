# Space: black hole

Per-pixel relativistic ray tracing around a spinning black hole with a lit accretion disc, lensed star field, Milky Way and a companion star. Raw WebGPU, no libs.

## Method
- Pass A (render-scale res, `wgsl.js` TRACE): each pixel integrates a null geodesic with the Schwarzschild
  acceleration a = -1.5 rs h^2 x / r^5 (exact for photon orbits) plus a Lense-Thirring frame-drag term for spin.
  Adaptive step: coarse far away, fine near the horizon and inside the disc slab. Capture at r < rs.
- Disc: thin flared slab with log-spiral arms + Keplerian-sheared ring noise (inner gas laps the outer), turbulent
  density, volumetric absorption/emission. ISCO inner edge is the Kerr prograde value for the chosen spin.
  Colour = blackbody of T(r) shifted by combined Doppler + gravitational redshift; intensity beamed ~g^3.6
  (approaching side is brighter and whiter, receding side redder and dimmer).
- Pass A writes disc light + transmittance and the escape direction (rgba32f). Pass B (full res) looks up the sky
  along that direction: cube-grid star layers (5 densities, blackbody colours), footprint-aware splatting from the
  lensed direction derivatives (keeps surface brightness, no aliasing in the Einstein ring), Milky Way band with
  dust lanes and nebulae, blue giant companion that smears into an arc/ring.
- Engine bloom + AgX tonemap finish the frame.

## Controls
Drag orbit, wheel zoom (clamped outside the horizon), WASD/QE slide target, H panel (spin, mass, disc brightness /
temperature / tilt / thickness, lensing steps, star density, exposure, bloom, render scale, auto scale).
Camera auto-rotates slowly when idle. URL: `cam=tx,ty,tz,yaw,pitch,dist`, any param key, `bench=1` logs GPU ms.

## Performance
1080p on this dev GPU: trace ~11.7 ms @ scale 1, ~7.5 ms @ 0.75, ~4.6 ms @ 0.5; composite ~3 ms.
Auto scale lowers the trace resolution if frames stay over ~20 ms; hi-dpi canvases start at 0.6.

## Known limits
- Schwarzschild geometry; spin only shifts the ISCO and adds a weak frame-drag term (horizon / shadow shape not exact Kerr).
- Disc light is not re-scattered; no volumetric corona beyond a faint gaussian halo; no temporal accumulation, so the
  photon-ring rim can shimmer slightly at scale < 1 (raise render scale for stills).
- Star positions are not re-blueshifted by the hole.
