# Campfire scene

Open `fire.html` (dev server) or `dist/fire.min.html` (after `node build.mjs fire`). Hero shot: `docs/fire.png`.

## How it is made
- **Flame**: raymarched volume (24-88 adaptive steps) inside a camera-facing billboard. Density comes from
  anisotropic 3-D noise (tight in xz, long in y) so it breaks into vertical tongues; every tongue has its own
  height (two "lick" noises), the whole body swirls/leans with a slow warp, the base is ragged (per-column floor)
  so flames emerge from between the logs. Temperature -> black-body ramp (deep red -> orange -> yellow-white), a
  thin blue sheet at the fuel, per-step emission/absorption `(1-exp(-w))`. The billboard sits 0.75 m in front of the
  fire so the flame draws over the sticks inside it.
- **Wood**: five star logs (inner ends raised), three cross logs, teepee sticks, kindling. Surface is procedural:
  charred blocks (Worley cell edges from a second 128^3 volume, stretched along the log) with glowing cracks that
  pulse, brighter towards the core and per-cell random; raw bark further out; end grain with rings + radial checks.
  Charcoal has a small specular sheen from the fire light.
- **Stones**: boxy boulders (superellipsoid + lumps, poles turned sideways), granite palette with mica speckle,
  soot on the fire side and underneath.
- **Ground**: trodden soil around the pit (grit, pebbles, litter), scorched ring, grey ash bed with a warped network
  of glowing embers partly covered by ash. Grass is thinned and shortened near the ring.
- **Particles** (compute sim, curl-ish noise from the noise volume): 2200 sparks (streaks stretched along
  velocity), 900 flame fragments, 2400 smoke puffs. Smoke is shaded in the particle plane (lit side faces the
  fire) with `1/(d^2+3)` falloff so it glows orange near the flame and goes grey/blue higher up. A 14 s warm-up runs
  at init.
- **Firelight**: one flickering point light (intensity + position jitter, flame height coupled). Analytic soft
  shadows from 12 stones + 16 log spheres, closed-form in-scattering through mist, `1/(d^2+0.8)` falloff.
- **Grass**: 200k instanced blades (wind gusts, translucency, root AO), dry meadow palette (straw/gold with a few
  olive clumps), blades seen through the hot plume shimmer (vertex-space heat distortion). Sky has the same heat
  wobble.
- **Sky**: dusk -> night, isotropic 3-D-cell star field, Milky Way, cratered moon, afterglow clouds, treeline,
  fireflies.

## Controls
Click: mouse look, WASD walk, Shift run, wheel = walk speed, H = panel (size, light, turbulence, wind, sparks, smoke,
grass density/wind, mist, heat haze, dusk->night, exposure, bloom). URL: `cam=x,y,z,yaw,pitch`, `eye=`, `tod=`,
any slider key, `ui=0`, `blobs=0` (hide flame fragments).

## Known limits
- Flame is lit by itself only as emission; it does not shade the logs it overlaps (logs get the point light).
- Heat distortion covers sky + grass only (ground/stones/logs are not refracted).
- Smoke is unsorted soft sprites; very close views show overlap banding.
- Grass is a fixed 52 m patch; beyond that only the ground texture is drawn. Camera cannot enter the 1.5 m stone ring.
- Flame raymarch is the cost driver when standing close; use `dpr=1` in the URL on weak GPUs.
- Headless timing on an Apple GPU at 1080p was ~18 ms/frame, which is the vsync floor of that harness; true
  uncapped frame rate is unmeasured.
