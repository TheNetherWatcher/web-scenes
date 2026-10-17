# Campfire scene

Open `fire.html` (dev server) or `dist/fire.min.html` (after `node build.mjs fire`). Hero shot: `docs/fire.png`.

## How it is made
- **Flame**: raymarched volume (72 steps) inside a camera-facing billboard. Density = tapered cylinder warped by a
  scrolling 128^3 tileable noise volume (generated on the GPU at start-up), eroded by two finer octaves, tongue
  height from a slow "lick" noise. Temperature -> black-body ramp (red -> orange -> yellow-white), blue base, soot
  absorption. The billboard sits 0.75 m in front of the fire so the flame draws over the sticks inside it.
- **Particles** (compute sim, curl-ish noise from the same volume, no per-frame allocation): 2200 sparks (streaks
  stretched along velocity), 900 flame fragments torn off the tip, 2400 smoke puffs lit by the fire. A 14 s
  warm-up runs at init so smoke/sparks are alive on frame one.
- **Firelight**: one flickering point light (intensity + position jitter, flame height coupled). Analytic soft
  shadows from the 12 stones and 9 log segments, closed-form in-scattering of the light through mist (sky too),
  1/(d^2+0.8) falloff. Logs have glowing cracks driven by distance to the core.
- **World**: 200k instanced grass blades (wind gusts, translucency, root AO, lit per vertex), warped-grid ground
  (flat near the pit, rolling beyond), ash bed with pulsing coals, procedural sky (dusk -> night: stars, Milky Way,
  moon with craters, afterglow clouds), distant treeline, fireflies, heat-haze wobble of the background.

## Controls
Click: mouse look, WASD walk, Shift run, wheel = walk speed, H = panel (size, light, turbulence, wind, sparks, smoke,
grass density/wind, mist, heat haze, dusk->night, exposure, bloom). URL: `cam=x,y,z,yaw,pitch`, `eye=`, `tod=`,
any slider key, `ui=0`, `blobs=0` (hide flame fragments).

## Known limits
- Flame is lit by itself only as emission; it does not shade the logs it overlaps (logs get the point light).
- Heat haze only distorts the sky/treeline, not the grass behind the flame.
- Smoke is unsorted soft sprites; very close views show overlap banding.
- Grass is a fixed 52 m patch; beyond that only the ground texture is drawn. Camera cannot enter the 1.5 m stone ring.
- Flame raymarch is the cost driver when standing close; lower `dpr=1` in the URL on weak GPUs.
