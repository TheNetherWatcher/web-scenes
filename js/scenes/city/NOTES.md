# New York skyline (js/scenes/city)

Raw WebGPU, no libraries. Hero view (preset 1): lower Manhattan + Midtown seen from the Upper
Bay at dusk, sun just set, windows lit. Golden hour is preset 2. Sky, water and lights run
from full day to night.

## How it is built
- `gen.js` — all geometry, generated once on the CPU: land mask (signed distance, Hudson / East
  River / Upper Bay / piers / islands), Manhattan grid (29 deg, 275 x 80 m blocks), ~60k building
  tiers with district heights (FiDi, Midtown, Hudson Yards, 57th St), setbacks, pyramid crowns,
  rooftop tanks and bulkheads. Hand-built: Empire State, Chrysler, One WTC (+3/4/7 WTC),
  Woolworth, BofA, One Vanderbilt, 432 Park, Central Park Tower, Steinway, Hudson Yards,
  Statue of Liberty, Brooklyn Bridge (stone arches, cables, diagonal stays), Manhattan Bridge.
- `shaders.js` — shared sky/fog/shadow preamble + instanced building shader (procedural windows
  with per-cell lit state, glass reflections, far-field window averaging, landmark up-lighting).
- `shaders2.js` — cable ribbons, light sprites (aircraft beacons, boats, bridge lamps, helicopter,
  jet), ground (streets, traffic, Central Park) + river, sky pass (stars, moon, clouds, sun).
- Sun shadows: a max-pooled height map (r16float mips) is ray-marched toward the sun per pixel.
- River: skyline is re-rendered mirrored (y flipped) into a half-res target and sampled with
  wave-distorted, vertically blurred lookups; analytic GGX sun glitter; boat wakes.

## Controls
Free fly (click = mouse look, WASD, Space/E up, Q/C down, Shift fast, wheel = speed).
Keys 1-8 = presets (dusk hero, Midtown golden hour, Downtown from Jersey City, Brooklyn Bridge,
Midtown night, sunset glitter, aerial dusk, Empire State close-up). T = time-lapse. H = panel (sun elevation / azimuth, haze, window glow,
clouds, water roughness, exposure, bloom). URL: `?preset=3`, `?sunElev=-10`, `?cam=x,y,z,yaw,pitch`.

## Known limits
- Realism pass: desaturated masonry/glass/steel, spiky height distribution with low gaps, landmark
  height caps around ESB/Chrysler/One WTC, floor banding + mullions + blinds, rooftop clutter,
  warm sun rim, cool ambient, blue-grey aerial perspective.
- Buildings are boxes / frustums with shader detail; no real facades, street furniture or trees.
- Far shorelines (NJ, Brooklyn, Queens) are low generic blocks; no terrain (Palisades are flat).
- Shadows come from a 10 m height map: soft and approximate, not exact for thin spires.
- Reflection is half-res; very close to piers its streaks can look smeared.
- Daytime shows a few lit windows as bright dots on towers (intentional baseline).
