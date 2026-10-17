# New York skyline (js/scenes/city)

Raw WebGPU, no libraries. Hero view (preset 1): Midtown from the East River at sunset, sun low behind
the towers, light shafts through haze, glitter path on the water. Golden hour, night, a Hudson
sunset, Brooklyn Bridge, downtown dusk and an aerial are the other presets. Sky, water and lights
run from full day to night.

## How it is built
- `gen.js` — all geometry, generated once on the CPU: land mask (signed distance, Hudson / East
  River / Upper Bay / piers / islands), Manhattan grid (29 deg, 275 x 80 m blocks), ~60k building
  tiers with district heights (FiDi, Midtown, Hudson Yards, 57th St), setbacks, pyramid crowns,
  rooftop tanks and bulkheads. Dense districts put a real tower on most lots (slim 24-60 m
  footprints, a few supertalls) so the skyline has a jagged crest. Hand-built: Empire State,
  Chrysler, One WTC (+3/4/7 WTC), Woolworth, BofA, One Vanderbilt, 432 Park, Central Park Tower,
  Steinway, Hudson Yards, 30 Rock, MetLife, Citigroup Center, Time Warner Center, United
  Nations, Statue of Liberty, Brooklyn Bridge (stone arches, cables, diagonal stays), Manhattan
  Bridge.
- `shaders.js` — shared preamble: sky (keyframed by sun elevation + sunlit cirrus + clouds), aerial
  perspective, sun shadows, instanced building shader.
- `shaders2.js` — cable ribbons, light sprites, ground (streets, crosswalks, lane lines, cars,
  Central Park) + river, sky pass.
- Sun shadows: a max-pooled height map (r16float mips) is ray-marched toward the sun per pixel.

## What makes it read as real
- **Aerial perspective.** Height-stratified extinction (slightly stronger in blue), horizon-sky
  in-scatter without the solar aureole, plus a forward-peaked Henyey-Greenstein lobe of sunlight
  and a thin low mist layer.
- **Light shafts.** The sunlit part of that haze is marched along the view ray (14 steps, jittered)
  and each sample asks the height map whether a tower shadows it, so Midtown casts shadow into the
  haze and gaps leak shafts. Same march darkens the sky pass in front of the skyline. Toggle with
  the `shafts` slider / `?shafts=0`.
- **Directional ambient.** With the sun low, faces looking toward it catch the glowing sky and faces
  looking away only see the dim anti-solar sky, so backlit towers go to true silhouettes.
- **Glass.** Reflectance ~0.24 (not mirror) with Fresnel; reflection ray picks up neighbouring towers
  (height-map lookup at 55 / 205 / 505 m: dark or sunlit faces looking back) and the street when
  it points down. Bay-scale patchwork, window reveals / sills, corner AO, cornice bands, grime,
  street-level storefronts.
- **Night.** Windows resolve per cell up close and fade to floor-group lit patterns far away
  (never a uniform glow); landmark floodlights; aircraft beacons; sodium street glow.
- **River.** Skyline is re-rendered mirrored (y flipped) into a half-res target and sampled with
  wave-distorted, vertically blurred lookups; wave field is 8 wind-aligned components with
  non-octave wavelengths and wavelength-scaled phase warping (no tiling), capillary ripples near
  the camera, analytic GGX sun glitter, boat wakes, sunlit cirrus and clouds in the reflection.
- **Streets.** Asphalt with curbs, double-yellow, lane dashes, zebra crossings, daytime cars
  (yellow cabs), wet-look sheen; night headlights / tail lights as traffic streaks.

## Controls
Free fly (click = mouse look, WASD, Space/E up, Q/C down, Shift fast, wheel = speed).
Keys 1-9 = presets (1 East River sunset, 2 Midtown golden hour, 3 Downtown from Jersey City,
4 Brooklyn Bridge, 5 Midtown night, 6 Hudson sunset, 7 aerial dusk, 8 Empire State close-up,
9 Lower Manhattan dusk). T = time-lapse. H = panel (sun elevation / azimuth, haze, light shafts,
window glow, clouds, water roughness, exposure, bloom). URL: `?preset=3`, `?sunElev=-10`,
`?cam=x,y,z,yaw,pitch`, `?shafts=0`.

## Known limits
- Buildings are boxes / frustums with shader detail; no real facades, street furniture or trees.
  Close to a wall the windows are procedural and flat (no depth parallax).
- Far shorelines (NJ, Brooklyn, Queens) are low generic blocks; no terrain (Palisades are flat).
- Shadows come from a 10 m height map: soft and approximate, not exact for thin spires.
- Mirror reflection is half-res; fine detail (bridge lamps) can show as dashes on the water.
- Cars are flat decals on the road, boats are simple boxes.
- Frame rate only measured headless on an Apple GPU (vsync-limited 60 fps at 1080p); the haze march
  is the main extra cost and can be switched off.
