# Car scene: "Vesper" (original mid-engine hypercar, wet road, golden hour)

## Design
- 4.58 m wedge. Body = polygon cross-sections lofted along the car (`loftMain`): flat floor, rocker, widest flank, a crisp
  beltline crease, crowned fenders over the wheels that dip to a hood spine, rounded with a 2 cm radius. Raked nose with a
  carbon splitter, cab-forward canopy, swept-up tail.
- Detail carved into the SDF: hood channels, hood vents, headlamp housings, door scoop + shut lines, side intake with
  louvres, rear arch intake, twin titanium exhaust tips, tail-light bar. The DRL blade, projector and housing edge are
  painted in the fragment shader from the same plan-view shapes (`HEAD_SHAPES` in sdf.js) so they stay crisp.
- Rear wing: cambered aerofoil section, gurney flap, end plates, swan-neck pylons (all carbon).
- Wheels (3 rim styles): thin-sidewall tyre with tread, moulded ridge and sidewall lettering, graphite spokes with a
  diamond-cut lip, drilled disc, red caliper. Interior (dash, seats, tunnel) is ray-cast through the glass.

## How it renders (per frame)
1. Environment cube (512^2, 10 mips): sky, clouds, hills, tree line, street lamps and a Fresnel "wet road" below the
   horizon are baked when the sun or lamps change (`ENV_BAKE`, `ENV_MIP` in shaders.js). The sun itself stays analytic
   (`sunLobe`) so it can be shadowed.
2. Car distance texture (208x60x100 rgba16float, 2.5 cm, built once): `carShadow` sphere-traces it for soft sun shadows on
   the body and on the road; `carAO` gives contact occlusion on the road.
3. Mirror pass: the car is drawn upside-down under the road into an offscreen target (colour + height above road), then
   mip-chained. The road samples it with 12 rotated taps; blur radius grows with roughness and with height above the road
   (contact hardening), and the lookup is offset by the road normal.
4. Main pass: car (clear coat over metallic flake over pigment, orange peel, rain beads, carbon twill, tinted glass with a
   cabin behind it, hex-mesh cavities), analytic sky + lamp posts, wet road (aggregate, cracks, patches, puddles, grass
   shoulders, GGX-style sky / lamp / sun reflection through the same roughness model).
5. Engine post: HDR bloom + AgX.

Body is meshed once on one side only (SDF -> GPU compute grid in 5 z-slabs -> CPU surface nets, 8 mm cells, ~1-2 s) and drawn
twice, the second instance mirrored in z.

## Controls
Drag orbit, wheel zoom, WASD slide the view, H panel. Sliders: paint hue/lightness, flake, orange peel, rain beads, sun
elevation/azimuth, wheel spin (road scrolls with it), rim style, headlights, street lamps, floor reflectivity, exposure,
bloom. URL: `cam=tx,ty,tz,yawDeg,pitchDeg,dist`, any slider key, `res=0.8` (body cell in cm). Dev: `perf=1` logs the frame
time to the dev server, `off=car,sky,road,mirror` skips passes for profiling.

## Measured
Apple M4 Pro, 1920x1080: about 6.5 ms/frame (road ~2.1, mirror pass ~1.2, car ~1.1, sky ~0.7, rest post). Not measured on
weaker GPUs; the road shader is the first thing to trim (`?dpr=1` helps most).

## Known limits
- Fine creases are still limited by the 8 mm grid (tight panel-edge radii look slightly bumpy at the 2.4 m minimum zoom).
- Panel gaps are shader lines (not geometry); the cabin is a handful of boxes, so it only reads through the side glass.
- Mirror reflection is planar (road only); the sky, lamps and sun are analytic / baked.
- Rim style change re-meshes the wheel (about 100 ms hitch).
