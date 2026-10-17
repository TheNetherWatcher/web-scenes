# Car scene: "Vesper" (original mid-engine hypercar, wet road, golden hour)

## Design
- 4.55 m wedge: lofted superellipse body (width/top/bottom curves per x), separate canopy, front
  fender volumes, carved arches, door scoop, side air intake, front mouth + brake ducts, engine-cover vent,
  twin exhausts, slim LED head blades, full-width tail bar, mirrors, pitched wing on swan-neck pylons.
- Wheels: thin-sidewall tyre, 3 rim styles (5 twin-spoke / 10-spoke / 7-spoke), drilled disc, red calipers.

## How it renders
- Models are SDFs (sdf.js). A compute shader samples them on a 1.4 cm grid (6 mm for wheels) and bakes
  material masks (glass, carbon, cavity, lamp, trim) plus ambient occlusion; mesher.js runs CPU surface nets
  and builds vertex buffers (about 1 s at start-up). `?res=1.0` meshes finer (cm).
- Raster frame: car drawn twice (second pass mirrored under the road), sky, then the road blended over
  the mirrored scene (fresnel x wetness x floor reflectivity).
- Shading (shaders.js): one analytic dusk environment drives everything. Paint = clearcoat + metallic
  base + footprint-faded flakes; carbon twill weave; tinted glass; hex-mesh cavities; panel gap lines in-shader;
  emissive lamps. Road: aggregate noise, puddles, lane paint, contact AO and soft sun shadow from a proxy.
- Engine post: HDR, 8-level bloom, AgX.

## Controls
Drag orbit, wheel zoom, WASD slide the view, H panel. Sliders: paint hue/lightness, flake, sun elevation/azimuth,
wheel spin (road scrolls with it), rim style, headlights, floor reflectivity, exposure, bloom.
URL: `cam=tx,ty,tz,yawDeg,pitchDeg,dist`, any slider key, `res`.

## Known limits
- Body is smooth-shaded from a 1.4 cm grid: very fine creases are rounded; shut lines are shader-drawn.
- No interior (glass is a dark tint); no sun shadow on the body itself (AO + analytic light only).
- Rim style change re-meshes the wheel (about 100 ms hitch). Reflection is a planar mirror pass (no blur).
- Not measured on a laptop GPU; load is two 400k-triangle passes plus one road quad.
