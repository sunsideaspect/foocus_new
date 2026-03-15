# Realism and Face-Preservation Tuning

This fork is tuned to prioritize photorealistic outputs and stronger face preservation from a reference image.

## Quick start

Use the new preset:

```bash
python entry_with_update.py --preset realistic_identity
```

The preset enables:

- `Quality` performance mode
- realistic stock-photo model defaults
- image prompt panel by default
- first image prompt slot set to `FaceSwap` with stronger stop/weight

## Recommended workflow for face-preserving generations

1. Open **Input Image** -> **Image Prompt**.
2. Put the target face in slot #1 (`FaceSwap`).
3. Use a clean, front-facing source portrait for best identity retention.
4. Keep prompts focused on scene, lens, lighting, and clothing instead of face shape.
5. Start with the preset defaults:
   - `FaceSwap stop`: `0.97`
   - `FaceSwap weight`: `1.15`
6. If identity is weak:
   - increase weight in small steps (`+0.05`)
   - keep stop high (`0.95-0.99`)
7. If composition is too rigid or artifacts appear:
   - decrease weight slightly
   - lower stop to `0.92-0.95`

## Quality controls (most impactful)

- **Performance**: keep `Quality` for detail fidelity.
- **Steps**: higher steps improve texture consistency (default override is set high in realistic presets).
- **Sampler/Scheduler**: `dpmpp_2m_sde_gpu + karras` is kept as default.
- **Styles**: realistic presets use photograph-focused styles and avoid aggressive stylization.

## Practical limits

- This pipeline can strongly preserve identity, but it cannot guarantee exact 1:1 identity in every pose, angle, or lighting setup.
- Extreme prompt changes (e.g., profile view from frontal-only source) will reduce similarity.
