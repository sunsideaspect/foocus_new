import os
import re
import tempfile
from typing import Optional

import args_manager


_PIPE = None
_LOAD_ERR: Optional[str] = None


_BLOCKED_REPO_RE = re.compile(r"\b(nsfw|adult|porn|uncensored)\b", flags=re.IGNORECASE)


def _assert_lora_allowed(lora_id: str) -> None:
    if not lora_id:
        return
    if _BLOCKED_REPO_RE.search(lora_id):
        raise RuntimeError("Blocked LoRA repo id (NSFW/adult). Provide a non-NSFW LoRA id or leave empty.")
    if lora_id.strip().lower() == "lkzd7/wan2.2_loraset_nsfw".lower():
        raise RuntimeError("Blocked LoRA set (NSFW). Provide a non-NSFW LoRA id or leave empty.")


def get_pipe():
    global _PIPE, _LOAD_ERR
    if _PIPE is not None:
        return _PIPE
    if _LOAD_ERR is not None:
        raise RuntimeError(_LOAD_ERR)

    model_id = (args_manager.args.wan_model_id or "").strip()
    if not model_id:
        raise RuntimeError("Missing --wan-model-id. Set it to a Hugging Face Wan I2V model id.")

    try:
        import torch
        from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline

        token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip() or None

        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            token=token,
        )

        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()

        lora_id = (args_manager.args.wan_lora_id or "").strip()
        if lora_id:
            _assert_lora_allowed(lora_id)
            pipe.load_lora_weights(lora_id, token=token)

        _PIPE = pipe
        return pipe
    except Exception as e:
        _LOAD_ERR = f"Wan I2V load failed: {e}"
        raise


def generate_video(
    image_pil,
    prompt: str,
    negative_prompt: str,
    steps: int,
    duration_seconds: float,
    guidance_scale: float,
    guidance_scale_2: float,
    seed: int,
):
    import numpy as np
    import torch
    from diffusers.utils.export_utils import export_to_video

    if image_pil is None:
        raise ValueError("Input image is required.")

    pipe = get_pipe()

    fps = 24
    num_frames = max(8, int(round(duration_seconds * fps)) + 1)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(seed))
    frames = pipe(
        image=image_pil,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        generator=generator,
    ).frames[0]

    if isinstance(frames, np.ndarray):
        frames_list = list(frames)
    else:
        frames_list = frames

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    export_to_video(frames_list, path, fps=fps)
    return path

