"""WanGP runtime for dual-conditioned Krea 2 Identity Edit.

The implementation deliberately reuses WanGP's Krea 2 transformer, scheduler,
VAE, callback and MMGP behavior. Identity-specific code is limited to the full
Qwen3-VL vision path and the clean reference-token stream.
"""

from __future__ import annotations

import hashlib
import math
import os
import types
import warnings
import weakref

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from accelerate import init_empty_weights
from PIL import Image
from safetensors import safe_open
from transformers import (
    AutoTokenizer,
    Qwen2VLImageProcessor,
    Qwen2VLImageProcessorFast,
    Qwen2VLProcessor,
)
from transformers.video_processing_utils import BaseVideoProcessor

from mmgp import offload
from shared.utils import files_locator as fl

from models.ideogram4.qwen3_vl_configuration import (
    Qwen3VLConfig,
    register_qwen3_vl_config,
)
from models.ideogram4.qwen3_vl_transformers import (
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)
from models.krea2.krea2_main import (
    Krea2Qwen3VLProcessor,
    Krea2Pipeline,
    _TEXT_ENCODER_SELECT_LAYERS,
    _TRANSFORMER_CONFIG_PATH,
    _load_transformer,
    _load_vae,
    _pack_image_latents,
)
from models.krea2.krea2_mmdit import attention, key_padding_mask, ropeapply

from .krea2_advanced_settings import expand_advanced_settings
from .krea2_custom_depth_mask import load_custom_depth_mask
from .krea2_identity_features import (
    REID_EXPERIMENTS_DISABLED_MESSAGE,
    reid_experiments_enabled,
)
from .krea2_registered_outpaint import canvas_bbox, composite, prepare_source

from .krea2_identity_utils import (
    DEFAULT_GROUNDING_PX,
    DEPTH_CONTROL_LORA_FILENAME,
    REID_INFERENCE_STEPS,
    REID_QWEN_MAX_PIXELS,
    REID_VAE_MAX_PIXELS,
    TWO_REFERENCE_RECOMMENDED_PIXELS,
    delay_user_loras_for_depth,
    schedule_builtin_identity_depth_adapters,
    depth_control_mask_selected,
    depth_control_selected,
    depth_control_lora_url,
    direct_image_control_selected,
    fit_identity_reference_geometry,
    fit_reference_pixel_budget,
    identity_lora_url,
    match_reference_dimensions,
    outpaint_lora_url,
    preprocess_krea2_adapter_state_dict,
    reid_lora_url,
    resolve_generation_process,
    resolve_identity_reference_boosts,
    resolve_wangp_checkpoint,
    subject_attention_boost_for_step,
    three_phase_value_for_step,
    validate_builtin_adapter_timing,
    validate_builtin_depth_ramp,
    validate_builtin_identity_ramp,
    validate_depth_control_strength,
    validate_depth_user_lora_ramp,
    validate_depth_user_lora_timing,
    validate_direct_image_denoising_strength,
    validate_depth_mask_feather_px,
    validate_grounding_px,
    validate_identity_method,
    validate_outpaint_seam_px,
    validate_reference_images,
    validate_reid_reference_images,
    validate_reid_lora_strength,
    validate_subject_attention_ramp,
    validate_subject_attention_timing,
)


_DEFAULT_NEGATIVE_PROMPT = ""
_LEGACY_VISION_FILENAME = "qwen3vl_4b_fp8_scaled.safetensors"


def _content_fingerprint(value):
    """Return a short, non-reversible fingerprint for generation diagnostics."""
    digest = hashlib.sha256()
    if torch.is_tensor(value):
        tensor = value.detach().to("cpu").contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return f"{tuple(tensor.shape)}:{digest.hexdigest()[:12]}"
    if isinstance(value, Image.Image):
        digest.update(value.mode.encode("ascii", errors="replace"))
        digest.update(str(value.size).encode("ascii"))
        digest.update(value.tobytes())
        return f"{value.width}x{value.height}:{digest.hexdigest()[:12]}"
    return "none" if value is None else f"unsupported:{type(value).__name__}"


def _sampled_conditioning_diagnostics(value, sample_count=4096):
    """Fingerprint grounded conditioning without copying the full tensor to CPU."""
    if not torch.is_tensor(value):
        return f"unsupported:{type(value).__name__}"
    flat = value.detach().reshape(-1)
    count = min(max(1, int(sample_count)), int(flat.numel()))
    if count == int(flat.numel()):
        sample = flat
    else:
        indices = torch.linspace(
            0,
            int(flat.numel()) - 1,
            steps=count,
            device=flat.device,
            dtype=torch.float64,
        ).round().to(torch.long)
        sample = flat.index_select(0, indices)
    sample = sample.float().cpu().contiguous()
    digest = hashlib.sha256(sample.numpy().tobytes()).hexdigest()[:12]
    rms = sample.square().mean().sqrt().item()
    return (
        f"shape={tuple(value.shape)}, dtype={value.dtype}, samples={count}, "
        f"sha256={digest}, mean={sample.mean().item():.6f}, rms={rms:.6f}"
    )


def _sampled_kv_cache_diagnostics(caches, samples_per_tensor=128):
    """Fingerprint isolated reference K/V without copying the full cache."""
    if not caches:
        return "blocks=0"
    key_samples = []
    value_samples = []
    for cached in caches:
        if not isinstance(cached, tuple) or len(cached) != 2:
            return f"unsupported:{type(cached).__name__}"
        for tensor, destination in zip(cached, (key_samples, value_samples)):
            if not torch.is_tensor(tensor):
                return f"unsupported:{type(tensor).__name__}"
            flat = tensor.detach().reshape(-1)
            count = min(max(1, int(samples_per_tensor)), int(flat.numel()))
            if count == int(flat.numel()):
                sample = flat
            else:
                indices = torch.linspace(
                    0,
                    int(flat.numel()) - 1,
                    steps=count,
                    device=flat.device,
                    dtype=torch.float64,
                ).round().to(torch.long)
                sample = flat.index_select(0, indices)
            destination.append(sample.float().cpu())
    keys = torch.cat(key_samples).contiguous()
    values = torch.cat(value_samples).contiguous()
    digest = hashlib.sha256()
    digest.update(keys.numpy().tobytes())
    digest.update(values.numpy().tobytes())
    return (
        f"blocks={len(caches)}, samples={keys.numel() + values.numel()}, "
        f"sha256={digest.hexdigest()[:12]}, "
        f"k_rms={keys.square().mean().sqrt().item():.6f}, "
        f"v_rms={values.square().mean().sqrt().item():.6f}"
    )


def _load_custom_depth_mask_tensor(source, control, video_prompt_type):
    """Load a custom mask as WanGP's CxTxHxW mask contract."""
    if not torch.is_tensor(control) or control.ndim not in (4, 5):
        raise TypeError("Custom depth mask requires a processed Control Image")
    control_height, control_width = map(int, control.shape[-2:])
    outside = "N" in str(video_prompt_type or "")
    mask_image, channel = load_custom_depth_mask(source, invert=outside)
    mask_width, mask_height = mask_image.size
    mask_ratio = mask_width / mask_height
    control_ratio = control_width / control_height
    ratio_error = abs(mask_ratio / control_ratio - 1.0)
    if ratio_error > 0.02:
        raise ValueError(
            "Custom depth mask aspect ratio must match the Control Image/output "
            f"aspect (mask {mask_width}x{mask_height}, processed control "
            f"{control_width}x{control_height})"
        )
    pixels = np.array(mask_image, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(pixels).div_(255.0).unsqueeze(0).unsqueeze(0)
    return tensor, {
        "channel": channel,
        "outside": outside,
        "size": mask_image.size,
    }


def _fingerprint_list(values):
    return "[" + ", ".join(_content_fingerprint(value) for value in (values or [])) + "]"


def _with_plugin_lora_weights(loras_slists, weights):
    """Clone WanGP schedules and replace the leading plugin-adapter weights."""
    phase_weights = {
        phase: list(weights) for phase in ("phase1", "phase2", "phase3")
    }
    return _with_plugin_lora_phase_weights(loras_slists, phase_weights)


def _with_plugin_lora_phase_weights(loras_slists, phase_weights):
    """Clone WanGP schedules and set per-phase plugin-adapter weights."""
    if loras_slists is None:
        raise ValueError("WanGP did not provide LoRA schedules")
    import copy

    result = copy.deepcopy(loras_slists)
    adapter_count = max((len(value) for value in phase_weights.values()), default=0)
    dictionaries = [result]
    dictionaries.extend(value for value in result.values() if isinstance(value, dict))
    for schedules in dictionaries:
        for phase in ("phase1", "phase2", "phase3"):
            values = schedules.get(phase)
            weights = phase_weights.get(phase)
            if (
                weights is not None
                and isinstance(values, list)
                and len(values) >= len(weights)
            ):
                values[: len(weights)] = list(weights)
        shared = schedules.get("shared")
        if isinstance(shared, list) and len(shared) >= adapter_count:
            shared[:adapter_count] = [True] * adapter_count
    return result


def _with_phase_lora_weights(
    loras_slists,
    plugin_weights,
    *,
    user_scale=1.0,
):
    """Clone a schedule, set plugin weights, and uniformly scale user LoRAs."""
    result = _with_plugin_lora_weights(loras_slists, plugin_weights)
    dictionaries = [result]
    dictionaries.extend(value for value in result.values() if isinstance(value, dict))
    visited = set()

    def scale(value):
        if isinstance(value, list):
            return [item * user_scale for item in value]
        return value * user_scale

    for schedules in dictionaries:
        identity = id(schedules)
        if identity in visited:
            continue
        visited.add(identity)
        for phase in ("phase1", "phase2", "phase3"):
            values = schedules.get(phase)
            if not isinstance(values, list):
                continue
            for index in range(len(plugin_weights), len(values)):
                values[index] = scale(values[index])
    return result


def _decoded_image_to_pil(images):
    """Convert the first WanGP BxCxHxW uint8 result into an RGB PIL image."""
    if images is None or len(images) != 1:
        raise ValueError("Two-phase Depth then ReID requires one phase-1 image")
    source_tensor = images[0].detach().cpu()
    if source_tensor.ndim != 3 or source_tensor.shape[0] not in (3, 4):
        raise ValueError(
            "WanGP returned an unsupported phase-1 image tensor shape: "
            f"{tuple(source_tensor.shape)}"
        )
    source_tensor = source_tensor[:3].permute(1, 2, 0)
    return Image.fromarray(source_tensor.to(torch.uint8).numpy(), mode="RGB")


def _wangp_control_to_pil(control, label):
    """Convert WanGP's first processed Control Image frame into RGB PIL."""
    if isinstance(control, Image.Image):
        return control.convert("RGB")
    if not torch.is_tensor(control):
        raise TypeError(f"{label} must be a PIL image or WanGP processed tensor")
    from shared.utils.utils import convert_tensor_to_image

    image = convert_tensor_to_image(control)
    if not isinstance(image, Image.Image):
        raise TypeError(f"WanGP returned an unsupported {label} image type")
    return image.convert("RGB")


def _resolve_wangp_checkpoint(path):
    return resolve_wangp_checkpoint(
        path,
        lambda candidate: fl.locate_file(candidate, error_if_none=False),
    )


class IdentityQwen3VLModel(Qwen3VLModel):
    """The reusable Qwen3-VL visual/language core without generation-only heads."""

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self._ar_cache = None


class GroundedQwen3VLConditioner(torch.nn.Module):
    """Produce the twelve Krea 2 hidden layers with image grounding."""

    template_prefix = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
        "<|im_end|>\n<|im_start|>user\n"
    )
    template_suffix = "<|im_end|>\n<|im_start|>assistant\n"
    prefix_tokens = 34

    def __init__(
        self,
        qwen,
        tokenizer,
        processor,
        select_layers=_TEXT_ENCODER_SELECT_LAYERS,
        max_prompt_tokens=512,
    ):
        super().__init__()
        self.qwen = qwen
        self.tokenizer = tokenizer
        self.processor = processor
        self.select_layers = tuple(select_layers)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.reference_images = []
        self.name_references = False
        self.grounding_px = DEFAULT_GROUNDING_PX
        self._interrupt = False
        self.allow_text_only = False

    def set_references(
        self,
        images,
        grounding_px,
        max_pixels=None,
        *,
        name_references=False,
    ):
        self.grounding_px = validate_grounding_px(grounding_px)
        self.reference_images = []
        self.name_references = bool(name_references)
        for image in images:
            image = image.convert("RGB")
            width, height = image.size
            if max_pixels is not None:
                scale = min(1.0, (int(max_pixels) / (width * height)) ** 0.5)
            elif max(width, height) > self.grounding_px:
                scale = self.grounding_px / max(width, height)
            else:
                scale = 1.0
            if scale < 1.0:
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    resample=Image.Resampling.LANCZOS,
                )
            self.reference_images.append(image)

    def clear_references(self):
        self.reference_images = []
        self.name_references = False

    def _bounded_prompt(self, prompt):
        prompt = str(prompt)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_prompt_tokens:
            return prompt
        return self.tokenizer.decode(
            ids[: self.max_prompt_tokens], skip_special_tokens=False
        )

    def _template(self, prompt):
        if self.name_references:
            # ReID's pinned Ostris pipeline and training contract name every
            # Qwen vision placeholder before the user prompt. Keep this
            # opt-in so the established Identity Edit message is unchanged.
            vision = "".join(
                f"Picture {index + 1}: "
                "<|vision_start|><|image_pad|><|vision_end|>"
                for index, _image in enumerate(self.reference_images)
            )
        else:
            vision = "".join(
                "<|vision_start|><|image_pad|><|vision_end|>"
                for _ in self.reference_images
            )
        return self.template_prefix + vision + self._bounded_prompt(prompt) + self.template_suffix

    def _encode_one(self, prompt, device):
        if self._interrupt:
            return None, None
        processor_kwargs = dict(
            text=[self._template(prompt)], padding=True, return_tensors="pt"
        )
        if self.reference_images:
            processor_kwargs["images"] = self.reference_images
        encoded = self.processor(**processor_kwargs).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        pixel_values = encoded.get("pixel_values")
        image_grid_thw = encoded.get("image_grid_thw")

        self.qwen.language_model._interrupt = self._interrupt
        self.qwen.visual._interrupt = self._interrupt
        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)
        deepstack = None
        if pixel_values is not None:
            image_embeds, deepstack = self.qwen.get_image_features(
                pixel_values, image_grid_thw
            )
            if image_embeds is None or self._interrupt:
                return None, None
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.qwen.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            image_mask = torch.zeros_like(inputs_embeds, dtype=torch.bool)
        visual_pos_masks = image_mask[..., 0]
        position_ids, _ = self.qwen.get_rope_index(
            input_ids, image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        selected = [layer - 1 for layer in self.select_layers]
        states = self.qwen.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack,
            return_mid_results_layers=selected,
        )
        if states.last_hidden_state is None or self._interrupt:
            return None, None
        hiddens = torch.stack(states.mid_results, dim=2)
        states.mid_results = None
        return hiddens[:, self.prefix_tokens :], attention_mask[:, self.prefix_tokens :]

    @torch.inference_mode()
    def forward(self, text: list[str], device):
        if not self.reference_images and not self.allow_text_only:
            raise ValueError("Grounded Qwen3-VL encoding requires reference images")
        hidden_batches, mask_batches = [], []
        for prompt in text:
            hidden, mask = self._encode_one(prompt, device)
            if hidden is None:
                return None, None
            hidden_batches.append(hidden[0])
            mask_batches.append(mask[0])
        return torch.stack(hidden_batches), torch.stack(mask_batches)


def _identity_build_stream(model, img, context, pos, mask, source_imgs, freqs=None):
    """Build [text | refs(frame 1..N) | target(frame 0)] with MMGP padding."""
    txtlen, target_len = context.shape[1], img.shape[1]
    batch = img.shape[0]
    target_pos = pos[:, -target_len:].clone()
    target_grid_h = int(target_pos[..., 1].amax().item()) + 1
    target_grid_w = int(target_pos[..., 2].amax().item()) + 1
    source_grids = getattr(model, "_identity_source_grids", None)
    if source_grids is None:
        source_grids = [(target_grid_h, target_grid_w)] * len(source_imgs)
    if len(source_grids) != len(source_imgs):
        raise ValueError("Identity reference token/grid counts do not match")
    source_positions = []
    source_masks = []
    for frame, (source, source_grid) in enumerate(
        zip(source_imgs, source_grids), start=1
    ):
        grid_h, grid_w = map(int, source_grid)
        if grid_h * grid_w != source.shape[1]:
            raise ValueError(
                "Identity reference grid does not match its packed token count"
            )
        source_pos = torch.zeros(
            batch, grid_h * grid_w, 3, device=pos.device, dtype=pos.dtype
        )
        source_pos[..., 0] = frame
        offset_h = (target_grid_h - grid_h) // 2
        offset_w = (target_grid_w - grid_w) // 2
        source_pos[..., 1] = (
            torch.arange(grid_h, device=pos.device) + offset_h
        ).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        source_pos[..., 2] = (
            torch.arange(grid_w, device=pos.device) + offset_w
        ).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        source_positions.append(source_pos)
        source_masks.append(
            torch.ones(batch, source.shape[1], device=mask.device, dtype=torch.bool)
        )
    combined = torch.cat([context, *source_imgs, img], dim=1)
    combined_pos = torch.cat(
        [pos[:, :txtlen], *source_positions, target_pos], dim=1
    )
    combined_mask = torch.cat(
        [mask[:, :txtlen], *source_masks, mask[:, -target_len:]], dim=1
    )
    source_len = sum(source.shape[1] for source in source_imgs)
    padlen = (-combined.shape[1]) % 256
    if padlen:
        combined = F.pad(combined, (0, 0, 0, padlen))
        combined_pos = F.pad(combined_pos, (0, 0, 0, padlen))
        combined_mask = F.pad(combined_mask, (0, padlen), value=False)
    if freqs is None:
        freqs = model.posemb(combined_pos).to(combined.dtype)
    return (
        combined,
        txtlen,
        source_len,
        target_len,
        freqs,
        key_padding_mask(combined_mask),
    )


def _identity_reference_attention_mask(
    model,
    base_mask,
    combined,
    txtlen,
    source_imgs,
    target_len,
):
    """Bias target queries toward reference keys while preserving padding.

    WanGP's Krea attention and shared masked-attention wrappers each transpose
    the two middle mask axes.  The two transposes cancel, so query-dependent
    masks must enter the Krea block as ``[B, 1, Q, K]``.  The ordinary padding
    mask is ``[B, 1, 1, K]`` and hid this distinction because transposing its
    singleton axes has no effect.
    """
    subject_boost = float(
        getattr(
            model,
            "_identity_effective_subject_boost",
            getattr(model, "_identity_subject_boost", 1.0),
        )
    )
    scene_boost = float(getattr(model, "_identity_scene_boost", 1.0))
    if subject_boost == 1.0 and scene_boost == 1.0:
        return base_mask

    source_lengths = [int(source.shape[1]) for source in source_imgs]
    if not source_lengths:
        return base_mask
    total_len = int(combined.shape[1])
    source_end = txtlen + sum(source_lengths)
    target_end = source_end + target_len
    bias = combined.new_zeros((1, 1, total_len, total_len))
    offset = txtlen
    for index, source_len in enumerate(source_lengths):
        boost = subject_boost if index == len(source_lengths) - 1 else scene_boost
        if boost != 1.0:
            bias[:, :, source_end:target_end, offset : offset + source_len] = (
                math.log(max(boost, 1e-4))
            )
        offset += source_len

    if base_mask is not None:
        if base_mask.dtype == torch.bool:
            padding_bias = combined.new_zeros(base_mask.shape)
            padding_bias.masked_fill_(~base_mask, float("-inf"))
        else:
            padding_bias = base_mask.to(device=combined.device, dtype=combined.dtype)
        bias = bias + padding_bias

    phase = int(getattr(model, "_identity_subject_attention_phase", 0))
    logged_phases = getattr(model, "_identity_reference_boost_logged_phases", set())
    if phase not in logged_phases:
        timing = getattr(model, "_identity_subject_attention_timing", "constant")
        phase_label = "constant" if timing == "constant" else f"{phase + 1}/3"
        print(
            "[Krea2 Identity][Identity Edit] reference attention boost active: "
            f"subject={subject_boost:.2f}x, scene={scene_boost:.2f}x, "
            f"references={len(source_lengths)}, timing={timing}, phase={phase_label}"
        )
        logged_phases.add(phase)
        model._identity_reference_boost_logged_phases = logged_phases
    return bias


def _identity_advance_subject_attention(model, timestep):
    """Advance step-dependent identity attention and direct depth strength."""
    if torch.is_tensor(timestep):
        flat = timestep.detach().reshape(-1)
        key = None if flat.numel() == 0 else float(flat[0].item())
    else:
        key = float(timestep)
    if key != getattr(model, "_identity_last_timestep", None):
        model._identity_last_timestep = key
        model._identity_step_index = min(
            int(getattr(model, "_identity_step_index", -1)) + 1,
            max(int(getattr(model, "_identity_num_steps", 1)) - 1, 0),
        )
    boost, phase = subject_attention_boost_for_step(
        getattr(model, "_identity_subject_boost", 1.0),
        getattr(model, "_identity_subject_attention_timing", "constant"),
        getattr(model, "_identity_subject_attention_ramp", (1.0, 2.0, 8.0)),
        getattr(model, "_identity_step_index", 0),
        getattr(model, "_identity_num_steps", 1),
    )
    model._identity_effective_subject_boost = boost
    model._identity_subject_attention_phase = phase
    depth_scale = float(getattr(model, "_depth_control_base_scale", 1.0))
    depth_phase = 0
    if getattr(model, "_depth_control_timing", "simultaneous") == "depth_then_identity":
        depth_multiplier, depth_phase = three_phase_value_for_step(
            getattr(model, "_depth_control_ramp", (1.0, 0.5, 0.0)),
            getattr(model, "_identity_step_index", 0),
            getattr(model, "_identity_num_steps", 1),
        )
        depth_scale *= depth_multiplier
    model._depth_control_effective_scale = depth_scale
    model._depth_control_phase = depth_phase


def _match_token_batch(tokens, batch, label):
    if tokens.shape[0] == batch:
        return tokens
    if tokens.shape[0] == 1:
        return tokens.expand(batch, -1, -1).contiguous()
    raise ValueError(
        f"{label} batch {tokens.shape[0]} does not match target batch {batch}"
    )


def _project_identity_inputs(model, img, sources):
    """Project clean references natively and depth-condition target tokens only."""
    batch = img.shape[0]
    target = model.first(img)
    depth = getattr(model, "_depth_control_tokens", None)
    if depth is not None:
        depth = _match_token_batch(depth, batch, "Depth control")
        weight = getattr(model, "_depth_control_runtime_weight", None)
        if weight is None:
            raise RuntimeError("Depth control projection weights were not prepared")
        if depth.shape[1] != img.shape[1]:
            raise ValueError(
                "Depth control token count must match the noisy target token count"
            )
        if depth.shape[-1] != weight.shape[1] or target.shape[-1] != weight.shape[0]:
            raise ValueError(
                "Depth control projection dimensions are incompatible with Krea 2"
            )
        depth_scale = float(
            getattr(model, "_depth_control_effective_scale", 1.0)
        )
        contribution = F.linear(depth.to(target.dtype), weight) * depth_scale
        depth_mask = getattr(model, "_depth_control_mask_tokens", None)
        if depth_mask is not None:
            depth_mask = _match_token_batch(depth_mask, batch, "Depth mask")
            if depth_mask.shape[1] != contribution.shape[1]:
                raise ValueError(
                    "Depth mask token count must match the target token count"
                )
            if depth_mask.shape[-1] != 1:
                raise ValueError("Depth mask tokens must contain one channel")
            contribution = contribution * depth_mask.to(
                device=contribution.device,
                dtype=contribution.dtype,
            )
        if not getattr(model, "_depth_diagnostics_logged", False):
            target_rms = target.float().square().mean().sqrt().item()
            contribution_rms = contribution.float().square().mean().sqrt().item()
            ratio = contribution_rms / max(target_rms, 1e-12)
            mask_coverage = (
                1.0
                if depth_mask is None
                else depth_mask.float().mean().item()
            )
            print(
                "[Krea2 Identity][Depth] target projection active: "
                f"target_rms={target_rms:.6f}, "
                f"control_rms={contribution_rms:.6f}, "
                f"control/target={ratio:.6f}, "
                f"scale={depth_scale:.4f}, "
                f"mask_coverage={mask_coverage:.4f}"
            )
            model._depth_diagnostics_logged = True
        target = target + contribution
    source_imgs = [
        model.first(_match_token_batch(source, batch, "Identity reference"))
        for source in sources
    ]
    return target, source_imgs


def _identity_forward(
    self, img, context, t, tvec, pos, mask=None,
    NAG=None, neg_context=None, neg_mask=None,
    target_len=None,
):
    # WanGP v12.34 supplies target_len because its native edit path can append
    # references before calling the transformer. This plugin appends and slices
    # its own independent reference stream, so img already defines our target.
    del target_len
    _identity_advance_subject_attention(self, t)
    sources = getattr(self, "_identity_source_tokens", None)
    if not sources:
        raise RuntimeError("Identity Edit source tokens were not prepared")
    target, source_imgs = _project_identity_inputs(self, img, sources)
    combined, txtlen, srclen, target_len, freqs, stream_mask = _identity_build_stream(
        self, target, context, pos, mask, source_imgs
    )
    if NAG is not None and (
        getattr(self, "_identity_effective_subject_boost", 1.0) != 1.0
        or getattr(self, "_identity_scene_boost", 1.0) != 1.0
    ):
        raise ValueError(
            "Identity Edit reference-fidelity boosts currently require NAG scale 1.0"
        )
    stream_mask = _identity_reference_attention_mask(
        self, stream_mask, combined, txtlen, source_imgs, target_len
    )
    del target, context, pos
    for block in self.blocks:
        combined = block(
            combined, tvec, freqs, stream_mask, txt_len=txtlen,
            NAG=NAG, neg_context=neg_context, neg_mask=neg_mask,
        )
        if getattr(self, "_interrupt", False):
            return None
    start = txtlen + srclen
    return self.last([combined[:, start : start + target_len]], t)


def _identity_forward_cfg(
    self, img, context, uncond_context, t, tvec, pos, uncond_pos,
    mask, uncond_mask,
    target_len=None,
):
    del target_len
    _identity_advance_subject_attention(self, t)
    sources = getattr(self, "_identity_source_tokens", None)
    if not sources:
        raise RuntimeError("Identity Edit source tokens were not prepared")
    target, source_imgs = _project_identity_inputs(self, img, sources)
    cond = _identity_build_stream(self, target, context, pos, mask, source_imgs)
    uncond = _identity_build_stream(
        self, target, uncond_context, uncond_pos, uncond_mask, source_imgs,
        freqs=cond[4] if pos.shape == uncond_pos.shape else None,
    )
    cond = (*cond[:5], _identity_reference_attention_mask(
        self, cond[5], cond[0], cond[1], source_imgs, cond[3]
    ))
    uncond = (*uncond[:5], _identity_reference_attention_mask(
        self, uncond[5], uncond[0], uncond[1], source_imgs, uncond[3]
    ))
    del target, context, uncond_context, pos, uncond_pos
    cond_stream, uncond_stream = cond[0], uncond[0]
    for block in self.blocks:
        cond_stream = block(cond_stream, tvec, cond[4], cond[5])
        if getattr(self, "_interrupt", False):
            return None, None
        uncond_stream = block(uncond_stream, tvec, uncond[4], uncond[5])
        if getattr(self, "_interrupt", False):
            return None, None
    cond_start, uncond_start = cond[1] + cond[2], uncond[1] + uncond[2]
    return (
        self.last([cond_stream[:, cond_start : cond_start + cond[3]]], t),
        self.last([uncond_stream[:, uncond_start : uncond_start + uncond[3]]], t),
    )


def _depth_forward(
    self, img, context, t, tvec, pos, mask=None,
    NAG=None, neg_context=None, neg_mask=None,
    target_len=None,
):
    """Run normal Krea text conditioning with depth projected onto the target."""
    target, _sources = _project_identity_inputs(self, img, [])
    combined, txtlen, imglen, freqs, stream_mask = self._build_stream(
        target, context, pos, mask
    )
    del target, context, pos
    for block in self.blocks:
        combined = block(
            combined, tvec, freqs, stream_mask, txt_len=txtlen,
            NAG=NAG, neg_context=neg_context, neg_mask=neg_mask,
        )
        if getattr(self, "_interrupt", False):
            return None
        self.txtfusion._interrupt = getattr(self, "_interrupt", False)
    target_len = imglen if target_len is None else target_len
    return self.last(
        [combined[:, txtlen + imglen - target_len : txtlen + imglen]], t
    )


def _depth_forward_cfg(
    self, img, context, uncond_context, t, tvec, pos, uncond_pos,
    mask, uncond_mask,
    target_len=None,
):
    """Run the CFG pair for reference-free depth + prompt generation."""
    target, _sources = _project_identity_inputs(self, img, [])
    share_freqs = pos.shape == uncond_pos.shape
    combined, txtlen, imglen, freqs, stream_mask = self._build_stream(
        target, context, pos, mask
    )
    uncond, uncond_txtlen, uncond_imglen, uncond_freqs, uncond_stream_mask = (
        self._build_stream(
            target,
            uncond_context,
            uncond_pos,
            uncond_mask,
            freqs=freqs if share_freqs else None,
        )
    )
    del target, context, uncond_context, pos, uncond_pos
    for block in self.blocks:
        combined = block(combined, tvec, freqs, stream_mask)
        if getattr(self, "_interrupt", False):
            return None, None
        uncond = block(uncond, tvec, uncond_freqs, uncond_stream_mask)
        if getattr(self, "_interrupt", False):
            return None, None
    target_len = imglen if target_len is None else target_len
    return (
        self.last(
            [combined[:, txtlen + imglen - target_len : txtlen + imglen]], t
        ),
        self.last(
            [
                uncond[
                    :,
                    uncond_txtlen + uncond_imglen - target_len :
                    uncond_txtlen + uncond_imglen,
                ]
            ],
            t,
        ),
    )


def _outpaint_attention(
    attn_module,
    hidden,
    freqs,
    mask=None,
    cached_kv=None,
    capture=False,
):
    """WanGP attention with optional isolated registered-reference K/V."""
    q = rearrange(attn_module.wq(hidden), "B L (H D) -> B H L D", H=attn_module.heads)
    k = rearrange(attn_module.wk(hidden), "B L (H D) -> B H L D", H=attn_module.kvheads)
    v = rearrange(attn_module.wv(hidden), "B L (H D) -> B H L D", H=attn_module.kvheads)
    q, k, v = attn_module.qknorm([q, k, v])
    if freqs is not None:
        q, k = ropeapply(q, k, freqs)
    captured = (k, v) if capture else None
    if cached_kv is not None:
        k = torch.cat((k, cached_kv[0].to(k.dtype)), dim=2)
        v = torch.cat((v, cached_kv[1].to(v.dtype)), dim=2)
        if mask is not None:
            reference_mask = torch.ones(
                (*mask.shape[:-1], cached_kv[0].shape[2]),
                device=mask.device,
                dtype=mask.dtype,
            )
            mask = torch.cat((mask, reference_mask), dim=-1)
    output = attention([q, k, v], mask=mask, gqa=attn_module.gqa)
    output.mul_(torch.sigmoid(attn_module.gate(hidden)))
    return attn_module.wo(output), captured


def _outpaint_block(
    block,
    hidden,
    vec,
    freqs,
    mask=None,
    cached_kv=None,
    capture=False,
):
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
    normalized = block.prenorm(hidden)
    normalized = normalized * (1 + prescale) + preshift
    attn_output, captured = _outpaint_attention(
        block.attn,
        normalized,
        freqs,
        mask=mask,
        cached_kv=cached_kv,
        capture=capture,
    )
    hidden = hidden + attn_output * pregate
    normalized = block.postnorm(hidden)
    normalized = normalized * (1 + postscale) + postshift
    hidden = hidden + block.mlp(normalized) * postgate
    return hidden, captured


def _registered_reference_positions(tokens, grid_size, bbox_normalized, device):
    ref_height, ref_width = grid_size
    x0, y0, x1, y1 = [float(value) for value in bbox_normalized]
    target_height, target_width = tokens
    ys = (torch.arange(ref_height, device=device) + 0.5) * (
        (y1 - y0) * target_height / ref_height
    ) + y0 * target_height - 0.5
    xs = (torch.arange(ref_width, device=device) + 0.5) * (
        (x1 - x0) * target_width / ref_width
    ) + x0 * target_width - 0.5
    positions = torch.zeros(ref_height, ref_width, 3, device=device)
    positions[..., 1] = ys[:, None]
    positions[..., 2] = xs[None, :]
    return positions.reshape(1, -1, 3)


def _reid_reference_positions(grid_size, device):
    """Build Kontext index positions for ReID's one isolated reference."""
    ref_height, ref_width = grid_size
    positions = torch.zeros(ref_height, ref_width, 3, device=device)
    positions[..., 0] = 1
    positions[..., 1] = torch.arange(ref_height, device=device)[:, None]
    positions[..., 2] = torch.arange(ref_width, device=device)[None, :]
    return positions.reshape(1, -1, 3)


def _mmgp_adapter_status(model, adapter_no):
    """Report whether one MMGP adapter is bound, selected and scheduled."""
    adapter = str(adapter_no)
    owner = model
    visited = set()
    while id(owner) not in visited:
        visited.add(id(owner))
        candidate = getattr(owner, "_lora_owner", None)
        if candidate is None or candidate is owner:
            break
        owner = candidate

    active_adapters = getattr(owner, "_loras_active_adapters", None)
    active = None
    if active_adapters is not None:
        active = adapter in {str(value) for value in active_adapters}

    module_data = getattr(owner, "_loras_model_data", None)
    bindings = None
    if isinstance(module_data, dict):
        bindings = sum(
            1
            for adapter_data in module_data.values()
            if isinstance(adapter_data, dict) and adapter in adapter_data
        )

    scaling = getattr(owner, "_loras_scaling", None)
    schedule = scaling.get(adapter) if isinstance(scaling, dict) else None
    if torch.is_tensor(schedule):
        schedule = schedule.detach().float().reshape(-1).tolist()
    elif isinstance(schedule, (tuple, list)):
        schedule = list(schedule)
    if isinstance(schedule, list):
        numeric = [float(value) for value in schedule]
        schedule_summary = (
            f"steps={len(numeric)}, first={numeric[0]:.4f}, "
            f"last={numeric[-1]:.4f}, min={min(numeric):.4f}, "
            f"max={max(numeric):.4f}"
            if numeric
            else "steps=0"
        )
    elif schedule is None:
        schedule_summary = "unavailable"
    else:
        schedule_summary = f"constant={float(schedule):.4f}"

    return {
        "adapter": adapter,
        "active": active,
        "bindings": bindings,
        "schedule": schedule_summary,
        "step": getattr(owner, "_lora_step_no", None),
    }


def _log_required_mmgp_adapter(model, adapter_no, label):
    """Log adapter evidence and reject a definitively inactive functional LoRA."""
    status = _mmgp_adapter_status(model, adapter_no)
    print(
        f"[Krea2 Identity][{label}] MMGP adapter status: "
        f"id={status['adapter']}, active={status['active']}, "
        f"module_bindings={status['bindings']}, "
        f"schedule=({status['schedule']}), step={status['step']}"
    )
    if status["active"] is False:
        raise RuntimeError(
            f"Krea 2 {label} adapter {status['adapter']} was loaded but is not active"
        )
    if status["bindings"] == 0:
        raise RuntimeError(
            f"Krea 2 {label} adapter {status['adapter']} did not bind to any "
            "transformer modules"
        )
    return status


def _precompute_registered_kv(transformer, ref_tokens, ref_pos):
    """Run the isolated reference stream once at flow time zero."""
    ref_hidden = transformer.first(ref_tokens)
    ref_freqs = transformer.posemb(ref_pos).to(ref_hidden.dtype)
    zero = torch.zeros(ref_hidden.shape[0], device=ref_hidden.device, dtype=ref_hidden.dtype)
    _t0, ref_vec = transformer.prepare_timestep(zero)
    caches = []
    for block in transformer.blocks:
        ref_hidden, captured = _outpaint_block(
            block, ref_hidden, ref_vec, ref_freqs, capture=True
        )
        caches.append(captured)
        if getattr(transformer, "_interrupt", False):
            return None
    return caches


def _reid_build_joint_stream(model, target, context, pos, mask, reference, ref_pos):
    """Build the official-style [text | target | reference] ReID stream."""
    txtlen = context.shape[1]
    target_len = target.shape[1]
    reference_len = reference.shape[1]
    batch = target.shape[0]
    target_pos = pos[:, -target_len:]
    if ref_pos.shape[0] == 1 and batch > 1:
        ref_pos = ref_pos.expand(batch, -1, -1)
    elif ref_pos.shape[0] != batch:
        raise ValueError("ReID reference position batch does not match target batch")
    combined = torch.cat((context, target, reference), dim=1)
    combined_pos = torch.cat((pos[:, :txtlen], target_pos, ref_pos), dim=1)
    combined_mask = None
    if mask is not None:
        reference_mask = torch.ones(
            batch,
            reference_len,
            device=mask.device,
            dtype=torch.bool,
        )
        combined_mask = torch.cat(
            (mask[:, :txtlen], mask[:, -target_len:], reference_mask),
            dim=1,
        )
    padlen = (-combined.shape[1]) % 256
    if padlen:
        combined = F.pad(combined, (0, 0, 0, padlen))
        combined_pos = F.pad(combined_pos, (0, 0, 0, padlen))
        if combined_mask is not None:
            combined_mask = F.pad(combined_mask, (0, padlen), value=False)
    freqs = model.posemb(combined_pos).to(combined.dtype)
    return (
        combined,
        txtlen,
        target_len,
        reference_len,
        freqs,
        None if combined_mask is None else key_padding_mask(combined_mask),
    )


def _reid_joint_timestep_zero_block(
    block,
    hidden,
    active_vec,
    zero_vec,
    reference_start,
    freqs,
    mask=None,
):
    """Process target and reference jointly, holding only the reference at t=0."""
    active = block.mod(active_vec)
    zero = block.mod(zero_vec)
    normalized = block.prenorm(hidden)
    normalized = torch.cat(
        (
            normalized[:, :reference_start] * (1 + active[0]) + active[1],
            normalized[:, reference_start:] * (1 + zero[0]) + zero[1],
        ),
        dim=1,
    )
    attn_output, _captured = _outpaint_attention(
        block.attn,
        normalized,
        freqs,
        mask=mask,
    )
    gated_attention = torch.cat(
        (
            attn_output[:, :reference_start] * active[2],
            attn_output[:, reference_start:] * zero[2],
        ),
        dim=1,
    )
    hidden = hidden + gated_attention
    normalized = block.postnorm(hidden)
    normalized = torch.cat(
        (
            normalized[:, :reference_start] * (1 + active[3]) + active[4],
            normalized[:, reference_start:] * (1 + zero[3]) + zero[4],
        ),
        dim=1,
    )
    mlp_output = block.mlp(normalized)
    gated_mlp = torch.cat(
        (
            mlp_output[:, :reference_start] * active[5],
            mlp_output[:, reference_start:] * zero[5],
        ),
        dim=1,
    )
    return hidden + gated_mlp


def _outpaint_forward(self, img, context, t, tvec, pos, mask=None, **_kwargs):
    caches = getattr(self, "_outpaint_ref_kv", None)
    if not caches:
        raise RuntimeError("Registered Outpaint reference K/V was not prepared")
    img = self.first(img)
    combined, txtlen, imglen, freqs, stream_mask = self._build_stream(
        img, context, pos, mask
    )
    for block, cached_kv in zip(self.blocks, caches):
        combined, _ = _outpaint_block(
            block,
            combined,
            tvec,
            freqs,
            mask=stream_mask,
            cached_kv=cached_kv,
        )
        if getattr(self, "_interrupt", False):
            return None
    return self.last([combined[:, txtlen : txtlen + imglen]], t)


def _outpaint_forward_cfg(
    self, img, context, uncond_context, t, tvec, pos, uncond_pos,
    mask, uncond_mask,
    target_len=None,
):
    del target_len
    cond = _outpaint_forward(self, img, context, t, tvec, pos, mask)
    if cond is None:
        return None, None
    uncond = _outpaint_forward(
        self, img, uncond_context, t, tvec, uncond_pos, uncond_mask
    )
    return cond, uncond


def _reid_forward(
    self, img, context, t, tvec, pos, mask=None,
    NAG=None, neg_context=None, neg_mask=None,
    target_len=None,
):
    """Predict the target with selectable official or legacy ReID reference flow."""
    del target_len
    target, _sources = _project_identity_inputs(self, img, [])
    method = getattr(self, "_reid_reference_method", "isolated_cache")
    if method == "joint_timestep_zero":
        ref_tokens = getattr(self, "_reid_ref_tokens", None)
        ref_pos = getattr(self, "_reid_ref_pos", None)
        if ref_tokens is None or ref_pos is None:
            raise RuntimeError("ReID joint reference stream was not prepared")
        reference = self.first(
            _match_token_batch(ref_tokens, target.shape[0], "ReID reference")
        )
        combined, txtlen, target_len, reference_len, freqs, stream_mask = (
            _reid_build_joint_stream(
                self,
                target,
                context,
                pos,
                mask,
                reference,
                ref_pos,
            )
        )
        zero_t = torch.zeros(
            target.shape[0],
            device=target.device,
            dtype=target.dtype,
        )
        _zero_timestep, zero_vec = self.prepare_timestep(zero_t)
        reference_start = txtlen + target_len
        if not getattr(self, "_reid_joint_logged", False):
            print(
                "[Krea2 Identity][ReID] joint timestep-zero stream active: "
                f"text={txtlen}, target={target_len}, reference={reference_len}, "
                f"padded_total={combined.shape[1]}"
            )
            self._reid_joint_logged = True
        for block in self.blocks:
            combined = _reid_joint_timestep_zero_block(
                block,
                combined,
                tvec,
                zero_vec,
                reference_start,
                freqs,
                mask=stream_mask,
            )
            if getattr(self, "_interrupt", False):
                return None
        return self.last([combined[:, txtlen : txtlen + target_len]], t)

    caches = getattr(self, "_reid_ref_kv", None)
    if not caches:
        raise RuntimeError("ReID isolated reference K/V was not prepared")
    combined, txtlen, imglen, freqs, stream_mask = self._build_stream(
        target, context, pos, mask
    )
    for block, cached_kv in zip(self.blocks, caches):
        combined, _ = _outpaint_block(
            block, combined, tvec, freqs, mask=stream_mask, cached_kv=cached_kv
        )
        if getattr(self, "_interrupt", False):
            return None
    return self.last([combined[:, txtlen : txtlen + imglen]], t)


def _reid_forward_cfg(
    self, img, context, uncond_context, t, tvec, pos, uncond_pos,
    mask, uncond_mask,
    target_len=None,
):
    del target_len
    cond = _reid_forward(self, img, context, t, tvec, pos, mask)
    if cond is None:
        return None, None
    uncond = _reid_forward(
        self, img, uncond_context, t, tvec, uncond_pos, uncond_mask
    )
    return cond, uncond


def _attach_identity_transformer_methods(transformer):
    def dispatch_forward(target, *args, **kwargs):
        mode = getattr(target, "_conditioning_mode", "identity")
        if mode == "depth_prompt":
            return _depth_forward(target, *args, **kwargs)
        if mode == "outpaint":
            return _outpaint_forward(target, *args, **kwargs)
        if mode == "reid":
            return _reid_forward(target, *args, **kwargs)
        return _identity_forward(target, *args, **kwargs)

    def dispatch_forward_cfg(target, *args, **kwargs):
        mode = getattr(target, "_conditioning_mode", "identity")
        if mode == "depth_prompt":
            return _depth_forward_cfg(target, *args, **kwargs)
        if mode == "outpaint":
            return _outpaint_forward_cfg(target, *args, **kwargs)
        if mode == "reid":
            return _reid_forward_cfg(target, *args, **kwargs)
        return _identity_forward_cfg(target, *args, **kwargs)

    transformer.forward = types.MethodType(dispatch_forward, transformer)
    transformer.forward_cfg = types.MethodType(dispatch_forward_cfg, transformer)
    transformer._conditioning_mode = "identity"
    transformer._identity_source_tokens = None
    transformer._identity_source_grids = None
    transformer._identity_subject_boost = 1.0
    transformer._identity_effective_subject_boost = 1.0
    transformer._identity_scene_boost = 1.0
    transformer._identity_reference_boost_logged = False
    transformer._identity_reference_boost_logged_phases = set()
    transformer._identity_subject_attention_timing = "constant"
    transformer._identity_subject_attention_ramp = (1.0, 2.0, 8.0)
    transformer._identity_subject_attention_phase = 0
    transformer._identity_num_steps = 1
    transformer._identity_step_index = -1
    transformer._identity_last_timestep = None
    transformer._outpaint_ref_kv = None
    transformer._reid_ref_kv = None
    transformer._reid_ref_tokens = None
    transformer._reid_ref_pos = None
    transformer._reid_reference_method = "isolated_cache"
    transformer._reid_joint_logged = False
    transformer._depth_control_weight_cpu = None
    transformer._depth_control_runtime_weight = None
    transformer._depth_control_tokens = None
    transformer._depth_control_mask_tokens = None
    transformer._depth_diagnostics_logged = False
    transformer._depth_control_base_scale = 1.0
    transformer._depth_control_effective_scale = 1.0
    transformer._depth_control_timing = "simultaneous"
    transformer._depth_control_ramp = (1.0, 0.5, 0.0)
    transformer._depth_control_phase = 0
    transformer_ref = weakref.ref(transformer)

    def preprocess_loras(_model_type, state_dict):
        converted, control_weight = preprocess_krea2_adapter_state_dict(state_dict)
        if control_weight is not None:
            target = transformer_ref()
            if target is None:
                raise RuntimeError("Krea 2 transformer was released during LoRA loading")
            target._depth_control_weight_cpu = control_weight.detach().cpu().clone()
        return converted

    transformer.preprocess_loras = preprocess_loras


class IdentityKrea2Pipeline(Krea2Pipeline):
    def _encode_prompts(self, prompts, device, dtype, images=None):
        # WanGP v12.34 added the image keyword to the base Krea 2 pipeline.
        # This plugin owns the grounded reference lifecycle through
        # encoder.set_references(), so the base-pipeline value is intentionally
        # unused while retaining the updated host method contract.
        del images
        self.encoder._interrupt = self._interrupt
        hidden, masks = self.encoder(prompts, device=device)
        if hidden is None:
            return None, None
        print(
            "[Krea2 Identity][Encoder] grounded conditioning: "
            f"{_sampled_conditioning_diagnostics(hidden)}, "
            f"active_tokens={int(masks.sum().item())}/{masks.numel()}"
        )
        return hidden.to(device=device, dtype=dtype), masks.to(device=device)

    def _encode_identity_reference(
        self,
        image,
        width,
        height,
        device,
        dtype,
        *,
        fit,
    ):
        """Encode a clean reference without distorting secondary subjects."""
        image = image.convert("RGB")
        encode_width, encode_height = width, height
        if fit:
            align = self.compression * self.transformer.config.patch
            (encode_width, encode_height), crop = fit_identity_reference_geometry(
                image.size,
                (width, height),
                align=align,
            )
            if crop is not None:
                image = image.crop(crop)
            image = image.resize(
                (encode_width, encode_height),
                resample=Image.Resampling.BICUBIC,
            )
        latent = self._encode_image_to_latents(
            image,
            encode_width,
            encode_height,
            device,
            dtype,
        )
        patch = self.transformer.config.patch
        return latent, (latent.shape[-2] // patch, latent.shape[-1] // patch)

    def _encode_reid_reference(self, image, width, height, device, dtype, seed):
        """Encode ReID exactly like its pinned pipeline: posterior sample, not mode."""
        from shared.utils.utils import convert_image_to_tensor

        image = image.convert("RGB")
        tensor = (
            convert_image_to_tensor(image)
            .unsqueeze(0)
            .unsqueeze(2)
            .to(device=device, dtype=self.vae.dtype)
        )
        generator = torch.Generator(device=device).manual_seed(int(seed))
        latents = self.vae.encode(tensor).latent_dist.sample(generator)
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.channels, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(
            1, self.channels, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) / latents_std
        return latents[:, :, 0].to(device=device, dtype=dtype)

    @staticmethod
    def _first_control_frame(control, label):
        if not torch.is_tensor(control):
            raise TypeError(f"{label} must be a WanGP processed tensor")
        if control.ndim == 4:
            # WanGP control contract: channels, frames, height, width.
            if control.shape[1] < 1:
                raise ValueError(f"{label} has no frames")
            return control[:, 0].unsqueeze(0)
        if control.ndim == 5:
            # Tolerate the equivalent batched contract for direct API callers.
            if control.shape[0] != 1 or control.shape[2] < 1:
                raise ValueError(f"{label} must contain one non-empty batch")
            return control[:, :, 0]
        raise ValueError(f"{label} must be CxTxHxW or BxCxTxHxW")

    @classmethod
    def _prepare_depth_mask(
        cls,
        mask,
        width,
        height,
        device,
        feather_px,
    ):
        if mask is None:
            return None
        mask_pixels = cls._first_control_frame(mask, "Depth mask")
        if mask_pixels.shape[1] not in (1, 3, 4):
            raise ValueError("Depth mask must have 1, 3 or 4 channels")
        mask_pixels = mask_pixels[:, :1].to(device=device, dtype=torch.float32)
        if mask_pixels.amin() < 0:
            mask_pixels = mask_pixels.add(1).mul(0.5)
        mask_pixels = F.interpolate(
            mask_pixels,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp_(0, 1).ge_(0.5).to(torch.float32)
        hard_mask = mask_pixels
        radius = validate_depth_mask_feather_px(feather_px)
        if radius:
            mask_pixels = F.avg_pool2d(
                F.pad(
                    mask_pixels,
                    (radius, radius, radius, radius),
                    mode="replicate",
                ),
                kernel_size=radius * 2 + 1,
                stride=1,
            )
        return hard_mask, mask_pixels

    @torch.inference_mode()
    def _encode_depth_control(
        self,
        control,
        mask,
        feather_px,
        width,
        height,
        device,
        dtype,
    ):
        """VAE-encode WanGP's native Depth Anything V2 control tensor."""
        if self._interrupt:
            return None, None
        depth_pixels = self._first_control_frame(control, "Depth control")
        channels = depth_pixels.shape[1]
        if channels == 1:
            depth_pixels = depth_pixels.expand(-1, 3, -1, -1)
        elif channels in (3, 4):
            depth_pixels = depth_pixels[:, :3]
        else:
            raise ValueError(
                "Processed depth control must have 1, 3 or 4 channels"
            )
        depth_pixels = depth_pixels.to(device=device, dtype=self.vae.dtype)
        depth_pixels = F.interpolate(
            depth_pixels,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).clamp_(-1, 1)
        prepared_mask = self._prepare_depth_mask(
            mask,
            width,
            height,
            device,
            feather_px,
        )
        hard_mask = feathered_mask = None
        if prepared_mask is not None:
            hard_mask, feathered_mask = prepared_mask

        # WanGP normally supplies Depth Anything as three identical channels.
        # Its masked preview is a composite, however, and future host changes
        # could also hand us RGB data in the active area. Measure both the full
        # tensor and the area that will actually condition Krea, then collapse
        # to a canonical three-channel grayscale representation. This is an
        # exact no-op for valid replicated depth while making chroma leakage
        # through the depth adapter impossible.
        rgb_spread = depth_pixels.amax(dim=1, keepdim=True) - depth_pixels.amin(
            dim=1, keepdim=True
        )
        full_rgb_spread = rgb_spread.float().mean().item()
        if hard_mask is None:
            active_rgb_spread = full_rgb_spread
        else:
            active_pixels = hard_mask.sum().clamp_min(1)
            active_rgb_spread = (
                (rgb_spread.float() * hard_mask).sum() / active_pixels
            ).item()
        depth_pixels = depth_pixels.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        if hard_mask is not None:
            # WanGP's masked preview contains raw RGB outside the selected depth
            # area. Depth is normalized to [-1, 1], so excluded pixels must be
            # filled with -1 (black/far), not 0 (mid-grey/mid-depth). The latter
            # creates an artificial plane around the selected subject and can
            # overwhelm its pose. Token gating below remains the authoritative
            # spatial exclusion.
            hard_mask = hard_mask.to(depth_pixels.dtype)
            depth_pixels = depth_pixels * hard_mask - (1 - hard_mask)
        depth_pixels = depth_pixels.unsqueeze(2)
        latents = self.vae.encode(depth_pixels).latent_dist.mode()
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.channels, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(
            1, self.channels, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) / latents_std
        patch = self.transformer.config.patch
        tokens = _pack_image_latents(
            latents[:, :, 0].to(device=device, dtype=dtype),
            patch,
        )
        mask_tokens = None
        if feathered_mask is not None:
            token_height = latents.shape[-2] // patch
            token_width = latents.shape[-1] // patch
            mask_tokens = F.interpolate(
                feathered_mask,
                size=(token_height, token_width),
                mode="area",
            ).flatten(2).transpose(1, 2).contiguous()
        print(
            "[Krea2 Identity][Depth] prepared control: "
            f"pixels={width}x{height}, tokens={tuple(tokens.shape)}, "
            f"token_min={tokens.float().amin().item():.6f}, "
            f"token_max={tokens.float().amax().item():.6f}, "
            f"input_rgb_spread={full_rgb_spread:.6f}, "
            f"active_rgb_spread={active_rgb_spread:.6f}, "
            "channels=canonical-grayscale, "
            f"mask={'whole-frame' if mask_tokens is None else f'{mask_tokens.float().mean().item():.4f} mean'}"
        )
        return tokens, mask_tokens

    def _prepare_depth_runtime(
        self,
        depth_control,
        depth_mask,
        depth_mask_feather_px,
        width,
        height,
        device,
        dtype,
    ):
        if depth_control is None:
            return True
        control_weight = self.transformer._depth_control_weight_cpu
        if control_weight is None:
            raise RuntimeError(
                f"{DEPTH_CONTROL_LORA_FILENAME} did not provide its expanded "
                "Krea 2 input projection"
            )
        depth_tokens, depth_mask_tokens = self._encode_depth_control(
            depth_control,
            depth_mask,
            depth_mask_feather_px,
            width,
            height,
            device,
            dtype,
        )
        if depth_tokens is None:
            return False
        self.transformer._depth_control_tokens = depth_tokens
        self.transformer._depth_control_mask_tokens = depth_mask_tokens
        self.transformer._depth_control_runtime_weight = control_weight.to(
            device=device, dtype=dtype
        )
        self.transformer._depth_diagnostics_logged = False
        runtime_weight = self.transformer._depth_control_runtime_weight
        print(
            "[Krea2 Identity][Depth] adapter projection ready: "
            f"weight_shape={tuple(runtime_weight.shape)}, "
            f"weight_abs_mean={runtime_weight.float().abs().mean().item():.6f}, "
            f"weight_abs_max={runtime_weight.float().abs().amax().item():.6f}"
        )
        return True

    def _clear_depth_runtime(self):
        self.transformer._depth_control_tokens = None
        self.transformer._depth_control_mask_tokens = None
        self.transformer._depth_control_runtime_weight = None
        self.transformer._depth_diagnostics_logged = False
        self.transformer._depth_control_base_scale = 1.0
        self.transformer._depth_control_effective_scale = 1.0
        self.transformer._depth_control_timing = "simultaneous"
        self.transformer._depth_control_ramp = (1.0, 0.5, 0.0)
        self.transformer._depth_control_phase = 0

    @torch.inference_mode()
    def generate_identity(
        self,
        *args,
        reference_images,
        grounding_px,
        depth_control=None,
        depth_mask=None,
        depth_mask_feather_px=16,
        depth_control_strength=1.0,
        builtin_adapter_timing="simultaneous",
        builtin_depth_ramp=(1.0, 0.5, 0.0),
        reference_subject_boost=1.0,
        reference_scene_boost=1.0,
        subject_attention_timing="constant",
        subject_attention_ramp=(1.0, 2.0, 8.0),
        fit_all_references=False,
        fit_secondary_reference=True,
        **kwargs,
    ):
        references = validate_reference_images(reference_images)
        grounding_px = validate_grounding_px(grounding_px)
        width = int(kwargs.get("width", 1024))
        height = int(kwargs.get("height", 1024))
        device, dtype = self.runtime_device, self.dtype
        self.encoder.set_references(references, grounding_px)
        self.transformer._identity_subject_boost = float(reference_subject_boost)
        self.transformer._identity_effective_subject_boost = float(
            reference_subject_boost
        )
        self.transformer._identity_scene_boost = float(reference_scene_boost)
        self.transformer._identity_reference_boost_logged = False
        self.transformer._identity_reference_boost_logged_phases = set()
        self.transformer._identity_subject_attention_timing = (
            validate_subject_attention_timing(subject_attention_timing)
        )
        self.transformer._identity_subject_attention_ramp = (
            validate_subject_attention_ramp(*subject_attention_ramp)
        )
        if self.transformer._identity_subject_attention_timing == "ramp":
            early, middle, final = self.transformer._identity_subject_attention_ramp
            print(
                "[Krea2 Identity][Identity Edit] subject-attention ramp active: "
                f"early={early:.2f}x, middle={middle:.2f}x, final={final:.2f}x; "
                f"scene={float(reference_scene_boost):.2f}x"
            )
        self.transformer._identity_subject_attention_phase = 0
        self.transformer._identity_num_steps = max(int(kwargs.get("steps", 1)), 1)
        self.transformer._identity_step_index = -1
        self.transformer._identity_last_timestep = None
        self.transformer._depth_control_base_scale = validate_depth_control_strength(
            depth_control_strength
        )
        self.transformer._depth_control_effective_scale = (
            self.transformer._depth_control_base_scale
        )
        self.transformer._depth_control_timing = validate_builtin_adapter_timing(
            builtin_adapter_timing
        )
        self.transformer._depth_control_ramp = validate_builtin_depth_ramp(
            *builtin_depth_ramp
        )
        self.transformer._depth_control_phase = 0
        try:
            source_tokens = []
            source_grids = []
            for reference_index, image in enumerate(references):
                latent, source_grid = self._encode_identity_reference(
                    image,
                    width,
                    height,
                    device,
                    dtype,
                    fit=(
                        fit_all_references
                        or (fit_secondary_reference and reference_index >= 1)
                    ),
                )
                source_tokens.append(_pack_image_latents(latent, self.transformer.config.patch))
                source_grids.append(source_grid)
            self.transformer._identity_source_tokens = source_tokens
            self.transformer._identity_source_grids = source_grids
            if not self._prepare_depth_runtime(
                depth_control,
                depth_mask,
                depth_mask_feather_px,
                width,
                height,
                device,
                dtype,
            ):
                return None
            return super().__call__(*args, **kwargs)
        finally:
            self.transformer._identity_source_tokens = None
            self.transformer._identity_source_grids = None
            self.transformer._identity_subject_boost = 1.0
            self.transformer._identity_effective_subject_boost = 1.0
            self.transformer._identity_scene_boost = 1.0
            self.transformer._identity_reference_boost_logged = False
            self.transformer._identity_reference_boost_logged_phases = set()
            self.transformer._identity_subject_attention_timing = "constant"
            self.transformer._identity_subject_attention_ramp = (1.0, 2.0, 8.0)
            self.transformer._identity_subject_attention_phase = 0
            self.transformer._identity_num_steps = 1
            self.transformer._identity_step_index = -1
            self.transformer._identity_last_timestep = None
            self._clear_depth_runtime()
            self.encoder.clear_references()

    @torch.inference_mode()
    def generate_reid(
        self,
        *args,
        reference_images,
        reid_reference_method="isolated_cache",
        depth_control=None,
        depth_mask=None,
        depth_mask_feather_px=16,
        **kwargs,
    ):
        """Generate with one ReID reference using the selected reference flow."""
        if reid_reference_method not in {"joint_timestep_zero", "isolated_cache"}:
            raise ValueError(
                "reid_reference_method must be 'joint_timestep_zero' or "
                "'isolated_cache'"
            )
        references = validate_reid_reference_images(reference_images)
        width = int(kwargs.get("width", 1024))
        height = int(kwargs.get("height", 1024))
        device, dtype = self.runtime_device, self.dtype
        reference = references[0].convert("RGB")
        ref_width, ref_height = fit_reference_pixel_budget(
            reference.size,
            max_pixels=REID_VAE_MAX_PIXELS,
        )
        if reference.size != (ref_width, ref_height):
            # Match the released ReID pipeline's aspect-preserving resize.
            reference = reference.resize(
                (ref_width, ref_height), Image.Resampling.BILINEAR
            )
        latent = self._encode_reid_reference(
            reference,
            ref_width,
            ref_height,
            device,
            dtype,
            kwargs.get("seed", 0),
        )
        ref_tokens = _pack_image_latents(latent, self.transformer.config.patch)
        ref_grid = (
            latent.shape[-2] // self.transformer.config.patch,
            latent.shape[-1] // self.transformer.config.patch,
        )
        ref_pos = _reid_reference_positions(ref_grid, device)
        self.encoder.set_references(
            [reference],
            384,
            max_pixels=REID_QWEN_MAX_PIXELS,
            name_references=True,
        )
        self.transformer._conditioning_mode = "reid"
        try:
            if not self._prepare_depth_runtime(
                depth_control,
                depth_mask,
                depth_mask_feather_px,
                width,
                height,
                device,
                dtype,
            ):
                return None
            from shared.utils.loras_mutipliers import update_loras_slists

            update_loras_slists(
                self.transformer,
                kwargs.get("loras_slists"),
                int(kwargs.get("steps", REID_INFERENCE_STEPS)),
            )
            offload.set_step_no_for_lora(self.transformer, 0)
            _log_required_mmgp_adapter(self.transformer, 0, "ReID")
            self.transformer._reid_reference_method = reid_reference_method
            self.transformer._reid_joint_logged = False
            if reid_reference_method == "joint_timestep_zero":
                self.transformer._reid_ref_tokens = ref_tokens
                self.transformer._reid_ref_pos = ref_pos
                print(
                    "[Krea2 Identity][ReID] joint reference stream ready: "
                    f"vae_pixels={ref_width}x{ref_height}, "
                    f"vae_max_pixels={REID_VAE_MAX_PIXELS}, "
                    f"qwen_pixels={reference.width}x{reference.height}, "
                    f"tokens={tuple(ref_tokens.shape)}, "
                    f"qwen_max_pixels={REID_QWEN_MAX_PIXELS}, "
                    "reference_timestep=zero, qwen_grounding=Picture 1, "
                    "vae=posterior-sample"
                )
            else:
                self.transformer._reid_ref_kv = _precompute_registered_kv(
                    self.transformer, ref_tokens, ref_pos
                )
                if self.transformer._reid_ref_kv is None:
                    return None
                print(
                    "[Krea2 Identity][ReID] isolated reference cache ready: "
                    f"vae_pixels={ref_width}x{ref_height}, "
                    f"vae_max_pixels={REID_VAE_MAX_PIXELS}, "
                    f"qwen_pixels={reference.width}x{reference.height}, "
                    f"tokens={tuple(ref_tokens.shape)}, "
                    f"cache=({_sampled_kv_cache_diagnostics(self.transformer._reid_ref_kv)}), "
                    f"qwen_max_pixels={REID_QWEN_MAX_PIXELS}, "
                    "qwen_grounding=Picture 1, vae=posterior-sample"
                )
            return super().__call__(*args, **kwargs)
        finally:
            self.transformer._reid_ref_kv = None
            self.transformer._reid_ref_tokens = None
            self.transformer._reid_ref_pos = None
            self.transformer._reid_joint_logged = False
            self.transformer._conditioning_mode = "identity"
            self._clear_depth_runtime()
            self.encoder.clear_references()

    @torch.inference_mode()
    def generate_depth_prompt(
        self,
        *args,
        depth_control,
        depth_mask=None,
        depth_mask_feather_px=16,
        **kwargs,
    ):
        """Generate from text and depth without any identity/reference stream."""
        if depth_control is None:
            raise ValueError("Depth + prompt only requires processed depth control")
        width = int(kwargs.get("width", 1024))
        height = int(kwargs.get("height", 1024))
        device, dtype = self.runtime_device, self.dtype
        self.encoder.clear_references()
        self.encoder.allow_text_only = True
        self.transformer._conditioning_mode = "depth_prompt"
        try:
            if not self._prepare_depth_runtime(
                depth_control,
                depth_mask,
                depth_mask_feather_px,
                width,
                height,
                device,
                dtype,
            ):
                return None
            return super().__call__(*args, **kwargs)
        finally:
            self.transformer._conditioning_mode = "identity"
            self.encoder.allow_text_only = False
            self._clear_depth_runtime()
            self.encoder.clear_references()

    @torch.inference_mode()
    def generate_registered_outpaint(
        self,
        *args,
        source_image,
        outpainting_dims,
        seam_px=32,
        **kwargs,
    ):
        """Generate an expanded canvas with registered, isolated source attention."""
        width = int(kwargs.get("width", 1024))
        height = int(kwargs.get("height", 1024))
        bbox = canvas_bbox(source_image.size, (width, height), outpainting_dims)
        prepared = prepare_source(
            source_image, (width, height), bbox, seam_px=seam_px
        )
        condition_width = max(16, prepared.condition.width // 16 * 16)
        condition_height = max(16, prepared.condition.height // 16 * 16)
        condition = prepared.condition.resize(
            (condition_width, condition_height), Image.Resampling.LANCZOS
        )
        device, dtype = self.runtime_device, self.dtype
        latent = self._encode_image_to_latents(
            condition, condition_width, condition_height, device, dtype
        )
        ref_tokens = _pack_image_latents(latent, self.transformer.config.patch)
        ref_grid = (
            latent.shape[-2] // self.transformer.config.patch,
            latent.shape[-1] // self.transformer.config.patch,
        )
        target_grid = (
            height // (self.compression * self.transformer.config.patch),
            width // (self.compression * self.transformer.config.patch),
        )
        ref_pos = _registered_reference_positions(
            target_grid, ref_grid, prepared.bbox_normalized, device
        )
        self.encoder.clear_references()
        self.encoder.allow_text_only = True
        self.transformer._conditioning_mode = "outpaint"
        try:
            # The host normally activates schedules inside Krea2Pipeline.__call__,
            # but reference K/V is intentionally computed before that call.
            from shared.utils.loras_mutipliers import update_loras_slists

            update_loras_slists(
                self.transformer,
                kwargs.get("loras_slists"),
                int(kwargs.get("steps", 8)),
            )
            offload.set_step_no_for_lora(self.transformer, 0)
            self.transformer._outpaint_ref_kv = _precompute_registered_kv(
                self.transformer, ref_tokens, ref_pos
            )
            if self.transformer._outpaint_ref_kv is None:
                return None
            generated = super().__call__(*args, **kwargs)
            if generated is None:
                return None
            output = []
            for tensor in generated:
                array = tensor.detach().cpu()
                if array.ndim == 3 and array.shape[0] in (3, 4):
                    array = array[:3].permute(1, 2, 0)
                if array.dtype != torch.uint8:
                    array = array.clamp(0, 255).round().to(torch.uint8)
                output.append(composite(Image.fromarray(array.numpy()), prepared))
            from shared.utils.utils import convert_image_to_tensor

            return torch.stack(
                [convert_image_to_tensor(image).add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8) for image in output]
            )
        finally:
            self.transformer._outpaint_ref_kv = None
            self.transformer._conditioning_mode = "identity"
            self.encoder.allow_text_only = False
            self.encoder.clear_references()


def _uses_legacy_encoder_stack(text_encoder_filename) -> bool:
    basename = os.path.basename(os.fspath(text_encoder_filename)).lower()
    return "quanto" in basename or "int8" in basename


def _load_grounded_qwen(
    text_encoder_filename,
    config_path,
    dtype,
    *,
    visual_filename=None,
):
    register_qwen3_vl_config()
    config = Qwen3VLConfig.from_json_file(config_path)
    with init_empty_weights(include_buffers=True):
        qwen = IdentityQwen3VLModel(config)
    # These buffers are non-persistent and therefore absent from both
    # checkpoints. Materialize them before MMGP validates the meta-initialized
    # modules, matching WanGP's own Krea 2 language-model loader.
    qwen.language_model.rotary_emb.reset_inv_freq()
    qwen.visual.rotary_pos_emb.reset_inv_freq()
    if visual_filename is None:
        # Current WanGP v12.34 Edit contract: language and vision weights come
        # from the complete BF16 Qwen3-VL checkpoint.
        offload.load_model_data(
            qwen,
            text_encoder_filename,
            writable_tensors=False,
            default_dtype=dtype,
        )
    else:
        # Legacy plugin contract retained strictly as an A/B diagnostic: the
        # host-selected Quanto checkpoint supplies the language model while
        # Comfy-Org's scaled-FP8 checkpoint supplies the visual tower.
        offload.load_model_data(
            qwen.language_model,
            text_encoder_filename,
            modelPrefix="language_model",
            writable_tensors=False,
            default_dtype=dtype,
        )
        visual_prefix = "model.visual."
        with safe_open(visual_filename, framework="pt", device="cpu") as reader:
            visual_state_dict = {
                key[len(visual_prefix) :]: reader.get_tensor(key)
                for key in reader.keys()
                if key.startswith(visual_prefix)
            }
        if not visual_state_dict:
            raise RuntimeError(
                f"The legacy Qwen3-VL checkpoint has no {visual_prefix} weights: "
                f"{visual_filename}"
            )
        offload.load_model_data(
            qwen.visual,
            visual_state_dict,
            writable_tensors=False,
            default_dtype=dtype,
        )
    qwen.eval().requires_grad_(False)
    return qwen


class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        text_encoder_filename=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        save_quantized=False,
        **kwargs,
    ):
        dtype = torch.bfloat16
        self.base_model_type = base_model_type or model_type
        self.model_def = model_def
        transformer_filename = (
            model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        )
        transformer_filename = _resolve_wangp_checkpoint(transformer_filename)
        text_encoder_filename = _resolve_wangp_checkpoint(text_encoder_filename)
        legacy_encoder_stack = _uses_legacy_encoder_stack(text_encoder_filename)
        transformer = _load_transformer(
            transformer_filename, _TRANSFORMER_CONFIG_PATH, dtype
        )
        _attach_identity_transformer_methods(transformer)
        if save_quantized:
            raise ValueError(
                "Saving a quantized Identity Edit transformer is not supported; "
                "use WanGP's supplied Krea 2 checkpoint."
            )
        text_encoder_folder = model_def["text_encoder_folder"]
        config_path = fl.locate_file(os.path.join(text_encoder_folder, "config.json"))
        visual_filename = None
        if legacy_encoder_stack:
            visual_filename = fl.locate_file(
                os.path.join("text_encoders", _LEGACY_VISION_FILENAME)
            )
        qwen = _load_grounded_qwen(
            text_encoder_filename,
            config_path,
            dtype,
            visual_filename=visual_filename,
        )
        tokenizer_path = os.path.dirname(
            fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json"))
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            extra_special_tokens={},
        )
        if legacy_encoder_stack:
            image_processor = Qwen2VLImageProcessor(
                patch_size=16,
                temporal_patch_size=2,
                merge_size=2,
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
            processor = Qwen2VLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=BaseVideoProcessor(),
            )
            encoder_stack = "legacy-quanto-language+comfy-fp8-vision"
        else:
            image_processor = Qwen2VLImageProcessorFast.from_pretrained(tokenizer_path)
            processor = Krea2Qwen3VLProcessor(image_processor, tokenizer)
            encoder_stack = "current-unified-bf16"
        print(
            "[Krea2 Identity][Encoder] "
            f"stack={encoder_stack}, language={os.path.basename(text_encoder_filename)}, "
            f"vision={os.path.basename(visual_filename) if visual_filename else 'same-checkpoint'}"
        )
        vae = _load_vae(
            fl.locate_file("qwen_vae.safetensors"),
            fl.locate_file("qwen_vae_config.json"),
            VAE_dtype,
        )
        conditioner = GroundedQwen3VLConditioner(qwen, tokenizer, processor)
        self.pipeline = IdentityKrea2Pipeline(
            transformer, vae, conditioner, dtype=dtype
        )
        self.transformer = transformer
        self.text_encoder = qwen
        self.tokenizer = tokenizer
        self.vae = vae

    def get_loras_transformer(
        self,
        get_model_recursive_prop,
        custom_settings=None,
        video_prompt_type="",
        denoising_strength=1.0,
        **kwargs,
    ):
        settings = expand_advanced_settings(custom_settings)
        identity_profile = settings.get("identity_method")
        identity_method = validate_identity_method(identity_profile)
        _two_phase, _phase2_denoise, _phase2_depth, outpaint_mode = (
            resolve_generation_process(settings)
        )
        if not reid_experiments_enabled() and (
            identity_method == "reid"
            or _two_phase == "depth_then_reid"
            or direct_image_control_selected(video_prompt_type)
        ):
            raise ValueError(REID_EXPERIMENTS_DISABLED_MESSAGE)
        if identity_method == "reid":
            if self.base_model_type != "krea2_identity_turbo":
                raise ValueError("Krea 2 ReID is available only with Krea 2 Identity Turbo")
            if outpaint_mode != "off":
                raise ValueError("Krea 2 ReID cannot be combined with Registered Outpaint")
        if outpaint_mode == "outpaint_only":
            return [outpaint_lora_url()], [1.0]
        depth_strength = validate_depth_control_strength(
            settings.get("depth_control_strength", denoising_strength)
        )
        if identity_method == "depth_prompt":
            if not depth_control_selected(video_prompt_type) or depth_strength <= 0:
                return [], []
            return [depth_control_lora_url()], [depth_strength]
        functional_lora = (
            reid_lora_url()
            if identity_method == "reid"
            else identity_lora_url(
                settings.get("identity_lora_variant", "full_v1.2")
            )
        )
        loras = [functional_lora]
        reid_lora_strength = validate_reid_lora_strength(
            settings.get("reid_lora_strength")
        )
        multipliers = [
            reid_lora_strength if identity_method == "reid" else 1.0
        ]
        if depth_control_selected(video_prompt_type) and depth_strength > 0:
            loras.append(depth_control_lora_url())
            multipliers.append(depth_strength)
        if outpaint_mode == "identity_then_outpaint":
            loras.append(outpaint_lora_url())
            multipliers.append(0.0)
        return loras, multipliers

    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "",
        n_prompt: str | None = None,
        sampling_steps: int = 28,
        width: int = 1024,
        height: int = 1024,
        guide_scale: float = 4.5,
        batch_size: int = 1,
        callback=None,
        VAE_tile_size=None,
        loras_slists=None,
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        input_ref_images=None,
        original_input_ref_images=None,
        custom_settings=None,
        video_prompt_type="",
        denoising_strength=0.25,
        input_frames=None,
        input_masks=None,
        input_custom=None,
        outpainting_dims=None,
        **kwargs,
    ):
        settings = expand_advanced_settings(custom_settings)
        identity_profile = settings.get("identity_method")
        identity_method = validate_identity_method(identity_profile)
        reference_subject_boost, reference_scene_boost = (
            resolve_identity_reference_boosts(identity_profile)
        )
        if identity_method == "depth_prompt":
            original_references = []
            references = []
        else:
            original_references = validate_reference_images(original_input_ref_images)
            references = validate_reference_images(
                input_ref_images
                if input_ref_images is not None and len(input_ref_images) > 0
                else original_references
            )
            if len(references) != len(original_references):
                raise ValueError(
                    "WanGP processed reference count does not match the uploaded references"
                )
        if identity_method == "reid":
            original_references = validate_reid_reference_images(original_references)
            references = validate_reid_reference_images(references)
            if self.base_model_type != "krea2_identity_turbo":
                raise ValueError("Krea 2 ReID is available only with Krea 2 Identity Turbo")
        (
            two_phase_mode,
            phase2_denoising_strength,
            phase2_depth_mode,
            outpaint_mode,
        ) = resolve_generation_process(settings)
        if not reid_experiments_enabled() and (
            identity_method == "reid"
            or two_phase_mode == "depth_then_reid"
            or direct_image_control_selected(video_prompt_type)
        ):
            raise ValueError(REID_EXPERIMENTS_DISABLED_MESSAGE)
        if identity_method == "reid" and outpaint_mode != "off":
            raise ValueError("Krea 2 ReID cannot be combined with Registered Outpaint")
        if identity_method == "depth_prompt" and outpaint_mode != "off":
            raise ValueError("Depth + prompt only cannot use Registered Outpaint")
        seam_px = validate_outpaint_seam_px(settings.get("outpaint_seam_px"))
        grounding_px = validate_grounding_px(settings.get("grounding_px"))
        reid_lora_strength = validate_reid_lora_strength(
            settings.get("reid_lora_strength")
        )
        reid_reference_method = settings.get(
            "reid_reference_method", "isolated_cache"
        )
        secondary_reference_geometry = settings.get(
            "secondary_reference_geometry", "fit"
        )
        output_resolution_limit = settings.get(
            "output_resolution_limit", "safe_2mp"
        )
        if identity_method == "reid":
            reid_lora_url()
        elif identity_method == "identity_edit":
            identity_lora_url(settings.get("identity_lora_variant", "full_v1.2"))
        depth_strength = validate_depth_control_strength(
            settings.get("depth_control_strength", denoising_strength)
        )
        depth_mask_feather_px = validate_depth_mask_feather_px(
            settings.get("depth_mask_feather_px")
        )
        depth_user_lora_timing = validate_depth_user_lora_timing(
            settings.get("depth_user_lora_timing")
        )
        depth_user_lora_ramp = validate_depth_user_lora_ramp(
            settings.get("depth_user_lora_ramp_early"),
            settings.get("depth_user_lora_ramp_middle"),
            settings.get("depth_user_lora_ramp_final"),
        )
        builtin_adapter_timing = validate_builtin_adapter_timing(
            settings.get("builtin_adapter_timing")
        )
        builtin_depth_ramp = validate_builtin_depth_ramp(
            settings.get("builtin_depth_ramp_early"),
            settings.get("builtin_depth_ramp_middle"),
            settings.get("builtin_depth_ramp_final"),
        )
        builtin_identity_ramp = validate_builtin_identity_ramp(
            settings.get("builtin_identity_ramp_early"),
            settings.get("builtin_identity_ramp_middle"),
            settings.get("builtin_identity_ramp_final"),
        )
        subject_attention_timing = validate_subject_attention_timing(
            settings.get("subject_attention_timing")
        )
        subject_attention_ramp = validate_subject_attention_ramp(
            settings.get("subject_attention_ramp_early"),
            settings.get("subject_attention_ramp_middle"),
            settings.get("subject_attention_ramp_final"),
        )
        depth_active = depth_control_selected(video_prompt_type) and depth_strength > 0
        direct_image_active = direct_image_control_selected(video_prompt_type)
        direct_image_denoising = validate_direct_image_denoising_strength(
            denoising_strength
        )
        if identity_method == "depth_prompt" and not depth_active:
            raise ValueError(
                "Depth + prompt only requires Transfer Depth with depth strength greater than 0"
            )
        if two_phase_mode == "depth_then_reid":
            if self.base_model_type != "krea2_identity_turbo":
                raise ValueError("Two-phase Depth then ReID requires Krea 2 Turbo")
            if identity_method != "reid":
                raise ValueError("Two-phase Depth then ReID requires the ReID identity method")
            if outpaint_mode != "off":
                raise ValueError("Two-phase Depth then ReID cannot use Registered Outpaint")
            if not depth_active:
                raise ValueError(
                    "Two-phase Depth then ReID requires Transfer Depth with "
                    "depth strength greater than 0"
                )
            if int(batch_size) != 1:
                raise ValueError("Two-phase Depth then ReID currently requires batch size 1")
        if direct_image_active:
            if identity_method != "reid":
                raise ValueError(
                    "Direct Image → ReID Edit requires the ReID identity method"
                )
            if depth_control_mask_selected(video_prompt_type) or "A" in str(
                video_prompt_type or ""
            ):
                raise ValueError(
                    "Direct Image → ReID Edit currently supports Whole Frame only"
                )
            if input_frames is None:
                raise ValueError(
                    "Direct Image → ReID Edit requires a processed Control Image"
                )
        depth_control = input_frames if depth_active else None
        direct_image_control = input_frames if direct_image_active else None
        depth_mask_active = depth_control_mask_selected(video_prompt_type)
        custom_mask_mode = "Y" in str(video_prompt_type or "")
        painted_mask_mode = "A" in str(video_prompt_type or "")
        custom_mask_info = None
        custom_depth_mask = None
        if depth_active and painted_mask_mode and input_custom is not None:
            raise ValueError(
                "An uploaded Custom Depth Mask must use Custom Mask — White "
                "Area or Custom Mask — Black Area. Painted Mask preprocessing "
                "composites the control before the plugin receives it."
            )
        if depth_active and custom_mask_mode and input_custom is not None:
            custom_depth_mask, custom_mask_info = _load_custom_depth_mask_tensor(
                input_custom,
                depth_control,
                video_prompt_type,
            )
            print(
                "[Krea2 Identity][Depth] custom mask loaded: "
                f"size={custom_mask_info['size'][0]}x{custom_mask_info['size'][1]}, "
                f"channel={custom_mask_info['channel']}, "
                f"area={'outside-white' if custom_mask_info['outside'] else 'inside-white'}, "
                "host_depth=full-frame"
            )
        elif input_custom is not None:
            print(
                "[Krea2 Identity][Depth] custom mask ignored: select Transfer "
                "Depth and Custom Mask — White Area or Custom Mask — Black "
                "Area to use it"
            )
        if depth_active and custom_mask_mode and input_custom is None:
            raise ValueError(
                "The selected Custom Mask depth area requires an uploaded "
                "Custom Depth Mask"
            )
        if (
            depth_active
            and depth_mask_active
            and input_masks is None
            and custom_depth_mask is None
        ):
            raise ValueError(
                "Inside/Outside Mask depth control requires either a painted "
                "Control Image mask or an uploaded Custom Depth Mask"
            )
        depth_mask = None
        mask_source = "none"
        if depth_active and depth_mask_active:
            if custom_depth_mask is not None:
                depth_mask = custom_depth_mask
                mask_source = "custom-upload"
            else:
                depth_mask = input_masks
                mask_source = "painted-editor"
        output_resolution_policy = (
            "selected-resolution"
            if identity_method == "depth_prompt"
            else output_resolution_limit
        )
        print(
            "[Krea2 Identity][Inputs] "
            f"identity_method={identity_method}, "
            f"reid_lora_strength={reid_lora_strength:.2f}, "
            f"reid_reference_method={reid_reference_method}, "
            f"reference_boosts=subject:{reference_subject_boost:.2f}x/"
            f"scene:{reference_scene_boost:.2f}x, "
            f"secondary_geometry={secondary_reference_geometry}, "
            f"output_resolution_limit={output_resolution_policy}, "
            f"mode={video_prompt_type or 'none'}, "
            f"original_refs={_fingerprint_list(original_references)}, "
            f"processed_refs={_fingerprint_list(references)}, "
            f"processed_depth={_content_fingerprint(depth_control)}, "
            f"direct_source={_content_fingerprint(direct_image_control)}, "
            f"mask={_content_fingerprint(depth_mask)}, "
            f"mask_source={mask_source}"
        )
        requested_width, requested_height = width, height
        identity_width, identity_height = width, height
        resolution_kwargs = (
            {"max_pixels": None}
            if output_resolution_limit == "unlimited"
            else {}
        )
        if outpaint_mode == "off":
            if identity_method == "depth_prompt":
                # There are no reference tokens competing for memory in this
                # mode, so preserve the user's selected output dimensions.
                identity_width, identity_height = int(width), int(height)
            else:
                aspect_source = (
                    (width, height)
                    if identity_method == "reid"
                    else original_references[0].size
                )
                identity_width, identity_height = match_reference_dimensions(
                    width,
                    height,
                    aspect_source,
                    **resolution_kwargs,
                )
            width, height = identity_width, identity_height
        elif outpainting_dims is None:
            raise ValueError("Registered Outpaint requires WanGP spatial outpainting margins")
        elif outpaint_mode == "identity_then_outpaint":
            identity_width, identity_height = match_reference_dimensions(
                width,
                height,
                original_references[0].size,
                **resolution_kwargs,
            )
        print(
            "[Krea2 Identity][Resolution] "
            f"requested={requested_width}x{requested_height}, "
            f"identity_pass={identity_width}x{identity_height}, "
            f"final={width}x{height}, "
            f"policy={output_resolution_policy}"
        )
        if (
            identity_method == "identity_edit"
            and len(references) == 2
            and width * height > TWO_REFERENCE_RECOMMENDED_PIXELS
        ):
            warnings.warn(
                "Two-reference Identity Edit is most reliable near 1-1.5 MP; "
                f"the requested output resolves to {width}x{height}.",
                RuntimeWarning,
                stacklevel=2,
            )
        if VAE_tile_size is not None and hasattr(self.vae, "use_tiling"):
            if isinstance(VAE_tile_size, int):
                tiling, tile_size = VAE_tile_size > 0, max(VAE_tile_size, 0)
            else:
                tiling = bool(VAE_tile_size[0])
                tile_size = VAE_tile_size[1] if len(VAE_tile_size) > 1 else 0
            if tiling:
                self.vae.enable_tiling(
                    tile_sample_min_height=tile_size or None,
                    tile_sample_min_width=tile_size or None,
                )
            else:
                self.vae.disable_tiling()
        turbo = self.base_model_type == "krea2_identity_turbo"
        if turbo:
            guide_scale, mu = 0, 1.15
        else:
            mu = None
        if identity_method == "reid":
            if sampling_steps != REID_INFERENCE_STEPS:
                print(
                    "[Krea2 Identity][ReID] overriding sampling steps "
                    f"from {sampling_steps} to {REID_INFERENCE_STEPS}"
                )
            sampling_steps = REID_INFERENCE_STEPS
            guide_scale = 0
        generator_seed = seed if seed is not None and seed >= 0 else torch.seed()
        prompts = [input_prompt] * int(batch_size)
        common = dict(
            negative_prompts=[n_prompt or _DEFAULT_NEGATIVE_PROMPT] * len(prompts),
            width=width,
            height=height,
            steps=sampling_steps,
            guidance=guide_scale,
            seed=generator_seed,
            mu=mu,
            callback=callback,
            NAG_scale=NAG_scale,
            NAG_tau=NAG_tau,
            NAG_alpha=NAG_alpha,
        )
        if outpaint_mode == "outpaint_only":
            if len(original_references) != 1:
                raise ValueError("Outpaint-only requires exactly one uploaded source image")
            images = self.pipeline.generate_registered_outpaint(
                prompts,
                loras_slists=loras_slists,
                source_image=original_references[0],
                outpainting_dims=outpainting_dims,
                seam_px=seam_px,
                **common,
            )
            return None if images is None else images.transpose(0, 1)

        identity_pass_references = references
        if (
            outpaint_mode == "identity_then_outpaint"
            and identity_method == "identity_edit"
        ):
            # WanGP prepares reference inputs against the selected final canvas.
            # That is useful for a normal single pass, but it must not feed the
            # portrait-shaped identity pass: a portrait upload may already have
            # been stretched to the landscape outpaint canvas. Phase one should
            # encode the untouched upload at its own aspect ratio; phase two
            # expands the generated result onto the requested canvas.
            identity_pass_references = original_references
            print(
                "[Krea2 Identity][Outpaint] identity phase reference source="
                f"original-upload, refs={_fingerprint_list(identity_pass_references)}"
            )

        identity_slists = loras_slists
        if outpaint_mode == "identity_then_outpaint":
            identity_slists = _with_plugin_lora_weights(loras_slists, [1.0, 0.0])
        if (
            two_phase_mode == "off"
            and depth_active
            and depth_user_lora_timing == "depth_first"
        ):
            plugin_head_count = 1 if identity_method == "depth_prompt" else 2
            identity_slists, delayed_count = delay_user_loras_for_depth(
                identity_slists,
                plugin_head_count,
                sampling_steps,
                phase_scales=depth_user_lora_ramp,
            )
            if delayed_count:
                early, middle, final = depth_user_lora_ramp
                print(
                    "[Krea2 Identity][Depth] depth-first scheduling active for "
                    f"{delayed_count} user LoRA(s): early={early:.2f}, "
                    f"middle={middle:.2f}, final={final:.2f}"
                )
        if (
            two_phase_mode == "off"
            and identity_method == "identity_edit"
            and depth_active
            and builtin_adapter_timing == "depth_then_identity"
        ):
            identity_slists = schedule_builtin_identity_depth_adapters(
                identity_slists,
                sampling_steps,
                identity_ramp=builtin_identity_ramp,
                depth_ramp=builtin_depth_ramp,
            )
            depth_early, depth_middle, depth_final = builtin_depth_ramp
            identity_early, identity_middle, identity_final = builtin_identity_ramp
            print(
                "[Krea2 Identity][Depth] layout-to-identity adapter timing active: "
                f"depth={depth_early:.2f}/{depth_middle:.2f}/{depth_final:.2f}, "
                f"identity={identity_early:.2f}/{identity_middle:.2f}/{identity_final:.2f}"
            )
        identity_common = dict(common)
        identity_common.update(width=identity_width, height=identity_height)
        if two_phase_mode == "depth_then_reid":
            phase1_slists = _with_phase_lora_weights(
                loras_slists,
                [0.0, depth_strength],
                user_scale=0.0,
            )
            print(
                "[Krea2 Identity][Two Phase] phase 1/2: Depth + Prompt, "
                "identity=0.00, additional_loras=0.00"
            )
            phase1_images = self.pipeline.generate_depth_prompt(
                prompts,
                loras_slists=phase1_slists,
                depth_control=depth_control,
                depth_mask=depth_mask,
                depth_mask_feather_px=depth_mask_feather_px,
                **identity_common,
            )
            if phase1_images is None or self._interrupt:
                return None
            phase1_source = _decoded_image_to_pil(phase1_images)
            full_frame_mask = Image.new("L", phase1_source.size, 255)
            keep_phase2_depth = phase2_depth_mode == "keep"
            phase2_slists = _with_phase_lora_weights(
                loras_slists,
                [
                    reid_lora_strength,
                    depth_strength if keep_phase2_depth else 0.0,
                ],
                user_scale=1.0,
            )
            first_phase2_step = int(
                sampling_steps * (1.0 - phase2_denoising_strength)
            )
            effective_phase2_steps = sampling_steps - first_phase2_step
            print(
                "[Krea2 Identity][Two Phase] phase 2/2: ReID refinement, "
                f"denoising={phase2_denoising_strength:.2f}, "
                f"effective_steps={effective_phase2_steps}/{sampling_steps}, "
                f"depth={'on' if keep_phase2_depth else 'off'}, "
                f"reid_lora_strength={reid_lora_strength:.2f}, "
                "additional_loras=selected strengths"
            )
            images = self.pipeline.generate_reid(
                prompts,
                loras_slists=phase2_slists,
                reference_images=references,
                reid_reference_method=reid_reference_method,
                depth_control=depth_control if keep_phase2_depth else None,
                depth_mask=depth_mask if keep_phase2_depth else None,
                depth_mask_feather_px=depth_mask_feather_px,
                source_image=phase1_source,
                image_mask=full_frame_mask,
                denoising_strength=phase2_denoising_strength,
                masking_strength=1.0,
                model_mode=0,
                **identity_common,
            )
        elif identity_method == "depth_prompt":
            images = self.pipeline.generate_depth_prompt(
                prompts,
                loras_slists=identity_slists,
                depth_control=depth_control,
                depth_mask=depth_mask,
                depth_mask_feather_px=depth_mask_feather_px,
                **identity_common,
            )
        elif identity_method == "reid" and direct_image_active:
            direct_source = _wangp_control_to_pil(
                direct_image_control, "Direct Image Control"
            )
            full_frame_mask = Image.new("L", direct_source.size, 255)
            effective_steps = sampling_steps - int(
                sampling_steps * (1.0 - direct_image_denoising)
            )
            print(
                "[Krea2 Identity][Direct Image] ReID refinement: "
                f"source={direct_source.width}x{direct_source.height}, "
                f"denoising={direct_image_denoising:.2f}, "
                f"effective_steps={effective_steps}/{sampling_steps}"
            )
            images = self.pipeline.generate_reid(
                prompts,
                loras_slists=identity_slists,
                reference_images=references,
                reid_reference_method=reid_reference_method,
                source_image=direct_source,
                image_mask=full_frame_mask,
                denoising_strength=direct_image_denoising,
                masking_strength=1.0,
                model_mode=0,
                **identity_common,
            )
        elif identity_method == "reid":
            images = self.pipeline.generate_reid(
                prompts,
                loras_slists=identity_slists,
                reference_images=references,
                reid_reference_method=reid_reference_method,
                depth_control=depth_control,
                depth_mask=depth_mask,
                depth_mask_feather_px=depth_mask_feather_px,
                **identity_common,
            )
        else:
            images = self.pipeline.generate_identity(
                prompts,
                loras_slists=identity_slists,
                reference_images=identity_pass_references,
                grounding_px=grounding_px,
                reference_subject_boost=reference_subject_boost,
                reference_scene_boost=reference_scene_boost,
                subject_attention_timing=subject_attention_timing,
                subject_attention_ramp=subject_attention_ramp,
                builtin_adapter_timing=builtin_adapter_timing,
                builtin_depth_ramp=builtin_depth_ramp,
                fit_all_references=False,
                fit_secondary_reference=secondary_reference_geometry == "fit",
                depth_control=depth_control,
                depth_mask=depth_mask,
                depth_mask_feather_px=depth_mask_feather_px,
                depth_control_strength=depth_strength,
                **identity_common,
            )
        if images is not None and outpaint_mode == "identity_then_outpaint":
            # Identity output is BxCxHxW uint8. Convert the first result of each
            # batch independently; Registered Outpaint currently follows the
            # upstream one-source-per-call contract.
            if len(images) != 1:
                raise ValueError("Identity Edit + Outpaint currently requires batch size 1")
            source_tensor = images[0].detach().cpu()
            if source_tensor.shape[0] in (3, 4):
                source_tensor = source_tensor[:3].permute(1, 2, 0)
            source = Image.fromarray(source_tensor.to(torch.uint8).numpy())
            outpaint_slists = _with_plugin_lora_weights(loras_slists, [0.0, 1.0])
            images = self.pipeline.generate_registered_outpaint(
                prompts,
                loras_slists=outpaint_slists,
                source_image=source,
                outpainting_dims=outpainting_dims,
                seam_px=seam_px,
                **common,
            )
        return None if images is None else images.transpose(0, 1)

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = value
            self.pipeline.encoder._interrupt = value
            self.pipeline.encoder.qwen.language_model._interrupt = value
            self.pipeline.encoder.qwen.visual._interrupt = value
        if hasattr(self, "transformer"):
            self.transformer._interrupt = value
            self.transformer.txtfusion._interrupt = value
        if hasattr(self, "text_encoder"):
            self.text_encoder.language_model._interrupt = value
            self.text_encoder.visual._interrupt = value
