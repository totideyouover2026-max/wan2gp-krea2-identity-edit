"""WanGP runtime for dual-conditioned Krea 2 Identity Edit.

The implementation deliberately reuses WanGP's Krea 2 transformer, scheduler,
VAE, callback and MMGP behavior. Identity-specific code is limited to the full
Qwen3-VL vision path and the clean reference-token stream.
"""

from __future__ import annotations

import os
import types
import warnings

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2VLProcessor
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
    Krea2Pipeline,
    _TEXT_ENCODER_SELECT_LAYERS,
    _TRANSFORMER_CONFIG_PATH,
    _load_transformer,
    _load_vae,
    _pack_image_latents,
)
from models.krea2.krea2_mmdit import key_padding_mask

from .krea2_identity_utils import (
    DEFAULT_GROUNDING_PX,
    TWO_REFERENCE_RECOMMENDED_PIXELS,
    identity_lora_url,
    match_reference_dimensions,
    preprocess_identity_lora_state_dict,
    resolve_wangp_checkpoint,
    validate_grounding_px,
    validate_reference_images,
)


VISION_FILENAME = "qwen3vl_4b_fp8_scaled.safetensors"
_DEFAULT_NEGATIVE_PROMPT = ""


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
        self.grounding_px = DEFAULT_GROUNDING_PX
        self._interrupt = False

    def set_references(self, images, grounding_px):
        self.grounding_px = validate_grounding_px(grounding_px)
        self.reference_images = []
        for image in images:
            image = image.convert("RGB")
            width, height = image.size
            if max(width, height) > self.grounding_px:
                scale = self.grounding_px / max(width, height)
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    resample=Image.Resampling.LANCZOS,
                )
            self.reference_images.append(image)

    def clear_references(self):
        self.reference_images = []

    def _bounded_prompt(self, prompt):
        prompt = str(prompt)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_prompt_tokens:
            return prompt
        return self.tokenizer.decode(
            ids[: self.max_prompt_tokens], skip_special_tokens=False
        )

    def _template(self, prompt):
        vision = "".join(
            "<|vision_start|><|image_pad|><|vision_end|>"
            for _ in self.reference_images
        )
        return self.template_prefix + vision + self._bounded_prompt(prompt) + self.template_suffix

    def _encode_one(self, prompt, device):
        if self._interrupt:
            return None, None
        encoded = self.processor(
            text=[self._template(prompt)],
            images=self.reference_images,
            padding=True,
            return_tensors="pt",
        ).to(device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        pixel_values = encoded["pixel_values"]
        image_grid_thw = encoded["image_grid_thw"]

        self.qwen.language_model._interrupt = self._interrupt
        self.qwen.visual._interrupt = self._interrupt
        inputs_embeds = self.qwen.get_input_embeddings()(input_ids)
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
        if not self.reference_images:
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
    source_positions = []
    source_masks = []
    for frame, source in enumerate(source_imgs, start=1):
        source_pos = target_pos.clone()
        source_pos[..., 0] = frame
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


def _identity_forward(
    self, img, context, t, tvec, pos, mask=None,
    NAG=None, neg_context=None, neg_mask=None,
):
    sources = getattr(self, "_identity_source_tokens", None)
    if not sources:
        raise RuntimeError("Identity Edit source tokens were not prepared")
    target = self.first(img)
    source_imgs = [self.first(source) for source in sources]
    combined, txtlen, srclen, target_len, freqs, stream_mask = _identity_build_stream(
        self, target, context, pos, mask, source_imgs
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
):
    sources = getattr(self, "_identity_source_tokens", None)
    if not sources:
        raise RuntimeError("Identity Edit source tokens were not prepared")
    target = self.first(img)
    source_imgs = [self.first(source) for source in sources]
    cond = _identity_build_stream(self, target, context, pos, mask, source_imgs)
    uncond = _identity_build_stream(
        self, target, uncond_context, uncond_pos, uncond_mask, source_imgs,
        freqs=cond[4] if pos.shape == uncond_pos.shape else None,
    )
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


def _attach_identity_transformer_methods(transformer):
    transformer.forward = types.MethodType(_identity_forward, transformer)
    transformer.forward_cfg = types.MethodType(_identity_forward_cfg, transformer)
    transformer.preprocess_loras = (
        lambda _model_type, state_dict: preprocess_identity_lora_state_dict(state_dict)
    )


class IdentityKrea2Pipeline(Krea2Pipeline):
    def _encode_prompts(self, prompts, device, dtype):
        self.encoder._interrupt = self._interrupt
        hidden, masks = self.encoder(prompts, device=device)
        if hidden is None:
            return None, None
        return hidden.to(device=device, dtype=dtype), masks.to(device=device)

    @torch.inference_mode()
    def generate_identity(self, *args, reference_images, grounding_px, **kwargs):
        references = validate_reference_images(reference_images)
        grounding_px = validate_grounding_px(grounding_px)
        width = int(kwargs.get("width", 1024))
        height = int(kwargs.get("height", 1024))
        device, dtype = self.runtime_device, self.dtype
        self.encoder.set_references(references, grounding_px)
        try:
            source_tokens = []
            for image in references:
                latent = self._encode_image_to_latents(image, width, height, device, dtype)
                source_tokens.append(_pack_image_latents(latent, self.transformer.config.patch))
            self.transformer._identity_source_tokens = source_tokens
            return super().__call__(*args, **kwargs)
        finally:
            self.transformer._identity_source_tokens = None
            self.encoder.clear_references()


def _load_grounded_qwen(text_encoder_filename, visual_filename, config_path, dtype):
    register_qwen3_vl_config()
    config = Qwen3VLConfig.from_json_file(config_path)
    with init_empty_weights(include_buffers=True):
        qwen = IdentityQwen3VLModel(config)
    # These buffers are non-persistent and therefore absent from both
    # checkpoints. Materialize them before MMGP validates the meta-initialized
    # modules, matching WanGP's own Krea 2 language-model loader.
    qwen.language_model.rotary_emb.reset_inv_freq()
    qwen.visual.rotary_pos_emb.reset_inv_freq()
    offload.load_model_data(
        qwen.language_model, text_encoder_filename,
        modelPrefix="language_model", writable_tensors=False,
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
            f"The Qwen3-VL checkpoint has no {visual_prefix} weights: {visual_filename}"
        )
    offload.load_model_data(
        qwen.visual, visual_state_dict,
        writable_tensors=False, default_dtype=dtype,
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
        visual_filename = fl.locate_file(os.path.join("text_encoders", VISION_FILENAME))
        qwen = _load_grounded_qwen(
            text_encoder_filename, visual_filename, config_path, dtype
        )
        tokenizer_path = os.path.dirname(
            fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json"))
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, max_length=512, trust_remote_code=True,
            extra_special_tokens={},
        )
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

    def get_loras_transformer(self, get_model_recursive_prop, custom_settings=None, **kwargs):
        settings = custom_settings if isinstance(custom_settings, dict) else {}
        return [identity_lora_url(settings.get("identity_lora_variant", "r64"))], [1.0]

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
        original_input_ref_images=None,
        custom_settings=None,
        **kwargs,
    ):
        references = validate_reference_images(original_input_ref_images)
        settings = custom_settings if isinstance(custom_settings, dict) else {}
        grounding_px = validate_grounding_px(settings.get("grounding_px"))
        identity_lora_url(settings.get("identity_lora_variant", "r64"))
        width, height = match_reference_dimensions(width, height, references[0].size)
        if len(references) == 2 and width * height > TWO_REFERENCE_RECOMMENDED_PIXELS:
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
        generator_seed = seed if seed is not None and seed >= 0 else torch.seed()
        prompts = [input_prompt] * int(batch_size)
        images = self.pipeline.generate_identity(
            prompts,
            negative_prompts=[n_prompt or _DEFAULT_NEGATIVE_PROMPT] * len(prompts),
            width=width,
            height=height,
            steps=sampling_steps,
            guidance=guide_scale,
            seed=generator_seed,
            mu=mu,
            callback=callback,
            loras_slists=loras_slists,
            NAG_scale=NAG_scale,
            NAG_tau=NAG_tau,
            NAG_alpha=NAG_alpha,
            reference_images=references,
            grounding_px=grounding_px,
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
