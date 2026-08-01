"""Pure validation and sizing helpers for Krea 2 Identity Edit."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence


MIN_GROUNDING_PX = 384
MAX_GROUNDING_PX = 1536
DEFAULT_GROUNDING_PX = 768
MAX_OUTPUT_PIXELS = 2_000_000
TWO_REFERENCE_RECOMMENDED_PIXELS = 1_500_000
MIN_DEPTH_CONTROL_STRENGTH = 0.0
MAX_DEPTH_CONTROL_STRENGTH = 2.0
DEFAULT_DEPTH_CONTROL_STRENGTH = 1.0
MIN_DIRECT_IMAGE_DENOISING_STRENGTH = 0.0
MAX_DIRECT_IMAGE_DENOISING_STRENGTH = 1.0
DEFAULT_DIRECT_IMAGE_DENOISING_STRENGTH = 0.25
MIN_DEPTH_MASK_FEATHER_PX = 0
MAX_DEPTH_MASK_FEATHER_PX = 64
DEFAULT_DEPTH_MASK_FEATHER_PX = 16
DEPTH_CONTROL_LORA_FILENAME = "depth-control-lora.safetensors"
OUTPAINT_LORA_FILENAME = "krea2_outpaint_rank32.safetensors"
OUTPAINT_MODES = {"off", "outpaint_only", "identity_then_outpaint"}
DEPTH_USER_LORA_TIMINGS = {"depth_first", "all_steps"}
DEFAULT_DEPTH_USER_LORA_RAMP = (0.0, 0.25, 1.0)
BUILTIN_ADAPTER_TIMINGS = {"simultaneous", "depth_then_identity"}
DEFAULT_BUILTIN_DEPTH_RAMP = (1.0, 0.5, 0.0)
DEFAULT_BUILTIN_IDENTITY_RAMP = (0.25, 0.75, 1.0)
SUBJECT_ATTENTION_TIMINGS = {"constant", "ramp"}
DEFAULT_SUBJECT_ATTENTION_RAMP = (1.0, 2.0, 8.0)
MIN_SUBJECT_ATTENTION_BOOST = 1.0
MAX_SUBJECT_ATTENTION_BOOST = 8.0
IDENTITY_METHOD_PROFILES = {
    # value: (conditioning implementation, subject/last-ref boost, scene/early-ref boost)
    "identity_edit": ("identity_edit", 1.0, 1.0),
    "identity_edit_ref2": ("identity_edit", 2.0, 1.0),
    "identity_edit_ref4": ("identity_edit", 4.0, 1.0),
    "identity_edit_ref8": ("identity_edit", 8.0, 1.0),
    "identity_edit_ref4_scene2": ("identity_edit", 4.0, 2.0),
    "identity_edit_ref8_scene2": ("identity_edit", 8.0, 2.0),
    "depth_prompt": ("depth_prompt", 1.0, 1.0),
}
IDENTITY_METHODS = set(IDENTITY_METHOD_PROFILES)
DEFAULT_PHASE2_DENOISING_STRENGTH = 0.25
GENERATION_PROCESS_PROFILES = {
    "standard": ("off", DEFAULT_PHASE2_DENOISING_STRENGTH, "keep", "off"),
    "outpaint_only": ("off", DEFAULT_PHASE2_DENOISING_STRENGTH, "keep", "outpaint_only"),
    "identity_then_outpaint": (
        "off",
        DEFAULT_PHASE2_DENOISING_STRENGTH,
        "keep",
        "identity_then_outpaint",
    ),
}
def resolve_wangp_checkpoint(path, locate_file) -> str:
    """Resolve absolute, checkpoint-relative, and ckpts-prefixed host paths."""
    if path is None:
        raise FileNotFoundError("WanGP did not provide a checkpoint path")

    path = os.fspath(path)

    if os.path.isabs(path) and os.path.isfile(path):
        return path

    candidates = [path]
    normalized = path.replace("\\", "/").lstrip("./")
    parts = normalized.split("/")

    if parts and parts[0].lower() in {"ckpts", "checkpoints"} and len(parts) > 1:
        candidates.append(os.path.join(*parts[1:]))

    candidates.append(os.path.basename(path))

    for candidate in dict.fromkeys(candidates):
        resolved = locate_file(candidate)

        if resolved is not None:
            return os.path.abspath(resolved)

        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Unable to locate WanGP checkpoint {path!r}; tried {candidates}"
    )


def validate_grounding_px(value) -> int:
    """Return a bounded integer grounding resolution."""
    if value is None:
        return DEFAULT_GROUNDING_PX

    if isinstance(value, bool):
        raise ValueError("grounding_px must be an integer")

    if isinstance(value, float) and not value.is_integer():
        raise ValueError("grounding_px must be an integer")

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("grounding_px must be an integer") from exc

    if parsed < MIN_GROUNDING_PX or parsed > MAX_GROUNDING_PX:
        raise ValueError(
            f"grounding_px must be between {MIN_GROUNDING_PX} "
            f"and {MAX_GROUNDING_PX}"
        )

    return parsed


def validate_reference_images(images) -> list:
    """Validate the public one/two-reference contract without reordering inputs."""
    if images is None:
        images = []

    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        images = [images]

    result = [image for image in images if image is not None]

    if not 1 <= len(result) <= 2:
        raise ValueError(
            "Krea 2 Identity Edit requires one or two reference images"
        )

    for image in result:
        if not hasattr(image, "convert") or not hasattr(image, "size"):
            raise TypeError(
                "Each Krea 2 Identity Edit reference must be a PIL image"
            )

    return result


def match_reference_dimensions(
    requested_width: int,
    requested_height: int,
    reference_size: tuple[int, int],
    *,
    align: int = 16,
    max_pixels: int | None = MAX_OUTPUT_PIXELS,
) -> tuple[int, int]:
    """Match the first reference aspect ratio while respecting area and alignment.

    ``None`` disables the plugin safety cap while retaining the selected
    resolution's pixel budget.
    """
    ref_width, ref_height = map(int, reference_size)

    if ref_width <= 0 or ref_height <= 0:
        raise ValueError("The primary reference image has invalid dimensions")

    if max(ref_width, ref_height) / min(ref_width, ref_height) > 200:
        raise ValueError(
            "The primary reference aspect ratio must not exceed 200:1"
        )

    requested_area = max(
        align * align,
        int(requested_width) * int(requested_height),
    )
    pixel_limit = requested_area if max_pixels is None else int(max_pixels)
    if pixel_limit < align * align:
        raise ValueError("max_pixels is too small for the requested alignment")
    target_area = min(requested_area, pixel_limit)
    aspect = ref_width / ref_height

    width = math.sqrt(target_area * aspect)
    height = width / aspect

    width = max(align, int(round(width / align)) * align)
    height = max(align, int(round(height / align)) * align)

    while width * height > pixel_limit:
        if width >= height and width > align:
            width -= align
        elif height > align:
            height -= align
        else:
            break

    return width, height


def identity_lora_url(variant: str) -> str:
    """Return the download URL for a supported Krea 2 Identity Edit LoRA."""
    variant = validate_identity_lora_variant(variant)
    variants = {
        "full_v1.2": "krea2_identity_edit_v1_2.safetensors",
        "r128": "krea2_identity_edit_v1_2_r128.safetensors",
        "r64": "krea2_identity_edit_v1_2_r64.safetensors",
    }

    try:
        filename = variants[str(variant)]
    except KeyError as exc:
        raise ValueError(
            "identity_lora_variant must be one of: "
            "full_v1.2, r128, r64"
        ) from exc

    return (
        "https://huggingface.co/conradlocke/krea2-identity-edit/"
        f"resolve/main/{filename}"
    )


def validate_identity_lora_variant(value) -> str:
    """Validate and normalize the Identity Edit adapter selector."""
    variant = str(value or "full_v1.2")
    # Migrate old queues without retaining v1.1 as a selectable/downloadable
    # variant. Existing rank aliases now resolve to their v1.2 releases.
    if variant in {"full", "full_v1.1"}:
        variant = "full_v1.2"
    if variant not in {"full_v1.2", "r128", "r64"}:
        raise ValueError(
            "identity_lora_variant must be one of: "
            "full_v1.2, r128, r64"
        )
    return variant


def validate_identity_method(value) -> str:
    """Select one mutually exclusive Krea 2 conditioning implementation."""
    method = str(value or "identity_edit")
    try:
        implementation, _subject_boost, _scene_boost = IDENTITY_METHOD_PROFILES[
            method
        ]
    except KeyError as exc:
        raise ValueError(
            "identity_method must be an Identity Edit fidelity profile "
            "or depth_prompt"
        ) from exc
    return implementation


def resolve_identity_reference_boosts(value) -> tuple[float, float]:
    """Return subject/last-ref and scene/earlier-ref attention multipliers."""
    method = str(value or "identity_edit")
    try:
        _implementation, subject_boost, scene_boost = IDENTITY_METHOD_PROFILES[
            method
        ]
    except KeyError as exc:
        raise ValueError(
            "identity_method must be an Identity Edit fidelity profile "
            "or depth_prompt"
        ) from exc
    return subject_boost, scene_boost


def validate_generation_process(value) -> str:
    """Validate the single visible selector used for mutually exclusive tasks."""
    process = str(value or "standard")
    if process not in GENERATION_PROCESS_PROFILES:
        raise ValueError(
            "generation_process must be standard or a registered outpaint profile"
        )
    return process


def resolve_generation_process(settings):
    """Resolve the visible profile."""
    values = settings if isinstance(settings, dict) else {}
    process = migrate_generation_process(values)
    return GENERATION_PROCESS_PROFILES[process]


def migrate_generation_process(settings) -> str:
    """Map current and older settings to a supported visible profile."""
    values = settings if isinstance(settings, dict) else {}
    if "generation_process" in values:
        process = str(values.get("generation_process") or "standard")
        return process if process in GENERATION_PROCESS_PROFILES else "standard"
    outpaint = validate_outpaint_mode(values.get("outpaint_mode"))
    if outpaint != "off":
        return outpaint
    return "standard"


def fit_identity_reference_geometry(
    reference_size: tuple[int, int],
    target_size: tuple[int, int],
    *,
    align: int = 16,
    cover_threshold: float = 0.92,
) -> tuple[tuple[int, int], tuple[int, int, int, int] | None]:
    """Plan an aspect-preserving secondary-reference fit for Identity Edit.

    References whose contained image already covers at least 92% of both target
    axes are centre-cropped to the target aspect. More divergent references are
    contained within the canvas at an aligned size and positioned separately by
    the transformer stream builder.
    """
    reference_width, reference_height = map(int, reference_size)
    target_width, target_height = map(int, target_size)
    if min(reference_width, reference_height, target_width, target_height) <= 0:
        raise ValueError("Identity Edit reference and target dimensions must be positive")
    if align <= 0:
        raise ValueError("Identity Edit reference alignment must be positive")
    if not 0 < float(cover_threshold) <= 1:
        raise ValueError("Identity Edit cover threshold must be in (0, 1]")

    contain_scale = min(
        target_height / reference_height,
        target_width / reference_width,
    )
    contained_height = reference_height * contain_scale
    contained_width = reference_width * contain_scale
    if (
        contained_height >= target_height * cover_threshold
        and contained_width >= target_width * cover_threshold
    ):
        cover_scale = max(
            target_height / reference_height,
            target_width / reference_width,
        )
        crop_height = min(reference_height, round(target_height / cover_scale))
        crop_width = min(reference_width, round(target_width / cover_scale))
        top = (reference_height - crop_height) // 2
        left = (reference_width - crop_width) // 2
        return (
            (target_width, target_height),
            (left, top, left + crop_width, top + crop_height),
        )

    fit_height = min(
        max(align, int(contained_height) // align * align),
        target_height,
    )
    fit_width = min(
        max(align, int(contained_width) // align * align),
        target_width,
    )
    return (fit_width, fit_height), None


def depth_control_lora_url() -> str:
    """Return the authoritative Krea 2 depth ControlNet-LoRA URL."""
    return (
        "https://huggingface.co/Patil/Krea-2-depth-controlnet/resolve/main/"
        f"{DEPTH_CONTROL_LORA_FILENAME}"
    )


def outpaint_lora_url() -> str:
    return (
        "https://huggingface.co/yijunwang2/krea2-outpaint/resolve/main/"
        f"{OUTPAINT_LORA_FILENAME}"
    )


def validate_outpaint_mode(value) -> str:
    mode = str(value or "off")
    if mode not in OUTPAINT_MODES:
        raise ValueError("outpaint_mode must be off, outpaint_only, or identity_then_outpaint")
    return mode


def validate_outpaint_seam_px(value) -> int:
    if value is None:
        return 32
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError("outpaint_seam_px must be an integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("outpaint_seam_px must be an integer") from exc
    if not 0 <= value <= 128:
        raise ValueError("outpaint_seam_px must be between 0 and 128")
    return value


def depth_control_selected(video_prompt_type) -> bool:
    """Return whether WanGP's dedicated control-image dropdown selected depth."""
    prompt_type = str(video_prompt_type or "")
    return "V" in prompt_type and "D" in prompt_type


def direct_image_control_selected(video_prompt_type) -> bool:
    """Return whether WanGP selected an unchanged source image plus denoising."""
    prompt_type = str(video_prompt_type or "")
    return "V" in prompt_type and "G" in prompt_type and "D" not in prompt_type


def depth_control_mask_selected(video_prompt_type) -> bool:
    """Return whether depth should be limited to a painted or uploaded mask."""
    prompt_type = str(video_prompt_type or "")
    return depth_control_selected(prompt_type) and any(
        marker in prompt_type for marker in ("A", "Y")
    )


def validate_direct_image_denoising_strength(value) -> float:
    """Validate WanGP's native source-image denoising value."""
    if value is None:
        return DEFAULT_DIRECT_IMAGE_DENOISING_STRENGTH
    if isinstance(value, bool):
        raise ValueError("Direct Image denoising strength must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Direct Image denoising strength must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError("Direct Image denoising strength must be finite")
    if not (
        MIN_DIRECT_IMAGE_DENOISING_STRENGTH
        <= parsed
        <= MAX_DIRECT_IMAGE_DENOISING_STRENGTH
    ):
        raise ValueError(
            "Direct Image denoising strength must be between "
            f"{MIN_DIRECT_IMAGE_DENOISING_STRENGTH} and "
            f"{MAX_DIRECT_IMAGE_DENOISING_STRENGTH}"
        )
    return parsed


def validate_depth_control_strength(value) -> float:
    """Return a bounded multiplier for the depth block LoRA."""
    if value is None:
        return DEFAULT_DEPTH_CONTROL_STRENGTH
    if isinstance(value, bool):
        raise ValueError("depth_control_strength must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth_control_strength must be a number") from exc
    if not math.isfinite(parsed):
        raise ValueError("depth_control_strength must be finite")
    if parsed < MIN_DEPTH_CONTROL_STRENGTH or parsed > MAX_DEPTH_CONTROL_STRENGTH:
        raise ValueError(
            "depth_control_strength must be between "
            f"{MIN_DEPTH_CONTROL_STRENGTH} and {MAX_DEPTH_CONTROL_STRENGTH}"
        )
    return parsed


def validate_depth_user_lora_timing(value) -> str:
    """Choose whether user-added LoRAs may influence early depth geometry."""
    timing = str(value or "depth_first")
    if timing not in DEPTH_USER_LORA_TIMINGS:
        raise ValueError(
            "depth_user_lora_timing must be depth_first or all_steps"
        )
    return timing


def validate_depth_user_lora_ramp(early=None, middle=None, final=None) -> tuple[float, float, float]:
    """Return a bounded, non-decreasing three-phase user-LoRA ramp."""
    supplied = (early, middle, final)
    values = []
    labels = ("early", "middle", "final")
    for label, value, default in zip(labels, supplied, DEFAULT_DEPTH_USER_LORA_RAMP):
        if value is None:
            value = default
        if isinstance(value, bool):
            raise ValueError(f"depth_user_lora_ramp_{label} must be a number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"depth_user_lora_ramp_{label} must be a number"
            ) from exc
        if not math.isfinite(parsed):
            raise ValueError(f"depth_user_lora_ramp_{label} must be finite")
        if parsed < 0.0 or parsed > 1.0:
            raise ValueError(
                f"depth_user_lora_ramp_{label} must be between 0 and 1"
            )
        values.append(parsed)
    if values != sorted(values):
        raise ValueError(
            "depth-first LoRA ramp must be non-decreasing: early <= middle <= final"
        )
    return tuple(values)


def validate_builtin_adapter_timing(value) -> str:
    """Choose the legacy simultaneous path or staged depth-to-identity guidance."""
    timing = str(value or "simultaneous")
    if timing not in BUILTIN_ADAPTER_TIMINGS:
        raise ValueError(
            "builtin_adapter_timing must be simultaneous or depth_then_identity"
        )
    return timing


def _validate_unit_ramp(
    name,
    defaults,
    early=None,
    middle=None,
    final=None,
    *,
    direction,
) -> tuple[float, float, float]:
    values = []
    for label, value, default in zip(
        ("early", "middle", "final"),
        (early, middle, final),
        defaults,
    ):
        if value is None:
            value = default
        if isinstance(value, bool):
            raise ValueError(f"{name}_{label} must be a number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}_{label} must be a number") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{name}_{label} must be finite")
        if parsed < 0.0 or parsed > 1.0:
            raise ValueError(f"{name}_{label} must be between 0 and 1")
        values.append(parsed)
    ordered = sorted(values, reverse=direction == "down")
    if values != ordered:
        relation = "early >= middle >= final" if direction == "down" else (
            "early <= middle <= final"
        )
        raise ValueError(f"{name} must satisfy {relation}")
    return tuple(values)


def validate_builtin_depth_ramp(
    early=None, middle=None, final=None
) -> tuple[float, float, float]:
    """Return a bounded, non-increasing built-in Depth LoRA ramp."""
    return _validate_unit_ramp(
        "builtin_depth_ramp",
        DEFAULT_BUILTIN_DEPTH_RAMP,
        early,
        middle,
        final,
        direction="down",
    )


def validate_builtin_identity_ramp(
    early=None, middle=None, final=None
) -> tuple[float, float, float]:
    """Return a bounded, non-decreasing built-in Identity Edit LoRA ramp."""
    return _validate_unit_ramp(
        "builtin_identity_ramp",
        DEFAULT_BUILTIN_IDENTITY_RAMP,
        early,
        middle,
        final,
        direction="up",
    )


def three_phase_value_for_step(phase_values, step_index, num_steps) -> tuple[float, int]:
    """Resolve one of three phase values for a denoising step."""
    if len(phase_values) != 3:
        raise ValueError("phase_values must contain three values")
    if isinstance(num_steps, bool) or int(num_steps) < 1:
        raise ValueError("num_steps must be a positive integer")
    if isinstance(step_index, bool):
        raise ValueError("step_index must be an integer")
    step_index = max(0, min(int(step_index), int(num_steps) - 1))
    first_cut = math.ceil(int(num_steps) / 3)
    second_cut = math.ceil(2 * int(num_steps) / 3)
    phase = 0 if step_index < first_cut else 1 if step_index < second_cut else 2
    return float(phase_values[phase]), phase


def validate_subject_attention_timing(value) -> str:
    """Choose constant subject attention or an experimental three-stage ramp."""
    timing = str(value or "constant")
    if timing not in SUBJECT_ATTENTION_TIMINGS:
        raise ValueError("subject_attention_timing must be constant or ramp")
    return timing


def validate_subject_attention_ramp(
    early=None, middle=None, final=None
) -> tuple[float, float, float]:
    """Return bounded, non-decreasing absolute subject-attention boosts."""
    supplied = (early, middle, final)
    values = []
    labels = ("early", "middle", "final")
    for label, value, default in zip(
        labels, supplied, DEFAULT_SUBJECT_ATTENTION_RAMP
    ):
        if value is None:
            value = default
        if isinstance(value, bool):
            raise ValueError(f"subject_attention_ramp_{label} must be a number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"subject_attention_ramp_{label} must be a number"
            ) from exc
        if not math.isfinite(parsed):
            raise ValueError(f"subject_attention_ramp_{label} must be finite")
        if not MIN_SUBJECT_ATTENTION_BOOST <= parsed <= MAX_SUBJECT_ATTENTION_BOOST:
            raise ValueError(
                f"subject_attention_ramp_{label} must be between "
                f"{MIN_SUBJECT_ATTENTION_BOOST:g} and "
                f"{MAX_SUBJECT_ATTENTION_BOOST:g}"
            )
        values.append(parsed)
    if values != sorted(values):
        raise ValueError(
            "subject attention ramp must be non-decreasing: "
            "early <= middle <= final"
        )
    return tuple(values)


def subject_attention_boost_for_step(
    constant_boost,
    timing,
    phase_boosts,
    step_index,
    num_steps,
) -> tuple[float, int]:
    """Resolve the subject boost and phase for one denoising step."""
    timing = validate_subject_attention_timing(timing)
    constant_boost = float(constant_boost)
    if timing == "constant":
        return constant_boost, 0
    phase_boosts = validate_subject_attention_ramp(*phase_boosts)
    if isinstance(num_steps, bool) or int(num_steps) < 1:
        raise ValueError("num_steps must be a positive integer")
    if isinstance(step_index, bool):
        raise ValueError("step_index must be an integer")
    return three_phase_value_for_step(phase_boosts, step_index, num_steps)


def delay_user_loras_for_depth(
    loras_slists,
    plugin_head_count,
    num_steps,
    phase_scales=(0.0, 0.25, 1.0),
):
    """Clone schedules and give appended user LoRAs a depth-first step ramp."""
    if loras_slists is None:
        raise ValueError("WanGP did not provide LoRA schedules")
    if isinstance(plugin_head_count, bool) or int(plugin_head_count) < 1:
        raise ValueError("plugin_head_count must be a positive integer")
    if isinstance(num_steps, bool) or int(num_steps) < 1:
        raise ValueError("num_steps must be a positive integer")
    if len(phase_scales) != 3:
        raise ValueError("phase_scales must contain three values")
    phase_scales = validate_depth_user_lora_ramp(*phase_scales)
    import copy

    plugin_head_count = int(plugin_head_count)
    num_steps = int(num_steps)
    first_cut = math.ceil(num_steps / 3)
    second_cut = math.ceil(2 * num_steps / 3)
    timing = (
        [phase_scales[0]] * first_cut
        + [phase_scales[1]] * (second_cut - first_cut)
        + [phase_scales[2]] * (num_steps - second_cut)
    )

    def expand_weight(value):
        source = value if isinstance(value, list) else [value]
        if not source:
            raise ValueError("WanGP supplied an empty user LoRA schedule")
        expanded = []
        position = 0.0
        increment = len(source) / num_steps
        for _ in range(num_steps):
            try:
                expanded.append(source[int(position)] * timing[len(expanded)])
            except TypeError as exc:
                raise ValueError(
                    "WanGP supplied a non-numeric user LoRA phase weight"
                ) from exc
            position += increment
        return expanded

    result = copy.deepcopy(loras_slists)
    dictionaries = [result]
    dictionaries.extend(value for value in result.values() if isinstance(value, dict))
    delayed_count = 0
    visited = set()
    for schedules in dictionaries:
        identity = id(schedules)
        if identity in visited:
            continue
        visited.add(identity)
        phase1 = schedules.get("phase1")
        if not isinstance(phase1, list) or len(phase1) <= plugin_head_count:
            continue
        user_count = len(phase1) - plugin_head_count
        delayed_count = max(delayed_count, user_count)
        shared = schedules.get("shared")
        for index in range(plugin_head_count, len(phase1)):
            step_schedule = expand_weight(phase1[index])
            for phase in ("phase1", "phase2", "phase3"):
                values = schedules.get(phase)
                if isinstance(values, list) and len(values) > index:
                    values[index] = list(step_schedule)
            if isinstance(shared, list) and len(shared) > index:
                shared[index] = True
    return result, delayed_count


def schedule_builtin_identity_depth_adapters(
    loras_slists,
    num_steps,
    identity_ramp=(0.25, 0.75, 1.0),
    depth_ramp=(1.0, 0.5, 0.0),
):
    """Clone WanGP schedules and ramp its Identity Edit and Depth adapters."""
    if loras_slists is None:
        raise ValueError("WanGP did not provide LoRA schedules")
    if isinstance(num_steps, bool) or int(num_steps) < 1:
        raise ValueError("num_steps must be a positive integer")
    identity_ramp = validate_builtin_identity_ramp(*identity_ramp)
    depth_ramp = validate_builtin_depth_ramp(*depth_ramp)
    import copy

    num_steps = int(num_steps)

    def expand_weight(value, ramp):
        source = value if isinstance(value, list) else [value]
        if not source:
            raise ValueError("WanGP supplied an empty built-in LoRA schedule")
        return [
            source[min(index * len(source) // num_steps, len(source) - 1)]
            * three_phase_value_for_step(ramp, index, num_steps)[0]
            for index in range(num_steps)
        ]

    result = copy.deepcopy(loras_slists)
    dictionaries = [result]
    dictionaries.extend(value for value in result.values() if isinstance(value, dict))
    visited = set()
    for schedules in dictionaries:
        identity = id(schedules)
        if identity in visited:
            continue
        visited.add(identity)
        phase1 = schedules.get("phase1")
        if not isinstance(phase1, list) or len(phase1) < 2:
            continue
        for phase in ("phase1", "phase2", "phase3"):
            values = schedules.get(phase)
            if not isinstance(values, list) or len(values) < 2:
                continue
            values[0] = expand_weight(values[0], identity_ramp)
            values[1] = expand_weight(values[1], depth_ramp)
        shared = schedules.get("shared")
        if isinstance(shared, list) and len(shared) >= 2:
            shared[:2] = [True, True]
    return result


def validate_depth_mask_feather_px(value) -> int:
    """Return a bounded integer edge feather for depth-control masks."""
    if value is None:
        return DEFAULT_DEPTH_MASK_FEATHER_PX
    if isinstance(value, bool):
        raise ValueError("depth_mask_feather_px must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("depth_mask_feather_px must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("depth_mask_feather_px must be an integer") from exc
    if parsed < MIN_DEPTH_MASK_FEATHER_PX or parsed > MAX_DEPTH_MASK_FEATHER_PX:
        raise ValueError(
            "depth_mask_feather_px must be between "
            f"{MIN_DEPTH_MASK_FEATHER_PX} and {MAX_DEPTH_MASK_FEATHER_PX}"
        )
    return parsed


def preprocess_krea2_adapter_state_dict(state_dict: dict) -> tuple[dict, object | None]:
    """Normalize Identity/Depth adapter keys and extract depth projection weights."""
    prefix = "diffusion_model."
    converted = {}
    control_weight = None

    for key, value in state_dict.items():
        key = key[len(prefix) :] if key.startswith(prefix) else key
        if key == "first.weight":
            shape = getattr(value, "shape", ())
            if len(shape) != 2 or shape[1] <= 0 or shape[1] % 2:
                raise ValueError(
                    "Krea 2 depth control first.weight must have an even 2D input width"
                )
            control_weight = value[:, shape[1] // 2 :].contiguous()
            continue
        if key == "first.bias":
            continue
        if key.endswith(".A"):
            key = key[:-2] + ".lora_A.weight"
        elif key.endswith(".B"):
            key = key[:-2] + ".lora_B.weight"
        converted[key] = value

    return converted, control_weight


def preprocess_identity_lora_state_dict(state_dict: dict) -> dict:
    """Map supported adapter keys to WanGP's Krea 2 transformer names."""
    converted, _control_weight = preprocess_krea2_adapter_state_dict(state_dict)
    return converted
