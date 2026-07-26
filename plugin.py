"""WanGP UI extension for the Krea 2 Identity Edit model plugin.

WanGP v12.34 exposes five model-defined custom-setting slots.  Four remain
context-sensitive controls on the main form; the fifth is a hidden JSON value
edited through this plugin-owned modal.  This keeps the package standalone and
does not require a WanGP core patch.
"""

from __future__ import annotations

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin  # pyright: ignore[reportMissingImports]

from .models.krea2_advanced_settings import (
    ADVANCED_SETTINGS_DEFAULTS,
    encode_advanced_settings,
    normalize_advanced_settings,
)
from .models.krea2_identity_features import reid_experiments_enabled
from .models.krea2_depth_preview import (
    build_masked_control_preview,
    generate_effective_depth_preview,
)
from .models.krea2_input_metadata import (
    gallery_input_label,
    image_input_label,
)


_MODEL_TYPES = {"krea2_identity_raw", "krea2_identity_turbo"}
_REID_EXPERIMENTS_ENABLED = reid_experiments_enabled()

_IDENTITY_EDIT_METHOD_CHOICES = [
    ("Identity Edit — standard fidelity (1x / 1x)", "identity_edit"),
    ("Identity Edit — subject fidelity 2x", "identity_edit_ref2"),
    ("Identity Edit — subject fidelity 4x (experimental)", "identity_edit_ref4"),
    ("Identity Edit — subject fidelity 8x (experimental)", "identity_edit_ref8"),
    ("Identity Edit — subject 4x + scene 2x (two refs)", "identity_edit_ref4_scene2"),
    ("Identity Edit — subject 8x + scene 2x (two refs)", "identity_edit_ref8_scene2"),
]

# This dropdown is hidden outside Identity Edit, but Gradio still preprocesses
# its value for every queued callback.  Keep every synchronized mode value in
# the component contract so a mode switch cannot invalidate an event that was
# queued with the immediately preceding value.
_ALL_IDENTITY_METHOD_CHOICES = _IDENTITY_EDIT_METHOD_CHOICES + [
    ("Depth + prompt only (selected by reference mode)", "depth_prompt"),
]
if _REID_EXPERIMENTS_ENABLED:
    _ALL_IDENTITY_METHOD_CHOICES.insert(
        len(_IDENTITY_EDIT_METHOD_CHOICES),
        ("ReID (selected by reference mode)", "reid"),
    )

_GENERATION_PROCESS_CHOICES = [
    ("Standard single pass (recommended)", "standard"),
    ("Registered Outpaint only", "outpaint_only"),
    ("Identity edit, then Registered Outpaint", "identity_then_outpaint"),
]
if _REID_EXPERIMENTS_ENABLED:
    _GENERATION_PROCESS_CHOICES[1:1] = [
        ("Two phase: depth, then ReID — light refinement", "two_phase_015_keep"),
        ("Two phase: depth, then ReID — balanced refinement", "two_phase_025_keep"),
        ("Two phase: depth, then ReID — stronger refinement", "two_phase_035_keep"),
        ("Two phase: depth, then ReID — balanced, depth off in phase 2", "two_phase_025_off"),
    ]

_IDENTITY_LORA_VARIANTS = [
    ("v1.2 Full — 1.83 GB (recommended)", "full_v1.2"),
    ("v1.2 Rank 128 — 0.91 GB", "r128"),
    ("v1.2 Rank 64 — 0.46 GB", "r64"),
]

_USER_LORA_TIMING_CHOICES = [
    ("Depth first — delay additional LoRAs (recommended)", "depth_first"),
    ("All steps — apply additional LoRAs throughout", "all_steps"),
]

_STYLE = """
<style>
.krea2-advanced-launcher { margin-top: 0.35rem; }
.krea2-advanced-floating {
  position: fixed !important;
  top: 96px !important;
  right: 32px !important;
  left: auto !important;
  z-index: 1250 !important;
  width: min(900px, calc(100vw - 34px)) !important;
  max-height: min(82vh, 780px) !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: visible !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  pointer-events: none !important;
}
.krea2-advanced-card {
  display: flex !important;
  flex-direction: column !important;
  gap: 0 !important;
  width: 100% !important;
  max-height: min(82vh, 780px) !important;
  margin: 0 !important;
  overflow: hidden !important;
  padding: 0 !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 1rem !important;
  background: var(--background-fill-primary) !important;
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.42) !important;
  pointer-events: auto !important;
}
.krea2-advanced-title {
  flex: 0 0 auto !important;
  align-items: center !important;
  gap: 0.75rem !important;
  margin: 0 !important;
  padding: 0.6rem 0.75rem 0.55rem 1rem !important;
  border: 0 !important;
  background: var(--button-primary-background-fill) !important;
  cursor: move !important;
  user-select: none !important;
  touch-action: none !important;
}
.krea2-advanced-title h2 {
  margin: 0 !important;
  color: var(--button-primary-text-color) !important;
  font-size: 1rem !important;
}
.krea2-advanced-content {
  min-height: 0 !important;
  overflow-y: auto !important;
  padding: 0.9rem 1rem 1rem !important;
}
.krea2-advanced-close {
  min-width: 3rem !important;
  flex-grow: 0 !important;
  cursor: pointer !important;
}
.krea2-depth-preview-panel { margin-top: 0.4rem; }
.krea2-depth-preview-note { opacity: 0.86; }
.krea2-depth-mask-upload, .krea2-effective-depth-output { min-width: 0 !important; }
@media (max-width: 640px) {
  .krea2-advanced-floating {
    top: 10px !important;
    right: 10px !important;
    width: calc(100vw - 20px) !important;
    max-height: calc(100vh - 20px) !important;
  }
  .krea2-advanced-card { max-height: calc(100vh - 20px) !important; }
}
</style>
"""


_SCRIPT = r"""
(() => {
  const modalId = "krea2-advanced-floating-modal";
  const handleId = "krea2-advanced-drag-handle";
  const margin = 10;
  let drag = null;

    function root() {
        if (window.gradioApp) return window.gradioApp();
        const app = document.querySelector("gradio-app");
        return app ? (app.shadowRoot || app) : document;
    }

    if (window.krea2IdentityAdvancedDragInstalled) return;
    window.krea2IdentityAdvancedDragInstalled = true;

  function clampPosition(modal, left, top) {
    const rect = modal.getBoundingClientRect();
    const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
    const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
    return {
      left: Math.min(Math.max(margin, left), maxLeft),
      top: Math.min(Math.max(margin, top), maxTop),
    };
  }

  document.addEventListener("pointerdown", (event) => {
    const handle = event.target.closest(`#${handleId}`);
    if (!handle || event.target.closest("button, input, select, textarea, a")) return;
        const modal = root().getElementById(modalId);
    if (!modal) return;
    const rect = modal.getBoundingClientRect();
    modal.style.setProperty("left", `${rect.left}px`, "important");
    modal.style.setProperty("top", `${rect.top}px`, "important");
    modal.style.setProperty("right", "auto", "important");
    drag = {
      modal,
      pointerId: event.pointerId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    handle.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  });

  document.addEventListener("pointermove", (event) => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const position = clampPosition(
      drag.modal,
      event.clientX - drag.offsetX,
      event.clientY - drag.offsetY,
    );
    drag.modal.style.setProperty("left", `${position.left}px`, "important");
    drag.modal.style.setProperty("top", `${position.top}px`, "important");
    event.preventDefault();
  });

  function endDrag(event) {
    if (drag && drag.pointerId === event.pointerId) drag = null;
  }
  document.addEventListener("pointerup", endDrag);
  document.addEventListener("pointercancel", endDrag);

  window.addEventListener("resize", () => {
        const modal = root().getElementById(modalId);
    if (!modal || modal.offsetParent === null) return;
    const rect = modal.getBoundingClientRect();
    const position = clampPosition(modal, rect.left, rect.top);
    modal.style.setProperty("left", `${position.left}px`, "important");
    modal.style.setProperty("top", `${position.top}px`, "important");
    modal.style.setProperty("right", "auto", "important");
  });
})();
"""


def _model_type(state) -> str:
    if not isinstance(state, dict):
        return ""
    key = "model_type" if state.get("active_form", "add") == "add" else "edit_model_type"
    # WanGP's add and edit forms normally use separate keys. During form
    # transitions one key can be briefly absent, so fall back to the other
    # instead of retaining visibility from the previously selected model.
    alternate_key = "edit_model_type" if key == "model_type" else "model_type"
    return str(state.get(key) or state.get(alternate_key) or "")


def _plugin_active(state) -> bool:
    return _model_type(state) in _MODEL_TYPES


def _reference_mode(video_prompt_type) -> str:
    prompt_type = str(video_prompt_type or "")
    if "K" in prompt_type:
        return "identity_edit"
    if "I" in prompt_type:
        return "reid" if _REID_EXPERIMENTS_ENABLED else "disabled_reid"
    return "depth_prompt"


def _synchronized_identity_method(identity_method, video_prompt_type) -> str:
    mode = _reference_mode(video_prompt_type)
    current = str(identity_method or "")
    if mode == "identity_edit":
        return current if current.startswith("identity_edit") else "identity_edit"
    return mode


def _safe_advanced_values(payload):
    try:
        return normalize_advanced_settings(payload)
    except ValueError:
        return dict(ADVANCED_SETTINGS_DEFAULTS)


def _custom_mask_mode(video_prompt_type) -> bool:
    return "Y" in str(video_prompt_type or "")


def _custom_mask_inverted(video_prompt_type) -> bool:
    prompt_type = str(video_prompt_type or "")
    return "Y" in prompt_type and "N" in prompt_type


def _refresh_image_label(value, base_label):
    return gr.update(label=image_input_label(base_label, value))


def _refresh_gallery_label(value, base_label):
    return gr.update(label=gallery_input_label(base_label, value))


class Krea2IdentityAdvancedUI(WAN2GPPlugin):
    """Adds adaptive visibility and a modal editor to the model form."""

    def setup_ui(self):
        for component_id in (
            "state",
            "refresh_form_trigger",
            "image_guide",
            "image_refs",
            "video_prompt_type",
            "custom_guide",
            "custom_settings_visibility_trigger",
            "custom_setting_text_inputs",
            "custom_setting_dropdown_inputs",
        ):
            self.request_component(component_id)
        self.request_global("get_preprocessor")
        self.request_global("release_model")
        self.add_custom_js(_SCRIPT)

    def post_ui_setup(self, _components):
        text_inputs = getattr(self, "custom_setting_text_inputs", [])
        dropdown_inputs = getattr(self, "custom_setting_dropdown_inputs", [])
        if len(text_inputs) < 5 or len(dropdown_inputs) < 3:
            print(
                "[Krea2 Identity][UI] Advanced settings unavailable: "
                "WanGP did not expose the expected five custom-setting slots."
            )
            return {}

        payload = text_inputs[4]
        identity_method = dropdown_inputs[0]
        reference_preparation = dropdown_inputs[2]
        segmentation_phrase = text_inputs[3]

        # Keep the serialized fifth slot out of the form even if a host refresh
        # temporarily evaluates its visibility before the plugin event runs.
        payload.visible = False

        initial_active = _plugin_active(self.state.value)
        # WanGP's generic custom-guide slot is a filename-only gr.File.  The
        # plugin mirrors it through an image uploader below while active so the
        # uploaded mask itself remains visible.  The hidden host component
        # continues to own queue persistence and the generation input.
        self.custom_guide.visible = not initial_active
        initial_mode = _reference_mode(self.video_prompt_type.value)
        identity_method.value = _synchronized_identity_method(
            identity_method.value, self.video_prompt_type.value
        )
        identity_method.visible = initial_active and initial_mode == "identity_edit"
        # Gradio stores dropdown choices internally as [label, value] pairs.
        # Do not narrow this list when the selector is hidden; concurrent host
        # refresh events can still carry the value from the previous mode.
        identity_method.choices = [list(choice) for choice in _ALL_IDENTITY_METHOD_CHOICES]
        initial_reference_visible = initial_active and initial_mode in {
            "identity_edit",
            "reid",
        }
        reference_preparation.visible = initial_reference_visible
        segmentation_phrase.visible = (
            initial_reference_visible and reference_preparation.value == "sam3"
        )

        control_base_label = getattr(self.image_guide, "label", None) or "Control Image"
        references_base_label = (
            getattr(self.image_refs, "label", None) or "Reference Images"
        )
        self.image_guide.label = image_input_label(
            control_base_label, getattr(self.image_guide, "value", None)
        )
        self.image_refs.label = gallery_input_label(
            references_base_label, getattr(self.image_refs, "value", None)
        )
        self.image_guide.change(
            fn=_refresh_image_label,
            inputs=[self.image_guide, gr.State(control_base_label)],
            outputs=[self.image_guide],
            show_progress="hidden",
        )
        self.image_refs.change(
            fn=_refresh_gallery_label,
            inputs=[self.image_refs, gr.State(references_base_label)],
            outputs=[self.image_refs],
            show_progress="hidden",
        )

        def construct_depth_preview_ui():
            initially_visible = initial_active and _custom_mask_mode(
                self.video_prompt_type.value
            )
            with gr.Column(
                visible=initially_visible,
                elem_classes=["krea2-depth-preview-panel"],
            ) as preview_panel:
                with gr.Row(equal_height=True):
                    with gr.Column(elem_classes=["krea2-depth-mask-upload"]):
                        mask_upload = gr.Image(
                            value=self.custom_guide.value,
                            label="Optional Custom Depth Mask",
                            sources=["upload"],
                            type="filepath",
                            image_mode=None,
                            format="png",
                            interactive=True,
                        )
                        mask_base_label = "Optional Custom Depth Mask"
                        mask_upload.label = image_input_label(
                            mask_base_label, getattr(mask_upload, "value", None)
                        )
                        mask_upload.change(
                            fn=_refresh_image_label,
                            inputs=[mask_upload, gr.State(mask_base_label)],
                            outputs=[mask_upload],
                            show_progress="hidden",
                        )
                        gr.Markdown(
                            "White selects the controlled area; the Black Area "
                            "mode reverses it. The mask does not replace the "
                            "Control Image.",
                            elem_classes=["krea2-depth-preview-note"],
                        )
                    with gr.Column(elem_classes=["krea2-effective-depth-output"]):
                        effective_depth_preview = gr.Image(
                            label="Effective Depth Preview",
                            interactive=False,
                            type="pil",
                        )
                        generate_button = gr.Button(
                            "Generate Effective Depth Preview",
                            variant="secondary",
                        )
                        preview_status = gr.Markdown()

            def refresh_depth_preview(
                state,
                prompt_type,
                control_source,
                mask_source,
                packed_settings,
            ):
                visible = _plugin_active(state) and _custom_mask_mode(prompt_type)
                if not visible:
                    return gr.update(visible=False), gr.update(value=mask_source), None, ""
                if control_source is None or mask_source is None:
                    missing = []
                    if control_source is None:
                        missing.append("Control Image")
                    if mask_source is None:
                        missing.append("custom mask")
                    return (
                        gr.update(visible=True),
                        gr.update(value=mask_source),
                        None,
                        "Upload " + " and ".join(missing) + " to generate effective depth.",
                    )
                values = _safe_advanced_values(packed_settings)
                feather = values["depth_mask_feather_px"]
                try:
                    _discarded_preview, channel = build_masked_control_preview(
                        control_source,
                        mask_source,
                        invert=_custom_mask_inverted(prompt_type),
                        feather_px=feather,
                    )
                except (OSError, TypeError, ValueError) as exc:
                    return (
                        gr.update(visible=True),
                        gr.update(value=mask_source),
                        None,
                        f"Mask error: {exc}",
                    )
                edge = "hard edge" if feather == 0 else f"{feather}px feather"
                return (
                    gr.update(visible=True),
                    gr.update(value=mask_source),
                    None,
                    f"Mask ready — {channel}, {edge}. Generate the effective "
                    "depth preview when needed. On low VRAM, WanGP will unload "
                    "the active generation model first.",
                )

            preview_inputs = [
                self.state,
                self.video_prompt_type,
                self.image_guide,
                self.custom_guide,
                payload,
            ]
            preview_outputs = [
                preview_panel,
                mask_upload,
                effective_depth_preview,
                preview_status,
            ]
            mask_upload.input(
                fn=lambda source: source,
                inputs=[mask_upload],
                outputs=[self.custom_guide],
                show_progress="hidden",
            )
            for event in (
                self.image_guide.change,
                self.custom_guide.change,
                self.video_prompt_type.change,
                payload.change,
            ):
                event(
                    fn=refresh_depth_preview,
                    inputs=preview_inputs,
                    outputs=preview_outputs,
                    show_progress="hidden",
                )
            self.refresh_form_trigger.change(
                fn=refresh_depth_preview,
                inputs=preview_inputs,
                outputs=preview_outputs,
                show_progress="hidden",
            )

            def generate_depth_preview(
                state,
                prompt_type,
                control_source,
                mask_source,
                packed_settings,
            ):
                if not _plugin_active(state) or not _custom_mask_mode(prompt_type):
                    return None, "Select a Custom Mask control-area mode first."
                values = _safe_advanced_values(packed_settings)
                feather = values["depth_mask_feather_px"]
                try:
                    preview, channel = generate_effective_depth_preview(
                        control_source,
                        mask_source,
                        get_preprocessor=getattr(self, "get_preprocessor", None),
                        release_model=getattr(self, "release_model", None),
                        invert=_custom_mask_inverted(prompt_type),
                        feather_px=feather,
                    )
                except Exception as exc:
                    return None, f"Effective-depth preview error: {exc}"
                edge = "hard edge" if feather == 0 else f"{feather}px feather"
                return (
                    preview,
                    f"Effective depth ready — WanGP Depth Anything ran on the "
                    f"full control before the {channel} mask was applied ({edge}). "
                    "The generation model will reload automatically if it was "
                    "unloaded to free VRAM.",
                )

            generate_button.click(
                fn=generate_depth_preview,
                inputs=preview_inputs,
                outputs=[effective_depth_preview, preview_status],
                show_progress="full",
            )
            return preview_panel

        self.insert_after("custom_guide", construct_depth_preview_ui)

        self.refresh_form_trigger.change(
            fn=lambda state: gr.update(visible=not _plugin_active(state)),
            inputs=[self.state],
            outputs=[self.custom_guide],
            show_progress="hidden",
        )

        def construct_advanced_ui():
            initial_launcher_visible = _plugin_active(self.state.value)
            with gr.Column(
                # WanGP constructs this shared extension after the selected
                # model form. Use that initial snapshot for the active Krea
                # form, then let the host refresh listeners below hide/show it
                # as the user changes models.
                visible=initial_launcher_visible,
                elem_classes=["krea2-advanced-launcher"],
            ) as launcher_panel:
                gr.HTML(_STYLE)
                show_advanced = gr.Dropdown(
                    choices=[("No", "no"), ("Yes", "yes")],
                    value="no",
                    label="Show Advanced Settings",
                    info="Select Yes to open the Krea 2 Identity Edit advanced settings.",
                )

                with gr.Group(
                    visible=False,
                    elem_id="krea2-advanced-floating-modal",
                    elem_classes=["krea2-advanced-floating"],
                ) as modal:
                    with gr.Column(elem_classes=["krea2-advanced-card"]):
                        with gr.Row(
                            elem_id="krea2-advanced-drag-handle",
                            elem_classes=["krea2-advanced-title"],
                        ):
                            gr.Markdown("## Krea 2 Identity Edit — Advanced Settings")
                            close_button = gr.Button(
                                "Close", variant="secondary", elem_classes=["krea2-advanced-close"]
                            )

                        with gr.Column(elem_classes=["krea2-advanced-content"]):
                            with gr.Tabs():
                                with gr.Tab("Process & identity"):
                                    generation_process = gr.Dropdown(
                                        choices=_GENERATION_PROCESS_CHOICES,
                                        value=ADVANCED_SETTINGS_DEFAULTS["generation_process"],
                                        label="Generation process / additional task",
                                        info=(
                                            "Standard is the existing single-pass workflow. "
                                            "Two-phase and outpaint modes are experimental."
                                        ),
                                    )
                                    identity_lora_variant = gr.Dropdown(
                                        choices=_IDENTITY_LORA_VARIANTS,
                                        value=ADVANCED_SETTINGS_DEFAULTS["identity_lora_variant"],
                                        label="Identity Edit LoRA variant",
                                        info="Used by Identity Edit modes; ReID uses its own fixed LoRA.",
                                    )
                                    reid_lora_strength = gr.Slider(
                                        minimum=0,
                                        maximum=2,
                                        step=0.05,
                                        value=ADVANCED_SETTINGS_DEFAULTS[
                                            "reid_lora_strength"
                                        ],
                                        label="ReID LoRA strength (diagnostic)",
                                        info=(
                                            "ReID only. Compare 0 and 1 with the same seed to "
                                            "verify whether the adapter changes the result."
                                        ),
                                        visible=_REID_EXPERIMENTS_ENABLED,
                                    )
                                    reid_reference_method = gr.Dropdown(
                                        choices=[
                                            (
                                                "Official isolated K/V cache (recommended)",
                                                "isolated_cache",
                                            ),
                                            (
                                                "Joint timestep-zero stream (diagnostic A/B)",
                                                "joint_timestep_zero",
                                            ),
                                        ],
                                        value=ADVANCED_SETTINGS_DEFAULTS[
                                            "reid_reference_method"
                                        ],
                                        label="ReID reference injection",
                                        info=(
                                            "Joint stream matches the released ComfyUI graph: "
                                            "target and reference attend together, with timestep-zero "
                                            "modulation on the reference. Isolated cache preserves the "
                                            "plugin's previous implementation for comparisons."
                                        ),
                                        visible=_REID_EXPERIMENTS_ENABLED,
                                    )
                                    grounding_px = gr.Slider(
                                        minimum=384,
                                        maximum=1536,
                                        step=64,
                                        value=ADVANCED_SETTINGS_DEFAULTS["grounding_px"],
                                        label="Reference grounding budget (pixels)",
                                        info=(
                                            "Controls Qwen3-VL reference-image grounding detail. "
                                            "Higher values use more memory."
                                        ),
                                    )
                                    output_resolution_limit = gr.Dropdown(
                                        choices=[
                                            (
                                                "Safe 2 MP limit (recommended)",
                                                "safe_2mp",
                                            ),
                                            (
                                                "Use full selected resolution",
                                                "unlimited",
                                            ),
                                        ],
                                        value=ADVANCED_SETTINGS_DEFAULTS[
                                            "output_resolution_limit"
                                        ],
                                        label="Reference-mode output resolution limit",
                                        info=(
                                            "Reference-based modes default to the 2 MP "
                                            "safety limit. Use full selected resolution "
                                            "only when memory allows. Depth + Prompt "
                                            "always uses the selected resolution."
                                        ),
                                    )
                                    secondary_reference_geometry = gr.Dropdown(
                                        choices=[
                                            (
                                                "Aspect-preserving FIT (current/default)",
                                                "fit",
                                            ),
                                            (
                                                "Stretch to output geometry (legacy A/B)",
                                                "stretch",
                                            ),
                                        ],
                                        value=ADVANCED_SETTINGS_DEFAULTS[
                                            "secondary_reference_geometry"
                                        ],
                                        label="Picture 2 reference geometry",
                                        info=(
                                            "Temporary Identity Edit diagnostic. FIT preserves "
                                            "Picture 2's aspect ratio and centres its latent grid. "
                                            "Legacy stretch resizes Picture 2 to the output geometry."
                                        ),
                                    )
                                    subject_attention_timing = gr.Dropdown(
                                        choices=[
                                            (
                                                "Constant selected fidelity profile (default)",
                                                "constant",
                                            ),
                                            (
                                                "Ramp explicit boosts over denoising thirds",
                                                "ramp",
                                            ),
                                        ],
                                        value=ADVANCED_SETTINGS_DEFAULTS[
                                            "subject_attention_timing"
                                        ],
                                        label="Identity Edit subject-attention timing",
                                        info=(
                                            "Identity Edit only. Ramp mode overrides the selected "
                                            "subject fidelity strength with the three boosts below; "
                                            "scene attention remains at the selected profile strength."
                                        ),
                                    )
                                    with gr.Group(visible=False) as subject_attention_ramp_group:
                                        gr.Markdown(
                                            "Subject-attention boosts are applied during the three "
                                            "equal denoising thirds. A low early value preserves "
                                            "composition; a high final value restores identity."
                                        )
                                        with gr.Row():
                                            subject_attention_ramp_early = gr.Slider(
                                                minimum=1,
                                                maximum=8,
                                                step=0.25,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "subject_attention_ramp_early"
                                                ],
                                                label="Subject early third",
                                            )
                                            subject_attention_ramp_middle = gr.Slider(
                                                minimum=1,
                                                maximum=8,
                                                step=0.25,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "subject_attention_ramp_middle"
                                                ],
                                                label="Subject middle third",
                                            )
                                            subject_attention_ramp_final = gr.Slider(
                                                minimum=1,
                                                maximum=8,
                                                step=0.25,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "subject_attention_ramp_final"
                                                ],
                                                label="Subject final third",
                                            )

                                with gr.Tab("Depth & LoRA interaction"):
                                    builtin_adapter_timing = gr.Dropdown(
                                        choices=[
                                            (
                                                "Simultaneous built-in adapters (default)",
                                                "simultaneous",
                                            ),
                                            (
                                                "Depth layout → Identity refinement (experimental)",
                                                "depth_then_identity",
                                            ),
                                        ],
                                        value=ADVANCED_SETTINGS_DEFAULTS[
                                            "builtin_adapter_timing"
                                        ],
                                        label="Built-in Identity Edit and Depth timing",
                                        info=(
                                            "Experimental Identity Edit + Transfer Depth preset. "
                                            "It fades Depth while increasing Identity Edit over "
                                            "the three denoising thirds."
                                        ),
                                    )
                                    with gr.Group(visible=False) as builtin_adapter_ramp_group:
                                        gr.Markdown(
                                            "These multipliers apply to the built-in adapters. "
                                            "Depth also scales its direct target-token projection."
                                        )
                                        with gr.Row():
                                            builtin_depth_ramp_early = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "builtin_depth_ramp_early"
                                                ],
                                                label="Depth early third",
                                            )
                                            builtin_depth_ramp_middle = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "builtin_depth_ramp_middle"
                                                ],
                                                label="Depth middle third",
                                            )
                                            builtin_depth_ramp_final = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "builtin_depth_ramp_final"
                                                ],
                                                label="Depth final third",
                                            )
                                        with gr.Row():
                                            builtin_identity_ramp_early = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "builtin_identity_ramp_early"
                                                ],
                                                label="Identity Edit early third",
                                            )
                                            builtin_identity_ramp_middle = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "builtin_identity_ramp_middle"
                                                ],
                                                label="Identity Edit middle third",
                                            )
                                            builtin_identity_ramp_final = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "builtin_identity_ramp_final"
                                                ],
                                                label="Identity Edit final third",
                                            )
                                    depth_mask_feather_px = gr.Slider(
                                        minimum=0,
                                        maximum=64,
                                        step=1,
                                        value=ADVANCED_SETTINGS_DEFAULTS["depth_mask_feather_px"],
                                        label="Depth mask feather (pixels)",
                                        info=(
                                            "Zero keeps a hard edge. This applies to painted and "
                                            "uploaded custom depth masks."
                                        ),
                                    )
                                    depth_user_lora_timing = gr.Dropdown(
                                        choices=_USER_LORA_TIMING_CHOICES,
                                        value=ADVANCED_SETTINGS_DEFAULTS["depth_user_lora_timing"],
                                        label="Additional LoRA timing with depth",
                                        info=(
                                            "Depth first reduces interference from user LoRAs during "
                                            "early pose formation."
                                        ),
                                    )
                                    with gr.Group(visible=True) as depth_lora_ramp_group:
                                        gr.Markdown(
                                            "Depth-first ramp multipliers scale each additional "
                                            "LoRA's selected strength during the three equal "
                                            "denoising thirds. Values must not decrease."
                                        )
                                        with gr.Row():
                                            depth_user_lora_ramp_early = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "depth_user_lora_ramp_early"
                                                ],
                                                label="Early third",
                                            )
                                            depth_user_lora_ramp_middle = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "depth_user_lora_ramp_middle"
                                                ],
                                                label="Middle third",
                                            )
                                            depth_user_lora_ramp_final = gr.Slider(
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=ADVANCED_SETTINGS_DEFAULTS[
                                                    "depth_user_lora_ramp_final"
                                                ],
                                                label="Final third",
                                            )

                            with gr.Row():
                                cancel_button = gr.Button("Cancel", variant="secondary")
                                apply_button = gr.Button("Apply Settings", variant="primary")

            modal_fields = [
                generation_process,
                grounding_px,
                output_resolution_limit,
                identity_lora_variant,
                reid_lora_strength,
                reid_reference_method,
                secondary_reference_geometry,
                subject_attention_timing,
                subject_attention_ramp_early,
                subject_attention_ramp_middle,
                subject_attention_ramp_final,
                builtin_adapter_timing,
                builtin_depth_ramp_early,
                builtin_depth_ramp_middle,
                builtin_depth_ramp_final,
                builtin_identity_ramp_early,
                builtin_identity_ramp_middle,
                builtin_identity_ramp_final,
                depth_mask_feather_px,
                depth_user_lora_timing,
                depth_user_lora_ramp_early,
                depth_user_lora_ramp_middle,
                depth_user_lora_ramp_final,
            ]

            def open_advanced(selection, packed_settings):
                if selection != "yes":
                    return [gr.update()] * (4 + len(modal_fields))
                values = _safe_advanced_values(packed_settings)
                return [
                    gr.update(visible=True),
                    values["generation_process"],
                    values["grounding_px"],
                    values["output_resolution_limit"],
                    values["identity_lora_variant"],
                    values["reid_lora_strength"],
                    values["reid_reference_method"],
                    values["secondary_reference_geometry"],
                    values["subject_attention_timing"],
                    values["subject_attention_ramp_early"],
                    values["subject_attention_ramp_middle"],
                    values["subject_attention_ramp_final"],
                    values["builtin_adapter_timing"],
                    values["builtin_depth_ramp_early"],
                    values["builtin_depth_ramp_middle"],
                    values["builtin_depth_ramp_final"],
                    values["builtin_identity_ramp_early"],
                    values["builtin_identity_ramp_middle"],
                    values["builtin_identity_ramp_final"],
                    values["depth_mask_feather_px"],
                    values["depth_user_lora_timing"],
                    values["depth_user_lora_ramp_early"],
                    values["depth_user_lora_ramp_middle"],
                    values["depth_user_lora_ramp_final"],
                    gr.update(
                        visible=values["subject_attention_timing"] == "ramp"
                    ),
                    gr.update(
                        visible=values["builtin_adapter_timing"]
                        == "depth_then_identity"
                    ),
                    gr.update(
                        visible=values["depth_user_lora_timing"] == "depth_first"
                    ),
                ]

            show_advanced.change(
                fn=open_advanced,
                inputs=[show_advanced, payload],
                outputs=[modal]
                + modal_fields
                + [
                    subject_attention_ramp_group,
                    builtin_adapter_ramp_group,
                    depth_lora_ramp_group,
                ],
                show_progress="hidden",
            )

            subject_attention_timing.change(
                fn=lambda timing: gr.update(visible=timing == "ramp"),
                inputs=[subject_attention_timing],
                outputs=[subject_attention_ramp_group],
                show_progress="hidden",
            )

            builtin_adapter_timing.change(
                fn=lambda timing: gr.update(visible=timing == "depth_then_identity"),
                inputs=[builtin_adapter_timing],
                outputs=[builtin_adapter_ramp_group],
                show_progress="hidden",
            )

            depth_user_lora_timing.change(
                fn=lambda timing: gr.update(visible=timing == "depth_first"),
                inputs=[depth_user_lora_timing],
                outputs=[depth_lora_ramp_group],
                show_progress="hidden",
            )

            def apply_advanced(
                process,
                grounding,
                resolution_limit,
                variant,
                reid_strength,
                reid_method,
                reference_geometry,
                subject_timing,
                subject_ramp_early,
                subject_ramp_middle,
                subject_ramp_final,
                builtin_timing,
                builtin_depth_early,
                builtin_depth_middle,
                builtin_depth_final,
                builtin_identity_early,
                builtin_identity_middle,
                builtin_identity_final,
                feather,
                timing,
                ramp_early,
                ramp_middle,
                ramp_final,
            ):
                packed = encode_advanced_settings(
                    {
                        "generation_process": process,
                        "grounding_px": grounding,
                        "output_resolution_limit": resolution_limit,
                        "identity_lora_variant": variant,
                        "reid_lora_strength": reid_strength,
                        "reid_reference_method": reid_method,
                        "secondary_reference_geometry": reference_geometry,
                        "subject_attention_timing": subject_timing,
                        "subject_attention_ramp_early": subject_ramp_early,
                        "subject_attention_ramp_middle": subject_ramp_middle,
                        "subject_attention_ramp_final": subject_ramp_final,
                        "builtin_adapter_timing": builtin_timing,
                        "builtin_depth_ramp_early": builtin_depth_early,
                        "builtin_depth_ramp_middle": builtin_depth_middle,
                        "builtin_depth_ramp_final": builtin_depth_final,
                        "builtin_identity_ramp_early": builtin_identity_early,
                        "builtin_identity_ramp_middle": builtin_identity_middle,
                        "builtin_identity_ramp_final": builtin_identity_final,
                        "depth_mask_feather_px": feather,
                        "depth_user_lora_timing": timing,
                        "depth_user_lora_ramp_early": ramp_early,
                        "depth_user_lora_ramp_middle": ramp_middle,
                        "depth_user_lora_ramp_final": ramp_final,
                    }
                )
                return packed, gr.update(visible=False), "no"

            apply_button.click(
                fn=apply_advanced,
                inputs=modal_fields,
                outputs=[payload, modal, show_advanced],
                show_progress="hidden",
            )

            def dismiss_advanced():
                return gr.update(visible=False), "no"

            for button in (cancel_button, close_button):
                button.click(
                    fn=dismiss_advanced,
                    inputs=None,
                    outputs=[modal, show_advanced],
                    show_progress="hidden",
                )

            def refresh_launcher(state):
                active = _plugin_active(state)
                return (
                    gr.update(visible=active),
                    gr.update(visible=False),
                    "no",
                )

            for trigger in (
                # payload is one of the model-defined custom-setting slots, so
                # WanGP refreshes its value directly on every model switch (as
                # part of fill_inputs' return values) even though
                # refresh_form_trigger/custom_settings_visibility_trigger do
                # not fire on that path. Its value only exists as this
                # plugin's encoded JSON, so it reliably changes on both the
                # transition into and out of Krea2, making it the trigger
                # that actually catches a live model switch.
                payload.change,
                self.refresh_form_trigger.change,
                self.custom_settings_visibility_trigger.change,
            ):
                trigger(
                    fn=refresh_launcher,
                    inputs=[self.state],
                    outputs=[launcher_panel, modal, show_advanced],
                    show_progress="hidden",
                )
            return launcher_panel

        self.insert_after("custom_settings_visibility_trigger", construct_advanced_ui)

        def refresh_context_controls(state, method, preparation, prompt_type):
            active = _plugin_active(state)
            mode = _reference_mode(prompt_type)
            reference_visible = active and mode in {"identity_edit", "reid"}
            return (
                gr.update(
                    visible=active and mode == "identity_edit",
                    choices=_ALL_IDENTITY_METHOD_CHOICES,
                    value=_synchronized_identity_method(method, prompt_type),
                ),
                gr.update(visible=reference_visible),
                gr.update(visible=reference_visible and preparation == "sam3"),
            )

        context_inputs = [
            self.state,
            identity_method,
            reference_preparation,
            self.video_prompt_type,
        ]
        context_outputs = [
            identity_method,
            reference_preparation,
            segmentation_phrase,
        ]
        for trigger in (
            self.video_prompt_type.change,
            self.custom_settings_visibility_trigger.change,
        ):
            trigger(
                fn=refresh_context_controls,
                inputs=context_inputs,
                outputs=context_outputs,
                show_progress="hidden",
            )
        self.refresh_form_trigger.change(
            fn=refresh_context_controls,
            inputs=context_inputs,
            outputs=context_outputs,
            show_progress="hidden",
        )

        def refresh_segmentation_phrase(state, preparation, prompt_type):
            visible = (
                _plugin_active(state)
                and _reference_mode(prompt_type) in {"identity_edit", "reid"}
                and preparation == "sam3"
            )
            return gr.update(visible=visible)

        reference_preparation.change(
            fn=refresh_segmentation_phrase,
            inputs=[self.state, reference_preparation, self.video_prompt_type],
            outputs=[segmentation_phrase],
            show_progress="hidden",
        )
        return {}