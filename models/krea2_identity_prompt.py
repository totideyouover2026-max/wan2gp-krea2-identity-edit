"""Prompt defaults and host-rendered help for Krea 2 Identity Edit."""

from __future__ import annotations


DEFAULT_IDENTITY_PROMPT = (
    "Preserve the identity and defining facial features of the person from the "
    "subject reference. Show them [desired appearance or action] in [desired "
    "environment], using [camera framing], with [lighting, visual style, and "
    "background details]."
)


KREA2_IDENTITY_PROMPT_INFOS = (
    "Krea 2 Identity Edit Prompt Guide",
    """
# Krea 2 Identity Edit Prompt Guide

## Core approach

- Write a direct description or instruction for the **desired final image**.
- State what must remain consistent, what should change, and the intended
  environment, framing, lighting and style.
- Identity references supply visual identity and source content. Avoid spending
  the whole prompt describing details that are already clear in the references.
- Prefer concrete sentences over disconnected keyword lists.

## One reference

Use the reference as the identity and source-image authority. Describe the edit
and the final result clearly.

```text
Preserve the identity and defining facial features of the person from the
reference. Show them [desired appearance or action] in [desired environment],
using [camera framing], with [lighting, visual style, and background details].
```

## Two references

Reference 1 is the **scene** and reference 2 is the **subject**. Make that
relationship explicit when the requested composition could otherwise be
ambiguous.

```text
Place the person from the second reference into the scene from the first
reference. Preserve their identity and defining appearance. Describe the
requested action, composition, framing, lighting and final visual style.
```

Automatic reference-background removal, when selected, applies only to the
second/subject reference. For SAM3 isolation, use a short visual segmentation
phrase that identifies the subject rather than describing the desired output.

## Depth control

- Describe the desired output; the depth adapter supplies structure and
  composition directly.
- For **Depth + prompt only**, upload no identity reference. Describe the
  subject, appearance, environment and photographic/style qualities in the
  prompt because no reference supplies them.
- Do not rely on phrases such as “match the control image.” The control image is
  not an additional semantic identity reference for Qwen3-VL.
- Avoid text that requests geometry or a pose that conflicts with the depth map.
- Lower depth strength for more compositional freedom; raise it cautiously for
  tighter structural adherence.
- With an Unlocker or style LoRA, prefer depth-first additional-LoRA timing so
  early denoising establishes geometry before the extra adapter reaches full
  strength.
- **Inside Mask** and **Outside Mask** limit where depth influences the target.
  These are conditioning masks, not output/inpainting masks.

## Direct Image to Identity Edit

- Select **Direct Image → Identity Edit** to use the Control Image itself as the
  starting composition rather than converting it to depth.
- The normal Identity Edit reference or references remain the identity and
  visual-content authority. Describe the desired edit and the details that
  should remain from the Control Image.
- Start with denoising strength `0.25`. Lower values preserve the Control Image
  more closely; higher values permit larger identity, clothing and scene edits.
- This experimental path supports Whole Frame and cannot be combined with
  Transfer Depth or Registered Outpaint.

## Registered Outpaint

- Set the dedicated **Registered Outpaint prompt** in Advanced settings. The
  outpaint pass does not reuse this main generation prompt.
- Describe the complete expanded image as a conservative continuation of the
  existing source: preserve scene, subjects, perspective, lighting, palette,
  texture and style.
- State any important continuation explicitly, such as extending the same wall,
  floor, sky or landscape. Avoid requesting new subjects, objects, text, logos
  or unrelated details unless they are intentionally wanted in the expansion.
- Target-aspect-ratio direction sliders control source placement on the axis
  that needs expansion. Manual mixed horizontal-and-vertical expansion is not
  yet supported because it requires two registered passes.

## Useful details

Include whichever of these materially affect the result:

- requested edit, action or subject placement;
- clothing, expression or retained identity details;
- environment and important background content;
- close-up, portrait, medium, full-body, wide or overhead framing;
- camera angle and perspective;
- lighting, time of day, color palette and visual style;
- elements that should remain unobstructed or absent.

Describe exclusions plainly and reinforce the desired positive alternative.
For example, specify an open or uncluttered background as well as naming an
unwanted obstruction.

## Avoid

- vague prompts that only say “match,” “copy” or “use the reference”;
- conflicting instructions about identity, scene order, pose or camera angle;
- long lists of redundant quality tags;
- treating a depth mask as a guarantee that unselected output pixels will stay
  unchanged;
- leaving bracketed template placeholders in the prompt before generation.

## Raw and Turbo

- **Turbo:** use one clear positive instruction; it normally runs without a
  separate negative-conditioning pass.
- **Raw:** use a clear positive instruction and reserve the negative prompt for
  concise unwanted attributes when guidance is enabled.
""".strip(),
)


def prompt_infos() -> tuple[str, str]:
    return KREA2_IDENTITY_PROMPT_INFOS
