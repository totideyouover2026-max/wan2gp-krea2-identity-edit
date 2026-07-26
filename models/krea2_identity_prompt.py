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

## ReID identity method

- ReID is a Turbo-only alternative to Identity Edit and accepts exactly one
  identity reference. A close face/head crop usually gives the strongest
  identity signal, while a full reference can retain more appearance context.
- Describe the desired final image normally, including the intended head
  direction, gaze, expression and camera angle when those matter.
- ReID fixes sampling to 8 steps with no separate CFG pass. It may use depth
  control, but cannot be combined with Registered Outpaint in the same task.
- Background isolation, when selected in ReID mode, applies to its sole
  identity reference.

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

## Direct Image to ReID

- Select **Direct Image → ReID Edit** to use the Control Image itself as the
  starting composition rather than converting it to depth.
- The separate ReID reference remains the identity authority. Describe the
  desired edit and the details that should remain from the Control Image.
- Start with denoising strength `0.25`. Lower values preserve the Control Image
  more closely; higher values permit larger identity, clothing and scene edits.
- This experimental path is Turbo/ReID-only and currently supports Whole Frame.

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


def prompt_infos(*, include_reid: bool) -> tuple[str, str]:
  """Return public help without unfinished developer-only ReID routes."""
  title, markdown = KREA2_IDENTITY_PROMPT_INFOS
  if include_reid:
    return title, markdown

  for heading in ("## ReID identity method", "## Direct Image to ReID"):
    start = markdown.find(heading)
    if start == -1:
      continue
    end = markdown.find("\n## ", start + len(heading))
    markdown = markdown[:start] + (markdown[end + 1:] if end != -1 else "")
  return title, markdown.strip()
