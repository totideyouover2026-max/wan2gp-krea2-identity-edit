# Contributing

## Development principles

- Keep this repository independently installable beneath WanGP's `plugins/` directory.
- Do not require edits to WanGP core for normal installation.
- Keep model weights and generated outputs out of Git.
- Prefer small adapters around host WanGP components over copied implementations.
- Record the source and license of any adapted code.
- Do not make unfinished model definitions visible.

## Before submitting changes

Run:

```sh
python tools/validate_scaffold.py
python -m unittest discover -s tests -v
```

Also validate public assets and the targeted clean WanGP checkout when changing
runtime integration:

```sh
python tools/validate_remote_assets.py
python tools/validate_wangp_contract.py /path/to/Wan2GP
```

Inference changes must include the fields required by `GPU_ACCEPTANCE.md` in the
pull request description. Never use a contributor's personal/production WanGP
installation unless they explicitly designate it as the test installation.
