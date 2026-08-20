# Preserved recovery environment

`pip_freeze_20260820.txt` is the exact byte output of:

```bash
/home/hermes/hat-face-training/hat-face/bin/python -m pip freeze --all
```

It was captured without installing or upgrading anything. Its SHA-256 and the
command are declared in `recovery/run_state.json`. The run-state verifier checks
both the preserved file bytes and a fresh command output, as well as declared
package versions and the repository commit:

```bash
/home/hermes/hat-face-training/hat-face/bin/python \
  -m recovery.tools.verify_run_state
```

A mismatch is evidence of environment drift. Do not update the snapshot merely
to make verification pass; preserve the old state and deliberately record a new
environment snapshot if the recovery protocol changes.
