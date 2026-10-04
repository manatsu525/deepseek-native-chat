---
name: verification-before-completion
description: Require concrete verification evidence before claiming completion.
---

# Verification before completion

Never infer success from an edit alone. Run the narrowest useful test, syntax
check, build, or health check and inspect its exit code and output. Report what
was actually verified and what remains unverified; distinguish syntax success
from runtime or browser behavior.
