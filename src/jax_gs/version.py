__version__ = "0.1.0"

# Historical baseline from which this port started.
GSPLAT_BASELINE_VERSION = "v1.5.3"
GSPLAT_BASELINE_SHA = "937e29912570c372bed6747a5c9bf85fed877bae"

# Active compatibility target. The branch name records upstream intent while
# the full SHA keeps builds and compatibility tests reproducible during the
# staged migration.
GSPLAT_TARGET_BRANCH = "main"
GSPLAT_TARGET_SHA = "2b902ff1891fc7f73f0f9b8c8bfc932cef2b198c"
GSPLAT_TARGET_COMMIT_DATE = "2026-07-24"

# Backward-compatible name used by earlier jax-gs revisions.
GSPLAT_MAIN_REFERENCE_SHA = GSPLAT_TARGET_SHA
