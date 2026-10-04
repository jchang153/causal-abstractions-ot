# Historical experiment versions

Earlier research implementations are preserved under `v1/`, `v2/` and `v3/`.
See [the experiment catalog](../README.md) for their roles. Shared libraries
have evolved, so a version folder is not a complete frozen environment. Use
recorded commits and configurations for exact replay.

Original `experiments/<folder>` imports remain relative compatibility symlinks.
The decimal and fixed-C1 tasks stay together because they share helpers. Current
core benchmark packages remain in their established locations.
