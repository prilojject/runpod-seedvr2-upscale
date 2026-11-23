# Runpod Hub always builds from a Dockerfile. This thin file lets the Hub
# reuse the already-built image instead of compiling the worker again.
# The real build recipe lives in the repository-root Dockerfile.
#
# Pin the digest (not a floating :latest). Releases of this repo must still
# use targetCommitish=main — this digest is the image pin, not the git target.
#
# :v1.0.0 / :latest / :smoke-20261123-r3 all resolve to this Docker-v2
# manifest (the smoke image was already Docker v2, so docker push did not
# rewrite the media type). Image config and 26 layers are unchanged:
# config sha256:0272500851ff2110629eeb46928de4a973cffe8831fa72608786adf21a8fef0c.
FROM ghcr.io/prilojject/runpod-seedvr2-upscale:v1.0.0@sha256:a9e1c48dae820ae642a4136301b7ff186834ae5d54931374b04d2f2bc0a28f54
