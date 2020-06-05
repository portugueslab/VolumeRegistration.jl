# VolumeRegistration.jl

[![Build Status](https://travis-ci.com/portugueslab/VolumeRegistration.jl.svg?branch=master)](https://travis-ci.com/portugueslab/VolumeRegistration.jl)
![CI](https://github.com/portugueslab/VolumeRegistration.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/portugueslab/VolumeRegistration.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/portugueslab/VolumeRegistration.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://portugueslab.github.io/VolumeRegistration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://portugueslab.github.io/VolumeRegistration.jl/dev)

Calcium imaging registration pipeline, a rewrite of the [Suite2p](https://github.com/MouseLand/suite2p) approach in Julia.

## Exported functions

- reference creation: `make_reference`
- translation-only registration: `find_translation` and `translate`
- non-rigid piecewise-translation registration: `find_deformation_map` and `apply_deformation_map`

## Usage

```julia

shift = find_translation(moving, reference)

corrected = translate(moving, shift)

```
