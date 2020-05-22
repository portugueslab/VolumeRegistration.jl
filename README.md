# VolumeRegistration.jl

[![Build Status](https://travis-ci.com/portugueslab/VolumeRegistration.jl.svg?branch=master)](https://travis-ci.com/portugueslab/VolumeRegistration.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://vilim.github.io/VolumeRegistration.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vilim.github.io/VolumeRegistration.jl/dev)

Calcium imaging registration pipeline, a rewrite of the [Suite2p](https://github.com/MouseLand/suite2p) approach in Julia.

# Usage

```julia

shift = find_translation(moving, reference)

translated = translate()

```