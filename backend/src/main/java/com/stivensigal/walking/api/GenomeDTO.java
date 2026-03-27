package com.stivensigal.walking.api;

import jakarta.validation.constraints.NotNull;

public record GenomeDTO(
    @NotNull double[] amplitudes,
    @NotNull double[] omegas,
    @NotNull double[] phases
) {}

