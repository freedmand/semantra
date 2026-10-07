`mlx.metallib` (MLX's compiled Metal kernels, ~180 MB) is generated here by the
mlx-sys build (see `../.cargo/config.toml`) and bundled into the app as a
resource. It is gitignored; this README keeps the bundle's `mlx/*` resource glob
non-empty before the first build.
