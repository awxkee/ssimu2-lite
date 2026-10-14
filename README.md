# ssimu2-lite

A dependency-free Rust implementation of the [SSIMULACRA 2](https://github.com/cloudinary/ssimulacra2)
perceptual image quality metric.

Scores: 100 is identical, lower is worse. Roughly 90 is visually lossless, 70
high quality, 50 medium, 30 low. Very poor images can score below 0.

## API

```rust
use ssimu2_lite::RgbImage;

let score = ssimu2_lite::compare(
    RgbImage::srgb8(&reference_rgb, width, height, width * 3)?,
    RgbImage::srgb8(&test_rgb, width, height, width * 3)?,
)?;
```
## License

Licensed under either of

- BSD-3-Clause ([LICENSE.md](LICENSE.md))
- Apache License, Version 2.0 ([LICENSE-APACHE.md](LICENSE-APACHE.md))

at your option.
