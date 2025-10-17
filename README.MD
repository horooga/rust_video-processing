Cli tool for video processing

**Examples**

```bash
./target/release/rust_video-processing get-frames input.mp4 # get video's frames to ./frames
./target/release/rust_video-processing psychodelize 0 0 0 10 5 0.05 # psychodelize with 0 initial vals, 10 speed degree hue shift, 5 amplitude and 0.05 frequency for wave distortion
./target/release/rust_video-processing dither-ordered 175 # ordered dithering with 16x16 Bayer matrix and 175 koefficent of ./frames
./target/release/rust_video-processing merge-frames output.mp4 # merge ./frames to output.mp4
```
