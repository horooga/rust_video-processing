Cli tool for video processing

**Examples**

```bash
./target/release/rust_video-processing get-frames input.mp4 # get video's frames to ./frames
./target/release/rust_video-processing dither-ordered 175 # ordered dithering with 16x16 Bayer matrix and 175 koefficent of ./frames
./target/release/rust_video-processing merge-frames output.mp4 # merge ./frames to output.mp4
```
