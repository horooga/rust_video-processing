mod processing;
use image::open;
use processing::*;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage:
    - get-palette [image_path] [palette_size(2 as min)]: extract palette from the image
    - get-frames [video_path]: extract frames to ./frames directory from the video
    - get-frames-decrypt [video_path] [base64url_key]: extract-decrypt frames to ./frames directory from the video

    - upscale [factor]: upscale the image by factor
    - downscale [factor]: downscale the image by factor

    Following commands works with ./frame dir which contains frame{{n}}.png files
    - merge [video_path] [fps]: merge frames to a video. Video dimensions are defined by a first frame
    - merge-encrypt [video_path] [fps] [base64url_key]: merge-decrypt frames to a video. Video dimensions are defined by a first frame
    - dither-ordered [koefficient(0-256)]: ordered dithering of frames with ./palette.txt and 16x16 Bayer matrix
    - posterize [koefficient]: posterize the image
    - quadrate [koefficient]: quadrate the dimensions of the album oriented frames (keep middle (n*k) x (n*k) square from n x m rectangle)
    - psychodelize [hue_shift] [amplitude] [frequency] [hue_shift_speed] [amplitude_speed] [frequency_speed]: psychodelize the image");
        std::process::exit(1);
    }
    if args[1] == "get-palette" {
        let img_path = &args[2];
        let img_bind = open(img_path).unwrap();
        let img = img_bind.to_rgb8();
        let palette_size = if args.len() > 3 {
            args[3].parse::<usize>().unwrap()
        } else {
            0
        };
        get_palette(&img, palette_size);
    } else if args[1] == "get-frames" {
        let _ = fs::remove_dir("./frames");
        let _ = fs::create_dir("./frames");
        video_to_frames(args[2].as_str()).unwrap();
    } else if args[1] == "merge-frames" {
        frames_to_video(args[2].as_str(), args[3].parse::<usize>().unwrap()).unwrap();
    } else if args[1] == "dither-ordered" {
        process_frames_mod(
            &DitheringOrdered {},
            &mut FrameProcessingArgs::DitheringOrdered(args[2].parse::<f32>().unwrap()),
            None,
        )
        .unwrap();
    } else if args[1] == "posterize" {
        process_frames_mod(
            &Posterize {},
            &mut FrameProcessingArgs::Posterize(args[2].parse::<u32>().unwrap()),
            None,
        )
        .unwrap()
    } else if args[1] == "psychodelize" {
        process_frames_mod(
            &Psychodelize {},
            &mut FrameProcessingArgs::Psychodelize((
                args[2].parse::<f32>().unwrap(),
                args[3].parse::<f32>().unwrap(),
                args[4].parse::<f32>().unwrap(),
            )),
            Some(&VideoProcessingArgs::Psychodelize((
                args[5].parse::<f32>().unwrap(),
                args[6].parse::<f32>().unwrap(),
                args[7].parse::<f32>().unwrap(),
            ))),
        )
        .unwrap();
    } else if args[1] == "quadrate" {
        process_frames_new(
            &Quadrate {},
            &mut FrameProcessingArgs::Quadrate(args[2].parse::<f32>().unwrap()),
            None,
        )
        .unwrap();
    } else if args[1] == "downscale" {
        process_frames_new(
            &Downscale {},
            &mut FrameProcessingArgs::Downscale(args[2].parse::<u32>().unwrap()),
            None,
        )
        .unwrap();
    } else if args[1] == "upscale" {
        process_frames_new(
            &Upscale {},
            &mut FrameProcessingArgs::Upscale(args[2].parse::<u32>().unwrap()),
            None,
        )
        .unwrap();
    }
}
