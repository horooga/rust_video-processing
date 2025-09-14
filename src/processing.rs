use crate::misc::*;
use image::{ImageBuffer, ImageReader, Rgb};
use ndarray::Array3;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};
use video_rs::{
    encode::{Encoder, Settings},
    time::Time,
};

pub struct DitheringOrdered {}
pub struct Quadrate {}
pub struct Downscale {}
pub struct Upscale {}

pub enum ProcessingArgs {
    DitheringOrdered(f32),
    Quadrate(f32),
    Downscale(u32),
    Upscale(u32),
}

impl ProcessingArgs {
    fn get_dithering_ordered_args(&self) -> Option<f32> {
        if let ProcessingArgs::DitheringOrdered(r) = self {
            Some(*r)
        } else {
            None
        }
    }
    fn get_quadrate_args(&self) -> Option<f32> {
        if let ProcessingArgs::Quadrate(k) = self {
            Some(*k)
        } else {
            None
        }
    }
    fn get_downscale_args(&self) -> Option<u32> {
        if let ProcessingArgs::Downscale(size) = self {
            Some(*size)
        } else {
            None
        }
    }
    fn get_upscale_args(&self) -> Option<u32> {
        if let ProcessingArgs::Upscale(size) = self {
            Some(*size)
        } else {
            None
        }
    }
}

pub trait FrameModProcessing {
    fn process(
        &self,
        img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &ProcessingArgs,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub trait FrameNewProcessing {
    fn process(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &ProcessingArgs,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>>;
}

impl FrameModProcessing for DitheringOrdered {
    fn process(
        &self,
        img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &ProcessingArgs,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let r = args.get_dithering_ordered_args().unwrap();
        let (width, height) = img.dimensions();
        let file = File::open("./palette.txt")
        .expect("Unable to open ./palette.txt file (use get-palette command to generate palette from image)");
        let reader = BufReader::new(file);
        let mut palette = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            let mut line = line.split(' ');
            let r = line.next().unwrap().parse::<u8>().unwrap();
            let g = line.next().unwrap().parse::<u8>().unwrap();
            let b = line.next().unwrap().parse::<u8>().unwrap();
            palette.push([r, g, b]);
        }
        for y in 0..height {
            for x in 0..width {
                let mut rgb = img.get_pixel(x, y).0;
                for i in 0..3 {
                    rgb[i] = clamp_f32_to_u8(
                        rgb[i] as f32
                            + r * (BAYER_16X16[(y % 16) as usize][(x % 16) as usize] as f32
                                / 255.0
                                - 0.5),
                    );
                }
                let curr_color = Rgb(rgb);
                let mut min_dist = i32::MAX;
                let mut min_index = 0;
                for (i, color) in palette.iter().enumerate() {
                    let dist = (curr_color[0] as i32 - color[0] as i32).pow(2)
                        + (curr_color[1] as i32 - color[1] as i32).pow(2)
                        + (curr_color[2] as i32 - color[2] as i32).pow(2);
                    if dist < min_dist {
                        min_dist = dist;
                        min_index = i;
                    }
                }
                img.put_pixel(
                    x,
                    y,
                    Rgb([
                        palette[min_index][0],
                        palette[min_index][1],
                        palette[min_index][2],
                    ]),
                );
            }
        }
        Ok(())
    }
}

impl FrameNewProcessing for Quadrate {
    fn process(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &ProcessingArgs,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
        let k = args.get_quadrate_args().unwrap();
        let (width, height) = img.dimensions();
        let new_height = (height as f32 * k) as u32;
        let mut new_img = ImageBuffer::new(new_height, new_height);
        let width_offset = (width - new_height) / 2;
        for y in 0..new_height {
            for x in 0..new_height {
                let rgb = img.get_pixel(x + width_offset, y).0;
                new_img.put_pixel(x, y, Rgb(rgb));
            }
        }
        Ok(new_img)
    }
}

impl FrameNewProcessing for Downscale {
    fn process(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &ProcessingArgs,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
        let k = args.get_downscale_args().unwrap();
        let (width, height) = img.dimensions();
        let new_width = width / k;
        let new_height = height / k;
        let mut xc;
        let mut new_img = ImageBuffer::new(new_width, new_height);
        for (yc, y) in (0..height).step_by(k as usize).enumerate() {
            xc = 0;
            for x in (0..width).step_by(k as usize) {
                new_img.put_pixel(
                    xc.min(new_width - 1),
                    (yc as u32).min(new_height - 1),
                    *img.get_pixel(x.min(width - 1), y.min(height - 1)),
                );
                xc += 1;
            }
        }
        Ok(new_img)
    }
}

impl FrameNewProcessing for Upscale {
    fn process(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &ProcessingArgs,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
        let k = args.get_upscale_args().unwrap();
        let (width, height) = img.dimensions();
        let new_width = width * k;
        let new_height = height * k;
        let mut new_img = ImageBuffer::new(new_width, new_height);
        for y in 0..new_height {
            for x in 0..new_width {
                new_img.put_pixel(x, y, *img.get_pixel(x / k, y / k));
            }
        }
        Ok(new_img)
    }
}

pub fn process_frames_mod<T: FrameModProcessing>(
    mod_processing: &T,
    args: &ProcessingArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut frame_paths: Vec<PathBuf> = std::fs::read_dir("./frames")?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let file_name = path.file_name()?.to_str()?;
            if file_name.starts_with("frame") && file_name.ends_with(".png") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    frame_paths.sort_by_key(|path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem_str| {
                stem_str
                    .strip_prefix("frame")
                    .and_then(|num_str| num_str.parse::<usize>().ok())
            })
            .unwrap_or(0)
    });
    let mut progress_bar = ProgressBar::new(frame_paths.len());

    let info_img = ImageReader::open("./frames/frame0.png")?
        .decode()?
        .to_rgb8();
    let (width, height) = info_img.dimensions();

    for path in frame_paths {
        let mut img = ImageReader::open(path.clone())?.decode()?.to_rgb8();

        if img.width() != width || img.height() != height {
            panic!("Frame size mismatch at {:?}", path);
        }
        progress_bar.step();
        mod_processing.process(&mut img, args)?;
        img.save(path.to_str().unwrap()).unwrap();
    }

    Ok(())
}

pub fn process_frames_new<T: FrameNewProcessing>(
    new_processing: &T,
    args: &ProcessingArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut frame_paths: Vec<PathBuf> = std::fs::read_dir("./frames")?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let file_name = path.file_name()?.to_str()?;
            if file_name.starts_with("frame") && file_name.ends_with(".png") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    frame_paths.sort_by_key(|path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem_str| {
                stem_str
                    .strip_prefix("frame")
                    .and_then(|num_str| num_str.parse::<usize>().ok())
            })
            .unwrap_or(0)
    });
    let mut progress_bar = ProgressBar::new(frame_paths.len());

    let info_img = ImageReader::open("./frames/frame0.png")?
        .decode()?
        .to_rgb8();
    let (width, height) = info_img.dimensions();

    for path in frame_paths {
        let img = ImageReader::open(path.clone())?.decode()?.to_rgb8();

        if img.width() != width || img.height() != height {
            panic!("Frame size mismatch at {:?}", path);
        }
        progress_bar.step();
        let new_img = new_processing.process(&img, args)?;
        new_img.save(path.to_str().unwrap()).unwrap();
    }

    Ok(())
}

pub fn frames_to_video(video_path: &str, fps: usize) -> Result<(), Box<dyn std::error::Error>> {
    video_rs::init()?;

    let mut frame_paths: Vec<PathBuf> = std::fs::read_dir("./frames")?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let file_name = path.file_name()?.to_str()?;
            if file_name.starts_with("frame") && file_name.ends_with(".png") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    frame_paths.sort_by_key(|path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem_str| {
                stem_str
                    .strip_prefix("frame")
                    .and_then(|num_str| num_str.parse::<usize>().ok())
            })
            .unwrap_or(0)
    });
    let mut progress_bar = ProgressBar::new(frame_paths.len());

    let info_img = ImageReader::open("./frames/frame0.png")?
        .decode()?
        .to_rgb8();
    let (width, height) = info_img.dimensions();

    let settings = Settings::preset_h264_yuv420p(width as usize, height as usize, false);
    let mut encoder = Encoder::new(Path::new(video_path), settings)?;
    let frame_duration = Time::from_nth_of_a_second(fps);
    let mut position = Time::zero();
    for path in frame_paths {
        let img = ImageReader::open(path.clone())?.decode()?.to_rgb8();
        if img.width() != width || img.height() != height {
            return Err(Box::new(std::io::Error::other(format!(
                "Frame size mismatch at {:?}",
                path
            ))));
        }
        let arr: Array3<u8> =
            Array3::from_shape_vec((height as usize, width as usize, 3), img.into_raw())?;
        encoder.encode(&arr, position)?;
        position = position.aligned_with(frame_duration).add();
        progress_bar.step();
    }

    encoder.finish()?;
    Ok(())
}

pub fn video_to_frames(video_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    video_rs::init()?;
    let absolute_path = Path::new(video_path).canonicalize()?;
    let mut decoder = video_rs::decode::Decoder::new(absolute_path.clone())?;
    let mut i = 0;
    let mut frame_count = 0;
    for result in decoder.decode_iter() {
        match &result {
            Ok(_) => frame_count += 1,
            Err(video_rs::error::Error::DecodeExhausted) => break,
            Err(e) => return Err(Box::new(e.clone())),
        }
    }

    decoder = video_rs::decode::Decoder::new(absolute_path)?;
    let mut progress_bar = ProgressBar::new(frame_count);
    for frame_result in decoder.decode_iter() {
        match frame_result {
            Ok((_timestamp, frame)) => {
                let (height, width, channels) = frame.dim();
                if channels == 3 {
                    let raw_pixels = frame.into_raw_vec_and_offset();
                    let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
                        ImageBuffer::from_vec(width as u32, height as u32, raw_pixels.0).unwrap();
                    image.save(format!("./frames/frame{}.png", i)).unwrap();
                    i += 1;
                }
                progress_bar.step();
            }
            Err(video_rs::error::Error::DecodeExhausted) => break, // end of decoding
            Err(e) => return Err(Box::new(e)),
        }
    }
    Ok(())
}
