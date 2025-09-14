use image::{ImageBuffer, ImageReader, Rgb};
use ndarray::Array3;
use rust_image_codec::*;
use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};
use video_rs::{
    encode::{Encoder, Settings},
    time::Time,
};

pub const BAYER_16X16: [[u8; 16]; 16] = [
    [
        0, 191, 48, 239, 12, 203, 60, 251, 3, 194, 51, 242, 15, 206, 63, 254,
    ],
    [
        127, 64, 175, 112, 139, 76, 187, 124, 130, 67, 178, 115, 142, 79, 190, 127,
    ],
    [
        32, 223, 16, 207, 44, 235, 28, 219, 35, 226, 19, 210, 47, 238, 31, 222,
    ],
    [
        159, 96, 143, 80, 171, 108, 155, 92, 162, 99, 146, 83, 174, 111, 158, 95,
    ],
    [
        8, 199, 56, 247, 4, 195, 52, 243, 11, 202, 59, 250, 7, 198, 55, 246,
    ],
    [
        135, 72, 183, 120, 131, 68, 179, 116, 138, 75, 186, 123, 134, 71, 182, 119,
    ],
    [
        40, 231, 24, 215, 36, 227, 20, 211, 43, 234, 27, 218, 39, 230, 23, 214,
    ],
    [
        167, 104, 151, 88, 163, 100, 147, 84, 170, 107, 154, 91, 166, 103, 150, 87,
    ],
    [
        2, 193, 50, 241, 14, 205, 62, 253, 1, 192, 49, 240, 13, 204, 61, 252,
    ],
    [
        129, 66, 177, 114, 141, 78, 189, 126, 128, 65, 176, 113, 140, 77, 188, 125,
    ],
    [
        34, 225, 18, 209, 46, 237, 30, 221, 33, 224, 17, 208, 45, 236, 29, 220,
    ],
    [
        161, 98, 145, 82, 173, 110, 157, 94, 160, 97, 144, 81, 172, 109, 156, 93,
    ],
    [
        10, 201, 58, 249, 6, 197, 54, 245, 9, 200, 57, 248, 5, 196, 53, 244,
    ],
    [
        137, 74, 185, 122, 133, 70, 181, 118, 136, 73, 184, 121, 132, 69, 180, 117,
    ],
    [
        42, 233, 26, 217, 38, 229, 22, 213, 41, 232, 25, 216, 37, 228, 21, 212,
    ],
    [
        169, 106, 153, 90, 165, 102, 149, 86, 168, 105, 152, 89, 164, 101, 148, 85,
    ],
];

pub fn clamp_f32_to_u8(val: f32) -> u8 {
    if val < 0.0 {
        0
    } else if val > 255.0 {
        255
    } else {
        val as u8
    }
}

fn shift_hue(img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, shift: f32) {
    fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
        let r = r as f32 / 255.0;
        let g = g as f32 / 255.0;
        let b = b as f32 / 255.0;
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;
        let v = max;
        let s = if max == 0.0 { 0.0 } else { delta / max };
        let h = if delta == 0.0 {
            0.0
        } else if max == r {
            60.0 * (((g - b) / delta) % 6.0)
        } else if max == g {
            60.0 * (((b - r) / delta) + 2.0)
        } else {
            60.0 * (((r - g) / delta) + 4.0)
        };
        (h, s, v)
    }
    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
        let c = v * s;
        let h_prime = h / 60.0;
        let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
        let m = v - c;

        let (r1, g1, b1) = match h_prime.floor() as i32 % 6 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 => (c, 0.0, x),
            _ => (0.0, 0.0, 0.0),
        };

        (
            ((r1 + m) * 255.0).round() as u8,
            ((g1 + m) * 255.0).round() as u8,
            ((b1 + m) * 255.0).round() as u8,
        )
    }
    for p in img.pixels_mut() {
        let (mut h, s, v) = rgb_to_hsv(p[0], p[1], p[2]);
        h = (h + shift) % 360.0;
        let (r, g, b) = hsv_to_rgb(h, s, v);
        *p = Rgb([r, g, b]);
    }
}

fn accent_edges(image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (w, h) = image.dimensions();
    let mut out = image.clone();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = |dx: i32, dy: i32| {
                let mut sum = 0i32;
                for ky in -1..=1 {
                    for kx in -1..=1 {
                        let v = image
                            .get_pixel((x as i32 + dx + kx) as u32, (y as i32 + dy + ky) as u32)[0]
                            as i32;
                        let coeff = if kx == 1 {
                            1
                        } else if kx == -1 {
                            -1
                        } else {
                            0
                        };
                        sum += v * coeff;
                    }
                }
                sum
            };
            let magnitude = ((gx(0, 0).abs() + gx(0, 0).abs()) / 2) as u8;
            if magnitude > 100 {
                out.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
    }
    out
}

fn wave_distort(
    img: ImageBuffer<Rgb<u8>, Vec<u8>>,
    amplitude: f32,
    frequency: f32,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (w, h) = img.dimensions();
    let mut out = img.clone();
    for y in 0..h {
        for x in 0..w {
            let offset_x = ((amplitude * ((y as f32 * frequency).sin())) as i32).max(0);
            let src_x = ((x as i32 + offset_x) % w as i32).unsigned_abs();
            out.put_pixel(x, y, *img.get_pixel(src_x, y));
        }
    }
    out
}

pub fn get_palette(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, palette_size: usize) {
    let (width, height) = img.dimensions();
    let mut file = File::create("./palette.txt").expect("Unable to create palette file");
    let mut palette = Vec::new();
    if palette_size == 0 {
        for y in 0..height {
            for x in 0..width {
                let curr_color = img.get_pixel(x, y).0;
                if !palette.contains(&curr_color) {
                    palette.push(curr_color);
                    let _ = writeln!(
                        file,
                        "{} {} {}",
                        curr_color[0], curr_color[1], curr_color[2]
                    );
                }
            }
        }
        return;
    }
    let mut buckets = vec![Bucket::new(
        img.pixels().cloned().collect::<Vec<Rgb<u8>>>().to_vec(),
    )];
    while buckets.len() < palette_size {
        if let Some((idx, _)) = buckets
            .iter()
            .enumerate()
            .max_by_key(|&(_, b)| b.variance())
        {
            let bucket = buckets.swap_remove(idx);
            if bucket.pixels.len() <= 1 {
                buckets.push(bucket);
                break;
            }

            let (b1, b2) = bucket.split();
            buckets.push(b1);
            buckets.push(b2);
        } else {
            break;
        }
    }

    for b in &buckets {
        let color = b.average_color();
        let _ = writeln!(file, "{} {} {}", color[0], color[1], color[2]);
    }
}

pub struct DitheringOrdered {}
pub struct Posterize {}
pub struct Quadrate {}
pub struct Downscale {}
pub struct Upscale {}
pub struct Psychodelize {}

pub enum ProcessingArgs {
    DitheringOrdered(f32),
    Posterize(u32),
    Quadrate(f32),
    Downscale(u32),
    Upscale(u32),
    Psychodelize((f32, f32, f32)),
}

impl ProcessingArgs {
    fn get_dithering_ordered_args(&self) -> Option<f32> {
        if let ProcessingArgs::DitheringOrdered(k) = self {
            Some(*k)
        } else {
            None
        }
    }
    fn get_posterize_args(&self) -> Option<u32> {
        if let ProcessingArgs::Posterize(amount) = self {
            Some(*amount)
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
    fn get_psychodelize_args(&self) -> Option<(f32, f32, f32)> {
        if let ProcessingArgs::Psychodelize((shift, attitude, frequency)) = self {
            Some((*shift, *attitude, *frequency))
        } else {
            None
        }
    }
    fn set_psychodelize_args(&mut self, args: (f32, f32, f32)) {
        *self = ProcessingArgs::Psychodelize(args);
    }
}

pub trait FrameModProcessing {
    fn process(
        &self,
        img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &mut ProcessingArgs,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

pub trait FrameNewProcessing {
    fn process(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &mut ProcessingArgs,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>>;
}

impl FrameModProcessing for DitheringOrdered {
    fn process(
        &self,
        img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &mut ProcessingArgs,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let r = args.get_dithering_ordered_args().unwrap();
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
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let mut rgb = *pixel;

            for i in 0..3 {
                rgb[i] = clamp_f32_to_u8(
                    rgb[i] as f32
                        + r * (BAYER_16X16[(y % 16) as usize][(x % 16) as usize] as f32 / 255.0
                            - 0.5),
                );
            }
            let curr_color = *pixel;
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
            *pixel = Rgb([
                palette[min_index][0],
                palette[min_index][1],
                palette[min_index][2],
            ]);
        }

        Ok(())
    }
}

impl FrameModProcessing for Posterize {
    fn process(
        &self,
        img: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &mut ProcessingArgs,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (_x, _y, pixel) in img.enumerate_pixels_mut() {
            let step = 255 / args.get_posterize_args().unwrap();
            for i in 0..3 {
                pixel[i] = (pixel[i] as u32 / step * step) as u8;
            }
        }

        Ok(())
    }
}

impl FrameNewProcessing for Quadrate {
    fn process(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &mut ProcessingArgs,
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
        args: &mut ProcessingArgs,
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
        args: &mut ProcessingArgs,
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

impl FrameNewProcessing for Psychodelize {
    fn process(
        &self,
        img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        args: &mut ProcessingArgs,
    ) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
        let (shift, amplitude, frequency) = args.get_psychodelize_args().unwrap();
        let mut output_img = img.clone();

        if shift != 0.0 {
            shift_hue(&mut output_img, shift);
        }
        output_img = wave_distort(output_img.clone(), amplitude, frequency);
        //args.set_psychodelize_args((shift + 10.0, (amplitude + 1.0) % 50.0, (frequency + 0.05) % 1.0));

        Ok(accent_edges(&output_img))
    }
}

pub fn process_frames_mod<T: FrameModProcessing>(
    mod_processing: &T,
    args: &mut ProcessingArgs,
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
    args: &mut ProcessingArgs,
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
