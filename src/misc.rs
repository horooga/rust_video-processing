use image::{ImageBuffer, Rgb};
use itertools::Itertools;
use std::{fs::File, io::Write};

pub const PROGRESS_BAR_WIDTH: usize = 50;

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

pub struct ProgressBar {
    pub last_step: usize,
    current_step: usize,
}

impl ProgressBar {
    pub fn new(last_step: usize) -> Self {
        Self {
            last_step,
            current_step: 0,
        }
    }

    pub fn step(&mut self) {
        self.current_step = (self.current_step + 1).min(self.last_step);
        let percent = self.current_step as f32 / self.last_step as f32 * 100.0;
        let done_width = (percent / 100.0 * PROGRESS_BAR_WIDTH as f32) as usize;

        print!("\r{}", " ".repeat(PROGRESS_BAR_WIDTH));
        print!(
            "\rProcessing... [{}{}] ({}%)",
            "|".repeat(done_width),
            " ".repeat(PROGRESS_BAR_WIDTH - done_width),
            percent as usize
        );
        use std::io::{Write, stdout};
        stdout().flush().unwrap();
    }
}

struct Bucket {
    pixels: Vec<Rgb<u8>>,
}

impl Bucket {
    fn new(pixels: Vec<Rgb<u8>>) -> Self {
        Self { pixels }
    }

    fn largest_range_channel(&self) -> usize {
        let (min_r, max_r) = self
            .pixels
            .iter()
            .map(|p| p[0])
            .minmax()
            .into_option()
            .unwrap();
        let (min_g, max_g) = self
            .pixels
            .iter()
            .map(|p| p[1])
            .minmax()
            .into_option()
            .unwrap();
        let (min_b, max_b) = self
            .pixels
            .iter()
            .map(|p| p[2])
            .minmax()
            .into_option()
            .unwrap();

        let range_r = max_r - min_r;
        let range_g = max_g - min_g;
        let range_b = max_b - min_b;

        if range_r >= range_g && range_r >= range_b {
            0
        } else if range_g >= range_r && range_g >= range_b {
            1
        } else {
            2
        }
    }

    fn split(self) -> (Self, Self) {
        let ch = self.largest_range_channel();
        let mut pixels = self.pixels;
        pixels.sort_unstable_by_key(|p| p[ch]);

        let mid = pixels.len() / 2;
        let lower = pixels[..mid].to_vec();
        let upper = pixels[mid..].to_vec();

        (Self::new(lower), Self::new(upper))
    }

    fn average_color(&self) -> Rgb<u8> {
        let len = self.pixels.len() as u32;
        let (r_sum, g_sum, b_sum) =
            self.pixels
                .iter()
                .fold((0u32, 0u32, 0u32), |(r_acc, g_acc, b_acc), p| {
                    (
                        r_acc + p[0] as u32,
                        g_acc + p[1] as u32,
                        b_acc + p[2] as u32,
                    )
                });
        Rgb([
            (r_sum / len) as u8,
            (g_sum / len) as u8,
            (b_sum / len) as u8,
        ])
    }

    fn variance(&self) -> u32 {
        let len = self.pixels.len() as u32;
        if len == 0 {
            return 0;
        }

        let avg = self.average_color();
        self.pixels
            .iter()
            .map(|p| {
                let dr = p[0] as i32 - avg[0] as i32;
                let dg = p[1] as i32 - avg[1] as i32;
                let db = p[2] as i32 - avg[2] as i32;
                (dr * dr + dg * dg + db * db) as u32
            })
            .sum::<u32>()
            / len
    }
}

pub fn clamp_f32_to_u8(val: f32) -> u8 {
    if val < 0.0 {
        0
    } else if val > 255.0 {
        255
    } else {
        val as u8
    }
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
