use std::env;
use std::fs;
use std::path;
use std::process;
use std::time::{Duration, Instant};

use image::io::Reader as ImageReader;
use pareto_front::ParetoFront;
use rand::rngs::SmallRng;
use rand::SeedableRng;

mod image_ants;
#[allow(dead_code)]
mod image_arithmetic;
mod pareto_pheromones;
mod segment_generation;

static PACKAGE_NAME: &str = env!("CARGO_PKG_NAME");

fn usage(program_name: Option<&str>) {
    println!(
        "Usage: {} [options] <image-path> <results-directory>",
        program_name.unwrap_or(PACKAGE_NAME)
    );
    println!();
    println!(
        "Run an ant-colony algorithm to find a good segmentation of \
              the image at the given path."
    );
    println!();
    println!("Options:");
    println!("  -h, --help          print this help page instead of regular execution");
    println!("  -d, --detailed      export detailed pheromone images from each intermediate step");
    println!("  -e, --eval-steps    consider each intermediate step for evaluation");
    println!("  -o, --objective M|S use either [M]ulti or [S]ingle objective optimization");
    println!("  -s, --seed SEED     use the given integer as a seed, otherwise use a random one");
    println!("  -t, --timeout SECS  stop generating new solutions after SECS seconds");
    println!("  -p, --parallel NUM  run NUM threads in parallel");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program_name: Option<&str> = Some(args[0].as_str());

    let mut detailed = false;
    let mut evaluate_every_step = false;
    let mut rng = SmallRng::from_entropy();
    let mut soft_timeout = None;
    let mut parallelity = None;
    let mut multi_objective = true;

    let usage_and_exit = |problem: Option<&str>| {
        let mut code = 0;
        if problem != None {
            eprintln!("{}", problem.unwrap());
            code = 1;
        }
        usage(program_name);
        process::exit(code);
    };

    let mut parameters = Vec::new();
    let mut i = 1;
    while i < args.len() {
        let arg = &args[i];
        {
            let mut get_parameter = || -> &String {
                if i + 1 >= args.len() {
                    usage_and_exit(Some(format!("Option '{}' expects parameter!", arg).as_str()));
                }
                i += 1;
                return &args[i];
            };
            match arg.as_str() {
                s if !s.starts_with("-") => parameters.push(arg.clone()),
                "-h" | "--help" => usage_and_exit(None),
                "-d" | "--detailed" => detailed = true,
                "-e" | "--eval-steps" | "--evaluate-steps" => evaluate_every_step = true,
                "-o" | "--objective" => match get_parameter().to_lowercase().as_str() {
                    "m" | "multi" | "multiple" => multi_objective = true,
                    "s" | "single" => multi_objective = false,
                    _ => usage_and_exit(Some("Unknown objective!")),
                },
                "-s" | "--seed" => match get_parameter().parse::<u64>() {
                    Ok(seed) => rng = SmallRng::seed_from_u64(seed),
                    _ => usage_and_exit(Some("Seed must be a positive integer!")),
                },
                "-t" | "--timeout" => match get_parameter().parse::<u64>() {
                    Ok(secs) => soft_timeout = Some(Duration::from_secs(secs)),
                    _ => usage_and_exit(Some(
                        "Timeout must be an amount of seconds as a positive integer!",
                    )),
                },
                "-p" | "--parallel" => match get_parameter().parse::<usize>() {
                    Ok(0) => usage_and_exit(Some("Parallelity cannot be 0!")),
                    Ok(num) => parallelity = Some(num),
                    _ => usage_and_exit(Some("Parallelity must a positive integer!")),
                },
                _ => usage_and_exit(Some(format!("Unknown option '{}'!", arg).as_str())),
            }
        }
        i += 1;
    }

    if parameters.len() < 2 {
        usage_and_exit(Some("Too few arguments!"));
    }
    let image_path = &parameters[0];
    let results_path = path::PathBuf::from(&parameters[1]);

    let mut dirbuilder = fs::DirBuilder::new();
    dirbuilder.recursive(true);
    dirbuilder.create(&results_path).unwrap();

    let detailed_path = results_path.join("detailed");
    if detailed {
        dirbuilder.create(&detailed_path).unwrap();
    }

    let input_image = ImageReader::open(image_path).unwrap().decode().unwrap();
    let rgb_image = input_image.to_rgb8();

    let rules = segment_generation::create_rules(&rgb_image, parallelity, multi_objective);

    let start_time = Instant::now();
    let mut attempts = ParetoFront::new();
    loop {
        let mut pheromones = image_ants::initialize_pheromones(&mut rng, &rgb_image, &rules);
        for step in 0..50 {
            image_ants::run_colony_step(&mut rng, &rgb_image, &rules, &mut pheromones);
            if detailed {
                image_ants::visualize_pheromones(&pheromones)
                    .save(&detailed_path.join(format!("{}-step{}.png", attempts.len(), step)))
                    .unwrap();
                if pheromones.len() > 1 {
                    for (i, pheromone) in pheromones.iter().enumerate() {
                        image_ants::visualize_pheromones(std::slice::from_ref(pheromone))
                            .save(&detailed_path.join(format!(
                                "{}-step{}-pheromone{}.png",
                                attempts.len(),
                                step,
                                i
                            )))
                            .unwrap();
                    }
                }
            }
            if evaluate_every_step {
                attempts
                    .push(pareto_pheromones::ParetoPheromones::new(&rgb_image, pheromones.clone()));
            }
        }
        if !evaluate_every_step {
            attempts.push(pareto_pheromones::ParetoPheromones::new(&rgb_image, pheromones));
        }
        if soft_timeout == None || start_time.elapsed() >= soft_timeout.unwrap() {
            break;
        }
    }

    let mut segments_path = results_path.join("type_1_segments");
    dirbuilder.create(&segments_path).unwrap();
    for (i, attempt) in attempts.iter().enumerate() {
        segment_generation::contour_segmententation(&attempt.pheromones, 0.33)
            .save(&segments_path.join(format!("{}-{}.png", i, attempt.stat_info())))
            .unwrap();
    }

    segments_path = results_path.join("type_2_segments");
    dirbuilder.create(&segments_path).unwrap();
    for (i, attempt) in attempts.iter().enumerate() {
        segment_generation::overlayed_contour_segmententation(
            &rgb_image,
            &attempt.pheromones,
            0.33,
        )
        .save(&segments_path.join(format!("{}-{}.png", i, attempt.stat_info())))
        .unwrap();
    }

    segments_path = results_path.join("type_3_segments");
    dirbuilder.create(&segments_path).unwrap();
    for (i, attempt) in attempts.iter().enumerate() {
        segment_generation::colorized_region_segmententation(&rgb_image, &attempt.pheromones, 0.33)
            .0
            .save(&segments_path.join(format!("{}-{}.png", i, attempt.stat_info())))
            .unwrap();
    }
}
