# Setup

1. Have the rust programming language installed: https://www.rust-lang.org/tools/install
   I'm using version `1.66.1`.
2. Automatically build and run the program using `cargo run`.

# Usage

You can (build and) run the program directly through `cargo run --release --`:

```bash
$ # Run the program on image 353013 and save the segmentations in a newly created folder. Limit the amount of concurrent threads to 4.
$ cargo run --release -- -p 4 training_images/353013/Test\ image.jpg results/353013
$ # Also output visualisations of the pheromone layers after each generation and repeatedly start new within a "soft" timeout of 60 seconds.
$ cargo run --release -- -d -t 60 -p 4 training_images/353013/Test\ image.jpg results/353013
```

Program options can be viewed using `cargo run --release -- -h`!

# Project Structure

The project is split into three main modules:
- `image_ants.rs`: Provides a framework to run general ACO on images with variable rules and pheromone layers.
- `segment_generation.rs`: Concrete implementation for rules for ACO image segmentation. Also includes code to generate segmentations (type-1, type-2, and another one) from pheromone layers.
- `image_arithmetic`: Utilities to perform different operations on images.
  - `color_distances.rs`: A variety of different distance functions for the RGB color space. I used manhattan distance preferably, because it is computationally more efficient.
  - `types.rs`: Includes the `Point` type, a 2D vector with some utility functions.
  - `segments.rs`: Includes calculation for all three objective functions, and for computation of segments from a contour image.
- `main.rs`: The entry point to the program. Handles command line input.

Lastly, `pareto_pheromones.rs` includes a struct to make pheromone layers pareto-comparable.

# Idea behind Multi-Objective Ant Colony

I came up with my own approach:

1. Randomly spawn ants. Give each a random position as a target.
2. Each ant:
  - Selects a neighbouring pixel and walks to it, leaving three pheromones behind.
  - The selection is random and weighted according to:
    - the strength of pheromones
    - direction towards their target
    - image data (e.g. similarity of colors)
    - avoiding visiting the same pixel multiple times
3. Repeat step 2 until all ants reached their target. Then have the ants return to their initial position, preferably on a different path.
4. Update pheromones on path according to local rules (e.g. deposit pheromone).
5. Form segments from the pheromones and update the pheromones according to global rules (overall_deviation, edge_value & connectivity_measure).
6. Restart at step 1 until convergence / end condition is reached.

By objective:
- edge_value is maximized locally to encourage edges. Deposited pheromone is important to kick-start the edge-detection.
- connectivity_measure is minimized locally to encourage connected segments. Becomes more meaningful after clear edges have formed.
- deviation was hard to integrate. I would have liked to decrease edges of segments with high deviation, but with my data-structures it is hard to find the edge of a segment. As such I found it too difficult to integrate it as an objective.

# Differences to Weighted-Sum Single-Objective Ant Colony

- The single objective ant colony only has one pheromone layer, which jointly handles all objectives.
- The weighting of the objectives is not exactly the same, but similar.

# Discussion of MOACO vs. SOACO

The multiple layers of MOACO allowed to optimise the two objectives separately, which lead to more connected segments overall.
With SOACO it was hard to find the right balance so that one objective did not overtake the pheromone and irreversibly ruined information important for the other objective.

# Additional Implementation Notes

- Pheromone layers are (unnormalized) floating-point greyscale images.
- Pheromones are additive; each layer is added to the previous layer to influence the weight for the next step an ant takes. This allows weighing different pheromones' importance. Other factors are multiplicative.
- Have a number of ants running in parallel (for threads).
  They will not update the world state for the other parallel ants, but their pheromones will be combined at the end.
- Have a number of ants that need to be run before global rules are evaluated.
- After global rules are evaluated, remember solution in a Pareto front.
- If a soft timeout is given, the simulation is restarted from time to time to generate new solutions.
