#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
from tqdm import tqdm


class ParticleCollisionData:
    def __init__(
        self,
        num_particles: int = 2,
        num_colors: int = 5,
        grid_size: int = 16,
        speed: float = 0.5,
        color_change_prob: float = 0.05,
        easy_q: float = 0.4,
    ):
        self.num_particles = num_particles
        self.num_colors = num_colors
        self.grid_size = grid_size
        self.speed = speed
        self.color_change_prob = color_change_prob
        self.easy_q = easy_q

        self.particle_locations = torch.rand((num_particles, 2)) * (
            grid_size - 1
        )
        self.particle_grid_locations = torch.round(self.particle_locations).long()
        self.particle_velocities = torch.randn((num_particles, 2)) * speed
        self.particle_colors = torch.randint(0, num_colors, (num_particles,))
        # only keep the last one of each color, record position and timestep
        self.crossing_history = torch.zeros((num_colors, num_colors, 3)).long()
        self.most_recent_crossing = (-1, -1)
        self.step = 0

    def check_crossings(self):
        for p in range(self.num_particles):
            for q in range(p + 1, self.num_particles):
                ploc = self.particle_grid_locations[p]
                qloc = self.particle_grid_locations[q]
                if ploc[0] == qloc[0] and ploc[1] == qloc[1]:
                    cp = self.particle_colors[p]
                    cq = self.particle_colors[q]
                    self.crossing_history[cp, cq, :2] = ploc
                    self.crossing_history[cp, cq, 2] = self.step
                    self.most_recent_crossing = (cp, cq)

    def check_bounds(self, z, max_H=None, max_W=None):
        max_H = max_H or self.grid_size
        max_W = max_W or self.grid_size
        return z[0] > 0 and z[0] < max_H - 1 and z[1] > 0 and z[1] < max_W - 1

    def change_direction(self, particle):
        self.particle_velocities[particle] = torch.randn(2) * self.speed

    def move_particles(self):
        self.step += 1
        for p in range(self.num_particles):
            while not self.check_bounds(
                self.particle_locations[p] + self.particle_velocities[p]
            ):
                self.change_direction(p)
            self.particle_locations[p] += self.particle_velocities[p]
            self.particle_grid_locations = torch.round(self.particle_locations).long()
            if torch.rand(1) < self.color_change_prob:
                self.particle_colors[p] = torch.randint(0, self.num_colors, (1,))

    def bin_location(self, h, w):
        return 2 * round((h.item() + 0.5) / self.grid_size) + round(
            (w.item() + 0.5) / self.grid_size
        )

    def generate_sample(self, l):
        dlength = self.num_particles * 3 + 2
        L = dlength * l
        x = torch.zeros(L, dtype=torch.long)
        y = torch.zeros(L, dtype=torch.long)
        for i in tqdm(range(l)):
            self.move_particles()
            self.check_crossings()
            for p in range(self.num_particles):
                x[i * dlength + 3 * p : i * dlength + 3 * p + 2] = (
                    self.particle_grid_locations[p]
                    + torch.LongTensor((0, self.grid_size))
                )
                x[i * dlength + 3 * p + 2] = (
                    self.particle_colors[p] + 2 * self.grid_size
                )

            if torch.rand(1) < self.easy_q:
                cp, cq = self.most_recent_crossing
            else:
                cp, cq = torch.randint(0, self.num_colors, (2,))

            x[(i + 1) * dlength - 2 : (i + 1) * dlength] = (
                torch.LongTensor((cp, cq))
                + 2 * self.grid_size
                + self.num_colors
            )
            # will give junk in v. beginning, don't count those
            if self.crossing_history[cp, cq, 2] > 0:
                y[(i + 1) * dlength - 1] = (
                    self.bin_location(
                        self.crossing_history[cp, cq, 0],
                        self.crossing_history[cp, cq, 1],
                    )
                    + 1
                )
        return x, y


def gen_data(args, file_path: str, nsteps: int):
    if os.path.exists(file_path):
        print(f"{file_path} already exists! Skipping generation.")
        return
    print(f"Generating {file_path}")

    data = ParticleCollisionData(
        num_particles=args.num_particles,
        num_colors=args.num_colors,
        grid_size=args.grid_size,
        speed=args.speed,
        color_change_prob=args.color_change_prob,
        easy_q=args.easy_q,
    )

    f = open(file_path, "w")
    g = open(file_path + ".labels", "w")
    x, y = data.generate_sample(nsteps)
    xprint = " ".join([str(c) for c in x.tolist()])
    yprint = " ".join([str(c) for c in y.tolist()])
    f.write(xprint)
    g.write(yprint)
    f.close()
    g.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--num-particles", type=int, default=2)
    parser.add_argument("--num-colors", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--color-change-prob", type=float, default=0.05)
    parser.add_argument("--easy-q", type=float, default=0.4)
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Creating directory: {args.path}")
        os.mkdir(args.path)
    gen_data(args, os.path.join(args.path, "train.txt"), int(50e6))
    gen_data(args, os.path.join(args.path, "valid.txt"), int(5e6))
    gen_data(args, os.path.join(args.path, "test.txt"), int(5e6))
