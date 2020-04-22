#!/usr/bin/python
# -*- encoding: utf-8 -*-
from cProfile import Profile
import pstats

from PIL import Image
from memory_profiler import profile

import faceutils as futils
import makeup

images = []
with Image.open(f'refs/0.jpeg') as image:
    images.append(makeup.preprocess(image)[1])
with Image.open(f'refs/1.jpeg') as image:
    images.append(makeup.preprocess(image)[1])


def demo():
    return makeup.solver.test(*images[0], *images[1])


if __name__ == '__main__':
    # analyze memory
    demo()

    # analyze runtime
    # profiler = Profile()
    # profiler.runcall(demo)
    # stats = pstats.Stats(profiler)
    # stats.strip_dirs()
    # stats.sort_stats('time')
    # stats.print_stats()
