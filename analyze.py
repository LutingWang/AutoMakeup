#!/usr/bin/python
# -*- encoding: utf-8 -*-
from cProfile import Profile
import pstats

from PIL import Image
from memory_profiler import profile

import faceutils as futils
import makeup


def runtime_demo():
    images = []
    with Image.open(f'refs/0.png') as image:
        images.append(makeup.preprocess(image))
    with Image.open(f'refs/1.png') as image:
        images.append(makeup.preprocess(image))
    image = makeup.solver.test(*images[0], *images[1])
    return futils.fpp.beautify(image)


def runtime_analyzer():
    profiler = Profile()
    profiler.runcall(runtime_demo)
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('time')
    stats.print_stats()


@profile
def memory_demo():
    images = []
    with Image.open(f'refs/0.png') as image:
        images.append(makeup.preprocess(image))
    with Image.open(f'refs/1.png') as image:
        images.append(makeup.preprocess(image))
    image = makeup.solver.test(*images[0], *images[1])
    return futils.fpp.beautify(image)


if __name__ == '__main__':
    # memory_demo()
    runtime_analyzer()
