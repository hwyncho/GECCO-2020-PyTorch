"""A package containing Generator and Discriminator."""
from __future__ import absolute_import, division, print_function

from .class_discriminator import Discriminator
from .class_generator import Generator
from .functions import predict
from .functions import train

__all__ = ["Discriminator", "Generator", "predict", "train"]
