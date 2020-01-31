import numpy as np
from typing import List, Dict, Tuple, NewType

IntVec = List[int]

class Particle(object):
	def __init__(self, nParticles: int):
		self.nParticles = nParticles

	def normalize(self, weight_map: Dict, axis: int):
		for key in weight_map.keys():
		    self.weights[key] = (
                self.weights[key] / self.weights[key].sum(axis)
			)


PT = NewType("PT", Particle)

def filter(particle: PT, ) -> PT: