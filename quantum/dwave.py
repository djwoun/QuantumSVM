# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:28:17 2023

@author: djwou
"""

from dwave.system import DWaveSampler, EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler(token='DEV-43415957998392a7a2e71c0901630c8718d0d78e'))





response = sampler.sample(bqm)


best_solution = response.first.sample
energy = response.first.energy