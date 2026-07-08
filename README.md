# EvoCloud: A Distributed Evolutionary Computing Platform

## Problem Statement

Training neural network models using evolutionary algorithms selection, crossover, and mutation across a population of candidate models is computationally expensive, since each individual must be evaluated independently before a new generation can be produced. Running this process sequentially on a single machine limits how large a population or how many generations can be explored within a reasonable time.

This project aims to design and build a distributed evolutionary computing platform that generates a population of models, distributes their evaluation across multiple cloud based worker nodes running in parallel, aggregates the results at a central coordinator to perform selection and reproduction, and repeats this cycle across generations until a target performance is reached. The system will be validated on a simple task (e.g., CartPole or Snake) and will include a dashboard to monitor generation progress and evolutionary performance in real time.
