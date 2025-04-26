import torch
import copy
import numpy as np
from pyneuro.core.optim.optimizer import Optimizer

class GeneticOptimizer(Optimizer):
    """
    Genetic algorithm optimizer
    """
    def __init__(self, model, fitness_fn, population_size=10, mutation_rate=0.1):
        self.model = model
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [copy.deepcopy(model) for _ in range(population_size)]
        self.best_model = copy.deepcopy(model)
    
    def step(self):
        """Perform a single optimization step using genetic algorithm"""
        # Calculate fitness for each model in the population
        fitness_scores = np.array([self.fitness_fn(model) for model in self.population])
        
        # Select the best model
        best_idx = np.argmax(fitness_scores)
        self.best_model = copy.deepcopy(self.population[best_idx])
        
        # Create a new population starting with the best model
        new_population = [self.best_model]
        
        # Fill the rest of the population with children from crossover and mutation
        while len(new_population) < self.population_size:
            parent1, parent2 = np.random.choice(self.population, size=2, replace=False)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        
        # Update the original model with the best model's parameters
        self.copy_parameters(self.best_model, self.model)
    
    def crossover(self, parent1, parent2):
        """Create a child model by combining parameters from two parents"""
        child = copy.deepcopy(parent1)
        
        # Iterate through each layer's parameters
        for name, param in child.named_parameters():
            # Get corresponding parameters from both parents
            p1_param = dict(parent1.named_parameters())[name]
            p2_param = dict(parent2.named_parameters())[name]
            
            # Create a random mask for crossover
            mask = torch.rand_like(param) > 0.5
            
            # Apply crossover
            param.data = torch.where(mask, p1_param.data, p2_param.data)
        
        return child
    
    def mutate(self, model):
        """Apply random mutations to model parameters"""
        for param in model.parameters():
            # Create a mutation mask
            mutation_mask = torch.rand_like(param) < self.mutation_rate
            
            # Generate random noise
            noise = torch.randn_like(param) * 0.1
            
            # Apply mutation
            param.data += mutation_mask * noise
        
        return model
    
    def copy_parameters(self, src_model, tgt_model):
        """Copy parameters from source model to target model"""
        tgt_params = dict(tgt_model.named_parameters())
        for name, param in src_model.named_parameters():
            if name in tgt_params:
                tgt_params[name].data.copy_(param.data)