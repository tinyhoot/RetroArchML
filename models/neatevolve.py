#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""An implementation of the NEATevolve algorithm, as described in:
Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary
computation, 10(2), 99-127. https://doi.org/10.1162/106365602320169811
"""

import logging
import math
from random import Random
from typing import Iterable, Tuple, Union, List, Sequence

LOG = logging.getLogger(__name__)
RANDOM = Random()

EXCESS_GENES_COEFFICIENT = 1.0
DISJOINT_GENES_COEFFICIENT = 1.0
WEIGHT_DIFFERENCE_COEFFICIENT = 0.4
NORMALISE_COMPAT_DISTANCE_FOR_GENE_SIZE = False

COMPATIBILITY_THRESHOLD = 3.0           # The maximum distance between members of a species.
DISABLED_GENE_REENABLE_CHANCE = 0.25
TOTAL_POPULATION_SIZE = 150

INNOVATION = 1


class Connection:
    """A connection gene between two node genes."""

    def __init__(self, input_node: int, output_node: int, innovation: int, weight: float = 0.5):
        self.input_node = input_node
        self.output_node = output_node
        self.innovation = innovation
        self.weight = weight
        self.enabled = True

    def __eq__(self, other):
        if not isinstance(other, Connection):
            super.__eq__(self, other)
        return self.innovation == other.innovation

    def copy(self) -> "Connection":
        """Get a full copy of this gene."""
        duplicate = Connection(self.input_node, self.output_node, self.innovation, self.weight)
        duplicate.enabled = self.enabled
        return duplicate


class Node:
    """A node gene, connected to other nodes by connection genes."""

    def __init__(self, innovation: int):
        self.innovation = innovation
        self.value = 0.0

    def __eq__(self, other):
        if not isinstance(other, Node):
            super.__eq__(self, other)
        return self.innovation == other.innovation


class Genome:
    """A collection of node genes connected by connection genes."""

    def __init__(self):
        self.input_nodes: List[Node] = []
        self.hidden_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self.connections: List[Connection] = []
        self.fitness = 0

    def __len__(self):
        return len(self.connections)

    def mutate_add_connection(self, input_node: int, output_node: int, weight: float, generation: "Generation") \
            -> Connection:
        """Add a new connection linking two previously unconnected nodes.

        :param input_node: The id of the input node of the new connection.
        :param output_node: The id of the output node of the new connection.
        :param weight: The weight of the new connection.
        :param generation: The generation this genome is a part of.
        :return: A connection gene containing the newly mutated connection.
        """
        global INNOVATION
        # Check if the new connection already exists
        connex = get_connection(self.connections, input_node=input_node, output_node=output_node)
        if connex is not None:
            raise ValueError(f"A connection already exists between in: {input_node}, out: {output_node}")

        mutated_connection = Connection(input_node, output_node, INNOVATION, weight)

        # Only update the global innovation number if this mutation has not occurred in this generation.
        dupl_innov = generation.check_for_duplicate_connection(input_node, output_node)
        if dupl_innov:
            mutated_connection.innovation = dupl_innov
        else:
            INNOVATION += 1
        self.connections.append(mutated_connection)
        LOG.debug(f"Created mutated connection <in:{input_node}, out:{output_node}, weight:{weight}, inno:{INNOVATION}>.")

        return mutated_connection

    def mutate_add_node(self, connection_innovation: int, generation: "Generation") \
            -> Tuple[Node, Connection, Connection]:
        """Add a new node splitting an existing connection in two.

        :param connection_innovation: The innovation number of the old connection to be split up.
        :param generation: The generation this genome belongs to.
        :return: A tuple containing the new node and connections.
        """
        global INNOVATION
        # Store the old connection's information.
        old_connection = get_connection(self.connections, innovation=connection_innovation)
        old_input = old_connection.input_node
        old_output = old_connection.output_node

        # Check whether this mutation already occurred this generation.
        dupl_id = generation.check_for_duplicate_node(old_connection)
        if dupl_id:
            node_id = dupl_id
        else:
            node_id = INNOVATION
            INNOVATION += 1

        # Create the new intervening node.
        mutated_node = Node(node_id)
        self.hidden_nodes.append(mutated_node)

        # Disable the old connection and replace it with two new ones.
        old_connection.enabled = False
        if dupl_id:
            connection_to_node = get_connection(generation.mutated_connections, old_input, node_id)
            connection_from_node = get_connection(generation.mutated_connections, node_id, old_output)
        else:
            connection_to_node = Connection(old_input, node_id, INNOVATION, 1.0)
            connection_from_node = Connection(node_id, old_output, INNOVATION + 1, old_connection.weight)
            INNOVATION += 2

        self.connections.append(connection_to_node)
        self.connections.append(connection_from_node)
        LOG.debug(f"Created mutated node with id {node_id} between nodes {old_input} and {old_output}.")

        return mutated_node, connection_to_node, connection_from_node


class Generation:
    """A collection of genomes representing a single generation of mutations."""

    def __init__(self, genomes: List[Genome], previous_gen: "Generation" = None):
        self.genomes = genomes
        # Connections this generation has added.
        self.mutated_connections: List[Connection] = []
        # Nodes this generation has added, along with the connection that they each split by being added.
        self.mutated_nodes: List[Tuple[Node, Connection]] = []
        self.species: List[Species] = []
        self._speciate(previous_gen)

    def breed(self, first_parent: Genome, second_parent: Genome) -> Genome:
        """Combine two genomes to produce offspring inheriting their combined features.

        Disjoint or excess genes are inherited from the fitter parent.

        :param first_parent: The first parent genome.
        :param second_parent: The second parent genome.
        :return: The offspring genome.
        """
        offspring = Genome()
        # Find the parent with higher fitness to inherit genes from.
        fit_parent = get_fittest((first_parent, second_parent))
        bad_parent = first_parent if fit_parent is second_parent else second_parent

        # Inputs and outputs are always identical.
        offspring.input_nodes = first_parent.input_nodes
        offspring.output_nodes = first_parent.output_nodes
        # Hidden nodes are always inherited from the fitter parent.
        offspring.hidden_nodes = fit_parent.hidden_nodes

        # Connection genes are inherited randomly if they match, or from the fitter parent if they do not.
        for fit_connection in fit_parent.connections:
            bad_connection = get_connection(bad_parent.connections, innovation=fit_connection.innovation)

            if bad_connection is not None:
                # Gene matches, inherit randomly.
                random_gene = RANDOM.choice((fit_connection, bad_connection))
                inherited_gene = random_gene.copy()
                # If the gene is disabled in either parent, there is a chance to re-enable it.
                if fit_connection.enabled is False or bad_connection.enabled is False:
                    inherited_gene.enabled = DISABLED_GENE_REENABLE_CHANCE < RANDOM.random()
            else:
                # Gene is disjointed or excess, inherit directly.
                inherited_gene = fit_connection.copy()
                # If the gene is disabled, there is a chance to re-enable it.
                if fit_connection.enabled is False:
                    inherited_gene.enabled = DISABLED_GENE_REENABLE_CHANCE < RANDOM.random()

            offspring.connections.append(inherited_gene)

        return offspring

    def check_for_duplicate_connection(self, input_node: int, output_node: int) -> Union[int, None]:
        """Check whether a connection between two nodes has already been created by any genome in the generation.

        :param input_node: The input node of the connection.
        :param output_node: The output node of the connection.
        :return: The innovation number of the duplicate connection, or None if the connection is unique.
        """
        for connection in self.mutated_connections:
            if connection.input_node == input_node and connection.output_node == output_node:
                return connection.innovation

        return None

    def check_for_duplicate_node(self, split_connection: Connection) -> Union[int, None]:
        """Check whether a new node has already been created by any genome in the generation.

        :param split_connection: The connection gene that is split by the addition of the new node.
        :return: None if the mutation is unique, otherwise the id of the previously mutated node.
        """
        for node, connection in self.mutated_nodes:
            if connection == split_connection:
                return node.innovation

        return None

    def _speciate(self, previous_gen: "Generation" = None):
        """Divide the population of this generation into a number of distinct species.

        The speciation depends on the compatibility distance. Intended for just after a fresh generation is created.

        :param previous_gen: If given, inherit existing species from the previous generation.
        """
        self.species = []
        genomes = self.genomes.copy()

        # If inheriting from the previous generation, choose a random member of each species to represent it this gen.
        if previous_gen and previous_gen.species:
            for prev_species in previous_gen.species:
                representative = RANDOM.choice(prev_species.population)
                spec = Species(representative)
                self.species.append(spec)
                # The chosen genome can be assumed to always be present in the list kept by the generation, otherwise
                # something has gone wrong.
                genomes.remove(representative)

        # Assign every genome in this generation that was not chosen as a representative to a species.
        for genome in genomes:
            assigned = False
            for spec in self.species:
                if get_compatibility_distance(genome, spec.representative) <= COMPATIBILITY_THRESHOLD:
                    spec.add(genome)
                    assigned = True
                    break
            # If the genome did not fit in an existing species, create a new one for it.
            if not assigned:
                new_spec = Species(genome)
                self.species.append(new_spec)


class Species:
    """A collection of genomes with low compatibility distance and a representative genome to measure against."""

    def __init__(self, representative: Genome):
        self.population = [representative]
        self.representative = representative

    def __len__(self) -> int:
        return len(self.population)

    def add(self, genome: Genome):
        """Add a genome to the population."""
        self.population.append(genome)

    def get_adjusted_fitness(self, genome: Genome) -> float:
        """Get the fitness of a genome, adjusted for the general fitness of the rest of the population.

        This is a form of explicit fitness sharing; each organism must share its fitness with the rest of its
        ecological niche.

        :return: This genome's fitness adjusted for the population.
        :raise ValueError: If the given genome is not part of the species.
        """
        if genome not in self.population:
            raise ValueError("Given genome is not part of this species' population!")

        # The actual function for adjusted fitness is a fair bit more complex than this, but in a speciated population
        # the denominator reduces to the population size.
        return genome.fitness / len(self.population)

    def get_population_fitness(self) -> float:
        """Get the overall fitness of the population.

        This is equal to the sum of adjusted fitnesses of all member organisms.

        :return: The population fitness.
        """
        fitness = 0.0
        for organism in self.population:
            fitness += self.get_adjusted_fitness(organism)

        return fitness


def get_compatibility_distance(first_genome: Genome, second_genome: Genome) -> float:
    """Calculate the compatibility distance between two genomes (network structures).

    The distance increases with excess and disjoint genes, as well as weight differences between matching genes.

    :param first_genome: The first genome.
    :param second_genome: The second genome.
    :return: A floating point value representing the compatibility distance.
    """
    # N is the number of genes of the larger genome.
    n_genes = max(len(first_genome), len(second_genome))

    if not NORMALISE_COMPAT_DISTANCE_FOR_GENE_SIZE and n_genes < 20:
        n_genes = 1

    # Count the total number of excess and disjoint genes.
    excess, disjoint, weight = get_gene_differences(first_genome, second_genome)

    excess_distance = (EXCESS_GENES_COEFFICIENT * excess) / n_genes
    disjoint_distance = (DISJOINT_GENES_COEFFICIENT * disjoint) / n_genes
    weight_distance = WEIGHT_DIFFERENCE_COEFFICIENT * weight

    return excess_distance + disjoint_distance + weight_distance


def get_connection(connections: Iterable[Connection], input_node: int = None, output_node: int = None,
                   innovation: int = None) -> Union[Connection, None]:
    """Get the connection gene with either the specified innovation number or the in- and output nodes.

    :param connections: The iterable of connection genes to search.
    :param input_node: The id of an input node to look for. Only usable with output_node.
    :param output_node: The id of an output node to look for. Only usable with input_node.
    :param innovation: The innovation number to look for.
    :return: A connection gene, or None if the connection does not exist.
    :raise ValueError: If the given parameters are faulty or incomplete.
    """
    LOG.debug(f"Finding connection in:{input_node}, out:{output_node}, innovation:{innovation}")
    if innovation is not None:
        for connection in connections:
            if connection.innovation == innovation:
                return connection
        # No such connection exists.
        # _log.warning(f"No such connection with innovation {innovation}")
        return None

    if input_node is not None and output_node is not None:
        for connection in connections:
            if connection.input_node == input_node and connection.output_node == output_node:
                return connection
        # _log.warning(f"No such connection with in:{input_node}, out:{output_node}")
        return None

    # No parameters were given, or the parameters were somehow faulty.
    LOG.warning(f"No connections match the parameters innovation: {innovation}, input: {input_node},"
                f" output: {output_node}")
    raise ValueError(f"No connections match the parameters innovation: {innovation}, input: {input_node},"
                     f" output: {output_node}")


def get_gene_differences(first_genome: Genome, second_genome: Genome) -> Tuple[int, int, float]:
    """Get the number of excess and disjoint genes along with total weight difference for two genomes.

    :return: A tuple containing the number of excess genes, number of disjoint genes, and total weight difference.
    """
    excess, disjoint, weight = 0, 0, 0.0

    # Innovation numbers are unique, sets make them easy to compare.
    first_genome_innovations = set([x.innovation for x in first_genome.connections])
    second_genome_innovations = set([x.innovation for x in second_genome.connections])

    # Iterate through all innovations that appear in either of the genomes.
    for innovation in first_genome_innovations | second_genome_innovations:
        # Does this gene exist in just one set, or both?
        if innovation in first_genome_innovations & second_genome_innovations:
            # The genes are matching. Calculate weight difference.
            first_gene = get_connection(first_genome.connections, innovation=innovation)
            second_gene = get_connection(second_genome.connections, innovation=innovation)
            weight += math.fabs(first_gene.weight - second_gene.weight)
        else:
            # The gene is either excess or disjoint. Find out which one it is.
            if innovation in first_genome_innovations:
                if innovation > max(second_genome_innovations):
                    excess += 1
                else:
                    disjoint += 1
            if innovation in second_genome_innovations:
                if innovation > max(first_genome_innovations):
                    excess += 1
                else:
                    disjoint += 1

    return excess, disjoint, weight


def get_fittest(genomes: Sequence[Genome]) -> Genome:
    """Find the fittest genome in a sequence of genomes.

    Will choose a random one if two genomes have exactly the same fitness.

    :param genomes: A sequence of genomes.
    :return: The fittest genome in the input sequence.
    """
    fittest = None
    for candidate in genomes:
        # Special handling for the first candidate in the sequence, since there's nothing to compare it to yet.
        if fittest is None:
            fittest = candidate
            continue
        if candidate.fitness > fittest.fitness:
            fittest = candidate
        elif candidate.fitness == fittest.fitness:
            # If fitness is equal, take a random one.
            fittest = RANDOM.choice((candidate, fittest))

    return fittest


def sigmoid(x: float) -> float:
    """Get Stanley & Miikkulainen's modified sigmoid of an input value."""
    return 1 / (1 + math.exp(-4.9 * x))
