"""An implementation of the NEATevolve algorithm, as described in:
Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary
computation, 10(2), 99-127. https://doi.org/10.1162/106365602320169811
"""

import logging
import math
from random import Random
from typing import Tuple, Union, List, Sequence

LOG = logging.getLogger(__name__)
RANDOM = Random()

EXCESS_GENES_COEFFICIENT = 1.0
DISJOINT_GENES_COEFFICIENT = 1.0
WEIGHT_DIFFERENCE_COEFFICIENT = 0.4
NORMALISE_COMPAT_DISTANCE_FOR_GENE_SIZE = False

DISABLED_GENE_REENABLE_CHANCE = 0.25


class Connection:
    """A connection gene between two node genes."""

    def __init__(self, input_node: int, output_node: int, innovation: int, weight: float = 0.5):
        self.input_node = input_node
        self.output_node = output_node
        self.innovation = innovation
        self.weight = weight
        self.enabled = True

    def copy(self) -> "Connection":
        """Get a full copy of this gene."""
        duplicate = Connection(self.input_node, self.output_node, self.innovation, self.weight)
        duplicate.enabled = self.enabled
        return duplicate


class Node:
    """A node gene, connected to other nodes by connection genes."""

    def __init__(self, node_id: int):
        self.id = node_id
        self.value = 0.5


class Genome:
    """A collection of node genes connected by connection genes."""

    def __init__(self):
        self.input_nodes: List[Node] = []
        self.hidden_nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self.connections: List[Connection] = []
        self.fitness = 0

    def get_connection(self, input_node: int = None, output_node: int = None, innovation: int = None) \
            -> Union[Connection, None]:
        """Get the connection gene with either the specified innovation number or the in- and output nodes.

        :param innovation: The innovation number to look for.
        :param input_node: The id of an input node to look for. Only usable with output_node.
        :param output_node: The id of an output node to look for. Only usable with input_node.
        :return: A connection gene, or None if the connection does not exist.
        :raise ValueError: If the given parameters are faulty or incomplete.
        """
        LOG.debug(f"Finding connection in:{input_node}, out:{output_node}, innovation:{innovation}")
        if innovation is not None:
            for connection in self.connections:
                if connection.innovation == innovation:
                    return connection
            # No such connection exists.
            # _log.warning(f"No such connection with innovation {innovation}")
            # raise ValueError(f"There is no connection gene with the innovation number {innovation}!")
            return None

        if input_node is not None and output_node is not None:
            for connection in self.connections:
                if connection.input_node == input_node and connection.output_node == output_node:
                    return connection
            # _log.warning(f"No such connection with in:{input_node}, out:{output_node}")
            # raise ValueError(f"There is no connection gene with input node {input_node} and output node {output_node}!")
            return None

        # No parameters were given, or the parameters were somehow faulty.
        LOG.warning(f"No connections match the parameters innovation: {innovation}, input: {input_node},"
                     f" output: {output_node}")
        raise ValueError(f"No connections match the parameters innovation: {innovation}, input: {input_node},"
                         f" output: {output_node}")

    def get_size(self) -> int:
        """Get the total number of all connection genes in this genome."""
        return len(self.connections)

    def mutate_add_connection(self, in_node: int, out_node: int, innovation: int, weight: float) -> Connection:
        """Add a new connection linking two previously unconnected nodes."""
        # Check if the new connection already exists
        connex = self.get_connection(input_node=in_node, output_node=out_node)
        if connex is not None:
            raise ValueError(f"A connection already exists between in: {in_node}, out: {out_node}")

        mutated_connection = Connection(in_node, out_node, innovation, weight)

        # TODO: Check if connection matches another genome's mutation, do not update innovation number if so.
        self.connections.append(mutated_connection)
        LOG.debug(f"Created mutated connection <in:{in_node}, out:{out_node}, weight:{weight}, inno:{innovation}>.")

        return mutated_connection

    def mutate_add_node(self, connection_innovation: int, new_node_id: int, global_innovation: int) \
            -> Tuple[Node, Connection, Connection]:
        """Add a new node splitting an existing connection in two.

        :param connection_innovation: The innovation number of the old connection to be split up.
        :param new_node_id: The id of the new node to be created.
        :param global_innovation: The current global innovation number.
        :return: A tuple containing the new node and connections.
        """
        # Ensure the global innovation number is not already in use for some reason.
        connex = self.get_connection(innovation=global_innovation)
        if connex is not None:
            raise ValueError(f"The global innovation number {global_innovation} is already in use!")

        # Store the old connection's information.
        old_connection = self.get_connection(innovation=connection_innovation)
        old_input = old_connection.input_node
        old_output = old_connection.output_node

        # Create the new intervening node.
        mutated_node = Node(new_node_id)
        self.hidden_nodes.append(mutated_node)

        # Disable the old connection and replace it with two new ones.
        old_connection.enabled = False
        connection_to_node = Connection(old_input, new_node_id, global_innovation, 1.0)
        connection_from_node = Connection(new_node_id, old_output, global_innovation + 1, old_connection.weight)

        # TODO: Check if new connections match another genome's mutations, do not update innovation if they do.

        self.connections.append(connection_to_node)
        self.connections.append(connection_from_node)
        LOG.debug(f"Created mutated node with id {new_node_id} between nodes {old_input} and {old_output}.")

        return mutated_node, connection_to_node, connection_from_node


class Generation:
    """A collection of genomes representing a single generation of mutations."""

    def __init__(self):
        self.global_innovation = 1
        self.mutated_connections: List[Connection] = []
        self.genomes: List[Genome] = []

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
            bad_connection = bad_parent.get_connection(innovation=fit_connection.innovation)

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

    def check_for_duplicate_innovation(self, input_node: int, output_node: int) -> Union[int, None]:
        """Check whether a connection between two nodes has already been created by any genome in the generation.

        :param input_node: The input node of the connection.
        :param output_node: The output node of the connection.
        :return: The innovation number of the duplicate connection, or None if the connection is unique.
        """
        for connection in self.mutated_connections:
            if connection.input_node == input_node and connection.output_node == output_node:
                return connection.innovation

        return None


def get_compatibility_distance(first_genome: Genome, second_genome: Genome,
                               normalise_gene_size: bool = NORMALISE_COMPAT_DISTANCE_FOR_GENE_SIZE) -> float:
    """Calculate the compatibility distance between two genomes (network structures).

    The distance increases with excess and disjoint genes, as well as weight differences between matching genes.

    :param first_genome: The first genome.
    :param second_genome: The second genome.
    :param normalise_gene_size: If False, n_genes is set to 1 for genomes consisting of < 20 genes.
    :return: A floating point value representing the compatibility distance.
    """
    # N is the number of genes of the larger genome.
    if first_genome.get_size() > second_genome.get_size():
        n_genes = first_genome.get_size()
    else:
        n_genes = second_genome.get_size()

    if not normalise_gene_size and n_genes < 20:
        n_genes = 1

    # Count the total number of excess and disjoint genes. Prevent duplicates by keeping track of innovation numbers.
    # This feels like a dumb way to do it. Can definitely be optimised.
    processed_genes = []
    excess, disjoint, weight = _get_gene_differences(first_genome, second_genome, processed_genes)
    excess2, disjoint2, weight2 = _get_gene_differences(second_genome, first_genome, processed_genes)

    excess += excess2
    disjoint += disjoint2
    weight += weight2

    excess_distance = (EXCESS_GENES_COEFFICIENT * excess) / n_genes
    disjoint_distance = (DISJOINT_GENES_COEFFICIENT * disjoint) / n_genes
    weight_distance = WEIGHT_DIFFERENCE_COEFFICIENT * weight

    return excess_distance + disjoint_distance + weight_distance


def _get_gene_differences(first_genome: Genome, second_genome: Genome, processed_genes: List) -> Tuple[int, int, float]:
    """Get the number of excess and disjoint genes along with total weight difference for two genomes.

    Intended to be called twice, once for each genome in initial position. Directly edits the given list.
    """
    excess, disjoint, weight = 0, 0, 0.0
    second_genome_last_innovation = second_genome.connections[-1].innovation

    for first_gene in first_genome.connections:
        if first_gene.innovation in processed_genes:
            continue  # Duplicate, skip.

        processed_genes.append(first_gene.innovation)
        second_gene = second_genome.get_connection(innovation=first_gene.innovation)

        if second_gene is None:
            if first_gene.innovation > second_genome_last_innovation:
                # Gene is excess
                excess += 1
            else:
                # Gene is disjoint
                disjoint += 1
        else:
            # Genes are matching, calculate weight difference
            weight += math.fabs(first_gene.weight - second_gene.weight)

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
