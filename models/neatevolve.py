import logging
from typing import Tuple, Union

_log = logging.getLogger(__name__)


class Connection:
    """A connection gene between two node genes."""

    def __init__(self, input: int, output: int, weight: float, innovation: int):
        self.input_node = input
        self.output_node = output
        self.weight = weight
        self.enabled = True
        self.innovation = innovation


class Node:

    def __init__(self, node_id: int):
        self.id = node_id
        self.value = 0.5


class Genome:
    """A collection of node genes connected by connection genes."""

    def __init__(self):
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []
        self.connections = []

    def get_connection(self, input_node: int = None, output_node: int = None, innovation: int = None) -> Connection:
        """Get the connection gene with either the specified innovation number or the in- and output nodes.

        :param innovation: The innovation number to look for.
        :param input_node: The id of an input node to look for. Only usable with output_node.
        :param output_node: The id of an output node to look for. Only usable with input_node.
        :return: A connection gene.
        :raise ValueError: If no connection matches the given parameters.
        """
        _log.debug(f"Finding connection in:{input_node}, out:{output_node}, innovation:{innovation}")
        if innovation is not None:
            for connection in self.connections:
                if connection.innovation == innovation:
                    return connection
            # No such connection exists.
            # _log.warning(f"No such connection with innovation {innovation}")
            raise ValueError(f"There is no connection gene with the innovation number {innovation}!")

        if input_node is not None and output_node is not None:
            for connection in self.connections:
                if connection.input_node == input_node and connection.output_node == output_node:
                    return connection
            # _log.warning(f"No such connection with in:{input_node}, out:{output_node}")
            raise ValueError(f"There is no connection gene with input node {input_node} and output node {output_node}!")

        # None of the input parameters yielded anything, or no parameters were given.
        _log.warning(f"No connections match the parameters innovation: {innovation}, input: {input_node},"
                     f" output: {output_node}")
        raise ValueError(f"No connections match the parameters innovation: {innovation}, input: {input_node},"
                         f" output: {output_node}")

    def mutate_add_connection(self, in_node: int, out_node: int, weight: float, innovation: int) -> Connection:
        """Add a new connection linking two previously unconnected nodes."""
        # Check if the new connection already exists
        try:
            connex = self.get_connection(input_node=in_node, output_node=out_node)
        except ValueError:
            connex = None
        if connex is not None:
            raise ValueError(f"A connection already exists between in: {in_node}, out: {out_node}")

        mutated_connection = Connection(in_node, out_node, weight, innovation)

        # TODO: Check if connection matches another genome's mutation, do not update innovation number if so.
        self.connections.append(mutated_connection)
        _log.debug(f"Created mutated connection <in:{in_node}, out:{out_node}, weight:{weight}, inno:{innovation}>.")

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
        try:
            connex = self.get_connection(innovation=global_innovation)
        except ValueError:
            connex = None
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
        connection_to_node = Connection(old_input, new_node_id, 1.0, global_innovation)
        connection_from_node = Connection(new_node_id, old_output, old_connection.weight, global_innovation + 1)

        # TODO: Check if new connections match another genome's mutations, do not update innovation if they do.

        self.connections.append(connection_to_node)
        self.connections.append(connection_from_node)
        _log.debug(f"Created mutated node with id {new_node_id} between nodes {old_input} and {old_output}.")

        return mutated_node, connection_to_node, connection_from_node


class Generation:
    """A collection of genomes representing a single generation of mutations."""

    def __init__(self):
        self.global_innovation = 1
        self.mutated_connections = []
        self.genomes = []

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

