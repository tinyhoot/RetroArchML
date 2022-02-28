import logging
import pytest
import models.neatevolve as neat


class TestGenetics:

    @pytest.fixture
    def genome(self):
        test_genome = neat.Genome()
        # Five input nodes.
        for i in range(1, 6):
            node = neat.Node(i)
            test_genome.input_nodes.append(node)
        # Two output nodes.
        for i in range(6, 8):
            node = neat.Node(i)
            test_genome.output_nodes.append(node)
        # Three hidden nodes.
        for i in range(8, 11):
            node = neat.Node(i)
            test_genome.hidden_nodes.append(node)

        # Add connections between the layers.
        global_innovation = 1
        for in_node in test_genome.input_nodes:
            for out_node in test_genome.hidden_nodes:
                connex = neat.Connection(in_node.id, out_node.id, 0.5, global_innovation)
                test_genome.connections.append(connex)
                global_innovation += 1
        for in_node in test_genome.hidden_nodes:
            for out_node in test_genome.output_nodes:
                connex = neat.Connection(in_node.id, out_node.id, 0.5, global_innovation)
                test_genome.connections.append(connex)
                global_innovation += 1

        return test_genome

    def test_get_connection_innovation(self, genome):
        connex = genome.get_connection(innovation=8)
        assert isinstance(connex, neat.Connection)
        assert connex.input_node == 3
        assert connex.output_node == 9

    def test_get_connection_innovation_out_of_range(self, genome):
        with pytest.raises(ValueError):
            genome.get_connection(innovation=0)
        with pytest.raises(ValueError):
            genome.get_connection(innovation=100)

    def test_get_connection_nodes(self, genome):
        connex = genome.get_connection(input_node=1, output_node=8)
        assert isinstance(connex, neat.Connection)
        assert connex.input_node == 1
        assert connex.output_node == 8

    def test_get_connection_nodes_no_connection(self, genome):
        with pytest.raises(ValueError):
            genome.get_connection(input_node=1, output_node=2)
        with pytest.raises(ValueError):
            genome.get_connection(input_node=5, output_node=6)
        with pytest.raises(ValueError):
            genome.get_connection(input_node=8, output_node=1)

    def test_get_connection_bad_param(self, genome):
        with pytest.raises(ValueError):
            genome.get_connection()
        with pytest.raises(ValueError):
            genome.get_connection(input_node=3)
        with pytest.raises(ValueError):
            genome.get_connection(output_node=8)

    def test_mutate_add_connection(self, genome):
        connex = genome.mutate_add_connection(3, 7, 0.5, 100)
        assert isinstance(connex, neat.Connection)
        assert connex.input_node == 3
        assert connex.output_node == 7
        assert connex.innovation == 100

    def test_mutate_add_connection_exists(self, genome):
        with pytest.raises(ValueError):
            genome.mutate_add_connection(1, 8, 0.5, 100)

    def test_mutate_add_node(self, genome, caplog):
        # Try adding a new node between input node 1 and hidden node 1 (=id 8), which should be connection 1.
        old_connex = 1
        new_node_id = 11
        node, connex1, connex2 = genome.mutate_add_node(old_connex, new_node_id, 100)
        assert isinstance(node, neat.Node)
        assert isinstance(connex1, neat.Connection)
        assert isinstance(connex2, neat.Connection)
        assert node.id == new_node_id
        assert connex1.input_node == old_connex and connex1.output_node == new_node_id
        assert connex2.input_node == new_node_id and connex2.output_node == 8
        assert genome.get_connection(innovation=old_connex).enabled is False

    def test_mutate_add_note_already_used(self, genome):
        with pytest.raises(ValueError):
            genome.mutate_add_node(1, 11, 10)
