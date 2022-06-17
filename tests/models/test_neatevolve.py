#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
                connex = neat.Connection(in_node.innovation, out_node.innovation, global_innovation)
                test_genome.connections.append(connex)
                global_innovation += 1
        for in_node in test_genome.hidden_nodes:
            for out_node in test_genome.output_nodes:
                connex = neat.Connection(in_node.innovation, out_node.innovation, global_innovation)
                test_genome.connections.append(connex)
                global_innovation += 1

        return test_genome

    @pytest.fixture
    def parent_genomes(self):
        # Set up according to Figure 4 in Stanley & Miikkulainen.
        first_parent = neat.Genome()
        first_parent.fitness = 5
        for i in range(1, 4):
            first_parent.input_nodes.append(neat.Node(i))
        first_parent.output_nodes.append(neat.Node(4))
        first_parent.hidden_nodes.append(neat.Node(5))

        con = []    # Connections to be added.
        con.append(neat.Connection(1, 4, 1, 0.1))
        c_two = neat.Connection(2, 4, 2, 0.6)
        c_two.enabled = False
        con.append(c_two)
        con.append(neat.Connection(3, 4, 3, 0.7))
        con.append(neat.Connection(2, 5, 4, 0.8))
        con.append(neat.Connection(5, 4, 5, 0.2))
        con.append(neat.Connection(1, 5, 8, 0.4))
        first_parent.connections += con

        # Deal with the second parent.
        second_parent = neat.Genome()
        second_parent.fitness = 2
        for i in range(1, 4):
            second_parent.input_nodes.append(neat.Node(i))
        second_parent.output_nodes.append(neat.Node(4))
        second_parent.hidden_nodes.append(neat.Node(5))
        second_parent.hidden_nodes.append(neat.Node(6))

        con = []
        con.append(neat.Connection(1, 4, 1, 0.3))
        con.append(c_two)
        con.append(neat.Connection(3, 4, 3, 0.4))
        con.append(neat.Connection(2, 5, 4, 0.8))
        c_five = neat.Connection(5, 4, 5, 0.9)
        c_five.enabled = False
        con.append(c_five)
        con.append(neat.Connection(5, 6, 6, 0.1))
        con.append(neat.Connection(6, 4, 7, 0.6))
        con.append(neat.Connection(3, 5, 9, 0.7))
        con.append(neat.Connection(1, 6, 10, 1.0))
        second_parent.connections += con

        return first_parent, second_parent

    def test_get_connection_innovation(self, genome):
        connex = neat.get_connection(genome.connections, innovation=8)
        assert isinstance(connex, neat.Connection)
        assert connex.input_node == 3
        assert connex.output_node == 9

    def test_get_connection_innovation_out_of_range(self, genome):
        assert neat.get_connection(genome.connections, innovation=0) is None
        assert neat.get_connection(genome.connections, innovation=100) is None

    def test_get_connection_nodes(self, genome):
        connex = neat.get_connection(genome.connections, input_node=1, output_node=8)
        assert isinstance(connex, neat.Connection)
        assert connex.input_node == 1
        assert connex.output_node == 8

    def test_get_connection_nodes_no_connection(self, genome):
        assert neat.get_connection(genome.connections, input_node=1, output_node=2) is None
        assert neat.get_connection(genome.connections, input_node=5, output_node=6) is None
        assert neat.get_connection(genome.connections, input_node=8, output_node=1) is None

    def test_get_connection_bad_param(self, genome):
        with pytest.raises(ValueError):
            neat.get_connection(genome.connections)
        with pytest.raises(ValueError):
            neat.get_connection(genome.connections, input_node=3)
        with pytest.raises(ValueError):
            neat.get_connection(genome.connections, output_node=8)

    def test_mutate_add_connection(self, genome):
        neat.INNOVATION = 100
        connex = genome.mutate_add_connection(3, 7, 0.5, neat.Generation([]))
        assert isinstance(connex, neat.Connection)
        assert connex.input_node == 3
        assert connex.output_node == 7
        assert connex.innovation == 100

    def test_mutate_add_connection_exists(self, genome):
        with pytest.raises(ValueError):
            genome.mutate_add_connection(1, 8, 0.5, neat.Generation([]))

    def test_mutate_add_node(self, genome, caplog):
        # Try adding a new node between input node 1 and hidden node 1 (=id 8), which should be connection 1.
        old_connex = 1
        neat.INNOVATION = 100
        node, connex1, connex2 = genome.mutate_add_node(old_connex, neat.Generation([]))
        assert isinstance(node, neat.Node)
        assert isinstance(connex1, neat.Connection)
        assert isinstance(connex2, neat.Connection)
        assert node.innovation == 100
        assert connex1.input_node == old_connex and connex1.output_node == 100
        assert connex2.input_node == 100 and connex2.output_node == 8
        assert neat.get_connection(genome.connections, innovation=old_connex).enabled is False

    def test_breed(self, parent_genomes):
        generation = neat.Generation([])
        offspring = generation.breed(parent_genomes[0], parent_genomes[1])
        assert len(offspring.hidden_nodes) == 1
        assert len(offspring.connections) == 6

    def test_get_compatibility_distance(self, parent_genomes):
        # 2 excess, 3 disjoint, 1.2 weight, n_genes = 9
        neat.NORMALISE_COMPAT_DISTANCE_FOR_GENE_SIZE = True
        assert neat.get_compatibility_distance(parent_genomes[0], parent_genomes[1]) == 1.0355555555555556

    def test_get_compatibility_distance_no_normalise(self, parent_genomes):
        neat.NORMALISE_COMPAT_DISTANCE_FOR_GENE_SIZE = False
        assert neat.get_compatibility_distance(parent_genomes[0], parent_genomes[1]) == 5.4799999999999995
