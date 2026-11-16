import unittest

from kanm.networks import (
    agent_ids,
    complete,
    cycle,
    density,
    edges_from_pairs,
    erdos_renyi,
    generate,
    mean_clustering,
    star,
    watts_strogatz,
)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.ids = agent_ids(10)

    def _valid(self, pairs):
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(a != b for a, b in pairs))
        self.assertEqual(pairs, sorted(pairs))

    def test_basic_generators(self):
        self.assertEqual(len(complete(self.ids)), 90)
        self.assertEqual(len(cycle(self.ids)), 20)
        self.assertEqual(len(cycle(self.ids, bidirectional=False)), 10)
        self.assertEqual(len(star(self.ids, direction="out")), 9)
        for pairs in (complete(self.ids), cycle(self.ids), star(self.ids)):
            self._valid(pairs)

    def test_erdos_renyi_is_seeded_and_bounded(self):
        a = erdos_renyi(self.ids, 0.3, seed=1)
        self.assertEqual(a, erdos_renyi(self.ids, 0.3, seed=1))
        self.assertNotEqual(a, erdos_renyi(self.ids, 0.3, seed=2))
        self.assertEqual(erdos_renyi(self.ids, 0.0, seed=1), [])
        self.assertEqual(len(erdos_renyi(self.ids, 1.0, seed=1)), 90)
        undirected = erdos_renyi(self.ids, 0.4, seed=5, directed=False)
        self.assertTrue(all((b, a) in set(undirected) for a, b in undirected))
        self._valid(a)

    def test_watts_strogatz_limits(self):
        lattice = watts_strogatz(self.ids, 4, 0.0, seed=1)
        self.assertEqual(len(lattice), 10 * 4)
        self.assertAlmostEqual(mean_clustering(lattice, self.ids), 0.5)
        rewired = watts_strogatz(self.ids, 4, 1.0, seed=1)
        self.assertEqual(len(rewired), 10 * 4)  # rewiring preserves edge count
        self._valid(rewired)
        with self.assertRaises(ValueError):
            watts_strogatz(self.ids, 3, 0.1, seed=1)

    def test_density_and_clustering_of_complete_graph(self):
        pairs = complete(self.ids)
        self.assertEqual(density(pairs, 10), 1.0)
        self.assertAlmostEqual(mean_clustering(pairs, self.ids), 1.0)

    def test_generate_spec_and_edge_defaults(self):
        pairs = generate({"type": "erdos_renyi", "p": 0.5, "seed": 3}, self.ids)
        self.assertEqual(pairs, erdos_renyi(self.ids, 0.5, seed=3))
        with self.assertRaises(ValueError):
            generate({"type": "hypercube"}, self.ids)
        with self.assertRaises(ValueError):
            generate({"type": "cycle", "bogus": 1}, self.ids)
        edges = edges_from_pairs(pairs, listen_chance=0.4)
        self.assertTrue(all(e.listen_chance == 0.4 for e in edges))
        with self.assertRaises(ValueError):
            edges_from_pairs(pairs, listen=0.4)


if __name__ == "__main__":
    unittest.main()
