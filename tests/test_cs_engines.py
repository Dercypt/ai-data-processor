import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from analyzer import (
    DisjointSetUnion,
    HeuristicRule,
    benchmark_imputation_drift,
    build_feature_dependency_graph,
    compute_ks_statistic_1d,
    compute_population_stability_index,
    compute_wasserstein_1d,
    evaluate_heuristic_rules,
    generate_summary,
    synthesize_heuristic_rules,
)


class TestFeatureDependencyGraph(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        n = 150
        x1 = np.random.normal(10.0, 2.0, size=n)
        x2 = 2.0 * x1 + np.random.normal(0.0, 0.5, size=n)  # Strong positive corr with x1
        x3 = -1.5 * x1 + np.random.normal(0.0, 0.5, size=n)  # Strong negative corr with x1
        x4 = np.random.uniform(0.0, 100.0, size=n)  # Independent feature
        self.df = pd.DataFrame({"feat_1": x1, "feat_2": x2, "feat_3": x3, "feat_4": x4})

    def test_dsu_operations_and_cycle_detection(self) -> None:
        elements = ["a", "b", "c", "d"]
        dsu = DisjointSetUnion(elements)
        self.assertEqual(dsu.find("a"), "a")

        # Union operations
        self.assertTrue(dsu.union("a", "b"))
        self.assertEqual(dsu.find("a"), dsu.find("b"))
        self.assertFalse(dsu.union("a", "b"))  # Cycle / already united

        self.assertTrue(dsu.union("c", "d"))
        self.assertNotEqual(dsu.find("a"), dsu.find("c"))
        self.assertTrue(dsu.union("b", "c"))
        self.assertEqual(dsu.find("a"), dsu.find("d"))

    def test_build_feature_dependency_graph_structure(self) -> None:
        graph = build_feature_dependency_graph(self.df, threshold=0.4)
        self.assertIn("nodes", graph)
        self.assertIn("edges", graph)
        self.assertIn("metrics", graph)

        metrics = graph["metrics"]
        self.assertEqual(metrics["total_nodes"], 4)
        self.assertGreater(metrics["total_edges"], 0)
        self.assertGreater(metrics["density"], 0.0)

        # Check nodes
        nodes = graph["nodes"]
        self.assertEqual(len(nodes), 4)
        for n in nodes:
            self.assertIn("id", n)
            self.assertIn("degree", n)
            self.assertIn("degree_centrality", n)
            self.assertIn("strength", n)
            self.assertIn("community_id", n)
            self.assertIn("x", n)
            self.assertIn("y", n)
            self.assertTrue(-1.05 <= n["x"] <= 1.05)
            self.assertTrue(-1.05 <= n["y"] <= 1.05)

        # Check edges and Kruskal MST
        edges = graph["edges"]
        mst_edges = [e for e in edges if e["is_mst_backbone"]]
        self.assertGreaterEqual(len(mst_edges), 1)

    def test_graph_threshold_and_domain_errors(self) -> None:
        with self.assertRaises(ValueError):
            build_feature_dependency_graph(self.df, threshold=-0.1)
        with self.assertRaises(ValueError):
            build_feature_dependency_graph(self.df, threshold=1.5)

    def test_graph_edge_cases(self) -> None:
        # Less than 2 numeric columns
        single_col_df = pd.DataFrame({"col_a": [1.0, 2.0, 3.0], "text": ["a", "b", "c"]})
        graph_single = build_feature_dependency_graph(single_col_df)
        self.assertEqual(graph_single["metrics"]["total_nodes"], 1)
        self.assertEqual(graph_single["metrics"]["total_edges"], 0)

        # Empty dataframe
        empty_df = pd.DataFrame()
        graph_empty = build_feature_dependency_graph(empty_df)
        self.assertEqual(graph_empty["metrics"]["total_nodes"], 0)


class TestHeuristicRuleEngine(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        n = 100
        a = np.random.normal(50.0, 5.0, size=n)
        b = a * 2.0 + np.random.normal(0.0, 1.0, size=n)
        self.df = pd.DataFrame({"metric_a": a, "metric_b": b})

        # Inject extreme outlier
        self.df.loc[0, "metric_a"] = 250.0  # extreme high Z-score & IQR outlier
        # Inject correlation breaker / bivariate coupling anomaly
        self.df.loc[1, "metric_a"] = 50.0
        self.df.loc[1, "metric_b"] = -100.0  # Breaks coupling

    def test_rule_instantiation_and_validation(self) -> None:
        rule = HeuristicRule(
            rule_id="TEST_R1",
            name="Test Rule",
            description="Testing severity validation",
            severity=0.75,
            rule_type="test",
            features=["metric_a"],
            predicate=lambda d: d["metric_a"] > 100,
        )
        self.assertEqual(rule.severity, 0.75)

        with self.assertRaises(ValueError):
            HeuristicRule(
                rule_id="BAD_R",
                name="Bad",
                description="Bad",
                severity=1.5,
                rule_type="test",
                features=["a"],
                predicate=lambda d: d["a"] > 0,
            )

    def test_synthesize_heuristic_rules(self) -> None:
        rules = synthesize_heuristic_rules(self.df)
        self.assertGreater(len(rules), 0)
        rule_types = {r.rule_type for r in rules}
        self.assertIn("extreme_tail", rule_types)
        self.assertIn("cross_feature_coupling", rule_types)

    def test_evaluate_heuristic_rules_and_attribution(self) -> None:
        result = evaluate_heuristic_rules(self.df)
        self.assertGreater(result["total_anomalies"], 0)
        self.assertIn(0, result["anomalous_indices"])

        # Check explainability attribution
        top_anom = result["top_anomalies"]
        self.assertGreater(len(top_anom), 0)
        record_0 = next((r for r in top_anom if r["index"] == 0), None)
        self.assertIsNotNone(record_0)
        self.assertGreater(record_0["composite_score"], 0.0)
        self.assertGreater(record_0["violated_rules_count"], 0)

    def test_heuristic_rules_empty_dataset(self) -> None:
        empty_df = pd.DataFrame()
        result = evaluate_heuristic_rules(empty_df)
        self.assertEqual(result["total_anomalies"], 0)
        self.assertEqual(result["anomalous_indices"], [])


class TestImputationDriftBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        n = 200
        self.u = np.random.normal(0.0, 1.0, size=n)
        self.v = np.random.normal(0.5, 1.2, size=n)
        self.df = pd.DataFrame({
            "feature_x": np.random.normal(25.0, 4.0, size=n),
            "feature_y": np.random.exponential(5.0, size=n),
        })

    def test_compute_wasserstein_1d(self) -> None:
        # Same distribution should yield approximately 0 distance
        w_ident = compute_wasserstein_1d(self.u, self.u)
        self.assertAlmostEqual(w_ident, 0.0, places=2)

        # Shifted distribution
        w_shifted = compute_wasserstein_1d(self.u, self.u + 5.0)
        self.assertAlmostEqual(w_shifted, 5.0, places=1)

        # Empty array error
        with self.assertRaises(ValueError):
            compute_wasserstein_1d(np.array([]), self.u)

    def test_compute_ks_statistic_1d(self) -> None:
        ks_ident = compute_ks_statistic_1d(self.u, self.u)
        self.assertAlmostEqual(ks_ident, 0.0, places=2)

        ks_diff = compute_ks_statistic_1d(self.u, self.u + 10.0)
        self.assertAlmostEqual(ks_diff, 1.0, places=1)

        with self.assertRaises(ValueError):
            compute_ks_statistic_1d(self.u, np.array([]))

    def test_compute_population_stability_index(self) -> None:
        psi_ident = compute_population_stability_index(self.u, self.u)
        self.assertAlmostEqual(psi_ident, 0.0, places=2)

        psi_diff = compute_population_stability_index(self.u, self.u + 2.0)
        self.assertGreater(psi_diff, 0.1)

        with self.assertRaises(ValueError):
            compute_population_stability_index(self.u, self.v, bins=1)

    def test_benchmark_imputation_drift_evaluation(self) -> None:
        result = benchmark_imputation_drift(self.df, missing_rate=0.15, seed=42)
        self.assertIn("leaderboard", result)
        self.assertIn("optimal_strategy", result)
        self.assertIn("per_feature_results", result)

        leaderboard = result["leaderboard"]
        self.assertEqual(len(leaderboard), 6)  # 6 benchmarked strategies
        # Ranked by composite drift score ascending
        scores = [item["composite_drift_score"] for item in leaderboard]
        self.assertEqual(scores, sorted(scores))

    def test_benchmark_imputation_drift_domain_errors(self) -> None:
        with self.assertRaises(ValueError):
            benchmark_imputation_drift(self.df, missing_rate=0.0)
        with self.assertRaises(ValueError):
            benchmark_imputation_drift(self.df, missing_rate=0.75)


class TestSummaryIntegration(unittest.TestCase):
    def test_generate_summary_contains_cs_engines(self) -> None:
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0],
            "c": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y", "x"],
        })
        summary = generate_summary(df)

        self.assertIn("dependency_graph", summary)
        self.assertIn("heuristic_rules", summary)
        self.assertIn("imputation_drift", summary)
        self.assertIn("heuristic_rules", summary["outliers"])


if __name__ == "__main__":
    unittest.main()
