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

    def test_synthesize_heuristic_rules_uses_pearson_exclusively(self) -> None:
        """synthesize_heuristic_rules must use Pearson correlation exclusively for linear coupling rules."""
        # Nonlinear monotonic relationship: y = exp(x)
        # Spearman rank correlation is 1.0, but Pearson correlation is ~0.69 (< 0.70 threshold)
        x = np.linspace(0.0, 10.0, 100)
        y = np.exp(x)
        nonlinear_df = pd.DataFrame({"feat_x": x, "feat_y": y})

        rules = synthesize_heuristic_rules(nonlinear_df)
        coupling_rules = [r for r in rules if r.rule_type == "cross_feature_coupling"]

        # Because Pearson correlation is < 0.70, no linear coupling rule should be synthesized
        self.assertEqual(
            len(coupling_rules),
            0,
            "Spearman rank correlation must not trigger linear coupling rules when Pearson r < 0.70",
        )

        # For a truly linear relationship with Pearson r >= 0.70, rule description and slope must match Pearson
        np.random.seed(42)
        lin_a = np.linspace(0.0, 50.0, 100)
        lin_b = lin_a * 1.5 + np.random.normal(0.0, 2.0, size=100)
        linear_df = pd.DataFrame({"feat_a": lin_a, "feat_b": lin_b})
        pearson_val = float(linear_df["feat_a"].corr(linear_df["feat_b"], method="pearson"))
        lin_rules = synthesize_heuristic_rules(linear_df)
        lin_coupling = [r for r in lin_rules if r.rule_type == "cross_feature_coupling"]
        self.assertGreater(len(lin_coupling), 0)
        self.assertIn(f"r={pearson_val:.2f}", lin_coupling[0].description)

    def test_synthesize_heuristic_rules_constant_and_insufficient_features(self) -> None:
        """synthesize_heuristic_rules handles constant features and single-feature DataFrames without error."""
        # Single feature DataFrame
        single_df = pd.DataFrame({"feat_1": [1.0, 2.0, 3.0, 4.0, 5.0]})
        rules_single = synthesize_heuristic_rules(single_df)
        self.assertEqual(len([r for r in rules_single if r.rule_type == "cross_feature_coupling"]), 0)

        # Constant features (zero variance)
        constant_df = pd.DataFrame({"const_a": [5.0] * 50, "const_b": [10.0] * 50})
        rules_const = synthesize_heuristic_rules(constant_df)
        self.assertEqual(len([r for r in rules_const if r.rule_type == "cross_feature_coupling"]), 0)

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

    def test_evaluate_heuristic_rules_attribution_pruning(self) -> None:
        """Verify only anomalous rows crossing the threshold receive attribution entries."""
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "val_a": np.random.normal(10.0, 1.0, size=n),
            "val_b": np.random.normal(20.0, 1.0, size=n),
        })
        # Inject exactly two extreme outliers
        df.loc[7, "val_a"] = 150.0
        df.loc[99, "val_a"] = 180.0

        result = evaluate_heuristic_rules(df)
        anomalous_indices = result["anomalous_indices"]
        self.assertIn(7, anomalous_indices)
        self.assertIn(99, anomalous_indices)

        # Attribution dictionary must only contain keys for anomalous rows, not all 500 rows
        attributions = result["record_attributions"]
        self.assertLessEqual(len(attributions), len(anomalous_indices))
        for key in attributions.keys():
            self.assertIn(key, anomalous_indices)

    def test_evaluate_heuristic_rules_vectorized_scoring_parity(self) -> None:
        """Verify vectorized composite scores and severe rule thresholding match exact mathematical weights."""
        df = pd.DataFrame({
            "feature_x": [10.0, 20.0, 30.0, 5.0, 15.0],
            "feature_y": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        rule_a = HeuristicRule(
            rule_id="RULE_A",
            name="Rule A",
            description="Feature X > 25",
            severity=0.60,
            rule_type="test",
            features=["feature_x"],
            predicate=lambda d: d["feature_x"] > 25.0,
        )
        rule_b = HeuristicRule(
            rule_id="RULE_B",
            name="Rule B",
            description="Feature Y <= 2",
            severity=0.40,
            rule_type="test",
            features=["feature_y"],
            predicate=lambda d: d["feature_y"] <= 2.0,
        )
        # Total weight = 0.60 + 0.40 = 1.0
        # Row 0: feature_x=10, feature_y=1 -> violates B -> weight 0.40 -> score 0.40 (>= 0.20)
        # Row 1: feature_x=20, feature_y=2 -> violates B -> weight 0.40 -> score 0.40 (>= 0.20)
        # Row 2: feature_x=30, feature_y=3 -> violates A -> weight 0.60 -> score 0.60 (>= 0.20)
        # Row 3: feature_x=5,  feature_y=4 -> violates neither -> weight 0.0 -> score 0.0 (< 0.20)
        # Row 4: feature_x=15, feature_y=5 -> violates neither -> weight 0.0 -> score 0.0 (< 0.20)
        result = evaluate_heuristic_rules(df, rules=[rule_a, rule_b])
        self.assertEqual(result["total_anomalies"], 3)
        self.assertEqual(result["anomalous_indices"], [0, 1, 2])

        top_scores = {r["index"]: r["composite_score"] for r in result["top_anomalies"]}
        self.assertEqual(top_scores[2], 0.60)
        self.assertEqual(top_scores[0], 0.40)
        self.assertEqual(top_scores[1], 0.40)

    def test_evaluate_heuristic_rules_throughput_large_dataframe(self) -> None:
        """Verify evaluate_heuristic_rules handles large datasets efficiently without O(N) dict overhead."""
        np.random.seed(42)
        n = 10_000
        large_df = pd.DataFrame({
            "col_a": np.random.normal(50.0, 5.0, size=n),
            "col_b": np.random.normal(100.0, 10.0, size=n),
        })
        large_df.loc[5, "col_a"] = 500.0
        result = evaluate_heuristic_rules(large_df)
        self.assertGreater(result["total_rules"], 0)
        self.assertGreater(result["total_anomalies"], 0)
        self.assertIn(5, result["anomalous_indices"])
        self.assertLess(len(result["record_attributions"]), 51)

    def test_evaluate_heuristic_rules_edge_cases(self) -> None:
        """Verify edge cases: non-integer string indices, faulty predicate containment, and zero anomalies."""
        # 1. Non-integer string indices
        df_str_idx = pd.DataFrame(
            {"val": [10.0, 20.0, 1000.0]},
            index=["rec_alpha", "rec_beta", "rec_gamma"],
        )
        rule = HeuristicRule(
            rule_id="RULE_HIGH",
            name="High Value",
            description="val > 100",
            severity=0.90,
            rule_type="test",
            features=["val"],
            predicate=lambda d: d["val"] > 100.0,
        )
        res_str = evaluate_heuristic_rules(df_str_idx, rules=[rule])
        self.assertEqual(res_str["total_anomalies"], 1)
        self.assertEqual(res_str["anomalous_indices"], ["rec_gamma"])
        self.assertIn("rec_gamma", res_str["record_attributions"])

        # 2. Faulty rule predicate containment (swallows exception without crashing)
        def faulty_pred(d: pd.DataFrame) -> pd.Series:
            raise RuntimeError("Corrupted predicate logic")

        bad_rule = HeuristicRule(
            rule_id="RULE_BAD",
            name="Bad Rule",
            description="Throws error",
            severity=0.50,
            rule_type="test",
            features=["val"],
            predicate=faulty_pred,
        )
        res_faulty = evaluate_heuristic_rules(df_str_idx, rules=[bad_rule, rule])
        self.assertEqual(res_faulty["total_anomalies"], 1)
        self.assertEqual(res_faulty["anomalous_indices"], ["rec_gamma"])

        # 3. Zero anomalies scenario
        clean_df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        untriggered_rule = HeuristicRule(
            rule_id="RULE_UNTRIGGERED",
            name="Never True",
            description="val > 9999",
            severity=0.50,
            rule_type="test",
            features=["val"],
            predicate=lambda d: d["val"] > 9999.0,
        )
        res_zero = evaluate_heuristic_rules(clean_df, rules=[untriggered_rule])
        self.assertEqual(res_zero["total_anomalies"], 0)
        self.assertEqual(res_zero["anomalous_indices"], [])
        self.assertEqual(res_zero["record_attributions"], {})
        self.assertEqual(res_zero["top_anomalies"], [])


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
