import argparse
import copy
import json
import os
import shutil
import sys
import time
from itertools import product

import pandas as pd
import yaml

from src.evaluation.comparison import save_experiment_comparison
from src.training.config_loader import load_config
from src.training.train import BandpowerTrainingPipeline


class BenchmarkRunner:
    """Run a grid benchmark defined in a YAML file."""

    def __init__(self, benchmark_path, force=False, limit=None, top_n=15):
        self.benchmark_path = benchmark_path
        self.force = force
        self.limit = limit
        self.top_n = top_n
        self.benchmark_config = self.load_benchmark_config(benchmark_path)
        self.benchmark_name = self.benchmark_config["benchmark"]["name"]
        self.base_config = load_config(self.benchmark_config["benchmark"]["base_config"])
        self.experiments_dir = os.path.join("reports", "experiments")
        self.benchmarks_dir = os.path.join("reports", "benchmarks")

    def load_benchmark_config(self, benchmark_path):
        with open(benchmark_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def deep_merge(self, base, override):
        merged = copy.deepcopy(base)

        for key, value in override.items():
            base_value = merged.get(key)

            if isinstance(base_value, dict) and isinstance(value, dict):
                merged[key] = self.deep_merge(base_value, value)
            else:
                merged[key] = copy.deepcopy(value)

        return merged

    def build_experiment_name(self, feature_set_name, model_name, normalization_name, artifact_name):
        return (
            self.benchmark_name
            + "__fs_" + feature_set_name
            + "__model_" + model_name
            + "__norm_" + normalization_name
            + "__art_" + artifact_name
        )

    def build_experiment_config(
        self,
        feature_set_name,
        feature_groups,
        model_name,
        model_config,
        normalization_name,
        normalization_config,
        artifact_name,
        artifact_config,
    ):
        experiment_name = self.build_experiment_name(
            feature_set_name,
            model_name,
            normalization_name,
            artifact_name,
        )

        override = {
            "experiment": {
                "name": experiment_name,
                "benchmark": self.benchmark_name,
                "feature_set": feature_set_name,
                "model_name": model_name,
                "normalization_name": normalization_name,
                "artifact_mode": artifact_name,
            },
            "features": {
                "feature_groups": feature_groups,
            },
            "model": model_config,
            "normalization": normalization_config,
            "artifacts": artifact_config,
        }

        return self.deep_merge(self.base_config, override)

    def iter_experiment_configs(self):
        feature_sets = self.benchmark_config["feature_sets"]
        models = self.benchmark_config["models"]
        normalizations = self.benchmark_config["normalizations"]
        artifact_modes = self.benchmark_config["artifact_modes"]

        for (feature_set_name, feature_groups), (model_name, model_config), (normalization_name, normalization_config), (artifact_name, artifact_config) in product(
            feature_sets.items(),
            models.items(),
            normalizations.items(),
            artifact_modes.items(),
        ):
            yield self.build_experiment_config(
                feature_set_name,
                feature_groups,
                model_name,
                model_config,
                normalization_name,
                normalization_config,
                artifact_name,
                artifact_config,
            )

    def experiment_metrics_path(self, experiment_name):
        return os.path.join(self.experiments_dir, experiment_name, "metrics.json")

    def should_skip(self, config):
        if self.force:
            return False

        experiment_name = config["experiment"]["name"]
        return os.path.exists(self.experiment_metrics_path(experiment_name))

    def validate_feature_table_for_benchmark(self):
        input_file = self.base_config["data"]["input_file"]

        if not os.path.exists(input_file):
            raise FileNotFoundError("Feature table not found: " + input_file)

    def run_one(self, config, index, total):
        experiment_name = config["experiment"]["name"]

        if self.should_skip(config):
            print(f"[{index}/{total}] SKIP {experiment_name}")
            return "skipped"

        print(f"[{index}/{total}] RUN  {experiment_name}")
        start = time.time()

        pipeline = BandpowerTrainingPipeline(config)
        summary = pipeline.run()

        elapsed = time.time() - start
        print(
            f"[{index}/{total}] DONE {experiment_name} "
            f"balanced_accuracy={summary['mean_balanced_accuracy']:.6f} "
            f"f1={summary['mean_f1']:.6f} "
            f"seconds={elapsed:.1f}"
        )

        return "completed"

    def save_manifest(self, configs):
        os.makedirs(self.benchmarks_dir, exist_ok=True)

        rows = []
        for config in configs:
            experiment = config["experiment"]
            rows.append({
                "experiment": experiment["name"],
                "feature_set": experiment["feature_set"],
                "model_name": experiment["model_name"],
                "normalization_name": experiment["normalization_name"],
                "artifact_mode": experiment["artifact_mode"],
            })

        manifest_path = os.path.join(self.benchmarks_dir, self.benchmark_name + "_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)

        print("Saved benchmark manifest:", manifest_path)

    def parse_experiment_name(self, experiment_name):
        parsed = {
            "feature_set": None,
            "model": None,
            "normalization": None,
            "artifact_mode": None,
        }

        for part in experiment_name.split("__")[1:]:
            key, value = part.split("_", 1)

            if key == "fs":
                parsed["feature_set"] = value
            elif key == "model":
                parsed["model"] = value
            elif key == "norm":
                parsed["normalization"] = value
            elif key == "art":
                parsed["artifact_mode"] = value

        return parsed

    def build_benchmark_results_df(self, comparison_df):
        benchmark_df = comparison_df[
            comparison_df["experiment"].str.startswith(self.benchmark_name + "__")
        ].copy()

        if len(benchmark_df) == 0:
            return benchmark_df

        parsed_df = benchmark_df["experiment"].apply(self.parse_experiment_name).apply(pd.Series)
        benchmark_df = pd.concat([benchmark_df, parsed_df], axis=1)
        benchmark_df = benchmark_df.sort_values("mean_balanced_accuracy", ascending=False)
        benchmark_df = benchmark_df.reset_index(drop=True)
        benchmark_df.insert(0, "rank", range(1, len(benchmark_df) + 1))

        return benchmark_df

    def save_top_options(self, benchmark_df):
        top_df = benchmark_df.head(self.top_n).copy()
        output_dir = os.path.join(self.benchmarks_dir, f"{self.benchmark_name}_top{self.top_n}")

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        top_df.to_csv(os.path.join(output_dir, f"top{self.top_n}_summary.csv"), index=False)

        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as file:
            file.write(f"# Top {self.top_n} benchmark options\n\n")
            file.write("Ranking sorted by mean_balanced_accuracy.\n\n")
            file.write("| rank | feature_set | model | normalization | artifacts | balanced_acc | f1 | precision | recall | roc_auc |\n")
            file.write("|---:|---|---|---|---|---:|---:|---:|---:|---:|\n")

            for _, row in top_df.iterrows():
                file.write(
                    f"| {int(row['rank'])} | {row['feature_set']} | {row['model']} | "
                    f"{row['normalization']} | {row['artifact_mode']} | "
                    f"{row['mean_balanced_accuracy']:.6f} | {row['mean_f1']:.6f} | "
                    f"{row['mean_precision']:.6f} | {row['mean_recall']:.6f} | "
                    f"{row['mean_roc_auc']:.6f} |\n"
                )

        for _, row in top_df.iterrows():
            rank = int(row["rank"])
            experiment_name = row["experiment"]
            source_dir = os.path.join(self.experiments_dir, experiment_name)
            target_dir = os.path.join(output_dir, f"{rank:02d}_{experiment_name}")
            os.makedirs(target_dir, exist_ok=True)

            files_to_copy = [
                "metrics.json",
                "config_used.yaml",
                "model_metadata.json",
                "feature_columns.json",
                "metrics_per_subject.csv",
                "metrics_per_task.csv",
                "metrics_per_group.csv",
                "artifact_prediction_correlation.csv",
                "confusion_matrix.png",
                "roc_curve.png",
            ]

            for filename in files_to_copy:
                source = os.path.join(source_dir, filename)
                if os.path.exists(source):
                    shutil.copy2(source, os.path.join(target_dir, filename))

            summary = row.to_dict()
            summary["source_dir"] = source_dir
            with open(os.path.join(target_dir, "option_summary.json"), "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2)

        return output_dir

    def print_group_winners(self, benchmark_df, group_column, title):
        winners = (
            benchmark_df.groupby(group_column)["mean_balanced_accuracy"]
            .max()
            .sort_values(ascending=False)
        )

        print()
        print(title)
        for name, value in winners.items():
            print(f"  {name}: {value:.6f}")

    def print_top_table(self, benchmark_df):
        top_df = benchmark_df.head(self.top_n)

        print()
        print(f"Top {len(top_df)} options by balanced accuracy")
        print(
            top_df[[
                "rank",
                "feature_set",
                "model",
                "normalization",
                "artifact_mode",
                "mean_balanced_accuracy",
                "mean_f1",
                "mean_roc_auc",
            ]].to_string(index=False)
        )

    def print_benchmark_summary(self, comparison_df):
        benchmark_df = self.build_benchmark_results_df(comparison_df)

        print()
        print("=" * 80)
        print("Benchmark summary")
        print("=" * 80)

        if len(benchmark_df) == 0:
            print("No completed experiments found for benchmark:", self.benchmark_name)
            return

        best = benchmark_df.iloc[0]
        print("Benchmark:", self.benchmark_name)
        print("Completed benchmark experiments:", len(benchmark_df))
        print()
        print("Best option")
        print("  experiment:", best["experiment"])
        print("  feature_set:", best["feature_set"])
        print("  model:", best["model"])
        print("  normalization:", best["normalization"])
        print("  artifacts:", best["artifact_mode"])
        print(f"  balanced_accuracy: {best['mean_balanced_accuracy']:.6f}")
        print(f"  f1: {best['mean_f1']:.6f}")
        print(f"  precision: {best['mean_precision']:.6f}")
        print(f"  recall: {best['mean_recall']:.6f}")
        print(f"  roc_auc: {best['mean_roc_auc']:.6f}")

        self.print_top_table(benchmark_df)
        self.print_group_winners(benchmark_df, "model", "Best balanced accuracy by model")
        self.print_group_winners(benchmark_df, "feature_set", "Best balanced accuracy by feature set")
        self.print_group_winners(benchmark_df, "normalization", "Best balanced accuracy by normalization")
        self.print_group_winners(benchmark_df, "artifact_mode", "Best balanced accuracy by artifact mode")

        top_dir = self.save_top_options(benchmark_df)

        print()
        print("Saved files")
        print("  comparison:", os.path.join(self.experiments_dir, "comparison.csv"))
        print("  top options:", top_dir)
        print("=" * 80)

    def run(self):
        self.validate_feature_table_for_benchmark()

        configs = list(self.iter_experiment_configs())
        total = len(configs)

        if self.limit is not None:
            configs_to_run = configs[:self.limit]
        else:
            configs_to_run = configs

        print("Benchmark:", self.benchmark_name)
        print("Total experiments:", total)
        print("Scheduled experiments:", len(configs_to_run))
        print("Force:", self.force)

        self.save_manifest(configs)

        counts = {"completed": 0, "skipped": 0}
        for index, config in enumerate(configs_to_run, start=1):
            status = self.run_one(config, index, len(configs_to_run))
            counts[status] = counts.get(status, 0) + 1

        comparison_df = save_experiment_comparison(self.experiments_dir)
        print("Saved comparison for experiments:", len(comparison_df))
        print("Completed:", counts.get("completed", 0))
        print("Skipped:", counts.get("skipped", 0))
        self.print_benchmark_summary(comparison_df)


def parse_args(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_config")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args(arguments)


def main():
    args = parse_args(sys.argv[1:])
    runner = BenchmarkRunner(
        args.benchmark_config,
        force=args.force,
        limit=args.limit,
        top_n=args.top_n,
    )
    runner.run()


if __name__ == "__main__":
    main()
