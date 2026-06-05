import os
import subprocess
import sys

from src.evaluation.comparison import save_experiment_comparison


BANDPOWER_CONFIGS = [
    "configs/model_bandpower_logreg.yaml",
    "configs/model_bandpower_svm.yaml",
    "configs/model_bandpower_logreg_subject_relative.yaml",
    "configs/model_bandpower_logreg_no_artifact_rejection.yaml",
    "configs/model_bandpower_logreg_no_af3_af4.yaml",
    "configs/model_bandpower_logreg_no_high_beta.yaml",
    "configs/model_bandpower_logreg_occipital_only.yaml",
    "configs/model_bandpower_logreg_frontal_only.yaml",
]


TASK_TRANSFER_EXPERIMENTS = [
    ("memory", "words"),
    ("words", "memory"),
]


def run_command(command):
    """Run one experiment command and stop if it fails."""
    print("Running:", " ".join(command))

    completed_process = subprocess.run(command)

    if completed_process.returncode != 0:
        raise SystemExit(completed_process.returncode)


def run_bandpower_experiments():
    """Run all bandpower-based experiments."""
    for config_path in BANDPOWER_CONFIGS:
        run_command([
            sys.executable,
            "-m",
            "src.training.train",
            config_path,
        ])


def run_task_transfer_experiments():
    """Run task-transfer experiments from the project requirements."""
    base_config = "configs/model_bandpower_logreg.yaml"

    for train_task, test_task in TASK_TRANSFER_EXPERIMENTS:
        run_command([
            sys.executable,
            "-m",
            "src.evaluation.task_transfer",
            base_config,
            train_task,
            test_task,
        ])


def run_riemannian_experiment():
    """Run Riemannian Tangent Space + Logistic Regression."""
    run_command([
        sys.executable,
        "-m",
        "src.training.train_riemannian",
        "configs/model_riemannian_logreg.yaml",
    ])


def write_comparison():
    """Create comparison.csv from all experiment metrics."""
    experiments_dir = os.path.join("reports", "experiments")

    if not os.path.exists(experiments_dir):
        print("No reports/experiments directory found. Skipping comparison.")
        return

    comparison_df = save_experiment_comparison(experiments_dir)
    print("Saved comparison for experiments:", len(comparison_df))


def main():
    include_riemannian = "--include-riemannian" in sys.argv
    comparison_only = "--comparison-only" in sys.argv

    if not comparison_only:
        run_bandpower_experiments()
        run_task_transfer_experiments()

        if include_riemannian:
            run_riemannian_experiment()

    write_comparison()


if __name__ == "__main__":
    main()
