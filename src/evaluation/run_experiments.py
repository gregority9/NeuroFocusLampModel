import os
import subprocess
import sys

from src.evaluation.comparison import save_experiment_comparison


def run_command(command):
    """Run one command used in the experiment list."""
    print("Running:", " ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        print("Command failed.")
        sys.exit(result.returncode)


def add_training_commands(commands):
    """Append standard bandpower experiments to the command list."""
    configs = []
    configs.append("configs/model_bandpower_logreg.yaml")
    configs.append("configs/model_bandpower_svm.yaml")
    configs.append("configs/model_bandpower_logreg_subject_relative.yaml")
    configs.append("configs/model_bandpower_logreg_no_artifact_rejection.yaml")
    configs.append("configs/model_bandpower_logreg_no_af3_af4.yaml")
    configs.append("configs/model_bandpower_logreg_no_high_beta.yaml")
    configs.append("configs/model_bandpower_logreg_occipital_only.yaml")
    configs.append("configs/model_bandpower_logreg_frontal_only.yaml")

    for config_path in configs:
        command = []
        command.append(sys.executable)
        command.append("-m")
        command.append("src.training.train")
        command.append(config_path)

        commands.append(command)


def add_task_transfer_commands(commands):
    """Append task-transfer experiments to the command list."""
    first_command = []
    first_command.append(sys.executable)
    first_command.append("-m")
    first_command.append("src.evaluation.task_transfer")
    first_command.append("configs/model_bandpower_logreg.yaml")
    first_command.append("memory")
    first_command.append("words")
    commands.append(first_command)

    second_command = []
    second_command.append(sys.executable)
    second_command.append("-m")
    second_command.append("src.evaluation.task_transfer")
    second_command.append("configs/model_bandpower_logreg.yaml")
    second_command.append("words")
    second_command.append("memory")
    commands.append(second_command)


def add_riemannian_command(commands):
    """Append the Riemannian experiment command."""
    command = []
    command.append(sys.executable)
    command.append("-m")
    command.append("src.training.train_riemannian")
    command.append("configs/model_riemannian_logreg.yaml")
    commands.append(command)


def write_comparison():
    """Create comparison.csv from all experiment metrics."""
    experiments_dir = os.path.join("reports", "experiments")

    if not os.path.exists(experiments_dir):
        print("No reports/experiments directory found. Skipping comparison.")
        return

    comparison_df = save_experiment_comparison(experiments_dir)
    print("Saved comparison for experiments:", len(comparison_df))


def main():
    include_riemannian = False
    comparison_only = False

    for argument in sys.argv:
        if argument == "--include-riemannian":
            include_riemannian = True

        if argument == "--comparison-only":
            comparison_only = True

    commands = []

    if comparison_only == False:
        add_training_commands(commands)
        add_task_transfer_commands(commands)

        if include_riemannian == True:
            add_riemannian_command(commands)

    for command in commands:
        run_command(command)

    write_comparison()


if __name__ == "__main__":
    main()
