import os
import subprocess
import sys

from src.evaluation.comparison import save_experiment_comparison


class ExperimentRunner:
    """Run configured project experiments."""

    def __init__(self, include_riemannian=False, comparison_only=False):
        self.include_riemannian = include_riemannian
        self.comparison_only = comparison_only
        self.commands = []

    def add_training_commands(self):
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

            self.commands.append(command)

    def add_task_transfer_commands(self):
        first_command = []
        first_command.append(sys.executable)
        first_command.append("-m")
        first_command.append("src.evaluation.task_transfer")
        first_command.append("configs/model_bandpower_logreg.yaml")
        first_command.append("TASK1")
        first_command.append("TASK2")
        self.commands.append(first_command)

        second_command = []
        second_command.append(sys.executable)
        second_command.append("-m")
        second_command.append("src.evaluation.task_transfer")
        second_command.append("configs/model_bandpower_logreg.yaml")
        second_command.append("TASK2")
        second_command.append("TASK1")
        self.commands.append(second_command)

    def add_riemannian_command(self):
        command = []
        command.append(sys.executable)
        command.append("-m")
        command.append("src.training.train_riemannian")
        command.append("configs/model_riemannian_logreg.yaml")
        self.commands.append(command)

    def build_commands(self):
        if self.comparison_only == True:
            return

        self.add_training_commands()
        self.add_task_transfer_commands()

        if self.include_riemannian == True:
            self.add_riemannian_command()

    def run_command(self, command):
        print("Running:", " ".join(command))

        result = subprocess.run(command)

        if result.returncode != 0:
            print("Command failed.")
            sys.exit(result.returncode)

    def run_commands(self):
        for command in self.commands:
            self.run_command(command)

    def write_comparison(self):
        experiments_dir = os.path.join("reports", "experiments")

        if not os.path.exists(experiments_dir):
            print("No reports/experiments directory found. Skipping comparison.")
            return

        comparison_df = save_experiment_comparison(experiments_dir)
        print("Saved comparison for experiments:", len(comparison_df))

    def run(self):
        self.build_commands()
        self.run_commands()
        self.write_comparison()


def parse_arguments(arguments):
    include_riemannian = False
    comparison_only = False

    for argument in arguments:
        if argument == "--include-riemannian":
            include_riemannian = True

        if argument == "--comparison-only":
            comparison_only = True

    return include_riemannian, comparison_only


def main():
    include_riemannian, comparison_only = parse_arguments(sys.argv)

    runner = ExperimentRunner(include_riemannian, comparison_only)
    runner.run()


if __name__ == "__main__":
    main()
