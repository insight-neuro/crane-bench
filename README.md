# Crane Bench

CraneBench is a framework for building benchmarks to evaluate the performance of foundation models of neural data. The goal of this project is to provide a standardized way to evaluate the performance of foundation models on a variety of tasks, making it easier for researchers to compare different models on the same tasks. The framework is designed to be flexible and extensible, allowing researchers to easily add new tasks and evaluation metrics.

## Installation

To install CraneBench, you can use pip:

```bash
pip install "crane_bench @ git+https://github.com/insight-neuro/crane-bench"
```

## Defining a Benchmark

To define a benchmark, you need to create a subclass of `crane_bench.BrainBench`.


A BrainBench defines a fixed set of evaluation tasks, organized into
named task groups and annotated with tags for selection and filtering.
Task definitions are declared by subclasses via decorated methods.

Tasks are defined with the `@task_group` decorator and
collected eagerly at subclass initialization time.

The following attributes and methods must be defined in the subclass:
- `name`: A human-readable name of the benchmark.
- `version`: (Optional) Version of the benchmark.
- `reference`: (Optional) Reference or citation for the benchmark.
- `default_tags`: (Optional) Default tags to use if none are specified.

To define a task, you can use the `@task_group` decorator on a method that returns a list of `Task` objects. Each `Task` object represents a specific evaluation task and should include the necessary information for running the task, such as the dataset, evaluation metric, and any other relevant parameters.

Here is an example of how to define a benchmark with two task groups:

```python
from crane_bench import BrainBench, Task, task_group

class MyBenchmark(BrainBench):
    name = "My Benchmark"
    version = "1.0"
    reference = "https://example.com/my-benchmark"

    # 1) Simple 1→1 task (no expansion)
    @task_group
    cross_subject = [
        Task(
            train=Subject(0, session=0),
            test=Subject(1, session=0),
        )
    ]

    # 2) Explicit Cartesian expansion over iterables
    @task_group(expand="cartesian")
    cross_subject_cartesian = [
        Task(
            train=Subject(subject=[0, 1], session=0),
            test=Subject(subject=[2, 3], session=1),
        )
    ]

    # 3) Explicit paired (zip) expansion
    @task_group(expand="zip")
    paired_sessions = [
        Task(
            train=Subject(subject=[0, 1], session=[0, 1]),
            test=Subject(subject=[0, 1], session=[2, 3]),
        )
    ]

    # 4) Train / test splits within the same subject & session
    @task_group
    within_session_split = [
        Task(
            train=Subject(
                0,
                session=0,
                splits=[(0, 15)]
            ),
            test=Subject(
                0,
                session=0,
                splits=[(15, 30)]
            ),
        )
    ]

    # 5) Per-group trainer override + expansion
    @task_group(trainer=SpecializedTrainer(), expand="cartesian")
    specialized = [
        Task(
            train=Subject(subject=[0, 1], session=1),
            test=Subject(subject=[0, 1], session=2),
        )
    ]
```

## Disclaimer

This project is in its early stages and is not yet ready for production use. The API is subject to change, and there may be bugs and missing features. We welcome contributions and feedback from the community to help improve the framework.
