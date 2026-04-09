import torch
from crane import BrainFeatureExtractor, BrainModel

from crane_bench.artifacts import ExecutionPlan, TaskResult
from crane_bench.bench import BrainBench
from crane_bench.exec.base import Executor, TestRunner
from crane_bench.protocols import TrainFn


class SequentialExecutor(Executor):
    """Executor that runs tasks sequentially."""

    def run(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        plan: ExecutionPlan,
    ) -> list[TaskResult]:
        runner = TestRunner(bench, featurizer)
        results: list[TaskResult] = []

        train_fn: TrainFn = bench.train_fn

        for task_unit in plan:
            trained_model = train_fn(bench, model, featurizer, task_unit.train_data)

            with torch.no_grad():
                results.extend(runner.run(trained_model, task_unit.tasks))

        return results
