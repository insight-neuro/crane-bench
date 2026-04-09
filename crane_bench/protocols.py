from typing import Protocol, TypeVar, runtime_checkable

from crane import BrainFeatureExtractor, BrainModel

from crane_bench.artifacts import TaskResult
from crane_bench.bench import BrainBench
from crane_bench.data import NeuralData

ND = TypeVar("ND", bound=NeuralData, contravariant=True)


@runtime_checkable
class TrainFn(Protocol[ND]):
    def __call__(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        train_data: ND,
    ) -> BrainModel: ...


@runtime_checkable
class TestFn(Protocol[ND]):
    def __call__(
        self,
        bench: BrainBench,
        model: BrainModel,
        featurizer: BrainFeatureExtractor,
        test_data: ND,
    ) -> TaskResult: ...
