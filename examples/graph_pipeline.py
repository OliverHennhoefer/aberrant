"""Branched preprocessing, raw feature bypass and staged online initialization."""

from aberrant.compose import GraphNode, GraphPipeline, MergeNode
from aberrant.model.stat import MovingCovariance
from aberrant.transform import IncrementalPCA, Select, StandardScaler


def main() -> None:
    graph = GraphPipeline(
        [
            GraphNode("electrical", Select(features=["voltage", "current"])),
            GraphNode("scaled", StandardScaler(), "electrical"),
            GraphNode("projected", IncrementalPCA(n_components=1, warm_up=5), "scaled"),
            GraphNode("temperature", Select(features=["temperature"])),
            MergeNode("features", ("projected", "temperature")),
        ],
        MovingCovariance(window_size=32),
        output="features",
        model_minimum_updates=2,
    )
    for index in range(20):
        event = {
            "voltage": 230.0 + index % 3,
            "current": 4.0 + index * 0.1,
            "temperature": 20.0 + index * 0.05,
        }
        if graph.readiness.ready:
            print(f"{index}: score={graph.score_one(event):.6f}")
        graph.learn_one(event)
    assert graph.model_updates == 16
    assert graph.readiness.ready


if __name__ == "__main__":
    main()
