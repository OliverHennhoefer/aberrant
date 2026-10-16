"""Branched preprocessing, raw feature bypass and staged online initialization."""

from aberrant.compose import GraphNode, GraphPipeline, MergeNode, Select
from aberrant.model.stat import MovingCovariance
from aberrant.transform.preprocessing import StandardScaler
from aberrant.transform.projection import IncrementalPCA


def main() -> None:
    graph = GraphPipeline(
        [
            GraphNode("electrical", Select(["voltage", "current"])),
            GraphNode("scaled", StandardScaler(), "electrical"),
            GraphNode("projected", IncrementalPCA(1, n0=5), "scaled"),
            GraphNode("temperature", Select(["temperature"])),
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
