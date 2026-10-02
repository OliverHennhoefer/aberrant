"""Compare recent-window and frozen calibration on a deterministic level shift."""

from aberrant.transform import RollingRobustScaler


def main() -> None:
    adaptive = RollingRobustScaler(window_size=5, min_samples=5)
    frozen = RollingRobustScaler(window_size=5, min_samples=5)
    for value in [0, 2, 4, 6, 8]:
        adaptive.learn_one({"x": float(value)})
        frozen.learn_one({"x": float(value)})
    frozen.freeze()

    for value in [10, 12, 14, 16, 18]:
        event = {"x": float(value)}
        # Candidate representations use calibration from earlier events.
        print(
            f"value={value}: adaptive={adaptive.transform_one(event)['x']:.2f}, "
            f"frozen={frozen.transform_one(event)['x']:.2f}"
        )
        adaptive.learn_one(event)
        frozen.learn_one(event)

    assert adaptive.transform_one({"x": 18.0}) == {"x": 1.0}
    assert frozen.transform_one({"x": 18.0}) == {"x": 3.5}
    # A downstream model keeps its old representation unless the application
    # explicitly rebuilds it; the scaler does not migrate model state.


if __name__ == "__main__":
    main()
