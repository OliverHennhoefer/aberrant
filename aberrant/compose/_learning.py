"""Per-observation state and transaction ownership for one learning pass."""

from contextlib import ExitStack
from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal

from aberrant.compose._types import INPUT, NodeState


@dataclass
class LearningPass:
    available: dict[str, dict[str, float]]
    schemas: dict[str, tuple[str, ...]]
    updates: dict[str, int]
    states: list[NodeState] = field(default_factory=list)
    model_updated: bool = False
    node_id: str = INPUT
    phase: str = "learn"


class LearningTransactions(ExitStack):
    """Keep contexts open through the terminal and detect rollback failures."""

    def __init__(self) -> None:
        super().__init__()
        self.uncertain = False

    def enter_learner(self, component: object, current: dict[str, float]) -> None:
        transaction = getattr(component, "learning_transaction", None)
        if not callable(transaction):
            self.uncertain = True
            return
        try:
            self.enter_context(transaction(dict(current)))
        except BaseException:
            self.uncertain = True
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            if super().__exit__(exc_type, exc, traceback):
                self.uncertain = True
                raise RuntimeError("Learning transactions cannot suppress failures")
        except BaseException as error:
            if error is not exc:
                self.uncertain = True
            raise
        return False
