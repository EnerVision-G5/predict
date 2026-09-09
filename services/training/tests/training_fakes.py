from __future__ import annotations

from typing import Any


class FakeConnection:
    def __init__(self, executed: list[Any]) -> None:
        self._executed = executed

    def execute(self, statement: Any) -> None:
        self._executed.append(statement)

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeEngine:
    def __init__(self) -> None:
        self.executed: list[Any] = []
        self.transactions = 0

    def begin(self) -> FakeConnection:
        self.transactions += 1
        return FakeConnection(self.executed)

    def dispose(self) -> None:
        pass
