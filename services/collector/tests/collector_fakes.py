from __future__ import annotations

from typing import Any


class FakeConnection:
    def __init__(self, executed: list[Any], rows: list[Any]) -> None:
        self._executed = executed
        self._rows = rows

    def execute(self, statement: Any) -> list[Any]:
        self._executed.append(statement)
        return list(self._rows)

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False


class FakeEngine:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self.executed: list[Any] = []
        self.rows: list[Any] = list(rows or ())

    def begin(self) -> FakeConnection:
        return FakeConnection(self.executed, self.rows)

    def dispose(self) -> None:
        pass
