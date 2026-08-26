from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TenantContext:
    organization_id: str
    actor_id: str | None


class AnalysisRepository(ABC):
    """Persistence port. Implementations must scope every query to a tenant."""

    @abstractmethod
    def save_analysis(self, tenant: TenantContext, analysis: Mapping[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_analysis(self, tenant: TenantContext, analysis_id: str) -> Mapping[str, Any] | None:
        raise NotImplementedError


class UnitOfWork(ABC):
    """Transaction boundary used by application services."""

    @abstractmethod
    def __enter__(self) -> "UnitOfWork":
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        raise NotImplementedError
