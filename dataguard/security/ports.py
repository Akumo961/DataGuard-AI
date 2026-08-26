from abc import ABC, abstractmethod

from dataguard.database.ports import TenantContext


class AuthorizationPolicy(ABC):
    """Central authorization port; business services must not infer roles ad hoc."""

    @abstractmethod
    def require(self, tenant: TenantContext, permission: str, resource_id: str | None = None) -> None:
        raise NotImplementedError


class IdentityProvider(ABC):
    """OIDC/OAuth2-ready identity abstraction."""

    @abstractmethod
    def authenticate(self, credential: str) -> TenantContext:
        raise NotImplementedError
