import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from dataguard.database.auth_repository import MAX_FAILED_LOGINS, authenticate_local_user
from dataguard.database.models import User
from dataguard.security.passwords import hash_password


class FakeResult:
    def __init__(self, user):
        self.user = user

    def scalar_one_or_none(self):
        return self.user


class FakeSession:
    def __init__(self, user):
        self.user = user
        self.flush_count = 0

    async def execute(self, statement):
        return FakeResult(self.user)

    async def flush(self):
        self.flush_count += 1


def test_failed_password_attempts_lock_account() -> None:
    user = User(
        organization_id=uuid4(),
        email="analyst@example.test",
        password_hash=hash_password("correct-password"),
        display_name="Analyst",
        active=True,
        failed_login_count=0,
    )
    session = FakeSession(user)

    async def run() -> None:
        for _ in range(MAX_FAILED_LOGINS):
            assert (
                await authenticate_local_user(
                    session, user.organization_id, user.email, "wrong-password"
                )
                is None
            )

    asyncio.run(run())
    assert user.locked_until is not None
    assert user.locked_until > datetime.now(timezone.utc)
    assert session.flush_count == MAX_FAILED_LOGINS
