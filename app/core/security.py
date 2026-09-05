"""
Security utilities for authentication and authorization.
"""

from datetime import datetime, timedelta
from typing import Any, Optional, Union, cast

import bcrypt
from jose import jwt

from app.core.config import settings

# Password hashing is done with the `bcrypt` library directly rather than
# through passlib's CryptContext. passlib 1.7.4's bcrypt backend runs an
# internal self-test (`detect_wrap_bug`) on first use that is broken
# against bcrypt>=4.1 (see https://github.com/pyca/bcrypt/issues/684):
# it hashes an oversized canned string and, since newer bcrypt raises
# instead of silently truncating secrets over 72 bytes, that self-test
# itself throws - breaking password hashing for every user, regardless of
# their actual password length. Calling bcrypt directly sidesteps that
# entirely and keeps the app compatible with current bcrypt releases
# (including the musllinux wheels used by the Alpine-based Docker image).
_BCRYPT_MAX_BYTES = 72


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    password_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))


def get_password_hash(password: str) -> str:
    """Generate a bcrypt password hash."""
    # bcrypt only uses the first 72 bytes of the input; truncate explicitly
    # (rather than letting bcrypt raise) so behavior is predictable and
    # documented, matching the same truncation applied in verify_password.
    password_bytes = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")


def create_access_token(
    subject: Union[str, Any], user_id: str, expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token."""
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject), "user_id": user_id}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return cast(str, encoded_jwt)


def create_refresh_token(
    subject: Union[str, Any], user_id: str, expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT refresh token."""
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode = {"exp": expire, "sub": str(subject), "user_id": user_id}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return cast(str, encoded_jwt)
