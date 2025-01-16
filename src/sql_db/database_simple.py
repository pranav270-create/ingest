import os

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

DB_HOST = os.environ.get('AWS_DB_HOST')
DB_USER = os.environ.get('AWS_DB_USER')
DB_PASS = os.environ.get('AWS_DB_PASS')
DB_PORT = int(os.environ.get('AWS_DB_PORT', 3306))


def get_database_url(db_name: str, async_url: bool = False) -> str:
    """Generate database URL based on the database name and connection type."""
    dialect = "postgresql+asyncpg" if async_url else "postgresql+psycopg2"
    return f"{dialect}://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{db_name}"


def get_engine(db_name: str):
    """Get a synchronous engine for the specified database."""
    return create_engine(get_database_url(db_name), echo=True)


# Create the async engine
engine = get_engine("energy_data")
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


# Dependency to get a session for the energy database
def get_async_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine


class DatabaseSessionManager:
    def __init__(self, db_name: str):
        self._engine = self._create_engine(db_name)
        self._sessionmaker = async_sessionmaker(
            autocommit=False,
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    def _create_engine(self, db_name: str) -> AsyncEngine:
        # Create the connection URL for AWS RDS
        db_url = (
            f"postgresql+asyncpg://"
            f"{os.environ.get('AWS_DB_USER')}:{os.environ.get('AWS_DB_PASS')}@"
            f"{os.environ.get('AWS_DB_HOST')}:{os.environ.get('AWS_DB_PORT')}/{db_name}"
        )

        engine = create_async_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            echo=False,
        )
        return engine

    async def close(self):
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        await self._engine.dispose()
        self._engine = None
        self._sessionmaker = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise Exception("DatabaseSessionManager is not initialized")

        session = self._sessionmaker()
        try:
            yield session
        except SQLAlchemyError as e:
            await session.rollback()
            raise Exception(f"Database error: {str(e)}") from None
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()

# Session dependency generators with explicit cleanup
session_manager = DatabaseSessionManager(os.environ.get("ENERGY_DB_NAME"))
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with session_manager.session() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()
