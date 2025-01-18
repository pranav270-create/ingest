import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.engine import Engine
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
session_manager = DatabaseSessionManager("energy_data")

async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with session_manager.session() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            raise e
        finally:
            await session.close()


def get_engine() -> Engine:
    return DatabaseSessionManager("energy_data")._engine
