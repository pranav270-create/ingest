import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os
from sqlalchemy.orm import sessionmaker

DB_HOST = os.environ.get('AWS_DB_HOST')
DB_USER = os.environ.get('AWS_DB_USER')
DB_PASS = os.environ.get('AWS_DB_PASS')
DB_PORT = int(os.environ.get('AWS_DB_PORT', 3306))


def get_database_url(db_name: str, async_url: bool = False) -> str:
    """Generate database URL based on the database name and connection type."""
    dialect = "postgresql+asyncpg" if async_url else "postgresql+psycopg2"
    return f"{dialect}://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{db_name}"


def create_async_engine_for_db(db_name: str):
    """Create an async engine for the specified database."""
    return create_async_engine(
        get_database_url(db_name, async_url=True),
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def get_async_session_maker(db_name: str):
    """Create a session maker for the specified database."""
    engine = create_async_engine_for_db(db_name)
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session(db_name: str) -> AsyncSession:
    """Get an async session for the specified database."""
    session_maker = get_async_session_maker(db_name)
    async with session_maker() as session:
        yield session


def get_db(db_name: str):
    """Create a database session dependency that can be used with FastAPI."""
    async def db_dependency() -> AsyncSession:
        session_maker = get_async_session_maker(db_name)
        async with session_maker() as session:
            try:
                yield session
            finally:
                await session.close()
    return db_dependency


def get_engine(db_name: str):
    """Get a synchronous engine for the specified database."""
    return create_engine(get_database_url(db_name), echo=True)


def create_database(db_name: str):
    """Create a database if it doesn't exist."""
    # Connect to default 'postgres' database first
    conn = psycopg2.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        database='postgres'  # Connect to default postgres database
    )
    conn.autocommit = True  # Required for database creation
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        if not exists:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")
    except psycopg2.Error as err:
        print(f"Error creating database: {err}")
    finally:
        cursor.close()
        conn.close()


async def test_connection():
    from sqlalchemy.sql import text
    async for session in get_async_session('energy'):
        async with session.begin():
            # Execute query asynchronously
            result = await session.execute(text('SELECT * FROM pg_catalog.pg_tables'))
            # Fetch results (no need to await)
            tables = result.fetchall()
            print(tables)


if __name__ == '__main__':
    # create_database('my_database')
    import asyncio
    asyncio.run(test_connection())
