import os
from sqlalchemy import create_engine    
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy import create_engine

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
