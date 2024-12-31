import os
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import create_engine    
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from fastapi import Depends
from typing import Annotated
from sqlalchemy.orm import scoped_session


def get_gcp_engine(db_name: str) -> Engine:
    """connect to GCP SQL database with an sqlalchemy engine"""
    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC
    connector = Connector(ip_type)

    def getconn():
        conn = connector.connect(
            os.environ.get("GEOSPATIAL_INSTANCE_CONNECTION_NAME"),
            "pg8000",
            user=os.environ.get("GEOSPATIAL_DB_USER"),
            db=db_name,
            password=os.environ.get("GEOSPATIAL_DB_PASSWORD"),
        )
        return conn

    engine = create_engine(
        "postgresql+pg8000://", 
        creator=getconn, 
        pool_size=5, 
        max_overflow=10, 
        pool_timeout=30,
        pool_recycle=1200)
    return engine


# Create the async engine
engine = get_gcp_engine(os.environ.get('ETL_DB_NAME'))
# Create the async session factory
# async_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


# Dependency to get a session for the energy database
def get_async_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


ETLDBSessionDep = Annotated[Session, Depends(get_async_session)]


def get_gcp_engine_host() -> Engine:
    """connect to GCP SQL database with an sqlalchemy engine"""
    db_user = os.getenv("GEOSPATIAL_DB_USER")
    db_pass = os.getenv("GEOSPATIAL_DB_PASSWORD")
    db_host = os.getenv("POSTGRES_DB_HOST", "34.48.20.86")  # The public IP address of db
    db_port = os.getenv("POSTGRES_DB_PORT", "5432")  # The port your database listens on, default is 5432
    # Construct the database URL
    database_url = (f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}")
    # Create the engine
    engine = create_engine(database_url)
    return engine


def create_database(db_name: str):
    engine = get_gcp_engine_host()
    conn = engine.connect()
    cursor = conn.cursor()

    try:
        # Check if the database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        if not exists:
            # Create the database
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")
    except Exception as err:
        print(f"Error creating database: {err}")
    finally:
        cursor.close()
        conn.close()


if __name__ == '__main__':
    # create_database(db_name="etl_pipeline")
    # Show all databases in the PostgreSQL database server
    engine = get_gcp_engine_host()
    conn = engine.connect()
    cursor = conn.cursor()
    cursor.execute("SELECT datname FROM pg_database;")
    rows = cursor.fetchall()
    print("Databases:")
    for row in rows:
        print("   ", row[0])
