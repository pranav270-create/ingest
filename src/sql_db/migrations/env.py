import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.sql_db.etl_model import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = (
        f"postgresql://"  # Changed to sync driver
        f"{os.environ.get('AWS_DB_USER')}:{os.environ.get('AWS_DB_PASS')}@"
        f"{os.environ.get('AWS_DB_HOST')}:{os.environ.get('AWS_DB_PORT')}/{os.environ.get('AWS_DB_NAME')}"
    )
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    url = (
        f"postgresql://"  # Changed to sync driver
        f"{os.environ.get('AWS_DB_USER')}:{os.environ.get('AWS_DB_PASS')}@"
        f"{os.environ.get('AWS_DB_HOST')}:{os.environ.get('AWS_DB_PORT')}/{os.environ.get('AWS_DB_NAME')}"
    )

    connectable = engine_from_config(
        {"sqlalchemy.url": url},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
