import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# this is the Alembic Config object
config = context.config

# Update the URL with environment variables
section = config.config_ini_section
config.set_section_option(section, "DB_USER", os.getenv("DB_USER", ""))
config.set_section_option(section, "DB_PASS", os.getenv("DB_PASS", ""))
config.set_section_option(section, "DB_HOST", os.getenv("DB_HOST", ""))
config.set_section_option(section, "DB_PORT", os.getenv("DB_PORT", "3306"))
config.set_section_option(section, "DB_NAME", os.getenv("ETL_DB_NAME", ""))

# Import your models
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.sql_db.etl_model import Base

# Set target metadata
target_metadata = Base.metadata

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
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
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
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
