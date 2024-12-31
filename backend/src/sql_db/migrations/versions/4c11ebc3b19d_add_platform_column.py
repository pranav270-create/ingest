"""add_platform_column

Revision ID: 4c11ebc3b19d
Revises: 10c666424348
Create Date: 2024-11-19 01:44:26.995294

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = '4c11ebc3b19d'
down_revision: Union[str, None] = '10c666424348'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add platform column to assessment_cache
    op.add_column('assessment_cache',
        sa.Column('platform', sa.String(20), nullable=True)
    )
    
    # Add platform column to final_recommendation
    op.add_column('final_recommendation',
        sa.Column('platform', sa.String(20), nullable=True)
    )
    
    # Backfill existing rows with 'testing'
    connection = op.get_bind()
    connection.execute(text("""
        UPDATE assessment_cache
        SET platform = 'testing'
        WHERE platform IS NULL
    """))
    
    connection.execute(text("""
        UPDATE final_recommendation
        SET platform = 'testing'
        WHERE platform IS NULL
    """))
    
    # Make platform column non-nullable after backfill
    op.alter_column('assessment_cache', 'platform',
        existing_type=sa.String(20),
        nullable=False
    )
    
    op.alter_column('final_recommendation', 'platform',
        existing_type=sa.String(20),
        nullable=False
    )


def downgrade() -> None:
    # Remove platform column from both tables
    op.drop_column('assessment_cache', 'platform')
    op.drop_column('final_recommendation', 'platform')
