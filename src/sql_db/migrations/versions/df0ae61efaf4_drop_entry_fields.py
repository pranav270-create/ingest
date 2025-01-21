"""drop Entry fields

Revision ID: df0ae61efaf4
Revises: f73fab78d2e3
Create Date: 2025-01-18 13:14:08.328437

"""
from collections.abc import Sequence
from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'df0ae61efaf4'
down_revision: Union[str, None] = 'f73fab78d2e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('entries', 'context_summary_string')
    op.drop_column('entries', 'index_numbers')


def downgrade() -> None:
    op.add_column('entries', sa.Column('context_summary_string', sa.Text(), nullable=True))
    op.add_column('entries', sa.Column('index_numbers', sa.JSON(), nullable=True))
