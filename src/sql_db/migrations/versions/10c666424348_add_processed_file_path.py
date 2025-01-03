"""add_processed_file_path

Revision ID: 10c666424348
Revises: 
Create Date: 2024-11-07 16:54:43.271279

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '10c666424348'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('ingest', sa.Column('processed_file_path', sa.String(255), nullable=True, comment="Path to the processed file"))

def downgrade():
    op.drop_column('ingest', 'processed_file_path')
