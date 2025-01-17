"""change string fields to text

Revision ID: f73fab78d2e3
Revises: 
Create Date: 2025-01-17 16:05:03.614213

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f73fab78d2e3'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('ingest', 'creator_name',
        existing_type=sa.String(100),
        type_=sa.Text(),
        existing_nullable=True)

    op.alter_column('ingest', 'extracted_document_file_path',
        existing_type=sa.String(255),
        type_=sa.Text(),
        existing_nullable=True)


def downgrade() -> None:
    op.alter_column('ingest', 'creator_name',
        existing_type=sa.Text(),
        type_=sa.String(100),
        existing_nullable=True)

    op.alter_column('ingest', 'extracted_document_file_path',
        existing_type=sa.Text(),
        type_=sa.String(255),
        existing_nullable=True)