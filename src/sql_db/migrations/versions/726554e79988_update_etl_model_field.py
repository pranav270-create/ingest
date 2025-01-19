"""update etl_model field

Revision ID: 726554e79988
Revises: df0ae61efaf4
Create Date: 2025-01-18 22:53:11.527631

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '726554e79988'
down_revision: Union[str, None] = 'df0ae61efaf4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename the column from 'hash' to 'document_hash'
    op.alter_column('ingest', 'hash', new_column_name='document_hash')
    # Drop the old unique constraint
    op.drop_constraint('uq_ingest_hash', 'ingest', type_='unique')
    # Create new unique constraint with updated column name
    op.create_unique_constraint('uq_ingest_document_hash', 'ingest', ['document_hash'])


def downgrade() -> None:
    # Drop the new unique constraint
    op.drop_constraint('uq_ingest_document_hash', 'ingest', type_='unique')
    # Rename the column back from 'document_hash' to 'hash'
    op.alter_column('ingest', 'document_hash', new_column_name='hash')
    # Recreate the original unique constraint
    op.create_unique_constraint('uq_ingest_hash', 'ingest', ['hash'])
