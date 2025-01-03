"""add row detail

Revision ID: a29cd0ba8a75
Revises: b451a2db093a
Create Date: 2024-12-19 16:04:22.590697

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = 'a29cd0ba8a75'
down_revision: Union[str, None] = 'b451a2db093a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('rows', sa.Column('client_id', sa.Integer(), nullable=True))
    op.add_column('rows', sa.Column('assessment_id', sa.Integer(), nullable=True))
    op.add_column('rows', sa.Column('platform', sa.String(length=255), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    op.drop_column('rows', 'client_id')
    op.drop_column('rows', 'assessment_id')
    op.drop_column('rows', 'platform')
