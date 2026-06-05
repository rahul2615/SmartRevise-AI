"""Backfill user active status

Revision ID: d6a6b7c8e9f0
Revises: a71f000c5db4
Create Date: 2026-06-06

"""
from alembic import op
import sqlalchemy as sa


revision = 'd6a6b7c8e9f0'
down_revision = 'a71f000c5db4'
branch_labels = None
depends_on = None


def upgrade():
    op.execute(sa.text('UPDATE "user" SET is_active = true WHERE is_active IS NULL'))
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column(
            'is_active',
            existing_type=sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        )


def downgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.alter_column(
            'is_active',
            existing_type=sa.Boolean(),
            nullable=True,
            server_default=None,
        )
