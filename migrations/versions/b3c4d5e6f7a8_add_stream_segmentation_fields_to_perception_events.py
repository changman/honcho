"""Add stream segmentation fields to perception_events

Revision ID: b3c4d5e6f7a8
Revises: cd6fd35470ac
Create Date: 2026-05-09 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "b3c4d5e6f7a8"
down_revision: Union[str, None] = "cd6fd35470ac"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "perception_events",
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "perception_events",
        sa.Column("segment_id", sa.TEXT(), nullable=True),
    )
    op.add_column(
        "perception_events",
        sa.Column(
            "is_segment_start",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "perception_events",
        sa.Column(
            "is_segment_end",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_index(
        "ix_perception_events_captured_at",
        "perception_events",
        ["captured_at"],
        unique=False,
    )
    op.create_index(
        "ix_perception_events_segment_id",
        "perception_events",
        ["segment_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_perception_events_segment_id", table_name="perception_events")
    op.drop_index("ix_perception_events_captured_at", table_name="perception_events")
    op.drop_column("perception_events", "is_segment_end")
    op.drop_column("perception_events", "is_segment_start")
    op.drop_column("perception_events", "segment_id")
    op.drop_column("perception_events", "captured_at")
