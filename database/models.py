"""
추천 점수를 저장하기 위한 데이터베이스 모델
"""

from sqlalchemy import Index
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.sql import sqltypes


class Base(DeclarativeBase):
    pass


class RecommendationScore(Base):
    __tablename__ = "recommendation_score"

    id: Mapped[int] = mapped_column(
        sqltypes.BIGINT,
        primary_key=True,
        autoincrement=True,
    )

    member_id: Mapped[int] = mapped_column(
        sqltypes.BIGINT,
        nullable=False
    )

    product_id: Mapped[int] = mapped_column(
        sqltypes.BIGINT,
        nullable=False
    )

    score: Mapped[float] = mapped_column(
        sqltypes.Float,
        nullable=False
    )


Index(
    None,
    RecommendationScore.member_id,
    RecommendationScore.product_id,
    unique=True,
) 