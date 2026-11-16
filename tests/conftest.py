"""Gemeinsame Fixtures: das Handbeispiel (zwei Gruppen, je fünf Beobachtungen), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand_groups():
    """Gruppe 0: 2, 3, 4, 5, 6; Gruppe 1: 0, 4, 8, 12, 16 (Brüche). Handrechnung: Gruppenmittel 4 und 8, SSE = 10 + 160 = 170, s² = 170/8 = 85/4."""
    return [Fraction(v) for v in (2, 3, 4, 5, 6)], [Fraction(v) for v in (0, 4, 8, 12, 16)]
