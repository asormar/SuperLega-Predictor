"""Tests for SeasonSimulator — round-robin scheduling, standings, and half flow.

Covers:
  - _generate_return_leg swaps home/away
  - _accumulate_player_stats no-rotaciones regression (N2)
  - Standings round-trip (serialize → parse)
  - Two-pass half flow (half='first' → half='second' + first_half_state)

1 regression pin:
  - player-stats collision (N4 / N2)
"""

import numpy as np
import pytest

from src.simulation.season_simulator import (
    SeasonSimulator,
    TeamStanding,
    match_points,
    generate_round_robin,
    generate_jornadas,
)
from src.simulation.simulator import (
    MatchSimulator,
    MatchResult,
    SetResult,
)


# ─────────────────────────────────────────────────────────────
# _generate_return_leg
# ─────────────────────────────────────────────────────────────

class TestGenerateReturnLeg:
    """_generate_return_leg produces the return half of a double round robin.

    For N teams, the return half has N*(N-1)/2 matches, each being the
    reverse of a first-leg pairing.
    """

    def test_return_leg_count(self):
        """For 4 teams, return leg has 6 matches (4*3/2)."""
        teams = ["Trento", "Perugia", "Monza", "Lube"]
        season_sim = SeasonSimulator()
        return_leg = season_sim._generate_return_leg(teams)
        expected_count = len(teams) * (len(teams) - 1) // 2
        assert len(return_leg) == expected_count

    def test_return_leg_no_self_matches(self):
        """No team plays itself in the return leg."""
        teams = ["Trento", "Perugia", "Monza"]
        season_sim = SeasonSimulator()
        return_leg = season_sim._generate_return_leg(teams)
        for home, away in return_leg:
            assert home != away

    def test_return_leg_no_duplicates(self):
        """Each pairing appears at most once in the return leg."""
        teams = ["Trento", "Perugia", "Monza", "Lube"]
        season_sim = SeasonSimulator()
        return_leg = season_sim._generate_return_leg(teams)
        seen = set()
        for home, away in return_leg:
            pair = (home, away)
            assert pair not in seen, f"Duplicate pairing: {pair}"
            seen.add(pair)

    def test_return_leg_all_reverse_pairs(self):
        """Every return-leg match (H,A) has (A,H) absent — it's the reverse."""
        teams = ["Trento", "Perugia", "Monza"]
        season_sim = SeasonSimulator()
        return_leg = season_sim._generate_return_leg(teams)
        for home, away in return_leg:
            assert (away, home) not in return_leg, (
                f"({home},{away}) and reverse both in return leg"
            )

    def test_return_leg_every_team_appears(self):
        """Every team appears at least once in the return leg."""
        teams = ["Trento", "Perugia", "Monza", "Lube"]
        season_sim = SeasonSimulator()
        return_leg = season_sim._generate_return_leg(teams)
        appeared = set()
        for home, away in return_leg:
            appeared.add(home)
            appeared.add(away)
        for t in teams:
            assert t in appeared, f"{t} not in return leg"


# ─────────────────────────────────────────────────────────────
# _accumulate_player_stats — no-rotaciones regression (N2)
# ─────────────────────────────────────────────────────────────

class TestAccumulatePlayerStats:
    """_accumulate_player_stats correctly accumulates stats across sets."""

    def test_same_player_two_sets_accumulates(self):
        """A player appearing in multiple sets has summed stats (not overwritten)."""
        season_sim = SeasonSimulator()
        stats = {}

        # Build a match where "PlayerA" appears in 2 sets
        match = MatchResult(
            home_team="Trento",
            away_team="Perugia",
            sets_home=3,
            sets_away=1,
            winner="home",
            resultado="3-1",
            sets=[
                SetResult(1, 25, 20, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 5, "aces": 1}]),
                SetResult(2, 25, 22, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 3, "aces": 0}]),
                SetResult(3, 20, 25, "away",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 4, "aces": 0}]),
                SetResult(4, 25, 18, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 2, "aces": 0}]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match, "Trento", "home")
        key = "Trento|PlayerA"
        assert key in stats
        assert stats[key]["puntos"] == 14  # 5+3+4+2
        assert stats[key]["aces"] == 1
        assert stats[key]["sets"] == 4
        assert stats[key]["partidos"] == 1

    def test_two_players_accumulated_separately(self):
        """Two different players in the same match have separate stats."""
        season_sim = SeasonSimulator()
        stats = {}

        match = MatchResult(
            home_team="Trento",
            away_team="Perugia",
            sets_home=3,
            sets_away=0,
            winner="home",
            resultado="3-0",
            sets=[
                SetResult(1, 25, 18, "home",
                          home_player_stats=[
                              {"jugador": "PlayerA", "puntos": 5},
                              {"jugador": "PlayerB", "puntos": 3},
                          ]),
                SetResult(2, 25, 20, "home",
                          home_player_stats=[
                              {"jugador": "PlayerA", "puntos": 4},
                              {"jugador": "PlayerB", "puntos": 6},
                          ]),
                SetResult(3, 25, 22, "home",
                          home_player_stats=[
                              {"jugador": "PlayerA", "puntos": 3},
                              {"jugador": "PlayerB", "puntos": 2},
                          ]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match, "Trento", "home")
        assert stats["Trento|PlayerA"]["puntos"] == 12  # 5+4+3
        assert stats["Trento|PlayerB"]["puntos"] == 11  # 3+6+2

    def test_away_side_accumulated_separately(self):
        """Away players are stored under the away team key."""
        season_sim = SeasonSimulator()
        stats = {}

        match = MatchResult(
            home_team="Trento",
            away_team="Perugia",
            sets_home=3,
            sets_away=0,
            winner="home",
            resultado="3-0",
            sets=[
                SetResult(1, 25, 20, "home",
                          away_player_stats=[{"jugador": "OppPlayer", "puntos": 4}]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match, "Perugia", "away")
        assert "Perugia|OppPlayer" in stats
        assert stats["Perugia|OppPlayer"]["puntos"] == 4

    def test_no_stats_with_empty_sets(self):
        """Sets with empty player stats produce no entries."""
        season_sim = SeasonSimulator()
        stats = {}

        match = MatchResult(
            home_team="Trento",
            away_team="Perugia",
            sets_home=3,
            sets_away=0,
            winner="home",
            resultado="3-0",
            sets=[
                SetResult(1, 25, 20, "home", home_player_stats=[]),
                SetResult(2, 25, 22, "home", home_player_stats=[]),
                SetResult(3, 25, 18, "home", home_player_stats=[]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match, "Trento", "home")
        assert len(stats) == 0


# ─────────────────────────────────────────────────────────────
# Standings round-trip
# ─────────────────────────────────────────────────────────────

class TestStandingsRoundTrip:
    """TeamStanding serialization and parsing are lossless."""

    def _sample_standings_list(self):
        entries = [
            TeamStanding(team="Trento", points=15, matches_played=5, wins=5, losses=0,
                         sets_won=15, sets_lost=2, points_scored=400, points_conceded=300,
                         wins_3_0=3, wins_3_1=1, wins_3_2=1,
                         losses_2_3=0, losses_1_3=0, losses_0_3=0),
            TeamStanding(team="Perugia", points=12, matches_played=5, wins=4, losses=1,
                         sets_won=13, sets_lost=5, points_scored=380, points_conceded=320,
                         wins_3_0=2, wins_3_1=1, wins_3_2=1,
                         losses_2_3=0, losses_1_3=1, losses_0_3=0),
        ]
        return entries

    def test_serialize_standings_fields(self):
        """serialize_standings produces the expected keys."""
        entries = self._sample_standings_list()
        serialized = SeasonSimulator.serialize_standings(entries)
        assert len(serialized) == 2
        for s in serialized:
            assert "equipo" in s
            assert "puntos" in s
            assert "pj" in s
            assert "pg" in s
            assert "pp" in s
            assert "sg" in s
            assert "sp" in s
            assert "sr" in s
            assert "pts_favor" in s
            assert "pts_contra" in s

    def test_parse_standings_round_trip(self):
        """parse_standings(serialize_standings(x)) recovers all fields."""
        entries = self._sample_standings_list()
        serialized = SeasonSimulator.serialize_standings(entries)
        parsed = SeasonSimulator.parse_standings(serialized)

        for original in entries:
            recovered = parsed[original.team]
            assert recovered.team == original.team
            assert recovered.points == original.points
            assert recovered.matches_played == original.matches_played
            assert recovered.wins == original.wins
            assert recovered.losses == original.losses
            assert recovered.sets_won == original.sets_won
            assert recovered.sets_lost == original.sets_lost
            assert recovered.points_scored == original.points_scored
            assert recovered.points_conceded == original.points_conceded
            assert recovered.wins_3_0 == original.wins_3_0

    def test_empty_standings(self):
        """Empty standings serialize/parse cleanly."""
        serialized = SeasonSimulator.serialize_standings([])
        assert serialized == []
        parsed = SeasonSimulator.parse_standings(serialized)
        assert parsed == {}

    def test_parse_standings_none(self):
        """parse_standings(None) returns empty dict."""
        parsed = SeasonSimulator.parse_standings(None)
        assert parsed == {}


# ─────────────────────────────────────────────────────────────
# Two-pass half flow
# ─────────────────────────────────────────────────────────────

class TestTwoPassHalfFlow:
    """half='first' then half='second' + first_half_state = full season."""

    def test_two_pass_half_first_second(self):
        """Combining half='first' and half='second' produces a full round-robin."""
        teams = ["Trento", "Perugia", "Monza", "Lube", "Milano", "Verona"]

        # First pass: first half
        season_sim = SeasonSimulator(
            simulator=MatchSimulator(),
            team_strengths={t: 0.5 for t in teams},
        )
        first = season_sim.simulate_season(
            teams=teams,
            double_round_robin=True,
            seed=42,
            half="first",
        )

        first_half_state = {
            "standings": SeasonSimulator.serialize_standings(first["standings"]),
            "player_season_stats": first.get("player_season_stats", {}),
        }

        # Second pass: second half
        second = season_sim.simulate_season(
            teams=teams,
            double_round_robin=True,
            seed=42,
            half="second",
            first_half_state=first_half_state,
        )

        # Verify both halves have matches
        assert len(first["matches"]) > 0
        assert len(second["matches"]) > 0

        # Total matches from both halves = full double round-robin
        n = len(teams)
        n_first = n * (n - 1) // 2
        n_second = n * (n - 1) // 2
        assert len(first["matches"]) == n_first, (
            f"First half should have {n_first} matches, got {len(first['matches'])}"
        )
        assert len(second["matches"]) == n_second, (
            f"Second half should have {n_second} matches, got {len(second['matches'])}"
        )

        # Combined points should show teams having played both halves
        expected_matches_per_team = (n - 1) * 2  # double round-robin
        second_standings = {
            s.team: s for s in second["standings"]
        }
        for t in teams:
            assert second_standings[t].matches_played == expected_matches_per_team, (
                f"{t} played {second_standings[t].matches_played} matches, "
                f"expected {expected_matches_per_team}"
            )

    def test_full_season_without_half_is_identical(self):
        """One-pass full season and two-pass combined produce same total match count."""
        teams = ["Trento", "Perugia", "Monza", "Lube", "Milano"]
        strengths = {t: 0.5 for t in teams}

        # Full season (no half)
        full_sim = SeasonSimulator(
            simulator=MatchSimulator(),
            team_strengths=strengths,
        )
        full = full_sim.simulate_season(
            teams=teams,
            double_round_robin=True,
            seed=42,
        )

        # Two-pass
        half_sim = SeasonSimulator(
            simulator=MatchSimulator(),
            team_strengths=strengths,
        )
        first = half_sim.simulate_season(
            teams=teams, double_round_robin=True, seed=42, half="first",
        )
        second = half_sim.simulate_season(
            teams=teams, double_round_robin=True, seed=42, half="second",
            first_half_state={
                "standings": SeasonSimulator.serialize_standings(first["standings"]),
                "player_season_stats": first.get("player_season_stats", {}),
            },
        )

        assert len(first["matches"]) + len(second["matches"]) == len(full["matches"])


# ─────────────────────────────────────────────────────────────
# Regression pin: player-stats collision
# ─────────────────────────────────────────────────────────────

class TestPlayerStatsCollision:
    """Regression pin for player-stats collision bug (N4/N2)."""

    def test_pin_player_stats_no_collision(self):
        """REGRESSION N4: player stats accumulate without collision across matches.

        Two different matches with overlapping player names must not corrupt
        each other's accumulated stats.
        """
        season_sim = SeasonSimulator()
        stats = {}

        # Match 1: Trento vs Perugia
        match1 = MatchResult(
            home_team="Trento",
            away_team="Perugia",
            sets_home=3, sets_away=0, winner="home", resultado="3-0",
            sets=[
                SetResult(1, 25, 20, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 5}],
                          away_player_stats=[{"jugador": "OppA", "puntos": 4}]),
                SetResult(2, 25, 22, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 3}],
                          away_player_stats=[{"jugador": "OppA", "puntos": 4}]),
                SetResult(3, 25, 18, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 2}],
                          away_player_stats=[{"jugador": "OppA", "puntos": 5}]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match1, "Trento", "home")
        season_sim._accumulate_player_stats(stats, match1, "Perugia", "away")

        # Match 2: Monza vs Verona (different teams, same player name)
        match2 = MatchResult(
            home_team="Monza",
            away_team="Verona",
            sets_home=3, sets_away=1, winner="home", resultado="3-1",
            sets=[
                SetResult(1, 25, 21, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 4}]),
                SetResult(2, 23, 25, "away",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 3}]),
                SetResult(3, 25, 19, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 5}]),
                SetResult(4, 25, 20, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 2}]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match2, "Monza", "home")

        # Trento's PlayerA must have 10 points (5+3+2), unchanged by Monza's PlayerA
        assert stats["Trento|PlayerA"]["puntos"] == 10
        assert stats["Trento|PlayerA"]["partidos"] == 1
        assert stats["Trento|PlayerA"]["sets"] == 3

        # Monza's PlayerA must have 14 points (4+3+5+2)
        assert stats["Monza|PlayerA"]["puntos"] == 14
        assert stats["Monza|PlayerA"]["partidos"] == 1
        assert stats["Monza|PlayerA"]["sets"] == 4

        # Both are separate entries — no collision
        assert len(stats) == 3  # Trento|PlayerA, Perugia|OppA, Monza|PlayerA

    def test_pin_player_stats_no_collision_partidos_unchanged(self):
        """REGRESSION N4: partidos counter is not inflated by cross-match collision."""
        season_sim = SeasonSimulator()
        stats = {}

        match1 = MatchResult(
            home_team="Trento",
            away_team="Perugia",
            sets_home=3, sets_away=0, winner="home", resultado="3-0",
            sets=[
                SetResult(1, 25, 20, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 5}]),
                SetResult(2, 25, 22, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 3}]),
                SetResult(3, 25, 18, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 2}]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match1, "Trento", "home")

        assert stats["Trento|PlayerA"]["partidos"] == 1

        # Second match: same player on the same team
        match2 = MatchResult(
            home_team="Trento",
            away_team="Monza",
            sets_home=3, sets_away=1, winner="home", resultado="3-1",
            sets=[
                SetResult(1, 25, 20, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 4}]),
                SetResult(2, 25, 22, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 3}]),
                SetResult(3, 20, 25, "away",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 2}]),
                SetResult(4, 25, 18, "home",
                          home_player_stats=[{"jugador": "PlayerA", "puntos": 1}]),
            ],
        )

        season_sim._accumulate_player_stats(stats, match2, "Trento", "home")

        # After 2 matches, partidos should be 2
        assert stats["Trento|PlayerA"]["partidos"] == 2
        # Puntos should be 5+3+2 + 4+3+2+1 = 20
        assert stats["Trento|PlayerA"]["puntos"] == 20
