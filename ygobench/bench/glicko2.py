"""Small dependency-free Glicko-2 implementation for arena ratings."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GlickoPlayer:
    rating: float = 1500.0
    deviation: float = 350.0
    volatility: float = 0.06


class Glicko2:
    scale = 173.7178
    tau = 0.5
    epsilon = 1e-6

    @classmethod
    def update(
        cls,
        player: GlickoPlayer,
        results: list[tuple[GlickoPlayer, float]],
    ) -> GlickoPlayer:
        mu = (player.rating - 1500.0) / cls.scale
        phi = player.deviation / cls.scale
        if not results:
            phi_star = math.sqrt(phi**2 + player.volatility**2)
            return GlickoPlayer(player.rating, phi_star * cls.scale, player.volatility)

        rows = []
        for opponent, score in results:
            opp_mu = (opponent.rating - 1500.0) / cls.scale
            opp_phi = opponent.deviation / cls.scale
            g = 1.0 / math.sqrt(1.0 + 3.0 * opp_phi**2 / math.pi**2)
            expected = 1.0 / (1.0 + math.exp(-g * (mu - opp_mu)))
            rows.append((g, expected, score))
        variance = 1.0 / sum(g**2 * expected * (1 - expected) for g, expected, _ in rows)
        delta = variance * sum(g * (score - expected) for g, expected, score in rows)
        sigma = cls._volatility(player.volatility, phi, variance, delta)
        phi_star = math.sqrt(phi**2 + sigma**2)
        phi_new = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / variance)
        mu_new = mu + phi_new**2 * sum(
            g * (score - expected) for g, expected, score in rows
        )
        return GlickoPlayer(
            rating=mu_new * cls.scale + 1500.0,
            deviation=phi_new * cls.scale,
            volatility=sigma,
        )

    @classmethod
    def _volatility(cls, sigma: float, phi: float, variance: float, delta: float) -> float:
        a = math.log(sigma**2)

        def f(value: float) -> float:
            exp_value = math.exp(value)
            denominator = phi**2 + variance + exp_value
            return (
                exp_value * (delta**2 - phi**2 - variance - exp_value)
                / (2.0 * denominator**2)
                - (value - a) / cls.tau**2
            )

        left = a
        if delta**2 > phi**2 + variance:
            right = math.log(delta**2 - phi**2 - variance)
        else:
            step = 1
            while f(a - step * cls.tau) < 0:
                step += 1
            right = a - step * cls.tau
        f_left, f_right = f(left), f(right)
        while abs(right - left) > cls.epsilon:
            middle = left + (left - right) * f_left / (f_right - f_left)
            f_middle = f(middle)
            if f_middle * f_right < 0:
                left, f_left = right, f_right
            else:
                f_left /= 2.0
            right, f_right = middle, f_middle
        return math.exp(left / 2.0)
