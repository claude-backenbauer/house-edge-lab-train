"""Simple bookmaker market-maker model (simulation only).

A bookmaker has a private *model probability*, then offers prices with a spread
around it and collects a fee. It will trade until it hits a maximum exposure on
either side. This is a classic "house" model: the edge comes from the spread +
fee, the risk comes from holding inventory when the model is wrong.

Nothing here is a real book -- it never touches money or an exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class BookmakerQuote:
    """A two-sided quote for a binary outcome."""

    yes_price: float
    no_price: float

    @property
    def overround(self) -> float:
        """How much the two prices sum above 1.0 (the built-in margin)."""
        return self.yes_price + self.no_price - 1.0


@dataclass
class BookmakerModel:
    """A bookmaker quoting a binary market.

    Parameters
    ----------
    model_probability
        The book's private estimate of P(Yes).
    spread
        Half-width applied around the offered probability (each side).
    fee
        Proportional fee taken on each filled trade's notional.
    max_exposure
        Maximum net signed inventory (in notional units) the book will hold on
        either side before it refuses to deepen that side.
    """

    model_probability: float
    spread: float = 0.03
    fee: float = 0.02
    max_exposure: float = 1000.0

    # internal state
    net_yes_inventory: float = field(default=0.0)  # +ve => sold Yes (short Yes)
    realised_fees: float = field(default=0.0)
    volume: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.model_probability = _clip(self.model_probability)

    @property
    def offered_probability(self) -> float:
        """Mid probability the book is currently willing to deal around.

        Skews away from accumulated inventory to mean-revert exposure.
        """
        skew = 0.0
        if self.max_exposure > 0:
            skew = self.spread * (self.net_yes_inventory / self.max_exposure)
        return _clip(self.model_probability + skew)

    def quote(self) -> BookmakerQuote:
        mid = self.offered_probability
        yes = _clip(mid + self.spread)
        no = _clip((1.0 - mid) + self.spread)
        return BookmakerQuote(yes_price=yes, no_price=no)

    def _capacity(self, side: str) -> float:
        """Remaining notional the book will take on a given side."""
        if side == "yes":
            # Buying Yes from a trader makes us shorter Yes -> inventory up.
            return max(0.0, self.max_exposure - self.net_yes_inventory)
        else:
            return max(0.0, self.max_exposure + self.net_yes_inventory)

    def trade(self, side: str, notional: float) -> dict:
        """Simulate a trader taking `notional` on `side` ('yes'/'no').

        Returns a dict describing the fill. Fills are capped by exposure.
        """
        side = side.lower()
        want = max(0.0, notional)
        filled = min(want, self._capacity(side))
        quote = self.quote()
        price = quote.yes_price if side == "yes" else quote.no_price

        fee_taken = filled * self.fee
        self.realised_fees += fee_taken
        self.volume += filled
        if side == "yes":
            self.net_yes_inventory += filled
        else:
            self.net_yes_inventory -= filled

        return {
            "side": side,
            "requested": want,
            "filled": filled,
            "price": price,
            "fee": fee_taken,
            "net_yes_inventory": self.net_yes_inventory,
        }

    def settle(self, outcome_yes: bool) -> float:
        """Settle inventory at resolution and return total book P&L.

        We are *short* Yes by `net_yes_inventory` (we sold Yes to traders).
        At resolution Yes pays 1, No pays 0.

        Simplified payoff: we collected ~offered_probability per Yes share sold
        and must pay 1 if Yes wins; symmetric for No. Fees are pure profit.
        """
        mid = self.model_probability
        # Cash collected selling Yes shares (approx at mid) minus payout.
        yes_short = self.net_yes_inventory
        if outcome_yes:
            inventory_pnl = yes_short * (mid - 1.0)  # short Yes, Yes won -> loss
        else:
            inventory_pnl = yes_short * (mid - 0.0)  # short Yes, Yes lost -> gain
        return inventory_pnl + self.realised_fees

    def simulate_demand(
        self,
        trades: list[tuple[str, float]],
        outcome_yes: bool,
    ) -> dict:
        """Run a sequence of (side, notional) trades then settle.

        `trades` is the *simulated user demand*. Returns a summary dict.
        """
        fills = [self.trade(side, notional) for side, notional in trades]
        pnl = self.settle(outcome_yes)
        return {
            "fills": fills,
            "volume": self.volume,
            "fees": self.realised_fees,
            "final_inventory": self.net_yes_inventory,
            "pnl": pnl,
        }
