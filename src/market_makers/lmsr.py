"""LMSR-style automated market maker (binary, simulation only).

Implements Hanson's Logarithmic Market Scoring Rule for a binary market. This
mirrors the AMM style used by several prediction-market platforms: prices live
in (0,1), always sum to 1, and the market maker's worst-case loss is bounded by
``b * ln(n_outcomes)`` (for binary, ``b * ln 2``).

Cost function (binary, shares q_yes / q_no):

    C(q) = b * ln( exp(q_yes / b) + exp(q_no / b) )

Price of Yes:

    p_yes = exp(q_yes / b) / ( exp(q_yes / b) + exp(q_no / b) )

A trade that moves shares from q -> q' costs the trader C(q') - C(q).
The market maker's running P&L (before resolution payout) is the total cash
collected, i.e. the increase in C since inception. Bounded loss = b * ln 2.

Offline only -- no money, no chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LMSRMarket:
    """A binary LMSR market maker.

    Parameters
    ----------
    b
        Liquidity parameter. Larger ``b`` => deeper market (smaller price moves
        per unit traded) but larger bounded loss.
    fee
        Proportional fee added on top of the LMSR cost of each trade.
    p_init
        Initial probability of Yes (sets the starting share imbalance).
    """

    b: float = 100.0
    fee: float = 0.02
    p_init: float = 0.5

    q_yes: float = field(default=0.0)
    q_no: float = field(default=0.0)
    collected_fees: float = field(default=0.0)
    volume: float = field(default=0.0)
    _c0: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.b <= 0:
            raise ValueError("liquidity parameter b must be > 0")
        p = min(max(self.p_init, 1e-6), 1 - 1e-6)
        # Seed shares so that the initial price equals p_init.
        # p = sigmoid((q_yes - q_no)/b) => set q_no=0, q_yes = b*logit(p).
        self.q_yes = self.b * math.log(p / (1 - p))
        self.q_no = 0.0
        self._c0 = self._cost()

    # ------------------------------------------------------------------ #
    # Core LMSR maths
    # ------------------------------------------------------------------ #
    def _cost(self, q_yes: float | None = None, q_no: float | None = None) -> float:
        qy = self.q_yes if q_yes is None else q_yes
        qn = self.q_no if q_no is None else q_no
        # Numerically stable log-sum-exp.
        m = max(qy, qn) / self.b
        return self.b * (m + math.log(math.exp(qy / self.b - m) + math.exp(qn / self.b - m)))

    def price_yes(self) -> float:
        qy, qn = self.q_yes / self.b, self.q_no / self.b
        m = max(qy, qn)
        ey, en = math.exp(qy - m), math.exp(qn - m)
        return ey / (ey + en)

    def price_no(self) -> float:
        return 1.0 - self.price_yes()

    def prices(self) -> dict[str, float]:
        py = self.price_yes()
        return {"yes": py, "no": 1.0 - py}

    @property
    def max_loss(self) -> float:
        """Bounded worst-case subsidy loss for a binary LMSR: b * ln 2."""
        return self.b * math.log(2)

    def cost_to_buy(self, side: str, shares: float) -> float:
        """LMSR cost (excl. fee) to buy `shares` of `side` ('yes'/'no')."""
        side = side.lower()
        if side == "yes":
            new = self._cost(q_yes=self.q_yes + shares)
        else:
            new = self._cost(q_no=self.q_no + shares)
        return new - self._cost()

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    def buy(self, side: str, shares: float) -> dict:
        """Simulate buying `shares` of `side`. Returns the trade record."""
        side = side.lower()
        shares = max(0.0, shares)
        cost = self.cost_to_buy(side, shares)
        if side == "yes":
            self.q_yes += shares
        else:
            self.q_no += shares
        fee = abs(cost) * self.fee
        self.collected_fees += fee
        self.volume += abs(cost)
        return {
            "side": side,
            "shares": shares,
            "cost": cost,
            "fee": fee,
            "price_yes": self.price_yes(),
        }

    def trade_to_belief(self, target_p_yes: float, max_shares: float) -> dict:
        """Trade toward a target Yes probability, capped by `max_shares`.

        Models a trader who believes Yes should be priced at `target_p_yes`
        and pushes the market in that direction up to their size limit.
        """
        target = min(max(target_p_yes, 1e-6), 1 - 1e-6)
        cur = self.price_yes()
        if abs(target - cur) < 1e-9:
            return {"side": None, "shares": 0.0, "cost": 0.0, "fee": 0.0,
                    "price_yes": cur}
        # Shares needed to move price to target (closed form for binary LMSR).
        # logit(target) = (q_yes - q_no)/b ; solve for delta on the chosen side.
        target_diff = self.b * math.log(target / (1 - target))
        cur_diff = self.q_yes - self.q_no
        delta = target_diff - cur_diff
        if delta >= 0:
            side, shares = "yes", min(delta, max_shares)
        else:
            side, shares = "no", min(-delta, max_shares)
        return self.buy(side, shares)

    def maker_pnl(self, outcome_yes: bool) -> float:
        """Market-maker P&L at resolution.

        Cash collected from traders = C(now) - C(0). At resolution the maker
        pays out 1 per winning share held by traders. The net (subsidy) cost is
        bounded below by -max_loss; fees are added on top as pure profit.
        """
        cash_collected = self._cost() - self._c0
        # Initial seeded quantities (set the starting price). q_no seed is 0;
        # q_yes seed is b*logit(p_init) and may be negative if p_init < 0.5.
        p = min(max(self.p_init, 1e-6), 1 - 1e-6)
        q_yes0 = self.b * math.log(p / (1 - p))
        q_no0 = 0.0
        # Traders are net-long the (quantity - seed) of the winning outcome;
        # the maker pays them 1 per such share at resolution.
        if outcome_yes:
            traders_winning_shares = self.q_yes - q_yes0
        else:
            traders_winning_shares = self.q_no - q_no0
        return cash_collected - traders_winning_shares + self.collected_fees
