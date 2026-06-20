"""house-edge-lab.

A simulation and research tool for prediction-market design.

SAFETY / SCOPE
--------------
This package is a *research sandbox*. It must not, and does not:

  * place real bets
  * create real markets
  * connect to wallets
  * sign transactions
  * request or store private keys
  * interact with any real-money system or live trading API

Everything here is offline simulation and analysis. Platform profiles
(e.g. Polkamarkets) describe *publicly documented* mechanics so we can model
economics on paper -- they perform no network calls.
"""

__version__ = "0.1.0"
