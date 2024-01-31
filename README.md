We combine an LSTM neural network and the SARIMAX time series model in a joint manner through a unified state space representation. The state space is then solved with particle filtering for parameter estimation. The state vector is entirely comprised of parameters-to-be-estimated, which allows for a single pass solution (e.g., with Kalman/particle filtering) instead of numeric optimization methods with iterative solutions.

Here is an example usage where we wrap the online learning loop of the model in a function
and then compare its predictions against statsmodels' SARIMAX on a simulated ARMA process:

```py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.tsa.statespace.sarimax import SARIMAX

from lstm_sx import LSTM_SX

def filter_lstmax(ys,
                  lstm_hidden_size=20,
                  sx_order=(1, 0, 0), sx_seas_order=None,
                  exog=None,
                  n_particles=100, resampling_thre=0.5):
    """
    Wrap the learning loop of the particle filter based LSTM-SX
    model. It has 3 phases: update_particles (apply state transition
    equations), meet a new measurement and update_weights, and lastly
    resample. The "fitted" LSTM_SX model is returned.
    """
    # Exog passed?
    if exog is not None:
        exog = np.asarray(exog)
        assert exog.ndim == 2, "Exogenous non-2D"

    # Form the model
    pf = LSTM_SX(lstm_hidden_size,
                 sx_order, sx_seas_order,
                 exog,
                 n_particles, resampling_thre)

    # Online learning loop against a data stream
    for t, y_t in enumerate(ys):
        pf.update_particles(t)
        pf.update_weights(t, y_t)
        pf.resample()
    return pf

## Simulation
# Some AR and MA parameters
ar_params = np.array([0.75, -0.25])
ma_params = np.array([0.32])

# Simulate an ARMA process
n_sample = 100
order = len(ar_params), 0, len(ma_params)
ys = pd.Series(arma_generate_sample(ar=[1, *-ar_params], ma=[1, *ma_params], nsample=n_sample))

# Fit with our model and statsmodels'
pf = filter_lstmax(ys, sx_order=order)
stats_sx = SARIMAX(ys, order=order).fit()

# In sample predictions
our_preds = pd.Series(pf.predictions)
stats_preds = stats_sx.predict()

ax = ys.plot(label="true")
our_preds.plot(figsize=(14, 5), ls="--", label="pred, ours", ax=ax)
stats_preds.plot(ls="--", label="pred, statsmodels", ax=ax)
ax.legend()
```
