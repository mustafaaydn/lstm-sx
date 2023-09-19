import warnings

import numpy as np
from numpy import tanh
from scipy.special import expit as sigmoid

rng = np.random.default_rng()

MEAS_NOISE_STD = 0.01
UPD_NOISE_STD  = 0.1


class LSTM_SX:
    def __init__(self, lstm_hidden_size=20,
                 sx_order=(1, 0, 0), sx_seas_order=None,
                 exog=None,
                 n_particles=500, resampling_threshold=0.5):
        """
        Parameters
        ----------
        lstm_hidden_size : int, default=20
            Size of h_t and c_t

        sx_order : list-like, default=(1, 0, 0)
            Ordinary order of the SARIMAX, i.e., (p, d, q)

        sx_seas_order : list-like or None, default=None
            Seasonal order of the SARIMAX, i.e., (P, D, Q, m)

        exog : 2D array-like or None
            Exogenous (side information) variables. Each row should correspond
            to a time step's side information.

        n_particles : int, default=500
            Number of particles in the filter

        resampling_threshold : float, default=0.5
            If N_eff < N_particles * this, resample. (N_eff^-1 := weights^2.sum())
        """
        self.n_particles = n_particles
        self.resampling_threshold = resampling_threshold

        self._has_lstm = True
        # LSTM is only active if side information, X_t, is passed
        if exog is None:
            warnings.warn("No exogenous variable passed; LSTM will be inactive")
            self._has_lstm = False
            self.exog = None
        else:
            self.exog = np.asarray(exog)
            assert self.exog.ndim == 2, "non-2D exog"
            # LSTM's part
            self.lstm_hidden_size = lstm_hidden_size

        # SARIMAX's part
        self.sx_p, self.sx_d, self.sx_q = sx_order
        if sx_seas_order is not None:
            self.sx_P, self.sx_D, self.sx_Q, self.sx_m = sx_seas_order
        else:
            self.sx_P, self.sx_D, self.sx_Q, self.sx_m = 0, 0, 0, None

        # Determine the state dimension
        # State vector's order is
        #    AR, MA, seas_AR, seas_MA, X (of SARIMAX)
        #    h_t, c_t, theta             (of LSTM)
        state_dim = self.sx_p + self.sx_q + self.sx_P + self.sx_Q
        if self._has_lstm:
            state_dim += exog.shape[1]
            state_dim += 2*lstm_hidden_size + 4*(lstm_hidden_size * (lstm_hidden_size + exog.shape[1] + 1)) + lstm_hidden_size
            #            ^h_t + c_t           ^                              W_*                      b_*     ^w_t

        # Initialize particles & associated weights
        self.particles = rng.normal(size=(n_particles, state_dim))
        self.weights = rng.dirichlet([1] * n_particles)
        assert np.isclose(self.weights.sum(), 1), "dirichlet broken"
        assert np.alen(self.particles) == np.alen(self.weights), "p <!-> w"

        # Keep track of `y_t` and `\hat_e_t`. Also record `\hat_y_t`.
        self._measurements = []
        self._errors = []
        self.predictions = []

    def _lstm_state_transition(self, t):
        n_particles = self.n_particles
        n_hidden, n_input = self.lstm_hidden_size, self.exog.shape[1]
        W_size  = n_hidden * (n_hidden + n_input)
        W_shape = n_particles, n_hidden, -1
        w_size = b_size  = n_hidden

        # LSTM-related state elements start after those of SARIMAX end
        particles = self.particles
        sx_offset = self.sx_p + self.sx_q + self.sx_P + self.sx_Q + n_input

        # Unpack h_t, c_t and theta_t [W_*, b_*]
        h_t = particles[:, sx_offset: sx_offset + n_hidden]
        c_t = particles[:, sx_offset + n_hidden: sx_offset + 2*n_hidden]

        W_f = particles[:, sx_offset + 2*n_hidden + 0*W_size: sx_offset + 2*n_hidden + 1*W_size].reshape(W_shape)
        W_i = particles[:, sx_offset + 2*n_hidden + 1*W_size: sx_offset + 2*n_hidden + 2*W_size].reshape(W_shape)
        W_c = particles[:, sx_offset + 2*n_hidden + 2*W_size: sx_offset + 2*n_hidden + 3*W_size].reshape(W_shape)
        W_o = particles[:, sx_offset + 2*n_hidden + 3*W_size: sx_offset + 2*n_hidden + 4*W_size].reshape(W_shape)

        b_f = particles[:, sx_offset + 2*n_hidden + 4*W_size + 0*b_size: sx_offset + 2*n_hidden + 4*W_size + 1*b_size]
        b_i = particles[:, sx_offset + 2*n_hidden + 4*W_size + 1*b_size: sx_offset + 2*n_hidden + 4*W_size + 2*b_size]
        b_c = particles[:, sx_offset + 2*n_hidden + 4*W_size + 2*b_size: sx_offset + 2*n_hidden + 4*W_size + 3*b_size]
        b_o = particles[:, sx_offset + 2*n_hidden + 4*W_size + 3*b_size: sx_offset + 2*n_hidden + 4*W_size + 4*b_size]

        # Apply Eq. 2, i.e., the recurrent forward equations
        x_t = self.exog[t]
        h_and_x = np.hstack((h_t, x_t[None].repeat(self.n_particles, axis=0)))
        assert h_and_x.shape == (self.n_particles, n_hidden + n_input), "h_and_x stacked wrong"

        ein_str = "BhX,BX->Bh"
        f_t = sigmoid(np.einsum(ein_str, W_f, h_and_x) + b_f)
        i_t = sigmoid(np.einsum(ein_str, W_i, h_and_x) + b_i)
        tilde_c_t = tanh(np.einsum(ein_str, W_c, h_and_x) + b_c)
        o_t = sigmoid(np.einsum(ein_str, W_o, h_and_x) + b_o)

        c_t = f_t * c_t + i_t * tilde_c_t
        h_t = o_t * tanh(c_t)

        # Update states h_t, c_t and theta_t (Eq. 7)
        particles[:, sx_offset: sx_offset + n_hidden] = h_t + rng.normal(scale=UPD_NOISE_STD, size=(n_particles, n_hidden))
        particles[:, sx_offset + n_hidden: sx_offset + 2*n_hidden] = c_t + rng.normal(scale=UPD_NOISE_STD, size=(n_particles, n_hidden))
        particles[:, sx_offset + 2*n_hidden: sx_offset + 2*n_hidden + 4*W_size + 4*b_size + w_size] += rng.normal(scale=UPD_NOISE_STD, size=(n_particles, 4*W_size + 4*b_size + w_size))

    def _lstm_prediction(self):
        n_hidden, n_input = self.lstm_hidden_size, self.exog.shape[1]
        W_size  = n_hidden * (n_hidden + n_input)
        w_size = b_size  = n_hidden

        particles = self.particles
        sx_offset = self.sx_p + self.sx_q + self.sx_P + self.sx_Q + n_input

        h_t = particles[:, sx_offset: sx_offset + n_hidden]
        w_t = particles[:, sx_offset + 2*n_hidden + 4*W_size + 4*b_size: sx_offset + 2*n_hidden + 4*W_size + 4*b_size + w_size]
        lstm_preds = np.einsum("Nh,Nh->N", w_t, h_t)
        return lstm_preds

    def _sx_state_transition(self):
        # SARIMAX-related state components are at the beginning
        sx_size = self.sx_p + self.sx_q + self.sx_P + self.sx_Q
        if self.exog is not None:
            sx_size += self.exog.shape[1]
        self.particles[:, :sx_size] += rng.normal(scale=UPD_NOISE_STD, size=(self.n_particles, sx_size))

    def _sx_prediction(self, t):
        # Form r_t := [y_{t-1}, ..., y_{t-p},
        #              y_{t-m}, ..., y_{t-mP},
        #              e_{t-1}, ..., e_{t-q},
        #              e_{t-m}, ..., e_{t-mQ}]
        r_t = []

        # Check if we are at the start yet; fill with 0 if so. Otherwise,
        # last p/q items are taken.
        n_meas, n_errs = len(self._measurements), len(self._errors)

        # AR part
        if n_meas < self.sx_p:
            r_t.extend(self._measurements[::-1] + [0] * (self.sx_p - n_meas))
        elif self.sx_p:  # can say `else` as well but this saves a bit of time
            r_t.extend(self._measurements[:-1-self.sx_p:-1])

        # Seasonal AR part
        _sx_has_seasonal_part = self.sx_P != 0 or self.sx_Q != 0
        if _sx_has_seasonal_part:
            # say m = 12, P = 3
            # so, need -12, -24, -36th values
            if n_meas < self.sx_m * self.sx_P:
                r_t.extend(self._measurements[-self.sx_m::-self.sx_m] + [0] * np.ceil(self.sx_P - n_meas / self.sx_m).astype(int))
                assert self.sx_P - n_meas / self.sx_m > 0, "mP versus n_meas gone wrong"
            elif self.sx_P:
                r_t.extend(self._measurements[-self.sx_m:-1 - self.sx_m*self.sx_P:-self.sx_m])

        # MA part
        if n_errs < self.sx_q:
            r_t.extend(self._errors[::-1] + [0] * (self.sx_q - n_errs))
        elif self.sx_q:
            r_t.extend(self._errors[-1:-1-self.sx_q:-1])

        # Seasonal MA part
        if _sx_has_seasonal_part:
            if n_errs < self.sx_m * self.sx_Q:
                r_t.extend(self._errors[-self.sx_m::-self.sx_m] + [0] * np.ceil(self.sx_Q - n_errs / self.sx_m).astype(int))
                assert self.sx_Q - n_errs / self.sx_m > 0, "mQ versus n_errs gone wrong"
            elif self.sx_Q:
                r_t.extend(self._errors[-self.sx_m:-1 - self.sx_m*self.sx_Q:-self.sx_m])

        # Exogenous part
        if self.exog is not None:
            r_t.extend(self.exog[t])

        # Get SARIMAX's particles' predictions
        sx_size = self.sx_p + self.sx_q + self.sx_P + self.sx_Q
        if self.exog is not None:
            sx_size += self.exog.shape[1]
        sx_particles = self.particles[:, :sx_size]
        sx_pred = sx_particles @ r_t
        return sx_pred

    def _normalize_weights(self):
        """
        Assure the weights behave like a probability distribution by normalizing
        them with their sum. If sum is near 0, reinitialize the weights uniformly,
        i.e., w_i = 1 / N for all i in [1, N].
        """
        Z = self.weights.sum()
        if not np.isclose(Z, 0):
            self.weights /= Z
        else:
            warnings.warn("Sum of weights is nearly 0, can't normalize; will uniformize")
            self.weights = np.full_like(self.weights,
                                        fill_value=1 / self.n_particles)

    def _compute_likelihood(self, t, y_t):
        """
        Measure how likely each particle is, i.e., p(y_t | \vec{s}_t^{(i)}).
        Assume Gaussian for {y_t}, centered around the specific measurement; so
        compute the prediction and measure the likelihood with it.
        """
        # Record the measurement
        self._measurements.append(y_t)

        # Gather base predictions and combine them to get predictions of each particle
        sx_y_hats_t = self._sx_prediction(t)
        lstm_y_hats_t = self._lstm_prediction() if self._has_lstm else 0
        # print("SX preds:", sx_y_hats_t)
        # print("\nLSTM preds:", lstm_y_hats_t)
        y_hats_t = sx_y_hats_t + lstm_y_hats_t

        # Error and likelihood computation
        errs_t = y_t - y_hats_t
        likelihood = np.exp(-errs_t**2 / (2 * MEAS_NOISE_STD**2)) / MEAS_NOISE_STD / np.sqrt(2 * np.pi)
        # print("max likelihood:", likelihood.max())

        # Before leaving, get a collective estimate out of particles & record
        y_hat_t = self.weights @ y_hats_t
        self.predictions.append(y_hat_t)
        self._errors.append(y_t - y_hat_t)

        return likelihood

    def update_particles(self, t):
        """
        Perform the state transition equation, i.e.,
        \vec{s}_t = \vec{s}_{t-1} + \vec{e}_t
        """
        self._sx_state_transition()
        if self._has_lstm:
            self._lstm_state_transition(t)

    def update_weights(self, time_idx, measurement):
        """
        w_t^{(i)} = w_{t-1}^{(i)} p(y_t | \vec{s}_t^{(i)})
        Then normalize.
        """
        self.weights *= self._compute_likelihood(time_idx, measurement)
        self._normalize_weights()

    def resample(self):
        effective_size = 1 / (self.weights ** 2).sum()
        if effective_size < self.resampling_threshold * self.n_particles:
            # Weighted sampling with replacement
            self.particles = rng.choice(self.particles,
                                        size=self.n_particles,
                                        replace=True,
                                        p=self.weights,
                                        axis=0)
            # Uniformize weights
            self.weights = np.full_like(self.weights,
                                        fill_value=1 / self.n_particles)
