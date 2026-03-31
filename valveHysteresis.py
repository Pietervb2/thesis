"""
Thermostatic bypass valve model with hysteresis.

Physical behaviour:
- Valve starts OPEN (position 1.0) — bypass active, cold water flows
- Temperature rises → valve starts closing at t_close_start, fully closed at t_close_end
- Temperature falls → valve starts reopening at t_open_start, fully open at t_open_end
- The two bands are NON-OVERLAPPING with a dead zone between them

Temperature axis (low → high):

  fully open          reopening band           dead zone        closing band         fully closed
      |---[ t_open_end ... t_open_start ]---[           ]---[ t_close_start ... t_close_end ]---|

Key insight:
  t_open_end < t_open_start < t_close_start < t_close_end

  The dead zone between t_open_start and t_close_start means:
    - Once the valve starts closing (rising temp hits t_close_start), it won't
      reopen until the temperature has fallen all the way back to t_open_start.
    - This prevents the valve from hunting at the low threshold.

Reversal inside either active band: valve HOLDS.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class ValveConfig:
    """
    Independent thresholds for the closing and reopening curves.

    Example with t_setpoint=60, close_band=3, open_band=3, dead_zone=6:
      t_close_start = 60       valve starts closing  (rising)
      t_close_end   = 63       valve fully closed    (rising)
      t_open_start  = 54       valve starts reopening (falling)
      t_open_end    = 51       valve fully open      (falling)
      dead zone     = 54..60   valve holds in this range
    """
    t_setpoint:   float         # Temperature at which valve starts closing
    close_band:   float = 3.0   # Width of the closing ramp  (°C above setpoint)
    open_band:    float = 3.0   # Width of the reopening ramp (°C)
    dead_zone:    float = 6.0   # Gap between reopening-end and closing-start (°C)
    valve_open_min: float = 0.0
    valve_open_max: float = 1.0

    @property
    def t_close_start(self) -> float:
        return self.t_setpoint

    @property
    def t_close_end(self) -> float:
        return self.t_setpoint + self.close_band

    @property
    def t_open_start(self) -> float:
        return self.t_setpoint - self.dead_zone

    @property
    def t_open_end(self) -> float:
        return self.t_setpoint - self.dead_zone - self.open_band


class ThermostaticBypassValve:
    """
    Thermostatic bypass valve with non-overlapping hysteresis bands.

    States:
      RISING  — temp climbing through closing band  → valve closes
      FALLING — temp falling through reopening band → valve reopens
      HOLDING — temp in dead zone or reversed inside a band → valve freezes
    """

    RISING  = "rising"
    FALLING = "falling"
    HOLDING = "holding"

    def __init__(self, config: ValveConfig, initial_temp: float):
        self.cfg = config

        if initial_temp <= config.t_open_end:
            self.position = config.valve_open_max
        elif initial_temp >= config.t_close_end:
            self.position = config.valve_open_min
        elif initial_temp <= config.t_open_start:
            self.position = self._position_on_falling_curve(initial_temp)
        elif initial_temp <= config.t_close_start:
            self.position = config.valve_open_max   # in dead zone, assume open
        else:
            self.position = self._position_on_rising_curve(initial_temp)

        self._last_temp    = initial_temp
        self._active_curve = self.RISING
        self._state        = self.HOLDING

    # ------------------------------------------------------------------
    # Curve definitions
    # ------------------------------------------------------------------

    def _position_on_rising_curve(self, temp: float) -> float:
        """Valve closes over [t_close_start, t_close_end]."""
        lo, hi = self.cfg.t_close_start, self.cfg.t_close_end
        if temp <= lo:
            return self.cfg.valve_open_max
        if temp >= hi:
            return self.cfg.valve_open_min
        return np.interp(temp, [lo, hi],
                         [self.cfg.valve_open_max, self.cfg.valve_open_min])

    def _position_on_falling_curve(self, temp: float) -> float:
        """Valve reopens over [t_open_end, t_open_start]."""
        lo, hi = self.cfg.t_open_end, self.cfg.t_open_start
        if temp <= lo:
            return self.cfg.valve_open_max
        if temp >= hi:
            return self.cfg.valve_open_min
        return np.interp(temp, [lo, hi],
                         [self.cfg.valve_open_max, self.cfg.valve_open_min])

    # ------------------------------------------------------------------
    # State machine step
    # ------------------------------------------------------------------

    def step(self, temp: float) -> float:
        """
        Update valve position given the current temperature.

        Returns:
            Valve position [0.0 = closed … 1.0 = open]
        """
        dt = temp - self._last_temp

        in_closing_band  = self.cfg.t_close_start <= temp <= self.cfg.t_close_end
        in_reopening_band = self.cfg.t_open_end   <= temp <= self.cfg.t_open_start
        in_dead_zone     = self.cfg.t_open_start  <  temp <  self.cfg.t_close_start

        if temp >= self.cfg.t_close_end:
            # Above everything — fully closed, follow rising curve
            self._active_curve = self.RISING
            self._state        = self.RISING

        elif temp <= self.cfg.t_open_end:
            # Below everything — fully open, follow falling curve
            self._active_curve = self.FALLING
            self._state        = self.FALLING

        elif in_closing_band:
            if self._active_curve == self.RISING and dt >= 0:
                self._state = self.RISING        # continuing upward → close
            elif self._active_curve == self.RISING and dt < 0:
                self._state = self.HOLDING       # reversed inside closing band → hold
            else:
                # Arrived here from below (falling curve exited dead zone upward)
                # — treat as hold; only the rising curve drives closing
                self._state = self.HOLDING

        elif in_dead_zone:
            # Between the two bands — always hold
            self._state = self.HOLDING

        elif in_reopening_band:
            if self._active_curve == self.FALLING and dt <= 0:
                self._state = self.FALLING       # continuing downward → reopen
            elif self._active_curve == self.FALLING and dt > 0:
                self._state = self.HOLDING       # reversed inside reopening band → hold
            else:
                # Arrived here from above (rising curve, temp dropped into reopening band)
                # Switch to falling curve now
                self._active_curve = self.FALLING
                self._state        = self.FALLING

        # Apply position
        if self._state == self.RISING:
            self.position = self._position_on_rising_curve(temp)
        elif self._state == self.FALLING:
            self.position = self._position_on_falling_curve(temp)
        # HOLDING: unchanged

        self._last_temp = temp
        return self.position


# ----------------------------------------------------------------------
# Simulation & plotting
# ----------------------------------------------------------------------

def run_simulation():
    cfg = ValveConfig(
        t_setpoint  = 60.0,   # valve starts closing here
        close_band  = 3.0,    # fully closed at 63°C
        open_band   = 3.0,    # reopening ramp: 51°C → 54°C
        dead_zone   = 6.0,    # hold zone: 54°C → 60°C
    )

    print(f"Closing band  : {cfg.t_close_start:.1f}°C → {cfg.t_close_end:.1f}°C")
    print(f"Dead zone     : {cfg.t_open_start:.1f}°C → {cfg.t_close_start:.1f}°C  (valve holds)")
    print(f"Reopening band: {cfg.t_open_end:.1f}°C → {cfg.t_open_start:.1f}°C")

    valve = ThermostaticBypassValve(cfg, initial_temp=40.0)

    t_time = np.linspace(0, 100, 500)
    temperature = (
        55 + 15 * np.sin(0.08 * t_time)
        + 3  * np.sin(0.40 * t_time)
    )

    positions, states = [], []
    for temp in temperature:
        positions.append(valve.step(temp))
        states.append(valve._state)

    # ---- Plot --------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Thermostatic Bypass Valve — Non-overlapping Hysteresis Bands",
                 fontsize=13, fontweight="bold")

    # Temperature panel
    ax1.plot(t_time, temperature, color="#2563EB", linewidth=1.5, label="Fluid temperature")

    # Closing band (red)
    ax1.fill_between(t_time, cfg.t_close_start, cfg.t_close_end,
                     alpha=0.15, color="#EF4444")
    ax1.axhline(cfg.t_close_start, color="#EF4444", ls="--", lw=1,
                label=f"Starts closing ({cfg.t_close_start:.0f}°C)")
    ax1.axhline(cfg.t_close_end, color="#7C2D12", ls="--", lw=1,
                label=f"Fully closed ({cfg.t_close_end:.0f}°C)")

    # Dead zone (grey)
    ax1.fill_between(t_time, cfg.t_open_start, cfg.t_close_start,
                     alpha=0.08, color="#6B7280")
    ax1.axhline(cfg.t_open_start, color="#6B7280", ls=":", lw=1,
                label=f"Dead zone top ({cfg.t_open_start:.0f}°C)")

    # Reopening band (green)
    ax1.fill_between(t_time, cfg.t_open_end, cfg.t_open_start,
                     alpha=0.15, color="#10B981")
    ax1.axhline(cfg.t_open_end, color="#065F46", ls="--", lw=1,
                label=f"Fully open ({cfg.t_open_end:.0f}°C)")
    ax1.axhline(cfg.t_open_start, color="#10B981", ls="--", lw=1,
                label=f"Starts reopening ({cfg.t_open_start:.0f}°C)")

    ax1.set_ylabel("Temperature (°C)", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Valve position panel
    state_colors = {
        ThermostaticBypassValve.RISING:  "#EF4444",
        ThermostaticBypassValve.FALLING: "#10B981",
        ThermostaticBypassValve.HOLDING: "#9CA3AF",
    }
    for i in range(1, len(t_time)):
        ax2.plot(t_time[i-1:i+1], positions[i-1:i+1],
                 color=state_colors[states[i]], linewidth=2)

    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(color="#EF4444", label="Rising  (valve closing)"),
        Patch(color="#10B981", label="Falling (valve reopening)"),
        Patch(color="#9CA3AF", label="Holding"),
    ], fontsize=9, loc="upper right")

    ax2.set_ylabel("Valve position", fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["Closed\n(no bypass)", "25%", "50%", "75%", "Open\n(full bypass)"])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("/mnt/user-data/outputs/valve_simulation.png", dpi=150, bbox_inches="tight")
    # plt.close()
    plt.show()
    print("Saved: valve_simulation.png")


if __name__ == "__main__":
    run_simulation()