"""
Educational descriptions for every computed quantity in the dashboard.

Single source of truth: each concept maps to one markdown string with a consistent
three-part shape — *How it's computed* (plain words + a typeset formula and symbol key),
*What it means for you*, and *Why it's useful*. Sections render the subset they need via
`render_section_help`, so the text is written once but each section stays self-contained.

Written for a reader with no finance background: jargon is defined where it first appears
and examples are illustrative (no hard-coded amounts).

Notation used throughout:
  P_t = price at time t,  r = a period's simple return,  N = periods per year
  (252 daily, 52 weekly, 12 monthly),  r_f = risk-free rate,  w = portfolio weights.
"""

import streamlit as st


DESCRIPTIONS = {

    # ── §1 — Load Data ───────────────────────────────────────────────────────
    "price_spikes": r"""
**Stock splits & price-anomaly check**

This is two checks, because a *stock split* and a *bad price* are different things.

*Recorded stock splits (📐):* read straight from yfinance's `stock splits` column — the
exact ratio on the exact ex-date, identical across daily/weekly/monthly. Since the prices
here are **Adj Close**, which is already split-adjusted, a split creates *no jump* in the
series and is purely informational.

*Anomaly check (⚠️):* a fixed "flag moves > 60%" rule can't serve every asset and interval
at once — a monthly bar compounds ~21 daily moves, and Bitcoin routinely swings further in
a month than an equity asset does in a year. Instead we standardise each return against the
asset's *own* history using a fat-tail-resistant scale, and flag only the genuine outliers:

$$z_t=\frac{r_t-\mathrm{median}(r)}{1.4826\cdot\mathrm{MAD}(r)},\qquad
\left|z_t\right| > 8 \;\text{ and }\; \left|r_t\right| > 45\%$$

where $\mathrm{MAD}$ is the median absolute deviation. The absolute floor is set above the
largest genuine single-bar swings (even crypto rarely moves more than ~40% in a day),
since real glitches and unadjusted splits move price by roughly half or double. Because the
$z$ scale adapts per asset and per interval, normal high-volatility swings pass, while a
fat-finger tick, a currency mix-up, or an unadjusted split stands out. A flagged move that
lands on a recorded split date is almost certainly just that split.

*Why it's useful:* one bad price can quietly distort every return, volatility and risk
figure downstream. This is a cheap, self-calibrating sanity check on the raw inputs.
""",

    "data_window": r"""
**Common data window**

*How it's computed:* each asset has its own start and end date; the shared window runs from
the **latest** start to the **earliest** end across all your holdings:

$$\text{start}=\max_i(\text{start}_i), \qquad \text{end}=\min_i(\text{end}_i)$$

*What it means for you:* a portfolio can only be compared over dates where *every* asset has
data. The asset with the shortest history (the "binding" one) sets how far back you can look.
The window also keeps only dates shared by all assets, so if you mix a 7-day market (crypto) with
5-day markets (stocks), the crypto's weekend moves get folded into the next shared trading
day — making its daily figures slightly approximate.

*Why it's useful:* it tells you how much real, overlapping history your analysis rests on —
a frontier built on 11 months of common data is far less trustworthy than one built on 10 years.
""",

    # ── §2 — Per-Asset Analytics ─────────────────────────────────────────────
    "simple_return": r"""
**Simple return (single period)**

*How it's computed:* the percentage change in price from the start to the end of the period:

$$R = \frac{P_{\text{end}}}{P_{\text{start}}} - 1$$

*What it means for you:* the plain answer to "if I'd bought at the start and sold at the end,
what would I have made or lost?" — including the full effect of the whole period.

*Why it's useful:* it's the most intuitive return there is, and the building block for
everything else. The table shows it over several look-back windows so you can see recent vs.
long-run performance at a glance.
""",

    "calendar_year_return": r"""
**Calendar-year return**

*How it's computed:* the same simple return formula, but measured from the end of one calendar
year to the end of the next:

$$R_{\text{year}} = \frac{P_{\text{Dec 31}}}{P_{\text{Dec 31, prior}}} - 1$$

*What it means for you:* how the asset did in each individual year, the way performance is
usually quoted in fund fact-sheets.

*Why it's useful:* year-by-year figures reveal consistency. Two funds with the same long-run
return can feel completely different if one was steady and the other swung between big gains
and losses.
""",

    "cagr": r"""
**CAGR (Compound Annual Growth Rate)**

*How it's computed:* the single steady yearly rate that would have grown the price from its
start to its end over $n$ years:

$$\text{CAGR} = \left(\frac{P_{\text{end}}}{P_{\text{start}}}\right)^{1/n} - 1$$

*What it means for you:* the return per year this investment *actually* delivered, with
compounding already baked in — the honest number you'd compare against a savings rate or
another fund.

*Why it's useful:* it collapses a messy price history into one comparable yearly figure. It
differs from the *Average annual return* below because that figure is a **linear** mean × N (it
just averages each period's return, with no compounding), whereas CAGR reflects how gains and
losses actually compounded. Two effects pull in opposite directions — volatility drag lowers
CAGR, while the compounding the linear average omits raises it — so CAGR can land a little below
**or** above the average return.
""",

    "avg_annual_return": r"""
**Average annual return**

*How it's computed:* take the average of the per-period returns and scale it up to
a year by the number of periods per year:

$$\mu_{\text{ann}} = \bar{r}\times N$$

where $\bar{r}$ = average periodic return and $N$ = periods per year (252 daily, 52 weekly, 12
monthly).

*What it means for you:* a forward-looking estimate of the typical yearly return, in the form
the optimizer and the risk/return chart use. It's an *expected* return, not the realized CAGR: the
two differ because volatility drag pulls CAGR down (≈ $\sigma^2/2$) while intra-period compounding
pulls it up, so for volatile portfolios CAGR is usually the lower of the two. Don't read this
figure as the growth you'd actually compound.

*Why it's useful:* it pairs cleanly with annual volatility (both scale with the same calendar),
which is exactly what's needed to compare portfolios and build the efficient frontier.

*Assumption:* the linear $\times N$ scaling assumes per-period returns are **serially uncorrelated**
(i.i.d.). Real return series have some autocorrelation, so treat the annualised figure as an
estimate, not an exact forecast.
""",

    "annual_volatility": r"""
**Annualised volatility (risk)**

*How it's computed:* the standard deviation of the periodic returns — how much they bounce
around their average — scaled to a year:

$$\sigma_{\text{ann}} = \sigma \times \sqrt{N}$$

where $\sigma$ = standard deviation of periodic returns and $N$ = periods per year. (Risk grows
with the square root of time, not linearly, which is why we use $\sqrt{N}$.)

*What it means for you:* a rough sense of how wide the swings are. If volatility is 12%, a
typical year's return lands roughly within ±12% of the average — bigger number, bumpier ride.

*Why it's useful:* it's the standard measure of risk in portfolio theory and the denominator of
the Sharpe ratio. Lower volatility for the same return means a smoother, more predictable journey.

*Assumption:* the $\sqrt{N}$ rule assumes per-period returns are **serially uncorrelated** (i.i.d.).
Positive autocorrelation (trending) makes true annual volatility *higher* than $\sigma\sqrt{N}$,
while mean-reversion makes it *lower* — so the annualised figure is an approximation.
""",

    "max_drawdown": r"""
**Maximum drawdown**

*How it's computed:* track the running peak of your wealth, then find the largest percentage
drop from any peak to a later low:

$$\text{MDD} = \max_t \frac{\text{peak}_t - W_t}{\text{peak}_t}, \qquad W_t = \prod_{i\le t}(1+r_i)$$

where $W_t$ = cumulative wealth and $\text{peak}_t$ = its highest value so far.

*What it means for you:* the worst peak-to-trough loss you'd have suffered if you'd bought at
the worst possible moment and held through the bottom — the gut-check of "how bad did it get?"

*Why it's useful:* volatility treats ups and downs the same, but investors actually fear the
downs. Max drawdown captures the deepest hole you'd have had to sit through, which is often what
decides whether someone can stick with a strategy. (For a portfolio the basis — rebalanced to
target weights each period, or buy-and-hold that drifts — is stated in each section, and the two
can behave differently.)
""",

    "cumulative_return": r"""
**Cumulative-return chart**

*How it's computed:* compound the periodic returns into a growing wealth line, shown as total
gain since the start:

$$\text{Cumulative}_t = \prod_{i \le t}(1+r_i) - 1$$

*What it means for you:* the running story of €1 invested at the start — every dip and recovery,
not just the endpoints.

*Why it's useful:* a single CAGR number hides the path. The curve shows *when* the gains and
losses happened, which matters enormously for how an investment actually felt to hold.
""",

    "lookback_annual_metrics": r"""
**Annualised metrics by look-back period**

*How it's computed:* the same *Average annual return* and *Annualised volatility* as above, but
measured over a single backward window of a chosen length (the last 1, 3 or 5 years that fully fit
within the asset's history, plus its full history — longer windows are omitted rather than truncated)
ending at one fixed anchor date. The window doesn't slide — each row is one number for
one look-back horizon. (For genuinely sliding, date-by-date windows, see the *Per-Asset Rolling Returns*
section instead.)

*What it means for you:* how the asset's return and risk looked over recent stretches, rather
than its entire history — useful because markets change character over time.

*Why it's useful:* an asset's long-run average can mask a very different recent regime. Comparing
look-back horizons shows whether performance and risk have been improving, deteriorating, or steady.
""",

    # ── §3 — Per-Asset Prices ────────────────────────────────────────────────
    "normalized_prices": r"""
**Normalized prices (base = 1000)**

*How it's computed:* divide every price by the asset's first price in the window and multiply by
1000, so every asset starts at the same point:

$$\tilde{P}_t = \frac{P_t}{P_0}\times 1000$$

*What it means for you:* it answers "if I'd put the same money into each at the start, how would
they compare today?" — stripping away the fact that one share might cost €5 and another €50,000.

*Why it's useful:* raw price lines are impossible to compare across assets of different prices.
Rebasing to a common start makes relative performance instantly readable on one chart.
""",

    # ── §4 — Per-Asset Rolling Returns ───────────────────────────────────────
    "rolling_returns_asset": r"""
**Rolling returns (per asset)**

*How it's computed:* for every date, the **cumulative total** return an investor would have earned
over the preceding window of length $w$ (selectable: 1, 2, 3, 5, 7 or 10 years):

$$R_t = \frac{P_t}{P_{t-w}} - 1$$

This is the whole-window total, **not** a per-year rate, so a longer window naturally shows a
bigger number purely from compounding more years — **windows of different lengths are not directly
comparable.** For a per-year figure, see the CAGR in §2.

*What it means for you:* instead of one return from a single lucky or unlucky start date, this
shows the full range of *total* outcomes a holder of this window length could have experienced
depending on *when* they started.

*Why it's useful:* it exposes how much your result depends on timing. A wide, jumpy band means
outcomes were very start-date-dependent; a tight band means the holding period was reliably
rewarding.
""",

    "rolling_returns_portfolio": r"""
**Rolling returns (portfolio)**

*How it's computed:* build your portfolio's buy-and-hold value by holding your chosen mix without
rebalancing (see *Buy-and-hold basis*), then take the **cumulative total** window-over-window
return at every date:

$$R_t = \frac{V_t}{V_{t-w}} - 1$$

where $V_t$ = portfolio value and $w$ = the window length. As with the per-asset chart this is the
whole-window total, **not** a per-year rate, so longer windows show bigger figures and aren't
comparable across window lengths (see §2 CAGR for a per-year figure).

*What it means for you:* the rolling-window experience of holding *your specific mix* untouched,
not the individual assets.

*Why it's useful:* it shows whether diversifying actually smoothed the ride: the portfolio band
is often tighter than any single asset's, which is diversification working in your favour.
""",

    # ── §5 — Per-Asset Returns & Statistics ──────────────────────────────────
    "return_stats": r"""
**Minimum / maximum / mean / median / standard deviation of returns**

*How it's computed:* basic summary statistics of each asset's periodic returns — the smallest,
largest, average, middle value, and spread:

$$\bar{r}=\frac{1}{T}\sum_{t} r_t, \qquad \text{median}=r_{(T/2)}, \qquad \sigma=\sqrt{\frac{1}{T}\sum_t (r_t-\bar{r})^2}$$

*What it means for you:* the min and max show the best and worst single periods; the mean and
median both show the central tendency; the standard deviation shows how scattered returns are
around that centre. The **mean − median gap** reads skew: mean above median = a right tail
(rare big gains), below = a left tail (rare big losses).

*Why it's useful:* these numbers are a quick fingerprint of an asset's behaviour and the inputs
from which volatility, correlation and the optimization are built.

*Why no annualised median:* the mean annualises linearly (`mean × N`) because it's an expectation
and expectations add; a median is a quantile and quantiles don't add across a sum, so `median × N`
estimates nothing meaningful (it actually drifts toward the annualised *mean*). These figures stay
per-period; for an annual-scale central tendency use the geometric **CAGR** in §2.
""",

    "sortino": r"""
**Sortino ratio**

*How it's computed:* like the Sharpe ratio, but it only penalises *downside* movement. Both parts
use the same benchmark — the risk-free rate — so the denominator (downside deviation) is built
only from periods that fell short of it:

$$\text{Sortino}=\frac{\mu_{\text{ann}}-r_f}{\sigma_{\text{down}}}, \qquad \sigma_{\text{down}}=\sqrt{\tfrac{1}{T}\sum_t \min(r_t-r_f^{\text{period}},\,0)^2}\times\sqrt{N}$$

where $r_f$ = risk-free rate (the return of a "safe" asset), $r_f^{\text{period}}$ = its per-period
value, and $\sigma_{\text{down}}$ = downside deviation.

*What it means for you:* reward earned per unit of *bad* risk. Gains — and even small losses that
still beat the safe rate — don't count against you here; only shortfalls below the risk-free rate do.

*Why it's useful:* the Sharpe ratio treats a big gain as just as "risky" as a big loss. Sortino
fixes that, so it's the fairer score for assets with occasional large upside (like crypto).
""",

    "covariance": r"""
**Covariance matrix**

*How it's computed:* for every pair of assets, how their returns move together; the diagonal is
each asset's own variance (volatility squared):

$$\Sigma_{ij} = \frac{1}{T}\sum_t (r_{i,t}-\bar{r}_i)(r_{j,t}-\bar{r}_j)$$

*What it means for you:* a positive value means two assets tend to rise and fall together; a
negative value means one tends to zig when the other zags.

*Why it's useful:* it's the mathematical engine of diversification. The optimizer uses this matrix
to find mixes whose combined swings partly cancel out — lowering risk without necessarily
lowering return.

*Caveat:* for a basket mixing 7-day assets (crypto) with ~5-day assets (equity ETFs), this matrix
is computed on the shared-date (inner-join) calendar — the 7-day asset's weekend moves fold into
the next shared day — so the covariance (and the frontier/VaR built on it) is **approximate**.
Prefer weekly or monthly data for cleaner mixed-calendar figures.
""",

    "correlation": r"""
**Correlation matrix**

*How it's computed:* the covariance rescaled to a tidy −1…+1 range by dividing by the two assets'
volatilities:

$$\rho_{ij} = \frac{\Sigma_{ij}}{\sigma_i\,\sigma_j}$$

*What it means for you:* +1 means two assets move in lockstep, 0 means no relationship, −1 means
they move exactly opposite. The heatmap shows this at a glance.

*Why it's useful:* combining assets with low or negative correlation is the single most powerful
way to cut portfolio risk. Two strong-but-uncorrelated assets make a much smoother portfolio than
either alone. (Same mixed-calendar caveat as the covariance matrix above: correlations between
7-day and ~5-day assets are approximate on daily data — prefer weekly/monthly.)
""",

    # ── §6 — Input Portfolio Analysis ────────────────────────────────────────
    "rebalancing": r"""
**Rebalancing frequency**

*What it is:* how often your portfolio is reset back to its target weights. Pick it in the sidebar — **Never** (buy-and-hold), **Every 6 months**, **Yearly**, or **Every period**. Between resets the mix drifts: winners grow into a larger share and losers shrink.

*Where it applies:* this section — including its Tail Risk & Return Distribution subsection — follows your choice. The efficient-frontier sections (§7 Monte Carlo, §8 Scipy) always assume *per-period* rebalancing — the closed-form Modern Portfolio Theory math (annualised mean and $\sqrt{w^\top \Sigma w}$ volatility) only holds when the portfolio return is $\sum_i w_i r_i$ every period, which *is* per-period rebalancing. Rebalance less often and that identity breaks, so the frontier can't be re-derived for it.

*What it means for you:* less-frequent rebalancing lets the portfolio drift, usually raising its volatility and tail risk versus the per-period ideal. Comparing cadences here shows how much the rebalancing discipline actually matters for *your* allocation.
""",
    "buy_and_hold": r"""
**How the value series is built (this section, incl. its tail-risk subsection)**

*How it's computed:* we invest at your current weights and reset to them on your chosen cadence; between resets each holding drifts with its price. With **Never** selected it is pure buy-and-hold — bought once and held — so the value is the sum of the drifting holdings:

$$V_t = V_r \sum_i w_i\,\frac{P_{i,t}}{P_{i,r}}, \qquad V_0 = 1$$

where $w_i$ = your weight in asset $i$, $P_{i,t}$ = its price, and $r$ = the most recent rebalance date (for **Never**, $r = 0$ throughout).

*What it means for you:* this mirrors a real account you rebalance on a schedule (or leave alone) — winners grow into a larger share of the pot and losers shrink between resets. Every figure in this section derives from this single value series.

*Why it's useful:* it shows what your actual holdings would have done at that discipline. Note this differs from the optimization sections (Monte Carlo, Scipy), which always rebalance back to fixed weights every period — so the *same* portfolio's return, risk and drawdown can legitimately differ between sections unless you set the cadence to **Every period**.
""",

    "underwater_curve": r"""
**Underwater curve**

*How it's computed:* track the running peak of portfolio value, then plot how far below that peak you sit at every date:

$$U_t = \frac{V_t}{\max_{s\le t} V_s} - 1 \;\le\; 0$$

*What it means for you:* the line rests at 0 whenever you're at a new all-time high and dips negative through every losing stretch — the depth is exactly how far you're down from the best you'd ever seen.

*Why it's useful:* it turns the price history into a map of pain: how deep the holes were and, just as important, how long you spent climbing back out of them.
""",

    "max_underwater_period": r"""
**Maximum drawdown & underwater period**

*How it's computed:* the deepest drawdown is the largest fall from a peak to a later trough. Its *recovery time* is the days from that peak until value first regains it; separately, the *longest underwater period* is the most days spent below any peak before recovering:

$$\text{recovery days} = t_{\text{recover}} - t_{\text{peak}}$$

If value never returns to its peak within the data, the period is shown as "ongoing" and counted up to the last date.

*What it means for you:* depth tells you how bad it got; duration tells you how long you'd have waited, underwater, just to break even. The deepest fall and the longest wait are not always the same episode.

*Why it's useful:* many investors abandon a strategy not at the bottom but during the long, flat climb back. Knowing the worst recovery time sets expectations for how much patience the portfolio has historically demanded.
""",

    # ── §7 — Monte Carlo Efficient Frontier ──────────────────────────────────
    "monte_carlo": r"""
**Monte Carlo simulation (random portfolios)**

*How it's computed:* generate thousands of random weight combinations — drawn so they spread
evenly across every possible mix (each set is non-negative and sums to 100%) — and for every one
compute its annual return, volatility and Sharpe ratio:

$$w_i \ge 0, \qquad \sum_i w_i = 1$$

*What it means for you:* instead of guessing a few mixes, we scatter thousands across the map of
possibilities, so you can *see* the whole cloud of risk/return trade-offs your assets allow.

*Why it's useful:* the shape of the cloud reveals the best achievable trade-offs and where your
own portfolio sits relative to them — without needing any equations, just the picture.
""",

    "sharpe": r"""
**Sharpe ratio**

*How it's computed:* the portfolio's annual return above the risk-free rate, divided by its
volatility:

$$\text{Sharpe}=\frac{\mu_{\text{ann}}-r_f}{\sigma_{\text{ann}}}$$

where $r_f$ = risk-free rate (what a "safe" asset pays) and $\sigma_{\text{ann}}$ = annual
volatility.

*What it means for you:* how much reward you earned for each unit of risk taken. As a rough guide,
around 1 is decent and 2 is very good.

*Why it's useful:* it lets you compare wildly different investments fairly — a calm bond fund and
a volatile crypto position are judged by the same risk-adjusted yardstick. The colour scale on the
chart is this ratio.
""",

    "efficient_frontier": r"""
**Efficient frontier**

*How it's computed:* among all the simulated portfolios, the frontier is the upper-left edge of
the cloud — for each level of risk, the mix that delivered the highest return.

*What it means for you:* every portfolio *on* the frontier is "efficient": you can't get more
return without taking more risk. Anything *below* the frontier is wasteful — there's a better mix
with the same risk.

*Why it's useful:* it turns a vague goal ("good returns, not too risky") into a concrete menu. You
pick your comfort level of risk and read off the best portfolio for it.
""",

    "marked_portfolios": r"""
**The highlighted portfolios**

*How they're computed:* among all candidate portfolios we single out a few special ones — the
lowest-risk mix (**Min Volatility**), the highest-return mix (**Max Return**), the best
risk-adjusted mixes (**Max Sharpe**, **Max Sortino**), and **My Portfolio** (your actual current
weights).

*What it means for you:* they're reference points. *Max Sharpe* is the textbook "best bang for your
risk" portfolio; *Min Volatility* is the calmest; *My Portfolio* shows where you stand among them.

*Why it's useful:* seeing your portfolio plotted next to the optimal ones tells you, at a glance,
whether you're leaving return on the table or taking on risk you aren't being rewarded for.
""",

    "cvar": r"""
**CVaR (Conditional Value at Risk / Expected Shortfall)**

*How it's computed:* look at the worst slice of returns — the tail beyond the chosen confidence
level — and take their *average*:

$$\text{CVaR}_\alpha = -\,\text{average}\big(r \;\big|\; r \le q_{\alpha}\big)$$

where $q_\alpha$ = the cut-off return marking the worst $\alpha$ of outcomes (e.g. the worst 5%).

*What it means for you:* "when things go badly, how bad is the average bad period?" If the worst 5%
of periods average a −4% return, the 95% CVaR is 4%.

*Why it's useful:* Value at Risk only tells you the *threshold* of a bad outcome; CVaR tells you
how painful it is once you're past that threshold — a more honest picture of tail risk. (At
portfolio level it's computed on a constant-weight, per-period-rebalanced return series.)
""",

    # ── §8 — Scipy Efficient Frontier ────────────────────────────────────────
    "scipy_optimization": r"""
**Mathematical optimization (SLSQP)**

*How it's computed:* rather than sampling randomly, a solver (Sequential Least-Squares Quadratic
Programming) computes the exact weights that minimise risk or maximise the Sharpe ratio, subject to
the weights being positive and summing to 100%.

*What it means for you:* this finds the genuine best portfolios precisely, instead of the
best-among-the-random-guesses that the Monte Carlo cloud gives.

*Why it's useful:* the random simulation is great for intuition but rarely lands exactly on the
optimum. The solver pins it down, so the highlighted portfolios here are the true mathematical
best, not approximations.
""",

    "efficient_frontier_line": r"""
**Efficient frontier line**

*How it's computed:* the solver is run repeatedly, each time asked for the lowest-risk portfolio
that achieves a given target return; stringing those points together draws a smooth curve:

$$\min_{w}\ \sigma_{\text{ann}}(w) \quad \text{subject to}\quad \mu_{\text{ann}}(w)=\text{target}$$

*What it means for you:* the clean curve is the exact boundary of what's possible — the random
cloud only *approaches* it from below.

*Why it's useful:* it's the precise "menu" of best-possible portfolios. Any realistic portfolio
sits on or below this line; the closer to the line, the more efficient your mix.
""",

    # ── Tail Risk & Return Distribution (rendered inside §6; formerly standalone) ─
    "zscore": r"""
**z-score**

*How it's computed:* the point on a standard bell curve that cuts off the chosen tail. For 95%
confidence we take the 5% lower tail:

$$z = \Phi^{-1}(1-\alpha)$$

where $\Phi^{-1}$ is the inverse standard-normal (it turns a probability back into a number of
standard deviations) and $\alpha$ is your confidence level.

*What it means for you:* it's the multiplier that says how many "standard swings" away the bad tail
sits. For 95% it's about −1.65.

*Why it's useful:* it's the bridge between a confidence level you choose (e.g. 95%) and the
volatility of your portfolio, which together produce the Value at Risk figure below.
""",

    "var_parametric": r"""
**Parametric Value at Risk (VaR)**

*How it's computed:* assuming returns follow a bell curve, VaR scales volatility by the z-score to
mark the loss at the edge of your confidence level, keeping the (tiny per-period) expected return:

$$\text{VaR}_\alpha = \sigma\,|z| - \mu$$

Here $\sigma$ = volatility, $\mu$ = average return, $|z|$ = size of the z-score. Keeping $\mu$ (rather
than dropping it for a "drift-free" $\sigma|z|$) puts this figure on the **same basis as the historical
VaR/CVaR** shown beside it, so the comparison isolates tail *shape*, not a difference in convention.

*What it means for you:* "with 95% confidence, I won't lose more than X over this period." A 95%
VaR of 5% means only about 1 period in 20 should be worse than a 5% loss.

*Why it's useful:* it puts a single, intuitive figure on downside risk — widely used by banks
and funds. Caveat: it assumes a bell curve, so it can *understate* risk for fat-tailed assets
like crypto, where extreme periods happen more often than the curve predicts. Compare it with the
*Historical VaR/CVaR* shown beside it to see whether that assumption actually holds.
""",

    "var_historical": r"""
**Historical (empirical) VaR & CVaR**

*How it's computed:* no formula and no distribution assumption — we sort the portfolio's actual
past returns and read the tail off directly. Historical VaR is the loss it exceeded only on its
worst $(1-\text{confidence})$ of periods; historical CVaR is the average loss across exactly
those worst periods.

*What it means for you:* "based on what really happened, this is what the bad periods looked like."
It captures the crashes and fat tails that a smooth bell curve glosses over.

*Why it's useful:* it's the reality check on the parametric (normal) figures. When the historical
loss is clearly worse than the parametric one, the asset has fat tails and the normal model is
understating your risk — typical for crypto. The skew and excess-kurtosis figures quantify how
far from a bell curve the returns really are.
""",

    "return_distribution": r"""
**Return distribution & tail plot**

*How it's computed:* we histogram your portfolio's actual returns and overlay a fitted bell curve;
the shaded red region is the worst tail beyond the VaR threshold.

*What it means for you:* it shows how often each size of gain or loss occurred, and where the
"danger zone" of bad outcomes begins.

*Why it's useful:* it lets you eyeball whether the bell-curve assumption behind parametric VaR is
reasonable. If the real bars stick out well past the curve on the left, true tail risk is worse
than the formula suggests. The **median** beside the mean in the profile is a robust companion to
skew: mean above median points to a right tail, below to a left tail. Both stay per-period (a
median has no linear `×N` annualisation).
""",

    "real_returns": r"""
**Real (inflation-adjusted) terms**

*How it's computed:* with the sidebar toggle on, every price/value level is deflated by a constant
assumed inflation rate $\pi$ before any figure is computed — divided by a cumulative factor that
grows with calendar time since the start date:

$$V^{\text{real}}_t = \frac{V_t}{(1+\pi)^{\,(\text{years since start})}},\qquad r^{\text{real}} = \frac{1+r^{\text{nom}}}{1+\pi}-1$$

(the Fisher relation). The risk-free rate is deflated the same way so the Sharpe/Sortino numerator
stays consistent.

*What it means for you:* figures are expressed in today's purchasing power — what your money can
actually buy — rather than headline euros. A 6% nominal return with 2% inflation is really ~3.9%.

*Why it's useful — and what it does* ***not*** *change:* subtracting a **constant** rate shifts
every return down by about $\pi$, so **CAGR and average return drop** and **drawdowns deepen and
last longer** (flat nominal value is a real loss). But a constant offset can't change the *spread*
of returns, so **volatility and correlations are unchanged** (the parametric VaR's volatility term is
too; only its small per-period drift term shifts), and
because the risk premium $r^{\text{real}}-r_f^{\text{real}}\approx r^{\text{nom}}-r_f^{\text{nom}}$
is inflation-invariant, **Sharpe, Sortino and the §7/§8 efficient-frontier weights don't move** —
which is exactly why the optimization sections stay nominal.
""",

}


def render_section_help(overview, keys):
    """Render a collapsed 'How to read this section' expander.

    `overview` is a one-line plain-language summary of the section; `keys` is the
    ordered list of DESCRIPTIONS entries to show (self-contained per section).
    """
    with st.expander("📖 How to read this section", expanded=False):
        st.markdown(f"_{overview}_")
        for k in keys:
            md = DESCRIPTIONS.get(k)
            if md:
                st.markdown(md)
