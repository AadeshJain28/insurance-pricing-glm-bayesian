# ML-3 · Phases 4–5 — Pricing & Finalization

## Phase 4 — commercial price
- `pricing/optimize.py` — profit maximisation `(P − cost) × P(convert|P)`, elasticity helper.
- `pricing/run_pricing.py` — technical price from the Tweedie GLM → calibrated demand → optimal
  price, with a **sensitivity sweep over assumed elasticity** and a per-risk-decile view.
- `app/streamlit_app.py` — quote a policy, see frequency / severity / Tweedie technical price, then
  the profit-optimal commercial price with elasticity sliders and a profit curve.

**Design rule:** because Phase 2 proved elasticity is unidentifiable here, the demand curve is
*calibrated from stated assumptions* (`calibrate_logistic_demand`) and every pricing output is
labelled conditional. The sweep is the deliverable, not a single price.

Verified: calibration hits its targets exactly (q=0.2000, elasticity=−1.5000); optimal price falls
monotonically as demand becomes more elastic (600 → 269); markup never falls below cost.

## Phase 5 — finalization
- `README.md` — full results, five defensible findings, the endogeneity table, honest limitations.
- `docs/Interview_QA.md` — ~15 Q&As + resume bullet, each tied to a measured number.
- `.gitignore` excludes `data/`, `models/`, `*.joblib`, `*.nc`; `reports/*.md` are committed.

## Push
```powershell
git init
git add .
git commit -m "ML-3: insurance pricing with GLMs, Bayesian credibility and profit optimisation"
git branch -M main
git remote add origin https://github.com/AadeshJain28/insurance-pricing-glm-bayesian.git
git push -u origin main
```

## Final checklist
- [x] Frequency / severity / Tweedie GLMs + GBM challengers
- [x] Bayesian hierarchical credibility (converged primary models, 0 divergences)
- [x] Endogeneity investigation documented
- [x] Profit-optimal pricing with sensitivity
- [x] Streamlit demo
- [x] README + Interview QA
- [ ] Run `run_pricing`, launch the app, push to GitHub
