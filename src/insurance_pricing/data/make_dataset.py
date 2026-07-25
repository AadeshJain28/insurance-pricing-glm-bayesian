"""Phase 2 — build and persist processed datasets for both halves.

    python -m insurance_pricing.data.make_dataset
"""

from __future__ import annotations

from pathlib import Path


from insurance_pricing.data.load_fremtpl import build_policy_frame, fetch_fremtpl
from insurance_pricing.data.load_quotes import load_quotes
from insurance_pricing.evaluation.endogeneity import (
    price_response_corr, price_response_table, segmented_price_response, verdict,
)
from insurance_pricing.features.build_features import (
    prepare_demand_features, prepare_risk_features, severity_subset, train_test_split_frame,
)
from insurance_pricing.logger import get_logger

logger = get_logger("insurance_pricing.make_dataset")


def run(config_path: str = "config/config.yaml") -> None:
    from insurance_pricing.config import load_config

    cfg = load_config(config_path)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    reports = Path(cfg.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    # ---------------- risk side ----------------
    freq_raw, sev_raw = fetch_fremtpl(cfg.data.cache_dir)
    policies = build_policy_frame(
        freq_raw, sev_raw,
        exposure_cap=cfg.risk.exposure_cap,
        max_claim_amount=cfg.risk.max_claim_amount,
    )
    policies = prepare_risk_features(policies)
    risk_tr, risk_te = train_test_split_frame(policies, cfg.validation.test_size, cfg.project.seed)
    sev_tr, sev_te = severity_subset(risk_tr), severity_subset(risk_te)

    risk_tr.to_parquet(out / "risk_train.parquet", index=False)
    risk_te.to_parquet(out / "risk_test.parquet", index=False)
    sev_tr.to_parquet(out / "severity_train.parquet", index=False)
    sev_te.to_parquet(out / "severity_test.parquet", index=False)
    logger.info("Risk: train=%s test=%s | severity: train=%s test=%s",
                risk_tr.shape, risk_te.shape, sev_tr.shape, sev_te.shape)
    logger.info("Portfolio frequency (train) = %.4f claims/exposure-year",
                risk_tr["ClaimNb"].sum() / risk_tr["Exposure"].sum())
    logger.info("Mean severity per claim (train) = %.2f", sev_tr["AvgClaimAmount"].mean())

    # ---------------- demand side ----------------
    quotes = prepare_demand_features(load_quotes(cfg.data.quotes_file))
    dem_tr, dem_te = train_test_split_frame(
        quotes, cfg.validation.test_size, cfg.project.seed, stratify_col=cfg.demand.target_col
    )
    dem_tr.to_parquet(out / "demand_train.parquet", index=False)
    dem_te.to_parquet(out / "demand_test.parquet", index=False)
    logger.info("Demand: train=%s test=%s | conversion=%.4f",
                dem_tr.shape, dem_te.shape, float(dem_tr[cfg.demand.target_col].mean()))

    # ---------------- endogeneity report (a deliverable) ----------------
    price, target = cfg.demand.price_col, cfg.demand.target_col
    tbl = price_response_table(quotes, price, target)
    corr = price_response_corr(quotes, price, target)
    seg = segmented_price_response(quotes, price, target, ["Vehicle_Damage", "Vehicle_Age"])
    text = (
        "# Endogeneity check — can this data identify a price elasticity?\n\n"
        "A demand curve requires conversion to **fall** as price rises.\n\n"
        "## Conversion by price decile\n\n" + tbl.round(4).to_markdown(index=False)
        + f"\n\n**Overall correlation(price, conversion) = {corr:+.3f}**\n\n"
        "## Within segments\n\n" + seg.to_markdown(index=False)
        + f"\n\n## Verdict\n\n{verdict(corr, seg['corr_price_conversion'])}\n\n"
        "Consequently the pricing step uses a **stated elasticity with sensitivity bands**, "
        "and no causal elasticity is claimed from this data.\n"
    )
    (reports / "endogeneity_check.md").write_text(text, encoding="utf-8")
    logger.info("Wrote reports/endogeneity_check.md — verdict: %s", verdict(corr, seg["corr_price_conversion"]))


def main() -> None:
    run()


if __name__ == "__main__":
    main()
