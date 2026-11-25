# Marketing Mix Model: Where to Invest the Next Pound?

This project answers the main business question: **where a brand should increase or decrease its advertising budget** to efficiently attract new customers while keeping the target cost per acquisition (CPA) at or below **£40**.

---

## Key Findings

- **All channels are profitable:** At current spending, every channel acquires customers cheaper than the £40 limit on average — Bing (£15), Meta (£29), Google (£32), Pinterest (£33).
- **The main driver is Meta:** It shows minimal saturation at current spend levels. A +20% budget increase (+£15k over 30 days) brings ~526 new customers at ~£29 each.
- **Channel limitations:** Google saturates fastest (CPA exceeds £40 at ~1.5x spend). Bing is cheapest (£15) but small. Pinterest has high variance across model refits.
- **Do not cut total budget:** A 10% spend reduction loses ~616 customers (saving only £28 per lost customer). Reallocating current budget across all 4 channels adds ~145 customers (+1%).
- **Model validation:** Outperforms no-media baseline (MAPE 6.2% vs 13.7%). Recommendations remain robust across 5 model sensitivity refits.

---

## Tech Stack
- **Python 3.13**
- **PyMC 6.3 & PyMC-Marketing 1.3** (Bayesian marketing mix modeling)
