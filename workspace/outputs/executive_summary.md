# Executive Summary Direct Marketing Optimization

## Objective
Identify which of the bank's clients are most likely to buy a consumer loan, credit card, or mutual fund, and select the 97 clients (15% of the untapped client base) to contact in order to maximize expected campaign revenue.

## Approach
Three propensity models were built — one per product — using socio-demographic, product-holding, and account-activity data for the 969 clients (60%) with known outcomes. Each model blends a logistic regression with a gradient boosting model. For each client, propensity was combined with the average revenue earned from past buyers of each product to compute an expected revenue per offer. Clients were then ranked by their single best offer, and the top 97 were selected. The strategy was validated with a held-out back-test on the labeled data before being applied to the 646 untapped clients.

## Who is targeted, and with what
| Offer | Clients contacted | Typical profile | Existing holders in list |
|---|---|---|---|
| **Consumer loan (CL)** | 73 | Young (median age 26 vs. 40 overall), long-tenured (160 vs. 102 months), more transactionally active | 14% already hold a CL |
| **Credit card (CC)** | 24 | Established clients with a savings account (96% vs. 25% overall) and a substantial savings balance | 0% already hold a CC |
| **Mutual fund (MF)** | 0 | Active accounts with high turnover are the best MF prospects, but their expected revenue never clears the cutoff for a contact slot (see below) | — |

Across the client base, higher propensity for each product associates with:
- **Consumer loan:** younger clients with longer tenure — the single strongest predictor of the three products.
- **Credit card:** clients already holding a savings account, with higher current-account balances.
- **Mutual fund:** clients with high transaction volumes but comparatively low current-account balances, and existing MF holders.

## Why no mutual fund offers
Mutual funds have real propensity signal (the best MF prospects score up to 59% likely to buy), but MF's average revenue per buyer is lower than the other two products, so it never wins a scarce contact slot under a 97-client budget. If contact capacity increases, MF becomes the next offer to add, as the full ranked list (`all_targeting_clients_scored.csv`) shows exactly which clients that would be.

## Expected revenue
- **Projected revenue from the 97 contacts: ≈ 640 (range ≈ 530–790)**, based on a back-test that applied the identical strategy to the held-out labeled data and measured realized (not just predicted) revenue.
- This is **1.8× the revenue of contacting the same 97 clients with a single, undifferentiated consumer-loan offer** (≈ 350), which we use as the baseline for the models' added value.
- A simpler, consumer-loan-only strategy scores slightly higher in the back-test (≈ 770), but the gap is within statistical noise and driven by a small number of high-value credit-card buyers. It is reported here as a lower-variance alternative.

## Model quality
The three propensity models achieve moderate but real discrimination (AUC 0.60–0.65; the top 15% of ranked clients convert at 1.9–2.0× the base rate). This is consistent with dummy data of this size: the value of the strategy comes primarily from concentrating contacts among the highest-scoring clients, not from any single feature dominating the prediction.

## Key assumptions and risks
- Historical sale outcomes are treated as a proxy for how clients would respond to this campaign; no control group is available in the data to measure true causal uplift.
- Revenue estimates rely on the average revenue of past buyers, which is heavy-tailed for credit cards (54% of CC revenue comes from the top 5% of buyers) — this is the main source of uncertainty in the revenue projection.
- The back-test is mildly optimistic, since the modeling choices were made on the same data it is validated against.

## Recommendation
Launch the campaign with a small randomized holdout (e.g., 10% of the selected 97) that receives no offer. This directly measures incremental uplift and removes the largest assumption underlying this analysis for future campaigns.