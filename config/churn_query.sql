-- Churn pipeline source query.
-- Must return one row per customer with the columns the pipeline expects.
-- Use AS aliases to map your real schema onto these exact names.
-- The `churned` column is required for training, optional for scoring.

SELECT
    customer_id,
    subscription_type,
    monthly_revenue,
    tenure_months,
    country,
    industry,
    sessions_30d,
    sessions_90d,
    days_since_last_login,
    days_since_last_core_action,
    login_frequency_change,
    session_duration_change,
    ticket_count,
    support_sentiment,
    error_rate,
    failed_api_calls,
    rage_click_count,
    billing_page_visits,
    downgrade_page_visits,
    invoice_download_frequency,
    trial_expiry_days,
    tier_capacity_utilization,
    export_frequency,
    api_usage,
    api_limit_usage,
    seat_limit_usage,
    churned
FROM customer_features;
