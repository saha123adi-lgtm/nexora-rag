# NexSupport Product Manual

*Nexora Technologies Pvt. Ltd.*

- **Document ID:** PRD-001
- **Department:** Product
- **Version:** 4.7
- **Effective Date:** 2026-09-15
- **Owner:** Product Management - NexSupport
- **Classification:** Internal - All Employees

## 1. What is NexSupport?

NexSupport is Nexora's AI-powered customer support platform. It brings email, live chat and WhatsApp conversations into a single inbox, suggests or sends automatic replies using AI, routes tickets to the right team, and tracks SLA timers. Version 4.7 was released in September 2026.

## 2. Key Features

- Omnichannel inbox: email, web chat, WhatsApp and social messages.
- AI auto-reply with a confidence threshold, so low-confidence answers are sent to an agent.
- Smart ticket routing based on topic, language and customer tier.
- Knowledge base with article suggestions to agents and customers.
- SLA timers with breach alerts.

## 3. Plans

| Feature | Starter | Professional | Enterprise |
|---|---|---|---|
| API rate limit | 60 requests/min | 300 requests/min | 1,500 requests/min |
| Uptime SLA | None | 99.9% | 99.95% |
| SSO (SAML 2.0) | No | No | Yes |
| Data residency | India | India or EU | India or EU |
| Support | Email | Email and chat | 24x7 phone and dedicated manager |

## 4. API and Integrations

- The REST API uses Bearer token authentication.
- Webhooks are retried 5 times with exponential backoff if your endpoint fails.
- Maximum attachment size is 20 MB per file.
- Native integrations: Salesforce, HubSpot, Slack and Jira.

## 5. Data Residency

Customer data can be stored in the India (Mumbai) region or the EU (Frankfurt) region. The region is chosen at account creation and cannot be changed later without a migration project.

## 6. Release Cadence

New versions are released every 2 weeks. Customers on Enterprise can opt into a monthly release channel.
