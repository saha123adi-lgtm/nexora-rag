# Nexora Technologies - Enterprise RAG Dataset

A fictional company knowledge base for building and evaluating an Enterprise RAG assistant. All people, numbers, phone numbers and email addresses are invented (emails use the reserved `.example` domain).

## Folder structure

```
nexora_dataset/
  documents/        21 source documents (PDF, DOCX, MD, TXT) grouped by department
  metadata/
    documents_metadata.csv   one row per document, including access control fields
    users.json               7 sample users with roles, for testing role-based access
  evaluation/
    qa_testset.csv           question set with expected answers and source documents
```

## Documents

| ID | File | Dept | Format | Access | Allowed roles |
|---|---|---|---|---|---|
| HR-001 | HR_Leave_Policy.pdf | HR | pdf | public | all |
| HR-002 | HR_Code_of_Conduct.docx | HR | docx | public | all |
| HR-003 | HR_Remote_Work_Policy.md | HR | md | public | all |
| HR-004 | HR_Onboarding_Guide.txt | HR | txt | public | all |
| HR-005 | HR_Performance_Review.pdf | HR | pdf | public | all |
| HR-006 | HR_Compensation_Bands.docx | HR | docx | restricted | hr;admin |
| HR-007 | HR_Disciplinary_Procedure.pdf | HR | pdf | restricted | hr;manager;admin |
| HR-008 | HR_Employee_Benefits.docx | HR | docx | public | all |
| IT-001 | IT_Helpdesk_FAQ.md | IT | md | public | all |
| IT-002 | IT_Laptop_and_Device_Policy.pdf | IT | pdf | public | all |
| IT-003 | IT_VPN_Setup_Guide.txt | IT | txt | public | all |
| IT-004 | IT_Software_Request_Process.docx | IT | docx | public | all |
| SEC-001 | Information_Security_Policy.pdf | Security | pdf | public | all |
| SEC-002 | Incident_Response_Plan.docx | Security | docx | restricted | security;admin |
| SEC-003 | Data_Privacy_and_Retention_Policy.pdf | Security | pdf | public | all |
| FIN-001 | Expense_Reimbursement_Policy.pdf | Finance | pdf | public | all |
| FIN-002 | Travel_Policy.docx | Finance | docx | public | all |
| FIN-003 | Procurement_Policy.txt | Finance | txt | restricted | finance;admin |
| PRD-001 | NexSupport_Product_Manual.md | Product | md | public | all |
| PRD-002 | NexAnalytics_FAQ.pdf | Product | pdf | public | all |
| SAL-001 | Pricing_and_Discount_Policy.docx | Sales | docx | restricted | sales;manager;admin |

`access_level = public` means every employee can retrieve it (`allowed_roles = all`). `restricted` documents are visible only to the listed roles.

## Roles

Roles used in `allowed_roles`: `employee` (everyone), `hr`, `manager`, `finance`, `security`, `sales`, `admin`. A user may access a document if it is `all` or if any of the user's roles appears in `allowed_roles`. See `metadata/users.json` for sample users.

## Evaluation set

`qa_testset.csv` columns: `qid`, `question`, `expected_answer`, `source_doc_ids`, `category`, `asking_role`, `expected_behavior`, `evidence_snippet`.

| Category | Purpose |
|---|---|
| factual | Single fact in a single document |
| paraphrase | Question worded differently from the document |
| multi_hop | Needs reasoning or combining information (including across documents) |
| access_allowed | Restricted document, asked by an authorised role |
| access_denied | Restricted document, asked by an unauthorised role. The assistant must NOT reveal the content |
| unanswerable | Answer is not in the documents. The assistant should say it does not know |

`expected_behavior` is one of `answer`, `refuse_access`, `say_unknown`. `evidence_snippet` is an exact phrase from the source document, useful for checking retrieval (did the retrieved chunks contain it?).

## Suggested metadata per chunk

`doc_id`, `title`, `department`, `access_level`, `allowed_roles`, `version`, `source file`, `page` or `section heading`.

## Notes

- Amounts use `Rs.` and Indian digit grouping (for example Rs. 5,00,000) so that chunking and embedding see consistent text.
- Several facts are deliberately repeated or related across documents (for example the 30-day training deadline in HR-004 and SEC-001) to test multi-document retrieval.
- Tables appear in most documents, which makes chunking strategy matter.
