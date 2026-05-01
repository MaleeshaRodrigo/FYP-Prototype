# HARE Prototype Risk Management Notes

These notes support ISO 14971-style thinking for the thesis prototype. They are not a complete regulatory submission.

| Hazard | Potential harm | Mitigation | Verification evidence |
|---|---|---|---|
| User treats result as diagnosis | Delayed or inappropriate clinical care | Clear screening-only copy; no treatment advice; clinician review prompts | UI review and user guide |
| Unauthorized PHI access | Privacy breach | Authentication, RBAC, private storage, audit logging | Auth/RBAC tests and audit table review |
| Uploaded image exposed in repository | Privacy breach | Runtime `data/` ignored; Azure Blob recommended | `.gitignore`, storage configuration |
| Weak password storage | Account compromise | Passwords hashed with bcrypt/passlib | Code review |
| Adversarial perturbation changes result | Incorrect screening output | PGD-10 robustness check displayed with final result | Analysis Report test |
| Model underperforms on real-world phone photos | Misleading confidence | Dermoscopy validation limitation disclosed | UI notes and user guide |
| Bias across skin tones | Unequal performance | Marked as future fairness evaluation requiring labeled dataset | Traceability matrix |
| Audit tampering | Loss of accountability | Append-only UI behavior and hash chain fields | Audit log inspection |

## Residual Risk

The app remains a thesis research prototype. It is not a regulated medical device, does not integrate with EHR systems, and does not provide treatment or prescription advice. Full HIPAA and SaMD compliance would require organizational policies, formal validation, threat modeling, incident response, key management, deployment hardening, and clinical risk controls beyond this codebase.
