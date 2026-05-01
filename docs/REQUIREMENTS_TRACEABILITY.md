# HARE Thesis Requirements Traceability

This document maps the thesis requirements to the implemented Streamlit thesis-demo system. The implementation is HIPAA-aligned for prototype evidence, not a claim of certified HIPAA compliance.

| ID | Status | Implementation evidence |
|---|---|---|
| FR01 | Implemented | Patient registration with unique email and hashed password. |
| FR02 | Implemented | Login required before protected pages are shown. |
| FR03 | Implemented | Researcher Admin page approves, disables, deletes, and resets passwords. |
| FR04 | Implemented | Authenticated upload supports JPEG and PNG. |
| FR05 | Implemented | Image History lists active images owned by the user. |
| FR06 | Implemented | Users can soft-delete owned image records. |
| FR07 | Implemented | DICOM upload parsed with `pydicom` and converted to RGB PNG for inference. |
| FR08 | Implemented | Analysis Report runs the trained CNN/ViT model on selected stored image. |
| FR09 | Implemented | Binary output is NV/MEL with patient-facing benign/lower concern or malignant/high concern wording. |
| FR10 | Implemented | PGD-10 robustness verification is run during robust analysis. |
| FR11 | Implemented | Final output includes classification and robustness status. |
| FR12 | Implemented | Report displays melanoma probability and confidence score. |
| FR13 | Implemented | Critical events write append-only hash-chained audit rows. |
| FR14 | Implemented | Researcher Technical page supports FGSM and PGD simulation. |
| FR15 | Implemented | Researcher Technical page includes Grad-CAM heatmap. |
| FR16 | Documented | No external EHR integration is included. |
| FR17 | Documented | App avoids treatment, prescription, or diagnostic advice beyond screening classification. |
| NFR01 | Prototype aligned | Azure PostgreSQL and Blob Storage provide managed encryption at rest and TLS in transit when configured. Local fallback is for demo only. |
| NFR02 | Implemented | RBAC restricts patient data to owners and researcher functions to researchers; access failures are audited. |
| NFR03 | Documented | ISO 14971-style risk notes are maintained in `RISK_MANAGEMENT.md`. |
| NFR04 | Evidence required | Robustness workflow exists; dataset-level >90% retention must be supported by thesis evaluation results. |
| NFR05 | Partially implemented | PGD-10 demo is bounded; runtime depends on model size and hardware. |
| NFR06 | Partially implemented | Pages use cached model loading and simple Streamlit layouts; deployment timing must be measured. |
| NFR07 | Implemented | Patient workflow is upload, history, select, analysis report. |
| NFR08 | Implemented | Report highlights classification, confidence, and robustness status. |
| NFR09 | Evidence required | Model metrics are documented on Technical Research page; final dataset evidence remains a thesis artifact. |
| NFR10 | Could Have | Fairness testing requires a separate labeled skin-tone dataset. |
