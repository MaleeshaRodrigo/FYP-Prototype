# HARE Thesis Requirements Implementation Guide

## 1. Purpose

This guide explains how to implement the full thesis requirements plan for the HARE Streamlit prototype and how to set up the additional Azure resources needed for a stronger thesis-demo deployment.

The target is a HIPAA-aligned research prototype, not a certified HIPAA-compliant production system or regulated medical device. The implementation should clearly document this limitation in the user interface, user guide, and thesis report.

## 2. Target Architecture

The full thesis-demo implementation should use:

| Layer | Recommended service or module | Purpose |
|---|---|---|
| Web application | Azure App Service running the Streamlit app | Hosts the HARE UI |
| User/account database | Azure Database for PostgreSQL Flexible Server | Stores users, image metadata, analysis results, attack results, and audit logs |
| Image storage | Azure Blob Storage private container | Stores uploaded JPEG, PNG, and converted DICOM images |
| Model storage | Azure Blob Storage private container `hare-models` | Stores `.pth` model checkpoints |
| Secrets/config | Azure App Service application settings | Stores database URL, storage connection string, bootstrap admin credentials, and model settings |
| Audit trail | PostgreSQL append-only `audit_events` table | Records critical security and analysis events |

## 3. Implementation Plan by Requirement Area

### 3.1 User Management

Implement:

- Registration form for new patient users.
- Unique email validation.
- Password hashing using `passlib[bcrypt]`.
- Login form using email and password.
- Session-based authentication through Streamlit `st.session_state`.
- Roles: `patient` and `researcher`.
- New patient accounts default to `pending`.
- Researcher admin page for:
  - Approving users
  - Disabling users
  - Marking users deleted
  - Resetting passwords with a temporary password

Recommended database table:

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('patient', 'researcher')),
    status TEXT NOT NULL CHECK (status IN ('pending', 'active', 'disabled', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ,
    approved_by INTEGER REFERENCES users(id)
);
```

Required environment variables:

```env
APP_SECRET_KEY=change-me-to-a-long-random-secret
ADMIN_BOOTSTRAP_EMAIL=researcher@example.com
ADMIN_BOOTSTRAP_PASSWORD=ChangeThisPassword123
```

### 3.2 Image Management

Implement:

- Authenticated image upload page.
- Supported formats:
  - JPEG
  - PNG
  - DICOM, using `pydicom`
- Private Blob Storage upload for image bytes.
- PostgreSQL metadata record for each image.
- Image History page showing only the current user's active images.
- Soft delete for image records.
- Ownership checks before viewing, selecting, deleting, or analyzing an image.

Recommended database table:

```sql
CREATE TABLE image_records (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    original_filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    blob_path TEXT NOT NULL,
    storage_backend TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    image_format TEXT NOT NULL,
    dicom_metadata JSONB,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ
);
```

Required environment variables:

```env
AZURE_STORAGE_CONNECTION_STRING=<storage-account-connection-string>
AZURE_IMAGE_CONTAINER=hare-images
```

### 3.3 Core Analysis Pipeline

Implement:

- Analysis page that requires a selected stored image.
- Model inference using the trained CNN/ViT HARE model.
- Binary screening output:
  - `NV` / lower concern
  - `MEL` / high concern
- Numerical confidence/probability.
- GA-calibrated late-fusion probability using:
  - `GA_ALPHA`
  - `GA_TAU`
  - `GA_THETA`
- PGD-10 robustness verification during the main analysis workflow.
- Final report text such as:
  - `Result: High concern melanoma screening signal (Verified Robust)`
  - `Result: Lower concern melanoma screening signal (Robustness Warning)`

Recommended database table:

```sql
CREATE TABLE analysis_results (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_records(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    predicted_label TEXT NOT NULL,
    predicted_summary TEXT NOT NULL,
    melanoma_probability DOUBLE PRECISION NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL,
    ga_threshold DOUBLE PRECISION NOT NULL,
    robustness_status TEXT NOT NULL,
    robustness_attack TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
```

Required environment variables:

```env
MODEL_DIR=./models
ACTIVE_CHECKPOINT=stage2_best.pth
GA_THETA=0.1372
GA_TAU=1.8162
GA_ALPHA=0.0582
```

### 3.4 Researcher Technical Functions

Implement:

- Restrict Technical Research page to `researcher` users only.
- Keep thesis metrics, architecture notes, and failure mode notes visible to researchers.
- Keep Grad-CAM explainability.
- Allow researcher-triggered attacks:
  - FGSM
  - PGD-20 or configurable PGD
- Persist attack simulation results.
- Audit every attack simulation.

Recommended database table:

```sql
CREATE TABLE attack_simulations (
    id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL REFERENCES image_records(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    attack_type TEXT NOT NULL,
    epsilon DOUBLE PRECISION NOT NULL,
    before_label TEXT NOT NULL,
    after_label TEXT NOT NULL,
    changed BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
```

### 3.5 Audit Logging

Implement an append-only audit log for:

- Registration attempts
- Login success and failure
- Pending or disabled account access attempts
- Upload events
- Image delete events
- Analysis requests
- Model results
- Robustness checks
- FGSM/PGD simulations
- Researcher admin actions
- RBAC denied events

Recommended database table:

```sql
CREATE TABLE audit_events (
    id SERIAL PRIMARY KEY,
    actor_user_id INTEGER REFERENCES users(id),
    target_resource TEXT,
    event_type TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    ip_address TEXT,
    details JSONB NOT NULL,
    previous_hash TEXT NOT NULL,
    current_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
```

Hash-chain rule:

- `previous_hash` stores the previous row's `current_hash`.
- `current_hash` is computed from the event payload and `previous_hash`.
- The UI must not provide update or delete operations for audit rows.

## 4. Azure Resource Setup

### 4.1 Resource Group

Create one resource group for all project resources.

Recommended:

- Name: `hare-prototype-rg`
- Region: same region for App Service, Storage, and PostgreSQL

Azure Portal:

1. Go to Azure Portal.
2. Search for `Resource groups`.
3. Click `Create`.
4. Enter `hare-prototype-rg`.
5. Choose a nearby region.
6. Click `Review + create`.

Azure CLI:

```powershell
az group create `
  --name hare-prototype-rg `
  --location eastus
```

### 4.2 Azure Database for PostgreSQL Flexible Server

Create PostgreSQL for users, metadata, analysis results, attack results, and audit logs.

Recommended low-cost thesis-demo settings:

| Setting | Recommended value |
|---|---|
| Service | Azure Database for PostgreSQL Flexible Server |
| Server name | `hare-postgres-demo` or globally unique variant |
| PostgreSQL version | 15 or 16 |
| Compute tier | Burstable |
| Size | B1ms for demo, larger if performance is poor |
| Storage | 32 GB minimum |
| High availability | Disabled for student-credit demo |
| Authentication | PostgreSQL authentication |
| Public access | Allowed only from App Service outbound IPs if possible |
| SSL | Required |

Azure Portal steps:

1. Search for `Azure Database for PostgreSQL flexible servers`.
2. Click `Create`.
3. Select the project resource group.
4. Enter a server name, admin username, and strong password.
5. Select the same region as the App Service.
6. Choose a low-cost burstable compute tier for the thesis demo.
7. Create the server.
8. Open the PostgreSQL server resource.
9. Create a database named `hare`.
10. Configure networking:
    - For quick demo: allow public access from your current IP and App Service outbound IPs.
    - For stronger security: use private networking/VNet integration.
11. Require SSL/TLS connections.

Connection string format:

```env
DATABASE_URL=postgresql://<admin-user>:<password>@<server-name>.postgres.database.azure.com:5432/hare?sslmode=require
```

Add this value to:

- Local `.env` for testing
- Azure App Service application settings for deployment

### 4.3 Azure Storage Account

Use one Storage Account with private containers.

Recommended containers:

| Container | Purpose | Public access |
|---|---|---|
| `hare-models` | Model checkpoint files such as `.pth` | Private |
| `hare-images` | Uploaded patient/demo images | Private |

Portal steps:

1. Go to `Storage accounts`.
2. Click `Create`.
3. Select the project resource group.
4. Choose `Standard` performance.
5. Choose `Locally-redundant storage (LRS)` for low-cost demo use.
6. After creation, open `Containers`.
7. Create `hare-models` with private access.
8. Create `hare-images` with private access.
9. Copy the Storage Account connection string from `Access keys`.

App settings:

```env
AZURE_STORAGE_CONNECTION_STRING=<connection-string>
AZURE_IMAGE_CONTAINER=hare-images
```

Upload model files to `hare-models`:

```powershell
az storage blob upload `
  --account-name <storage-account-name> `
  --container-name hare-models `
  --name stage2_best.pth `
  --file .\models\stage2_best.pth
```

### 4.4 Azure App Service Settings

In Azure Portal:

1. Open the HARE App Service.
2. Go to `Settings` -> `Environment variables` or `Configuration`.
3. Add these application settings:

```env
APP_SECRET_KEY=<long-random-secret>
ADMIN_BOOTSTRAP_EMAIL=<researcher-email>
ADMIN_BOOTSTRAP_PASSWORD=<temporary-strong-password>
DATABASE_URL=postgresql://<user>:<password>@<server>.postgres.database.azure.com:5432/hare?sslmode=require
AZURE_STORAGE_CONNECTION_STRING=<storage-connection-string>
AZURE_IMAGE_CONTAINER=hare-images
MODEL_DIR=./models
ACTIVE_CHECKPOINT=stage2_best.pth
GA_THETA=0.1372
GA_TAU=1.8162
GA_ALPHA=0.0582
```

4. Save changes.
5. Restart the App Service.
6. Sign in with the bootstrapped researcher account.
7. Change or rotate the bootstrap password after first use.

## 5. Local Development Setup

Install dependencies:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

For local-only testing, leave `DATABASE_URL` and `AZURE_STORAGE_CONNECTION_STRING` blank if the implementation includes SQLite/local file fallback. For Azure integration testing, fill them with the actual Azure values.

Run the app:

```powershell
streamlit run app.py
```

## 6. Verification Checklist

### User Management

- Register a new patient user.
- Confirm duplicate email registration fails.
- Confirm pending patient cannot access protected pages.
- Approve patient from Researcher Admin.
- Confirm approved patient can sign in.
- Confirm disabled/deleted users cannot sign in.
- Confirm password reset generates a temporary password.

### Image Management

- Upload JPEG image.
- Upload PNG image.
- Upload valid DICOM image.
- Confirm invalid DICOM is rejected safely.
- Confirm uploaded image appears in Image History.
- Confirm patient cannot access another patient's image.
- Soft-delete an image and confirm it disappears from active history.

### Analysis

- Select a stored image.
- Run robust analysis.
- Confirm binary classification is shown.
- Confirm melanoma probability/confidence is shown.
- Confirm GA threshold is shown.
- Confirm PGD-10 robustness status is shown.
- Confirm analysis result is stored in PostgreSQL.

### Researcher Functions

- Confirm patient cannot access Technical Research, Researcher Admin, or Audit Log.
- Run FGSM simulation as researcher.
- Run PGD simulation as researcher.
- Confirm attack simulation is stored.
- Confirm attack simulation audit event is written.

### Audit

- Confirm login success/failure events are logged.
- Confirm upload, delete, analysis, robustness, and admin actions are logged.
- Confirm audit rows include `previous_hash` and `current_hash`.
- Confirm the UI has no edit/delete action for audit records.

### Azure

- Confirm App Service can connect to PostgreSQL.
- Confirm App Service can download model files from `hare-models`.
- Confirm uploaded images are stored in `hare-images`.
- Confirm containers are private.
- Confirm App Service uses HTTPS.

## 7. Thesis Documentation Notes

In the thesis, describe the implemented system as:

- A Streamlit-based HARE melanoma screening research prototype.
- A role-controlled thesis-demo system when authentication/RBAC is implemented.
- HIPAA-aligned in technical controls, but not legally certified HIPAA compliant.
- Not a regulated medical device.
- Not integrated with EHR systems.
- Not intended to provide treatment, prescriptions, or final diagnosis.

Include these supporting artifacts:

- Functional requirements table
- Non-functional requirements table
- Requirements traceability matrix
- Risk management notes
- Azure architecture diagram
- Database schema
- Test evidence screenshots
- Audit log screenshots
- Model performance and robustness evaluation results

## 8. Recommended Implementation Order

1. Add database layer and schema initialization.
2. Add authentication, registration, and researcher bootstrap.
3. Add RBAC page guards.
4. Add Azure Blob image storage and image metadata table.
5. Add Image History page.
6. Add DICOM parsing.
7. Add robust Analysis Report page.
8. Add audit logging across all critical events.
9. Add Researcher Admin page.
10. Restrict Technical Research page to researchers.
11. Persist attack simulation results.
12. Update user guide, deployment guide, and thesis traceability documents.

