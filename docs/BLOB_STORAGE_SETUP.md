# Azure Blob Storage Model Hosting Guide

This guide explains how to host your HARE model weights in Azure Blob Storage and configure the app to download them dynamically. This reduces your Docker image from 500MB+ to ~50MB and makes deployments 10x faster.

---

## Why Use Blob Storage?

| Aspect | Local in Docker | Blob Storage |
|--------|-----------------|--------------|
| **Docker Image Size** | 500-800MB | ~50MB |
| **Deployment Time** | 5-10 minutes | 30-60 seconds |
| **Model Updates** | Rebuild image | Upload new file |
| **Cost** | Free (within credits) | ~$0.01/month |
| **Scalability** | Limited | Unlimited |

---

## Part 1: Create Azure Blob Storage Account

### Step 1: Create Storage Account
1. Go to Azure Portal: https://portal.azure.com
2. Search for **"Storage accounts"**
3. Click **"+ Create"**
4. Fill in:
   - **Resource group**: `hare-prototype-rg` (same as your app)
   - **Storage account name**: `haremodelstorage` (must be globally unique)
   - **Region**: Same as your App Service
   - **Performance**: `Standard`
   - **Redundancy**: `Locally-redundant storage (LRS)` (cheapest)
5. Click **"Review + create"** → **"Create"**
6. Wait for deployment (~1-2 minutes)

### Step 2: Create Container
1. Go to your new Storage Account
2. In left sidebar, click **"Containers"** (under Data storage)
3. Click **"+ Container"**
4. Fill in:
   - **Name**: `hare-models`
   - **Public access level**: `Private` (secure)
5. Click **"Create"**

### Step 3: Get Connection String
1. Go to your Storage Account
2. In left sidebar, click **"Access keys"**
3. Under **Key1**, click the copy button next to **Connection string**
4. Save this securely - you'll need it for the App Service

---

## Part 2: Upload Model Files to Blob Storage

### Option A: Using Azure Portal (Simple)
1. Go to your Storage Account → **Containers** → **hare-models**
2. Click **"Upload"**
3. Select your model file (e.g., `stage2_best.pth`)
4. Click **"Upload"**
5. Repeat for each model file

### Option B: Using Azure CLI (Faster for Multiple Files)
```bash
# Install Azure CLI: https://learn.microsoft.com/cli/azure/install-azure-cli

# Login to Azure
az login

# Upload single model
az storage blob upload \
  --account-name haremodelstorage \
  --container-name hare-models \
  --name stage2_best.pth \
  --file ./models/stage2_best.pth

# Upload all models
az storage blob upload-batch \
  --account-name haremodelstorage \
  --destination hare-models \
  --source ./models \
  --pattern "*.pth"
```

### Option C: Using Python (Programmatic)
```python
from azure.storage.blob import BlobClient

connection_string = "your-connection-string-here"
model_path = "./models/stage2_best.pth"

blob_client = BlobClient.from_connection_string(
    connection_string,
    container_name="hare-models",
    blob_name="stage2_best.pth"
)

with open(model_path, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)

print("Model uploaded successfully!")
```

---

## Part 3: Configure App Service Environment Variable

### Step 1: Add Connection String to App Service
1. Go to Azure Portal → Your App Service (`hare-prototype-app`)
2. In left sidebar, click **"Environment variables"** or **"Configuration"**
3. Click **"+ New application setting"**
4. Fill in:
   - **Name**: `AZURE_STORAGE_CONNECTION_STRING`
   - **Value**: Paste the connection string from Part 1, Step 3
5. Click **"OK"** → **"Save"**

### Step 2: Verify Configuration
The App Service will automatically restart with the new environment variable. The app will now:
1. Check for models locally first
2. If not found, download from Blob Storage
3. Cache locally for future use

---

## Part 4: Update Your Application (Already Done ✓)

The app now automatically:
1. Tries to load models from the local `models/` folder
2. Falls back to Azure Blob Storage if not found locally
3. Caches downloaded models for fast subsequent loads

**Files Updated:**
- `blob_storage_utils.py` - New utility for Blob Storage downloads
- `.dockerignore` - Excludes `.pth` files from Docker image
- `requirements.txt` - Added `azure-storage-blob` dependency

**Your app code doesn't need changes** - model loading is automatic!

---

## Part 5: Verify It Works

### Local Testing (Before Deploying)
```bash
# Set the connection string locally
$env:AZURE_STORAGE_CONNECTION_STRING = "your-connection-string"

# Run the app (models will download automatically)
streamlit run app.py
```

You should see in the terminal:
```
Downloading stage2_best.pth from Azure Blob Storage...
✓ Downloaded to ./models/stage2_best.pth
Loaded thesis-aligned checkpoint from: stage2_best.pth
```

### After Azure Deployment
1. Go to your App Service URL
2. The app will download models on first load
3. Check **App Service → Logs** for download status
4. Subsequent loads will be fast (uses cache)

---

## Cost Breakdown

| Service | Cost | Notes |
|---------|------|-------|
| Storage Account | ~$1/month | Basic storage in US region |
| Blob Storage | ~$0.01/month | 50-100MB of models |
| Data Transfer | $0 | First 1GB/month free (redundancy) |
| **Total** | ~**$1-2/month** | Negligible with $100 credits |

---

## Troubleshooting

### Error: "AZURE_STORAGE_CONNECTION_STRING not set"
**Cause**: Environment variable not configured in App Service

**Solution**:
1. Go to App Service → Configuration
2. Verify `AZURE_STORAGE_CONNECTION_STRING` exists
3. Restart the App Service

### Error: "Cannot download from Blob Storage"
**Cause**: Connection string is invalid or container doesn't exist

**Solution**:
1. Verify connection string in Azure Portal → Storage Account → Access keys
2. Verify container name is `hare-models` (exact case match)
3. Verify model files actually exist in the container

### Docker Image Still Large
**Cause**: Models weren't excluded properly

**Solution**:
1. Make sure `.dockerignore` has `models/*.pth` and `*.pth`
2. Rebuild Docker image:
   ```bash
   docker build --no-cache -t hare-app .
   ```

### Models Download Every Time (Not Caching)
**Cause**: Local `models/` folder is read-only in Azure

**Solution**:
- This is normal - Azure App Service has ephemeral storage
- Models cache within the current session (~1 hour)
- Re-download takes <5 seconds
- Consider switching to use memory caching if needed

---

## Model Management

### Update a Model
1. Upload new version to Blob Storage (same filename, different content)
2. Restart App Service
3. App downloads updated model on next request

### Add a New Model
1. Upload to Blob Storage container
2. Update app code to reference it (if needed)
3. Restart App Service

### Delete Old Models
1. Go to Storage Account → hare-models container
2. Right-click model → Delete

---

## Production Best Practices

✅ **Recommended:**
- Store models in Blob Storage
- Use environment variables for connection strings
- Set Blob Storage container to Private
- Monitor storage costs in Azure Cost Management
- Backup critical models

❌ **Avoid:**
- Hardcoding connection strings in code
- Public Blob Storage containers
- Large number of model versions (clean up old ones)
- Using Standard_GRS (costs more, not needed for models)

---

## Next Steps

1. Create Storage Account (**Part 1**)
2. Upload models to Blob Storage (**Part 2**)
3. Add connection string to App Service (**Part 3**)
4. Redeploy your app (push to GitHub)
5. Verify models download on first load

Your HARE app is now production-ready with dynamic model loading! 🚀

---

**Last Updated**: May 1, 2026  
**Applies to**: HARE Azure Deployment
