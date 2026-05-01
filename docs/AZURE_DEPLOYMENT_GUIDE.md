# Azure Deployment Guide for HARE Prototype

## Overview
This guide walks you through deploying your Streamlit HARE application to Azure App Service using your $100 student credits.

---

## Prerequisites
- ✅ GitHub account with your code pushed
- ✅ Azure account with $100 credits
- ✅ Streamlit app ready
- ✅ Configuration files created (Dockerfile, .streamlit/config.toml, GitHub Actions workflow)

---

## Part 1: Application Configuration (Already Completed ✅)

The following files have been created:

### 1. **Dockerfile**
- Containerizes your Streamlit application
- Uses Python 3.9 slim image for minimal size
- Installs dependencies and configures Streamlit to run on port 8501

### 2. **.streamlit/config.toml**
- Configures Streamlit server for cloud deployment
- Sets server address to 0.0.0.0 to accept external connections
- Optimizes for web access

### 3. **.github/workflows/deploy.yml**
- Automated GitHub Actions workflow
- Builds Docker image on every push to main branch
- Pushes to Azure Container Registry
- Deploys to App Service

### 4. **requirements.txt** (Updated)
- Pinned PyTorch versions for consistency
- Ensures reproducible builds

---

## Important: Model Storage Setup (First, Read This!)

Before creating resources, decide how to host your model files:

**Option A: Azure Blob Storage (Recommended - 50MB Docker image)**
- Download models dynamically from cloud storage
- Makes deployments 10x faster
- Easy to update models without rebuilding
- See [BLOB_STORAGE_SETUP.md](BLOB_STORAGE_SETUP.md) for detailed instructions

**Option B: Bundle in Docker (Large image - 500MB+)**
- Simpler setup, no extra configuration
- Slower deployments
- Hard to update models

**Recommendation**: Follow Option A if you want production-grade hosting. Follow Option B if you want the quickest setup.

---

## Part 2: Azure Portal Setup

### Step 1: Sign in to Azure Portal
1. Navigate to https://portal.azure.com
2. Sign in with your university account
3. Verify your $100 credits are available

### Step 2: Create a Resource Group
1. In the search bar at the top, type **"Resource groups"**
2. Click **"Create"**
3. Fill in the details:
   - **Resource group name**: `hare-prototype-rg`
   - **Region**: Select the region closest to you (e.g., `East US`, `UK South`, `West Europe`)
4. Click **"Review + create"** → **"Create"**
5. Wait for deployment to complete

### Step 3: Create Container Registry
1. Search for **"Container Registries"** in the search bar
2. Click **"Create"**
3. Fill in the details:
   - **Resource group**: Select `hare-prototype-rg`
   - **Registry name**: `hareprototype` (must be lowercase, no hyphens, 5-50 characters)
   - **Region**: Same as your resource group
   - **SKU**: Select `Basic` (free tier, sufficient for your needs)
4. Click **"Review + create"** → **"Create"**
5. Wait for deployment to complete (~2 minutes)

### Step 4: Get Container Registry Credentials
1. Navigate to your newly created Container Registry
2. In the left sidebar, click **"Access keys"**
3. Enable the **"Admin user"** toggle
4. Copy and save the following information (you'll need it for GitHub):
   - **Login server** (e.g., `hareprototype.azurecr.io`)
   - **Username**
   - **Password**

---

## Part 3: Deploy to App Service

### Step 5: Create App Service Plan
1. Search for **"App Service plans"** in the search bar
2. Click **"Create"**
3. Fill in the details:
   - **Resource group**: Select `hare-prototype-rg`
   - **Name**: `hare-app-plan`
   - **Operating System**: `Linux`
   - **Region**: Same as your resource group
   - **Pricing tier**: Click **"Change size"**
     - Select **B1** (basic tier: ~$10/month)
     - Note: This is covered by your $100 credits
4. Click **"Review + create"** → **"Create"**
5. Wait for deployment (~1-2 minutes)

### Step 6: Create App Service
1. Search for **"App Services"** in the search bar
2. Click **"Create"**
3. Fill in the details:
   - **Resource group**: Select `hare-prototype-rg`
   - **Name**: `hare-prototype-app`
     - This becomes your public URL: `https://hare-prototype-app.azurewebsites.net`
   - **Publish**: Select `"Docker Container"`
   - **Operating System**: `Linux`
   - **Region**: Same as your resource group
   - **App Service Plan**: Select `hare-app-plan`
4. Click **"Next: Docker"**

### Step 7: Configure Docker Settings (Initial)
On the Docker tab:
- **Image Source**: Select `"Azure Container Registry"`
- **Registry**: Your registry will appear here (e.g., `hareprototype`)
- **Image**: `hare-app`
- **Tag**: `latest`
- Click **"Review + create"** → **"Create"**

Wait for deployment (~3-5 minutes).

---

## Part 4: GitHub Configuration

### Step 8: Add GitHub Secrets
Your GitHub repository needs credentials to deploy to Azure.

1. Go to your GitHub repository
2. Click **"Settings"** (in the top menu)
3. In the left sidebar, click **"Secrets and variables"** → **"Actions"**
4. Click **"New repository secret"** for each of the following:

**Secret 1 - Registry Login Server**
- **Name**: `REGISTRY_LOGIN_SERVER`
- **Value**: Copy from Step 4 (e.g., `hareprototype.azurecr.io`)
- Click **"Add secret"**

**Secret 2 - Registry Username**
- **Name**: `REGISTRY_USERNAME`
- **Value**: Copy from Step 4
- Click **"Add secret"**

**Secret 3 - Registry Password**
- **Name**: `REGISTRY_PASSWORD`
- **Value**: Copy from Step 4
- Click **"Add secret"**

**Secret 4 - Azure Publish Profile**
- **Name**: `AZURE_PUBLISH_PROFILE`
- **Value**: Paste the publish profile XML from Step 10
- Click **"Add secret"**

### Step 9: App Registration Is Not Required
If Azure shows a 401 or says you do not have access to App registrations, skip that step.
This deployment guide uses the App Service publish profile instead, so you do not need to create an app registration or service principal.

### Step 10: Get Azure Publish Profile
1. Go to Azure Portal and navigate to your App Service (`hare-prototype-app`)
2. In the top right, click **"Get publish profile"**
3. This downloads an XML file automatically
4. Open the XML file with a text editor and copy all the content

5. Go back to GitHub repo → **Settings** → **Secrets and variables** → **Actions**
6. Click **"New repository secret"**
7. Fill in:
   - **Name**: `AZURE_PUBLISH_PROFILE`
   - **Value**: Paste the entire XML content
8. Click **"Add secret"**

---

## Part 5: Deploy Your Application

### Step 12: Push Code to GitHub
Make sure all your files are committed and pushed:

```bash
# Add the deployment files (if not already done)
git add Dockerfile .streamlit/config.toml requirements.txt .github/workflows/deploy.yml

# Commit the changes
git commit -m "Add Azure deployment configuration"

# Push to GitHub
git push origin main
```

### Step 13: Monitor the Deployment
1. Go to your GitHub repository
2. Click the **"Actions"** tab (top menu)
3. You'll see a workflow running named **"Deploy to Azure"**
4. Watch the progress:
   - **Build and push Docker image** (1-2 minutes)
   - **Deploy to App Service** (2-3 minutes)

If you see a ✅ checkmark, deployment was successful!

5. Check Azure deployment:
   - Go to Azure Portal → Your App Service (`hare-prototype-app`)
   - Click **"Deployment"** → **"Deployment Center"** in the left sidebar
   - You should see the latest deployment with status "Active"

---

## Part 6: Access Your Application

### Step 14: Launch Your App
1. Go to Azure Portal → Your App Service (`hare-prototype-app`)
2. In the **"Overview"** section, you'll see the URL
3. Click the URL or copy it to your browser:
   - URL format: `https://hare-prototype-app.azurewebsites.net`
4. Your Streamlit HARE application should load!

### Step 15: Test the Application
- Navigate through your app's pages
- Test file uploads (if applicable)
- Verify model predictions work correctly
- Check that all CSS/styling loads properly

---

## Part 7: Monitoring and Maintenance

### Monitor Performance
1. App Service → **"Monitoring"** → **"Metrics"** (in left sidebar)
2. View:
   - **CPU Percentage**: Should stay below 80%
   - **Memory Percentage**: Should stay below 80%
   - **HTTP Server Errors**: Should be minimal
   - **HTTP 4xx/5xx**: Watch for errors

### View Application Logs
1. App Service → **"Logs"** (in left sidebar)
2. View real-time logs from your application

### Access Logs
1. App Service → **"App Service logs"** (in left sidebar)
2. Enable:
   - **Application Logging**: Set to **"File System"** (Free tier)
   - **Web Server Logging**: Optional
3. Set **Retention period**: 7 days
4. Click **"Save"**

---

## Troubleshooting

### App Not Starting
**Symptoms**: 502 Bad Gateway error or blank page

**Solution**:
1. Check Azure App Logs for errors
2. Verify all GitHub secrets are correctly set
3. Check if the Dockerfile has syntax errors
4. Ensure `requirements.txt` installs without errors

### Model Loading Issues
**Symptoms**: App times out or crashes

**Solutions**:
1. Add model caching to your app:
```python
@st.cache_resource
def load_model():
    return torch.load('models/hare_stage2_robust.pth', map_location='cpu')

model = load_model()
```

2. Use CPU instead of GPU:
```python
device = torch.device('cpu')
model.to(device)
```

3. Upgrade to B2 tier if B1 has memory issues

### Slow Deployment
**Symptoms**: GitHub Actions workflow takes >10 minutes

**Solution**:
- Reduce Docker image size by removing unnecessary dependencies
- Use `--no-cache-dir` in pip (already included)
- Consider using a smaller base image

### Connection Timeout
**Symptoms**: Can't reach the URL

**Solution**:
1. Wait 5-10 minutes after deployment (app needs to start)
2. Check Azure Portal if app is running (App Service → Overview → Status should be "Running")
3. Restart the app: Click **"Restart"** in the top menu

---

## Cost Breakdown (12 Months with $100 Credits)

| Resource | Est. Cost/Month | Annual Cost |
|----------|-----------------|-------------|
| App Service (B1 Linux) | $10 | $120 |
| Container Registry (Basic) | $5 | $60 |
| Storage | ~$2 | ~$24 |
| **Total** | ~**$17/month** | ~**$204/year** |

**With your $100 credits**: Covers approximately **6 months** of full usage

After 6 months, you'll pay ~$17/month, which is minimal for a deployed web application.

---

## Scaling Up (If Needed)

If you experience performance issues:

### Upgrade App Service Tier
1. App Service → **"Scale up"** (in left sidebar)
2. Select a higher tier:
   - **B2**: $20/month (2GB RAM, 2 CPU cores)
   - **B3**: $40/month (4GB RAM, 4 CPU cores)
3. Click **"Apply"**

The upgrade typically takes 1-2 minutes with no downtime.

---

## Next Steps

1. ✅ Push code to GitHub
2. ✅ Monitor first deployment in GitHub Actions
3. ✅ Test the application at your URL
4. ✅ Set up monitoring alerts (optional)
5. ✅ Share your public URL with stakeholders

---

## Support Resources

- **Azure Documentation**: https://docs.microsoft.com/azure/
- **Streamlit Deployment Guide**: https://docs.streamlit.io/deploy
- **GitHub Actions**: https://docs.github.com/actions
- **Docker Documentation**: https://docs.docker.com/

---

## Quick Reference Commands

```bash
# Check git status
git status

# Add files
git add .

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push origin main

# View git log
git log --oneline
```

---

**Last Updated**: May 1, 2026  
**Application**: HARE Skin Lesion Classifier  
**Deployment Platform**: Microsoft Azure
