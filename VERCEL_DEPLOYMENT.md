# Vercel Deployment Instructions

## ✅ Pre-Deployment Verification Complete

- ✅ **Build Test**: Successful (2.93 kB page, no errors)
- ✅ **package.json**: Has `build` and `start` scripts
- ✅ **next.config.js**: Properly configured with NEXT_PUBLIC_API_URL
- ✅ **Environment Variable**: Used correctly (localhost only as fallback)
- ✅ **vercel.json**: Created with root directory configuration

## 🚀 Deployment Steps

### Option 1: Vercel Dashboard (Recommended)

1. **Go to Vercel**: https://vercel.com/new

2. **Import Repository**:
   - Click "Import Git Repository"
   - Select: `Vedmeena21/SHL_Assessment`

3. **Configure Project**:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `.next` (auto-detected)

4. **Environment Variables** (CRITICAL):
   Click "Environment Variables" and add:
   ```
   Key: NEXT_PUBLIC_API_URL
   Value: https://your-backend-api-url.com
   ```
   
   **Temporary Placeholder**: If backend isn't deployed yet, use:
   ```
   Value: https://shl-api-placeholder.com
   ```
   (Update this later when backend is live)

5. **Deploy**:
   - Click "Deploy"
   - Wait 1-2 minutes for build
   - Get your live URL: `https://shl-assessment-xyz.vercel.app`

### Option 2: Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   cd /Users/ved/Documents/SHL_PROJECT
   vercel --prod
   ```

4. **Follow Prompts**:
   - Set up and deploy: `Y`
   - Project name: `shl-assessment` (or your choice)
   - Directory: `./frontend`
   - Framework: `Next.js`

5. **Set Environment Variable**:
   ```bash
   vercel env add NEXT_PUBLIC_API_URL production
   ```
   Enter value: `https://your-backend-api-url.com`

6. **Redeploy** (if needed):
   ```bash
   vercel --prod
   ```

## 🔧 Post-Deployment Checklist

After deployment, test the live URL:

- [ ] Page loads without errors
- [ ] UI renders correctly (gradient header, input box, buttons)
- [ ] Example queries populate when clicked
- [ ] Submit button is visible
- [ ] Developer console shows no errors (F12)
- [ ] API requests are sent to correct URL (check Network tab)

## 📝 For SHL Submission

You will receive a URL like:
```
https://shl-assessment.vercel.app
```

Use this as your **"URL of the webapp that you built"** in the submission form.

## ⚠️ Important Notes

1. **Backend Not Included**: Only frontend is deployed on Vercel. Backend must be deployed separately (e.g., Render, Railway, Heroku).

2. **Environment Variable**: Once backend is deployed, update `NEXT_PUBLIC_API_URL` in Vercel:
   - Go to Project Settings → Environment Variables
   - Edit `NEXT_PUBLIC_API_URL`
   - Add backend URL
   - Redeploy (Deployments → Three dots → Redeploy)

3. **Public Access**: The Vercel URL is publicly accessible. SHL reviewers can access it without authentication.

4. **Auto-Deploy**: Any push to GitHub main branch will trigger automatic redeployment.

## 🎯 Expected Live URL Format

```
https://shl-assessment-[random-id].vercel.app
```

This URL will remain accessible as long as the Vercel project exists.
