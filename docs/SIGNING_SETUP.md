# Plotwise — APK Signing Setup

## Generate Release Keystore (One-Time)

Run this on your machine (keep the .jks file safe — losing it means you can never update the app):

```bash
keytool -genkeypair -v \
  -keystore plotwise-release.jks \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -alias plotwise \
  -storepass YOUR_STORE_PASSWORD \
  -keypass YOUR_KEY_PASSWORD \
  -dname "CN=Limawapang Jamir, OU=Plotwise, O=Plotwise, L=Kohima, ST=Nagaland, C=IN"
```

**On Windows (Anaconda Prompt or PowerShell):**
```
keytool -genkeypair -v -keystore plotwise-release.jks -keyalg RSA -keysize 2048 -validity 10000 -alias plotwise
```
(It will prompt for passwords and name details interactively.)

## Upload to GitHub Secrets

1. Convert keystore to base64:
   ```bash
   # Linux/Mac:
   base64 -i plotwise-release.jks > keystore-base64.txt

   # Windows PowerShell:
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("plotwise-release.jks")) | Out-File keystore-base64.txt
   ```

2. Go to: `https://github.com/Limajmr8/plotwise/settings/secrets/actions`

3. Add these 4 secrets:
   | Secret Name | Value |
   |-------------|-------|
   | `KEYSTORE_BASE64` | Contents of keystore-base64.txt |
   | `KEYSTORE_PASSWORD` | Your store password |
   | `KEY_ALIAS` | `plotwise` |
   | `KEY_PASSWORD` | Your key password |

4. Push to main — the workflow will automatically build a signed APK.

## Verify Signed APK

Download the `plotwise-release-signed` artifact from GitHub Actions and check:

```bash
# Verify signature
apksigner verify --verbose app-release.apk

# Or using jarsigner
jarsigner -verify -verbose app-release.apk
```

## Install on Phone

```bash
adb install app-release.apk
```

Or transfer via USB/WhatsApp/Google Drive and tap to install.
Enable "Install from unknown sources" in phone settings.

## Important Notes

- **NEVER commit the .jks file to git** — it's in .gitignore
- Store a backup of the .jks file somewhere safe (Google Drive, USB)
- If you lose the keystore, you cannot update the app — users must uninstall and reinstall
- The keystore is valid for 10,000 days (~27 years)
