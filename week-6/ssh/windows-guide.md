# Guide for Windows: Creating an SSH Key

## Checking Your SSH Key

Before creating a new key, check if you already have one:

1. Open PowerShell or Command Prompt
2. Run this command to see if you have existing SSH keys:

   ```powershell
   dir ~/.ssh
   ```

or

   ```powershell
   dir C:\Users\YourUsername\.ssh
```

3. Look for files named `id_rsa.pub`, `id_ed25519.pub`, or similar files ending with `.pub` (Ed25519 is newer, more secure, and faster than RSA - use Ed25519 for all new keys.)
4. If these files exist, you already have SSH keys
5. **Best Practice**: Create a new SSH key for every new production environment. We recommend creating a new one for this tutorial as well, even if you already have existing SSH keys.

## Creating New SSH Keys

If you don't have an SSH key or want to create a new one:

1. **Open PowerShell**:
   - Search for "PowerShell" in the Start menu
   - Right-click and select "Run as administrator" (recommended)

2. **Generate a new SSH key**:

   ```powershell
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_genai_accelerator_prod -C "your_email@example.com"
   ```

   - Replace `id_ed25519_genai_accelerator_prod` with your actual file name for reference
   - Replace `your_email@example.com` with your actual email (this just serves as a label)

3. **When prompted for a passphrase**:
   - You can press Enter for no passphrase (easier but less secure)
   - Or enter a passphrase for additional security (you'll need to enter this when using the key)
   - A passphrase encrypts your private key file on your laptop - without it, anyone who steals your laptop can immediately use your SSH keys.

4. **View your public key**:

   ```powershell
   Get-Content ~/.ssh/id_ed25519_genai_accelerator_prod.pub
   ```

   - The output is your public key, which you'll add to servers you want to access

5. **Copy your public key to clipboard**:

   ```powershell
   Get-Content ~/.ssh/id_ed25519_genai_accelerator_prod.pub | clip
   ```

## Removing SSH Keys

**Delete the key files** from your local machine:

```powershell
Remove-Item ~/.ssh/id_ed25519_genai_accelerator_prod
Remove-Item ~/.ssh/id_ed25519_genai_accelerator_prod.pub
```
