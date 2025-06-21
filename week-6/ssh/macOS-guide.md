# Guide for macOS: Creating an SSH Key

## Checking for Existing SSH Keys

1. Open Terminal
2. Run this command:

```bash
ls -la ~/.ssh
```

3. Look for files named `id_rsa.pub`, `id_ed25519.pub`, or similar files ending with `.pub` (Ed25519 is newer, more secure, and faster than RSA - use Ed25519 for all new keys.)
4. If these files exist, you already have SSH keys
5. **Best Practice**: Create a new SSH key for every new production environment. We recommend creating a new one for this tutorial as well, even if you already have existing SSH keys.

## Creating New SSH Keys

1. **Open Terminal**:
   - Find Terminal in Applications → Utilities, or use Spotlight (Cmd+Space) and search for "Terminal"

2. **Generate a new SSH key**:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_genai_accelerator_prod -C "your_email@example.com"
```

   - Replace `id_ed25519_genai_accelerator_prod` with your actual file name for reference
   - Replace `your_email@example.com` with your actual email (this just serves as a label)

3. **When prompted for a passphrase**:
   - You can press Enter for no passphrase (easier but less secure)
   - Or enter a passphrase for additional security (you'll need to enter this when using the key)
   - A passphrase encrypts your private key file on your laptop - without it, anyone who steals your laptop can immediately use your SSH keys.

4. **View your public key**:

```bash
cat ~/.ssh/id_ed25519_genai_accelerator_prod.pub
```

   - The output is your public key, which you'll add to servers you want to access

5. **Copy your public key to clipboard**:

```bash
cat ~/.ssh/id_ed25519_genai_accelerator_prod.pub | pbcopy
```

## Removing SSH Keys

**Delete the key files** from your local machine:

```bash
rm ~/.ssh/id_ed25519_genai_accelerator_prod
rm ~/.ssh/id_ed25519_genai_accelerator_prod.pub
```