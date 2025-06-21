# Creating a Server on Hetzner

## Prerequisites: SSH Key Setup

Before creating a server on Hetzner, you'll need to have an SSH key ready.

**Create your SSH key first:**
- **macOS users**: Follow the [macOS SSH Key Guide](../ssh/macOS-guide.md)
- **Windows users**: Follow the [Windows SSH Key Guide](../ssh/windows-guide.md)

Once you have your SSH key created and copied to your clipboard, you can proceed with creating your Hetzner server below.

---

This guide will walk you through how to set up a server on Hetzner and connect to it via SSH. Instead of duplicating all the steps here, I'm linking directly to Hetzner's official documentation below. This ensures you always get the most up-to-date instructions straight from the source, and you can trust that the information is accurate and maintained by the Hetzner team themselves.

**Official Hetzner Documentation:**

- [Creating a Server](https://docs.hetzner.com/cloud/servers/getting-started/creating-a-server)
- [Connecting to the Server](https://docs.hetzner.com/cloud/servers/getting-started/connecting-to-the-server)
- [Creating a Firewall](https://docs.hetzner.com/cloud/firewalls/getting-started/creating-a-firewall)

Follow these guides in order: first create your server, then use the connection guide to SSH into it. Once you're connected, you'll be ready to deploy your application!

## Firewall Configuration - Best Practices

On your server, you have specific ports which are like doors you can access. Each service running on your server uses a designated port number - think of it as different entrances to a building where each door serves a specific purpose. A firewall acts as a security guard, controlling which doors are open and who can use them.

When you deploy an application, you need to consider which ports should be accessible from the internet and which should remain closed for security. Opening the wrong ports or leaving them unrestricted can expose your server to attacks.

### Understanding the Standard Port Configuration

Looking at this Hetzner firewall setup, here's what each rule does and why it's considered a best practice:

#### Rule 1: SSH Access (Port 22) - Restricted Access
- **Port:** 22 (SSH is always the default port 22 - this is how you connect to your server via terminal)
- **Access:** Restricted to specific IP address only
- **Why:** SSH gives you complete control over your server, so you want to limit access to only trusted locations. By restricting it to your IP address, you prevent random hackers from even attempting to connect to your server. This is one of the most important security measures you can implement.

#### Rule 2: HTTP Web Traffic (Port 80) - Public Access
- **Port:** 80 (standard web traffic - when someone visits http://yoursite.com)
- **Access:** Open to everyone (Any IPv4/IPv6)
- **Why:** This allows anyone on the internet to visit your website or application. Port 80 is the standard for regular web traffic, and most browsers automatically use this port when you don't specify https://. You want this open so users can actually reach your application.

#### Rule 3: HTTPS Web Traffic (Port 443) - Public Access  
- **Port:** 443 (secure web traffic with SSL encryption - when someone visits https://yoursite.com)
- **Access:** Open to everyone (Any IPv4/IPv6)
- **Why:** This handles encrypted web traffic and is essential for any serious application. Modern browsers prefer HTTPS, and many features (like camera access, location services) only work over secure connections. Having both 80 and 443 open ensures your users can access your site regardless of how they type the URL.

### Setting Up Your Own Firewall

#### Step 1: Find Your IP Address
To restrict SSH access to only your location, you need to know your public IP address:

1. Go to [whatismyipaddress.com](https://whatismyipaddress.com)
2. You'll see something like `203.0.113.42` (IPv4) or `2001:db8::1` (IPv6)
3. **Use IPv4** (the shorter number format) - it's more common and simpler to work with

Your IP address is like your internet "return address" - it identifies where your internet connection is coming from. When you restrict SSH to this IP, only connections from your current location will be allowed.

#### Step 2: Configure Your Firewall Rules
When setting up your own firewall, replace the example IP address with your actual IP address. This ensures only you can SSH into your server while still allowing public access to your web application.

## Note About Static IPs

At Datalumina, we use [NordLayer](https://nordlayer.com/) which gives all our developers the same static IP address. This means any team member can access our servers. If you're working solo, use your home/office IP. If it changes frequently, you might want to consider a VPN service with a static IP.