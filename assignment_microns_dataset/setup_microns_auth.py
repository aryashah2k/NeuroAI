"""
MicRons Authentication Setup Script
Step-by-step guide to authenticate with the MicRons dataset

IMPORTANT: This must be completed before running the main experiment!
"""

import sys
import webbrowser
from caveclient import CAVEclient

def setup_microns_authentication():
    """
    Interactive setup for MicRons authentication
    """
    print("MicRons Authentication Setup")
    print("=" * 40)
    print()
    
    print("STEP 1: Accept Terms of Service")
    print("-" * 35)
    print("You need to accept the MicRons terms of service first.")
    print("This will open in your web browser...")
    print()
    
    # Open terms of service page
    tos_url = "https://global.daf-apis.com/sticky_auth/api/v1/tos/2/accept"
    print(f"Opening: {tos_url}")
    
    try:
        webbrowser.open(tos_url)
        print("✓ Browser opened with terms of service page")
    except Exception as e:
        print(f"✗ Could not open browser automatically: {e}")
        print(f"Please manually visit: {tos_url}")
    
    print()
    input("Press Enter after you have accepted the terms of service in your browser...")
    print()
    
    print("STEP 2: Generate Authentication Token")
    print("-" * 38)
    print("Now we'll generate your authentication token...")
    print()
    
    try:
        # Initialize global client
        print("Initializing global CAVEclient...")
        client = CAVEclient(server_address="https://global.daf-apis.com")
        
        print("Generating new authentication token...")
        print("This will open a browser window for authentication.")
        print("Please follow the instructions in the browser.")
        print()
        
        # Generate new token
        token = client.auth.setup_token(make_new=True)
        
        if token:
            print("✓ Token generated successfully!")
            print(f"Token: {token[:20]}...")  # Show only first 20 chars for security
            
            # Save the token
            print("Saving token...")
            client.auth.save_token(token=token)
            print("✓ Token saved successfully!")
            
        else:
            print("✗ Failed to generate token")
            return False
            
    except Exception as e:
        print(f"✗ Error during token generation: {e}")
        print()
        print("MANUAL SETUP INSTRUCTIONS:")
        print("1. Run the following in Python:")
        print("   from caveclient import CAVEclient")
        print("   client = CAVEclient(server_address='https://global.daf-apis.com')")
        print("   token = client.auth.setup_token(make_new=True)")
        print("   client.auth.save_token(token=token)")
        print()
        return False
    
    print()
    print("STEP 3: Test Authentication")
    print("-" * 28)
    print("Testing connection to MicRons dataset...")
    
    try:
        # Test connection to minnie65_public
        test_client = CAVEclient('minnie65_public')
        info = test_client.info.get_datastack_info()
        
        print("✓ Authentication successful!")
        print(f"✓ Connected to: {info['datastack_name']}")
        print(f"✓ Version: {test_client.materialize.version}")
        print()
        print("🎉 MicRons authentication setup complete!")
        print("You can now run the main experiment:")
        print("python main.py --neurons 2 --learning-rules hebbian stdp --epochs 10")
        
        return True
        
    except Exception as e:
        print(f"✗ Authentication test failed: {e}")
        print()
        print("Please try the manual setup instructions above.")
        return False

def check_existing_authentication():
    """
    Check if authentication is already set up
    """
    try:
        client = CAVEclient('minnie65_public')
        info = client.info.get_datastack_info()
        print("✓ MicRons authentication already configured!")
        print(f"✓ Connected to: {info['datastack_name']}")
        return True
    except:
        return False

def main():
    """
    Main authentication setup function
    """
    print("Checking existing authentication...")
    
    if check_existing_authentication():
        print("Authentication is already working. You can run the experiment!")
        return
    
    print("No valid authentication found. Starting setup process...")
    print()
    
    success = setup_microns_authentication()
    
    if success:
        print("\n" + "="*50)
        print("SUCCESS: MicRons authentication is now configured!")
        print("You can run your experiment with:")
        print("python main.py")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("SETUP INCOMPLETE: Please follow manual instructions")
        print("or visit: https://caveconnectome.github.io/CAVEclient/tutorials/authentication/")
        print("="*50)

if __name__ == "__main__":
    main()
