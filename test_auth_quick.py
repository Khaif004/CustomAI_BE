#!/usr/bin/env python3
"""
Quick authentication test script

Tests all authentication endpoints without waiting for API startup.
Run: python test_auth_quick.py
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
TIMEOUT = 10

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
END = '\033[0m'


def print_header(text: str):
    print(f"\n{BLUE}{'='*60}{END}")
    print(f"{BLUE}{text:^60}{END}")
    print(f"{BLUE}{'='*60}{END}\n")


def print_success(text: str):
    print(f"{GREEN}✓ {text}{END}")


def print_error(text: str):
    print(f"{RED}✗ {text}{END}")


def print_info(text: str):
    print(f"{YELLOW}ℹ {text}{END}")


def test_dev_token():
    """Test getting a development token"""
    print_header("Test 1: Get Development Token")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/auth/dev/token?username=developer",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Dev token generated successfully!")
            print(f"  Access Token: {data['access_token'][:50]}...")
            print(f"  Expires in: {data['expires_in']} seconds")
            return data['access_token'], data['refresh_token']
        else:
            print_error(f"Failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None, None
            
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")
        print_info("Make sure the API is running: python -m uvicorn app.main:app --reload")
        return None, None


def test_login():
    """Test login with credentials"""
    print_header("Test 2: Login with Credentials")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            json={
                "username": "developer",
                "password": "developer123"
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Login successful!")
            print(f"  Access Token: {data['access_token'][:50]}...")
            print(f"  Refresh Token: {data['refresh_token'][:50]}...")
            return data['access_token'], data['refresh_token']
        else:
            print_error(f"Login failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None, None
            
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")
        return None, None


def test_verify_token(token: str):
    """Test verifying a token"""
    print_header("Test 3: Verify Token")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/verify",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Token is valid!")
            print(f"  User ID: {data['user_id']}")
            print(f"  Username: {data['username']}")
            print(f"  Expires at: {data['expires_at']}")
            return True
        else:
            print_error(f"Token verification failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")
        return False


def test_get_current_user(token: str):
    """Test getting current user info"""
    print_header("Test 4: Get Current User Info")
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("User info retrieved!")
            print(f"  User ID: {data['user_id']}")
            print(f"  Username: {data['username']}")
            if data.get('email'):
                print(f"  Email: {data['email']}")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")
        return False


def test_refresh_token(refresh_token: str):
    """Test refreshing a token"""
    print_header("Test 5: Refresh Token")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/refresh",
            json={"refresh_token": refresh_token},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Token refreshed successfully!")
            print(f"  New Access Token: {data['access_token'][:50]}...")
            print(f"  Expires in: {data['expires_in']} seconds")
            return data['access_token']
        else:
            print_error(f"Token refresh failed: {response.status_code}")
            return None
            
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")
        return None


def test_protected_endpoint(token: str):
    """Test calling a protected endpoint"""
    print_header("Test 6: Call Protected Endpoint (Chat)")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat/",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "message": "Hello!",
                "conversation_history": []
            },
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            print_success("Protected endpoint accessible!")
            data = response.json()
            print(f"  Response: {data.get('response', 'N/A')[:100]}...")
        elif response.status_code == 401:
            print_error("Unauthorized - token not accepted")
        else:
            print_error(f"Failed with status {response.status_code}")
            
    except Exception as e:
        print_error(f"Connection failed: {str(e)}")


def run_all_tests():
    """Run all authentication tests"""
    print_header("AUTHENTICATION TEST SUITE")
    print_info(f"Testing API at: {BASE_URL}")
    
    # Test getting token
    access_token, refresh_token = test_dev_token()
    if not access_token:
        print_error("Cannot continue without access token")
        return
    
    # Test verification
    if test_verify_token(access_token):
        
        # Test getting user info
        test_get_current_user(access_token)
        
        # Test refresh
        new_token = test_refresh_token(refresh_token)
        if new_token:
            test_verify_token(new_token)
        
        # Test protected endpoint
        test_protected_endpoint(access_token)
    
    print_header("SUMMARY")
    print_success("Authentication tests completed!")
    print(f"\n{YELLOW}Your access token:{END}")
    print(f"{GREEN}{access_token}{END}\n")
    print(f"{YELLOW}Use this in requests:{END}")
    print(f"curl -H 'Authorization: Bearer {access_token[:30]}...' http://localhost:8000/api/chat/")
    print()


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{END}")
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
