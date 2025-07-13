from dotenv import load_dotenv
import os

def load_environment():
    load_dotenv()

def get_env(var, default=None):
    return os.getenv(var, default)
