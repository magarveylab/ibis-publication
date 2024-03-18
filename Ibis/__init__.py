import os

from dotenv import load_dotenv

curdir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = f"{curdir}/.env"
load_dotenv(dotenv_path)
