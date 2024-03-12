from dotenv import load_dotenv
import os

curdir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = f"{curdir}/.env"
load_dotenv(dotenv_path)
