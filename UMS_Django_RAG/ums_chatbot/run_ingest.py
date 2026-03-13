import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ums_chatbot.settings')

import django
django.setup()

from chatbot.ingest import ingest_all
ingest_all()
