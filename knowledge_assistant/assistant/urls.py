from django.urls import path
from .views import ask_question, upload_file

urlpatterns = [
    path('ask/', ask_question, name='ask_question'),
    path('upload/', upload_file, name='upload_file'),
]
