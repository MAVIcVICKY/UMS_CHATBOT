from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .rag_pipeline import generate_answer
from django.shortcuts import render

def home(request):
    return render(request, "chatbot/index.html")


class ChatView(APIView):

    def post(self, request):
        query = request.data.get("query")

        if not query:
            return Response(
                {"error": "Please provide a 'query' field in the request body."},
                status=status.HTTP_400_BAD_REQUEST
            )

        result = generate_answer(query)
        return Response(result)
