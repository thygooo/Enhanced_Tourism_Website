from django.urls import path
from .views import (
    openai_chat,
    text_cnn_predict,
    log_recommendation_click,
    accommodation_booking_notifications,
    submit_usability_feedback,
)

urlpatterns = [
    path("chat/", openai_chat, name="openai_chat"),
    path("chat/recommendation-click/", log_recommendation_click, name="chat_recommendation_click"),
    path("chat/accommodation-booking-notifications/", accommodation_booking_notifications, name="chat_accommodation_booking_notifications"),
    path("chat/usability-feedback/", submit_usability_feedback, name="chat_usability_feedback"),
    path("text-cnn/predict/", text_cnn_predict, name="text_cnn_predict"),
]
