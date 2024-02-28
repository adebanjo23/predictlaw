from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('delete_user/', views.delete_user, name='delete_user'),
    path('resend_otp/', views.resend_otp, name='resend_otp'),
    path('verify_otp/', views.verify_otp, name='verify_otp'),
    path('login/', views.user_login, name='user_login'),
    path('upload/', views.upload_pdf, name='upload_pdf'),
    path('query/', views.query_pdf, name='query_pdf'),
    path('delete/', views.delete_pdf, name='delete_pdf'),
    re_path(r'^media/(?P<path>.*)$', serve, {
            'document_root': settings.MEDIA_ROOT,
        }),
]
