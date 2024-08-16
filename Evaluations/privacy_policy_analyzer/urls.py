from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include

from api.views import login,register,upload_file, check_similarity, check_gdpr_compliance,check_gdpr_compliance_bert

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/upload/', upload_file, name='upload_file'),
    path('api/check_similarity/', check_similarity, name='check_similarity'),
    path('api/check_gdpr_compliance/', check_gdpr_compliance, name='check_gdpr_compliance'),
    path('api/register/', register, name='register'),
    path('api/login/', login, name='login'),
    path('api/check_compliance_bert/', check_gdpr_compliance_bert, name='check_compliance_bert'),
    ]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
