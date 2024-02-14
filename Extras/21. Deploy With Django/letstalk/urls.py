from django.contrib import admin
from django.urls import path
from django.urls import include

urlpatterns = [
    # Rotas para os endpoints da aplicação
    path('admin/', admin.site.urls),
    path('analisechurn/', include('analisechurn.urls')),
]
