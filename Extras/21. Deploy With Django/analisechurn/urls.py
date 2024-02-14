from django.urls import path
from .views import HomePageView, DadosClienteView

urlpatterns = [
    # Mapa de rotas para os endpoints da aplicação
    # Transformar url em view (negócio)
    path('', HomePageView.as_view(), name='home'),
    path('dados_cliente/', DadosClienteView.as_view(), name='dados_cliente'),
]

