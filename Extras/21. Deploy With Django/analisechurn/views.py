import pickle
import pandas as pd
from django.shortcuts import render
from django.views import View

# Modelo de Machine Learning para prever o Churn
modelo_pipeline = pickle.load(open('./analisechurn/models/modelo_churn.pkl', 'rb'))

# Endpoint inicial
class HomePageView(View):
    # Método GET (passando pela URL)
    def get(self, request):
        return render(request, 'homepage.html')


# Endpoint para prever o Churn
class DadosClienteView(View):
    # Primeiro o GET para chegar no formulário com os dados do cliente
    def get(self, request):
        return render(request, 'form.html')

    # Depois o POST para enviar os dados do cliente e receber a previsão
    def post(self, request):
        tenure = request.POST.get('tenure')
        MonthlyCharges = request.POST.get('MonthlyCharges')
        TotalCharges = request.POST.get('TotalCharges')
        gender = request.POST.get('gender')
        SeniorCitizen = request.POST.get('SeniorCitizen')
        Partner = request.POST.get('Partner')
        Dependents = request.POST.get('Dependents')
        PhoneService = request.POST.get('PhoneService')
        MultipleLines = request.POST.get('MultipleLines')
        InternetService = request.POST.get('InternetService')
        OnlineSecurity = request.POST.get('OnlineSecurity')
        OnlineBackup = request.POST.get('OnlineBackup')
        DeviceProtection = request.POST.get('DeviceProtection')
        TechSupport = request.POST.get('TechSupport')
        StreamingTV = request.POST.get('StreamingTV')
        StreamingMovies = request.POST.get('StreamingMovies')
        Contract = request.POST.get('Contract')
        PaperlessBilling = request.POST.get('PaperlessBilling')
        PaymentMethod = request.POST.get('PaymentMethod')

        d_dict = {'tenure': [tenure], 'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges],
                  'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'Partner': [Partner],
                  'Dependents': [Dependents], 'PhoneService': [PhoneService],
                  'MultipleLines': [MultipleLines], 'InternetService': [InternetService],
                  'OnlineSecurity': [OnlineSecurity], 'OnlineBackup': [OnlineBackup],
                  'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport],
                  'StreamingTV': [StreamingTV], 'StreamingMovies': [StreamingMovies],
                  'Contract': [Contract], 'PaperlessBilling': [PaperlessBilling],
                  'PaymentMethod': [PaymentMethod]}

        df = pd.DataFrame.from_dict(d_dict, orient='columns')

        # Colocando as features em ordem
        # O modelo preditivo foi treinado com as features nessa ordem
        df = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]

        prediction = modelo_pipeline.predict(df)

        # Se prediction == 0, o cliente não vai cancelar
        if prediction == 0:
            mensagem = 'Ufa... esse cliente vai ficar!! Aproveita pra entubar uns serviços novos!'
            imagem = 'img/chefe_feliz.jpg'
        else:
            # Se prediction == 1, o cliente vai cancelar o serviço
            mensagem = 'DANGER!!! VAI VAZAR! TELEMARKETING NELE!!'
            imagem = 'img/chefe_brabo.jpg'

        return render(request, 'result.html', {'tables': [df.to_html(classes='data', header=True, col_space=10)],
                                               'result': mensagem, 'imagem': imagem})
