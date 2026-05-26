import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo e encoders
model = joblib.load('modelo_obesidade.pkl')
encoders = joblib.load('encoders.pkl')
le_target = joblib.load('target_encoder.pkl')

st.title("Sistema Preditivo de Diagnóstico de Obesidade")
st.write("Insira os dados do paciente para prever o risco de obesidade.")

# Dicionários de mapeamento (Traduzindo da Interface em Português para o Modelo em Inglês)
map_sn = {'Sim': 'yes', 'Não': 'no'}
map_freq = {'Não': 'no', 'Algumas Vezes': 'Sometimes', 'Frequentemente': 'Frequently', 'Sempre': 'Always'}
map_trans = {'Carro': 'Automobile', 'Moto': 'Motorbike', 'Bicicleta': 'Bike', 'Transporte Público': 'Public_Transportation', 'A Pé': 'Walking'}

# Coleta de Inputs da Equipe Médica
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gênero", ['Female', 'Male'])
    age = st.number_input("Idade", min_value=14, max_value=61, value=25)
    height = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.70)
    weight = st.number_input("Peso (kg)", min_value=30, max_value=200, value=70)
    family_history = st.selectbox("Histórico Familiar de Excesso de Peso", ['Sim', 'Não'])
    favc = st.selectbox("Consumo de alimentos altamente calóricos (FAVC)", ['Sim', 'Não'])
    fcvc = st.selectbox("Frequência de consumo de vegetais (FCVC)", [1, 2, 3])
    ncp = st.selectbox("Número de refeições principais (NCP)", [1, 2, 3, 4])

with col2:
    caec = st.selectbox("Consumo de alimentos entre refeições (CAEC)", ['Não', 'Algumas Vezes', 'Frequentemente', 'Sempre'])
    smoke = st.selectbox("Fumante", ['Sim', 'Não'])
    ch2o = st.selectbox("Consumo diário de água (CH2O)", [1, 2, 3])
    scc = st.selectbox("Monitora ingestão calórica (SCC)", ['Sim', 'Não'])
    faf = st.selectbox("Frequência de atividade física (FAF)", [0, 1, 2, 3])
    tue = st.selectbox("Tempo diário usando dispositivos eletrônicos (TUE)", [0, 1, 2])
    calc = st.selectbox("Consumo de álcool (CALC)", ['Não', 'Algumas Vezes', 'Frequentemente', 'Sempre'])
    mtrans = st.selectbox("Meio de transporte (MTRANS)", ['Carro', 'Moto', 'Bicicleta', 'Transporte Público', 'A Pé'])

# Processamento da Predição
if st.button("Realizar Diagnóstico"):
    # Estruturar os dados de entrada aplicando os mapeamentos de volta para o inglês
    input_data = pd.DataFrame({
        'Gender': [gender], 
        'Age': [age], 
        'Height': [height], 
        'Weight': [weight],
        'family_history': [map_sn[family_history]], 
        'FAVC': [map_sn[favc]], 
        'FCVC': [fcvc], 
        'NCP': [ncp],
        'CAEC': [map_freq[caec]], 
        'SMOKE': [map_sn[smoke]], 
        'CH2O': [ch2o], 
        'SCC': [map_sn[scc]],
        'FAF': [faf], 
        'TUE': [tue], 
        'CALC': [map_freq[calc]], 
        'MTRANS': [map_trans[mtrans]]
    })

    # Aplicar as codificações salvas
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

    # Fazer a predição
    predicao = model.predict(input_data)
    diagnostico = le_target.inverse_transform(predicao)[0]
    
    st.success(f"Diagnóstico Preditivo: {diagnostico}")
