import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Configuração da página para ficar mais larga, ideal para dashboards
st.set_page_config(layout="wide", page_title="Sistema de Saúde - Obesidade")

# --- CARREGAMENTO DE DADOS E MODELOS ---
# O cache evita que o Streamlit leia o CSV toda vez que você clicar em um botão
@st.cache_data
def carregar_dados_dashboard():
    df = pd.read_csv("Obesity.csv")
    # Arredondamento obrigatório conforme dicionário
    cols_to_round = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in cols_to_round:
        df[col] = df[col].round().astype(int)
    return df

df_analise = carregar_dados_dashboard()

# Carregamento do modelo de Machine Learning
model = joblib.load('modelo_obesidade.pkl')
encoders = joblib.load('encoders.pkl')
le_target = joblib.load('target_encoder.pkl')

# --- ESTRUTURA DA PÁGINA ---
st.title("Tech Challenge: Diagnóstico e Análise de Obesidade")
st.write("Navegue pelas abas abaixo para acessar o sistema de predição ou a visão analítica.")

# Criação das abas
tab1, tab2 = st.tabs(["🩺 Sistema Preditivo", "📊 Painel Analítico"])

# ==========================================
# ABA 1: SISTEMA PREDITIVO
# ==========================================
with tab1:
    st.header("Diagnóstico de Pacientes")
    
    # Dicionários de mapeamento (Português -> Inglês para o modelo)
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
    if st.button("Realizar Diagnóstico", type="primary"):
        input_data = pd.DataFrame({
            'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
            'family_history': [map_sn[family_history]], 'FAVC': [map_sn[favc]], 'FCVC': [fcvc], 'NCP': [ncp],
            'CAEC': [map_freq[caec]], 'SMOKE': [map_sn[smoke]], 'CH2O': [ch2o], 'SCC': [map_sn[scc]],
            'FAF': [faf], 'TUE': [tue], 'CALC': [map_freq[calc]], 'MTRANS': [map_trans[mtrans]]
        })

        for col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

        predicao = model.predict(input_data)
        diagnostico = le_target.inverse_transform(predicao)[0]
        
        st.success(f"Diagnóstico Preditivo: {diagnostico}")

# ==========================================
# ABA 2: PAINEL ANALÍTICO
# ==========================================
with tab2:
    st.header("Visão Analítica de Negócios")
    
    ordem_obesidade = [
        'Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
        'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'
    ]

    # Dividindo a tela em duas colunas para os gráficos ficarem lado a lado
    graf_col1, graf_col2 = st.columns(2)

    with graf_col1:
        # Gráfico 1: Distribuição
        fig1 = px.histogram(
            df_analise, x="Obesity_level", color="Obesity_level",
            title="Distribuição de Pacientes por Nível de Obesidade",
            category_orders={"Obesity_level": ordem_obesidade},
            text_auto=True
        )
        fig1.update_layout(xaxis_title="Nível de Obesidade", yaxis_title="Quantidade", showlegend=False)
        st.plotly_chart(fig1, use_container_width=True) # Renderiza no Streamlit

    with graf_col2:
        # Gráfico 2: Histórico Familiar
        fig2 = px.histogram(
            df_analise, x="Obesity_level", color="family_history", barmode="group",
            title="Impacto do Histórico Familiar",
            category_orders={"Obesity_level": ordem_obesidade},
            text_auto=True, color_discrete_map={"yes": "#EF553B", "no": "#636EFA"}
        )
        fig2.update_layout(xaxis_title="Nível de Obesidade", yaxis_title="Quantidade", legend_title="Histórico Familiar")
        st.plotly_chart(fig2, use_container_width=True)

    # Gráfico 3: Atividade Física (Ocupando a largura total embaixo)
    fig3 = px.box(
        df_analise, x="Obesity_level", y="FAF", color="Obesity_level",
        title="Frequência Semanal de Atividade Física por Nível de Obesidade",
        category_orders={"Obesity_level": ordem_obesidade}
    )
    fig3.update_layout(xaxis_title="Nível de Obesidade", yaxis_title="Frequência Semanal (0 a 3)", showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
