from fastapi import FastAPI
from model_input import AmesInput # Importa a classe do arquivo model_input.py
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="API de Predição de Preços de Imóveis - Ames", version="1.0")

# Carregar o modelo e as colunas esperadas ao iniciar a aplicação
try:
    model = joblib.load("ames_ridge_model.joblib")
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    model = None

try:
    expected_input_cols_for_df_creation = joblib.load("expected_columns.joblib") # Estas são as colunas APÓS feature engineering
    print(f"Colunas esperadas para DataFrame (após eng.): {expected_input_cols_for_df_creation}")
except Exception as e:
    print(f"Erro ao carregar expected_columns.joblib: {e}")
    expected_input_cols_for_df_creation = None


# Lista das 15 features originais que a API espera (de AmesInput)
# Deve corresponder aos campos em AmesInput
original_feature_names = [
    "Gr_Liv_Area", "Garage_Area", "Total_Bsmt_SF", "Year_Built",
    "Year_Remod_Add", "Full_Bath", "Fireplaces", "TotRms_AbvGrd",
    "Lot_Area", "Garage_Cars", "MS_Zoning", "Neighborhood",
    "House_Style", "Exter_Qual", "Kitchen_Qual"
]


@app.on_event("startup")
async def startup_event():
    if model is None:
        print("MODELO NÃO CARREGADO. O ENDPOINT /predict NÃO FUNCIONARÁ CORRETAMENTE.")
    if expected_input_cols_for_df_creation is None:
        print("COLUNAS ESPERADAS NÃO CARREGADAS. O ENDPOINT /predict PODE NÃO FUNCIONAR CORRETAMENTE.")


def perform_feature_engineering(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica a mesma engenharia de features que foi feita no treinamento.
    O input_data aqui é um DataFrame com as 15 features originais.
    Retorna um DataFrame com as features transformadas/criadas que o modelo espera.
    """
    df_engineered = input_data.copy()

    # Renomear colunas com ponto para corresponder ao que o Pydantic/DataFrame pode usar mais facilmente
    # e ao que o modelo foi treinado (se você removeu os pontos no X_full)
    # Se X_full usou colunas com ponto, ajuste aqui para corresponder.
    # Assumindo que no X_full as colunas NÃO tinham ponto:
    df_engineered.rename(columns={
        "Gr_Liv_Area": "Gr_Liv_Area", # Já está ok, mas para exemplo
        "Garage_Area": "Garage_Area",
        "Total_Bsmt_SF": "Total_Bsmt_SF",
        "Year_Built": "Year_Built",
        "Year_Remod_Add": "Year_Remod_Add",
        "Full_Bath": "Full_Bath",
        "Fireplaces": "Fireplaces",
        "TotRms_AbvGrd": "TotRms_AbvGrd",
        "Lot_Area": "Lot_Area",
        "Garage_Cars": "Garage_Cars",
        "MS_Zoning": "MS_Zoning",
        "Neighborhood": "Neighborhood",
        "House_Style": "House_Style",
        "Exter_Qual": "Exter_Qual",
        "Kitchen_Qual": "Kitchen_Qual"
    }, inplace=True)


    # 1. Criar novas features (igual à Célula 7 do notebook)
    #    Assegure-se que as colunas usadas aqui existem em df_engineered após o rename.
    df_engineered['Age'] = df_engineered['Year_Remod_Add'] - df_engineered['Year_Built']
    df_engineered['TotalSF'] = df_engineered['Gr_Liv_Area'] + df_engineered['Total_Bsmt_SF']
    df_engineered['Bath_Rooms'] = df_engineered['Full_Bath'] # Renomeado para Bath_Rooms para evitar conflito
    df_engineered['HasFireplace'] = (df_engineered['Fireplaces'] > 0).astype(int)
    df_engineered['HasGarage'] = (df_engineered['Garage_Area'] > 0).astype(int)
    df_engineered['RecentRemodel'] = (df_engineered['Year_Remod_Add'] > df_engineered['Year_Built']).astype(int)

    # Remover as colunas originais que foram usadas para criar as novas
    # se elas não fazem parte das features finais do modelo
    # Verifique as colunas em 'expected_input_cols_for_df_creation'
    # Por exemplo, se 'Year_Built' e 'Year_Remod_Add' não estão em expected_input_cols_for_df_creation:
    # E se 'Full.Bath', 'Fireplaces', 'Garage.Area' foram substituídas por 'Bath.Rooms', 'HasFireplace', 'HasGarage'
    # Esta etapa é CRUCIAL e deve espelhar EXATAMENTE o X_full com o qual o modelo foi treinado.
    
    # Para garantir, vamos retornar apenas as colunas que o modelo espera, na ordem correta.
    # Se expected_input_cols_for_df_creation foi carregado corretamente:
    if expected_input_cols_for_df_creation:
        # Adicionar colunas faltantes com NaN (serão imputadas pelo pipeline)
        for col in expected_input_cols_for_df_creation:
            if col not in df_engineered.columns:
                df_engineered[col] = np.nan
        return df_engineered[expected_input_cols_for_df_creation]
    else:
        # Fallback se expected_columns não carregou - pode dar erro no predict
        # Remova manualmente as colunas que NÃO estão em X_full do notebook
        # Exemplo (AJUSTE CONFORME SEU X_full):
        cols_to_drop_after_eng = ['Year_Built', 'Year_Remod_Add', 'Full_Bath', 'Fireplaces', 'Garage_Area'] # Ajuste esta lista!
        df_engineered.drop(columns=cols_to_drop_after_eng, inplace=True, errors='ignore')
        return df_engineered


@app.post("/predict/")
async def predict_price(data: AmesInput):
    if model is None:
        return {"error": "Modelo não carregado"}
    if expected_input_cols_for_df_creation is None:
        return {"error": "Configuração de colunas esperadas não carregada"}

    try:
        # 1. Converter o input Pydantic para um dicionário
        input_dict = data.dict()

        # 2. Criar um DataFrame do Pandas com as features originais
        #    Assegure-se que a ordem das colunas aqui não importa tanto,
        #    desde que o perform_feature_engineering e o preprocessor lidem com isso.
        #    Mas é bom manter a consistência.
        input_df_original = pd.DataFrame([input_dict], columns=original_feature_names)


        # 3. Aplicar a mesma engenharia de features do treinamento
        #    A função perform_feature_engineering deve retornar um DataFrame
        #    com as colunas EXATAMENTE como o modelo espera (as colunas de X_full)
        input_df_engineered = perform_feature_engineering(input_df_original)
        
        # Verificar se todas as colunas esperadas estão presentes após a engenharia
        missing_model_cols = set(expected_input_cols_for_df_creation) - set(input_df_engineered.columns)
        if missing_model_cols:
            return {"error": f"Colunas faltando após engenharia para o modelo: {missing_model_cols}"}
        
        # Reordenar para garantir a ordem correta, se necessário (o ColumnTransformer deve lidar com isso, mas não custa)
        input_df_for_model = input_df_engineered[expected_input_cols_for_df_creation]

        # 4. Fazer a predição (o pipeline já inclui o pré-processamento)
        log_price_prediction = model.predict(input_df_for_model)

        # 5. Transformar o resultado de volta para a escala original
        predicted_price = 10**(log_price_prediction[0]) # log10

        return {"predicted_sale_price": round(predicted_price, 2)}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def read_root():
    return {"message": "API para predição de preços de imóveis no dataset Ames. Use o endpoint /predict/ para fazer uma predição."}