import streamlit as st
import pickle

st.title("Suç Tipi Tahmini (NLP) :mag:")
st.write("Suçun işlendiği mekanı ve silahı İngilizce yazın. Örnek: STREET HANDGUN, SINGLE FAMILY DWELLING NO WEAPON")
model=pickle.load(open('suc_nlp_model.pkl','rb'))

text=st.text_area("Mekan ve silah açıklaması")
if text:
    tahmin=model.predict([text])[0]
    st.success(f'Tahmin edilen suç türü: {tahmin}')
    olasilik=model.predict_proba([text])[0]
    en_iyi=olasilik.argsort()[::-1][:3]
    st.write('En olası 3 tür:')
    for i in en_iyi:
        st.write(f'{model.classes_[i]}: %{olasilik[i]*100:.1f}')
