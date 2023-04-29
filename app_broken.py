# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model


def main():

	st.set_page_config(page_title="Health app")
	overview = st.container()
	left, right = st.columns(2)

	st.image("https://chgseniorliving.com/wp-content/uploads/2019/08/Assisted-Living-Improves-Heart-Health.jpg")

	with overview:
		st.title("Health app")

	with left:
		age_slider = st.slider("Wiek", value=1, min_value=11, max_value=77)
		objawy_slider = st.slider("Liczba objaw", min_value=1, max_value=5)
		illness_slider = st.slider("Liczba chorób", min_value=0, max_value=5)
		height_slider = st.slider("Wysokość", min_value=159, max_value=200)
		medicine_slider = st.slider("Liczba leków", min_value=1, max_value=4, step=1)

	data = [[age_slider, objawy_slider, illness_slider, height_slider, medicine_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with right:
		st.subheader("Czy taka osoba jest zdrowa?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()
