import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import plotly.express as px

from tools import ctc_greedy_decode
from model import CRNN
from data import tfm_val, tokenizer


device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')


st.set_page_config(layout="wide")
st.title("Пет-проект: Распознавание рукописного текста")

def predict(img):
    with torch.no_grad():
        img_reverse = torch.tensor(img, dtype=torch.float).sub_(255).abs_()
        img_x = tfm_val(img_reverse[None, None, :, :]).to(device)
        model = CRNN(tokenizer.vocab_size, 0.2).to(device)
        model.load_state_dict(torch.load('models/epoch(14|20)_cel0.31_cer0.09.pth', map_location=torch.device(device))['model'])
        model.eval()
        pred = model(img_x)
        decode_pred = torch.tensor(ctc_greedy_decode(pred))
        return decode_pred

st.subheader("Рисуй здесь:")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=300,
    width=2000,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img_gray = canvas_result.image_data[:, :, 0]
    pred = predict(img_gray)
    text_res = tokenizer.decode(predict(img_gray))
    st.text(text_res)




# col_values = st.columns(20)

# for i in range(tokenizer.vocab_size):
#     with col_values[i]:
#         st.subheader(f'{i}')
#         if i == values.index(max(values)):
#             st.text('')
#         # values[0] = st.slider('цвет 0', 1, 100, 25)
#         background_color = ('000000' if (values[i] > 45) else 'ffffff')
#         value_color = hex(int(values[i] * 2.55))[2:]
#         value_color = (f'0{value_color}' if len(value_color) == 1 else value_color) * 3
#         st.markdown(f":color[{values[i]:.2f}%]{{background='#{background_color}' foreground='#{value_color}'}}")
