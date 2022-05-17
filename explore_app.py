import streamlit as st
from streamlit import caching
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import random
import pickle
import zipfile

# Read in data
@st.experimental_memo
def read_data(path):
  with zipfile.ZipFile('model_region_2022-05-11-17-38_outputs.zip') as myzip:
    with myzip.open('model_region_2022-05-11-17-38_outputs.pkl') as f:
            proteins = pickle.load(f)
            protein_names = sorted(list(proteins.keys()))
  return proteins, protein_names

  # with open(path, 'rb') as f:
  #   proteins = pickle.load(f)
  #   protein_names = sorted(list(proteins.keys()))
  # return proteins, protein_names


if 'times_randomised' not in st.session_state:
    st.session_state['times_randomised'] = 0

# path = 'G:\\My Drive\\Thesis\\model_region_2022-05-07-22-59_outputs_full.pkl'
path = 'model_region_2022-05-11-17-38_outputs.pkl'
prots, prot_names = read_data(path)


# Basic layout

# Get random protein
@st.experimental_memo
def get_rand_id(change_to_refresh_value):
  valid_ids = [i for i, (k,v) in enumerate(sorted(prots.items())) if v['info']['split'] == 'Validation set' or v['info']['split'] == 'Test set']
  idx = random.choice(valid_ids) #list(range(len(prot_names)))
  return idx

if st.sidebar.button('Random test/validation set protein'):
  st.session_state['times_randomised'] += 1
idx = get_rand_id(st.session_state['times_randomised'])


prot_name = st.sidebar.selectbox('Choose protein', prot_names, index = idx)
idx = prot_names.index(prot_name)


# Shared items
basic = st.sidebar.checkbox('Simplify plot')
show_stats = st.sidebar.checkbox('Show stats in title', True)


def plot_preds(preds, labels, idx = None, basic = False, show_stats = True):
  probs = preds['region_probs']
  decoded = preds['region'].squeeze()
  labels = labels

  fig, ax = plt.subplots()
  ax.plot(labels, label = 'True regions', lw = 2)
  ax.plot(decoded, '--', label = 'Predicted regions', lw = 2)
  if basic:
    ax.plot(probs[0, :, 1], label = 'Region probability')
  else:
    ax.plot(probs[0, :, 1], label = 'Region probability (1)', alpha = 0.75)
    ax.plot(probs[0, :, 0], label = 'Non-region probability (0)', alpha = 0.75)
    emissions = preds['region_lstm_softmax']
    ax.plot(emissions[0, :, 0], label = 'Region LSTM output (1)', alpha = 0.2)
    ax.plot(emissions[0, :, 1], label = 'Non-region LSTM output (0)', alpha = 0.2)

  title = ""
  if show_stats:
    title += f" | F1: {skm.f1_score(y_true = labels, y_pred = decoded, zero_division=1):0.5f}, AUC: {skm.roc_auc_score(y_true = labels, y_score = probs[0, :, 1]) if np.sum(labels) > 0 else -1:0.5f}, AP: {skm.average_precision_score(y_true = labels, y_score = probs[0, :, 1]) if np.sum(labels) > 0 else -1:0.5f}"
  if idx is not None:
    title = f"Entry {idx} " + title
  if title != "":
    ax.set_title(title)
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol = 3)
  st.pyplot(fig)

st.title(f"{prot_name} | {prots[prot_name]['info']['split']}")
plot = plot_preds(preds = prots[prot_name]['output'], labels = prots[prot_name]['input'], idx = prot_name, basic = basic, show_stats = show_stats)
