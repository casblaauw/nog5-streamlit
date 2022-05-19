import streamlit as st
from streamlit import caching
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import random
import pickle
import zipfile
import time
import gdown
import os
start_time = time.perf_counter()

# Define functions ------------------------------------
@st.experimental_memo
def read_data(id):
  output_file = 'outputs.pkl'
  if not os.path.exists(output_file):
    gdown.download(id = id, output = output_file, quiet = False)
  with open('outputs.pkl', 'rb') as f:
    proteins = pickle.load(f)
    protein_names = sorted(list(proteins.keys()))
  return proteins, protein_names

# Randomisation 
@st.experimental_memo
def get_rand_id(change_to_refresh_value):
  valid_ids = [i for i, (k,v) in enumerate(sorted(prots.items())) if v['info']['split'] == 'Validation set' or v['info']['split'] == 'Test set']
  idx = random.choice(valid_ids) #list(range(len(prot_names)))
  return idx

# Plotting and visualisation
# ----- TEMP ----
def get_sites_protein(seq, mask, window_size = 15):
  assert window_size % 2 == 1
  window_wing = window_size//2
  prot_len = len(seq)
  sites = []
  for idx, (aa, pred) in enumerate(zip(seq, mask)): # Figure out cleaner way to loop over labels?
    if pred == 1 and aa in ['S', 'T']:
      start_idx = max(idx-window_wing, 0)
      end_idx = min(idx+window_wing+1, prot_len)
      sites.append(seq[start_idx:end_idx]) # Check indexing behaviour
  return sites
# ----- END TEMP ----

@st.experimental_memo
def base_plot(info, labels, preds, show_stats = True):
  # Initialize
  probs = preds['region_probs']
  decoded = preds['region'].squeeze()
  fig, ax = plt.subplots()

  # Plot basic data
  ax.plot(labels, label = 'True regions', lw = 2)
  ax.plot(decoded, '--', label = 'Predicted regions', lw = 2)
  ax.plot(probs[0, :, 1], label = 'Region probability')

  # Add title/legend
  title = ""
  if show_stats:
    title += f" | F1: {skm.f1_score(y_true = labels, y_pred = decoded, zero_division=1):0.5f}, AUC: {skm.roc_auc_score(y_true = labels, y_score = probs[0, :, 1]) if np.sum(labels) > 0 else -1:0.5f}, AP: {skm.average_precision_score(y_true = labels, y_score = probs[0, :, 1]) if np.sum(labels) > 0 else -1:0.5f}"
  if info['idx'] is not None:
    title = f"Entry {idx}" + title
  if title != "":
    ax.set_title(title)
  return fig, ax

def add_seen(fig, ax, info, labels, preds):
  ax.fill_between(np.arange(len(info['seen'])), 0, 1, where=info['seen'], alpha=0.15, transform=ax.get_xaxis_transform(), label = 'Evidenced regions')
  return fig, ax

def add_emissions(fig, ax, info, labels, preds):
  emissions = preds['region_lstm_softmax']
  ax.plot(emissions[0, :, 0], label = 'Region LSTM output (1)', alpha = 0.2)
  ax.plot(emissions[0, :, 1], label = 'Non-region LSTM output (0)', alpha = 0.2)
  return fig, ax

def add_accessibility(fig, ax, info, labels, preds):
  if info['accessibility']:
    acc = np.array(info['accessibility'])
    acc = acc/np.max(acc)
    ax.plot(acc, label = 'Surface accessibility')
  return fig, ax

def add_disorder(fig, ax, info, labels, preds):
  if info['disorder']:
    dis = np.array(info['disorder'])
    dis = dis/np.max(dis)
    ax.plot(dis, label = 'Disorder')
  return fig, ax

def finish_plot(fig, ax):
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol = 3)
  return fig, ax

# Runtime -----------------------------------------------

# Initialise data and states
if 'times_randomised' not in st.session_state:
    st.session_state['times_randomised'] = 0

id = '12i6RF_8dQo8L1cf-3Q3zBJgfw1GXoiJp'
prots, prot_names = read_data(id)

# Protein selection
if st.sidebar.button('Random test/validation set protein'):
  st.session_state['times_randomised'] += 1
idx = get_rand_id(st.session_state['times_randomised']) # Needs to be between random button and sidebar selectbox

prot_name = st.sidebar.selectbox('Choose protein', prot_names, index = idx)
idx = prot_names.index(prot_name)



# Plot options
disable_acc = prots[prot_name]['info']['accessibility'] is None
disable_dis = prots[prot_name]['info']['disorder'] is None

show_stats = st.sidebar.checkbox('Show stats in title', True)
show_seen = st.sidebar.checkbox('Show seen regions', True)
show_acc = st.sidebar.checkbox('Show surface accessibility', False, disabled = disable_acc)
show_dis = st.sidebar.checkbox('Show disorder', False, disabled = disable_dis)
show_emissions = st.sidebar.checkbox('Show emissions', False)

# Plotting area
st.title(f"{prot_name} | {prots[prot_name]['info']['split']}")

fig, ax = base_plot(info = prots[prot_name]['info'], labels = prots[prot_name]['input'], preds = prots[prot_name]['output'], show_stats = show_stats)
if show_seen:
  fig, ax = add_seen(fig, ax, prots[prot_name]['info'], prots[prot_name]['input'], prots[prot_name]['output'])
if show_acc:
  fig, ax = add_accessibility(fig, ax, prots[prot_name]['info'], prots[prot_name]['input'], prots[prot_name]['output'])
if show_dis:
  fig, ax = add_disorder(fig, ax, prots[prot_name]['info'], prots[prot_name]['input'], prots[prot_name]['output'])
if show_emissions:
  fig, ax = add_emissions(fig, ax, prots[prot_name]['info'], prots[prot_name]['input'], prots[prot_name]['output'])
fig, ax = finish_plot(fig, ax)
st.pyplot(fig)

st.subheader('S/T in predicted regions')
st.code('\n'.join(get_sites_protein(seq = prots[prot_name]['info']['seq'], mask = prots[prot_name]['output']['region'].squeeze().tolist())))
st.subheader('S/T in true regions')
st.code('\n'.join(get_sites_protein(seq = prots[prot_name]['info']['seq'], mask = prots[prot_name]['input'].squeeze().tolist())))
end_time = time.perf_counter()
st.write(f"App took {end_time-start_time} seconds to run")