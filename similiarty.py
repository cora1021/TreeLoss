import json
import gensim.models as gsm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

emoticon_vector = []
e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)
emoticon = ['😂', '😷', '😮', '🙋', '🙄', '😐', '🙃', '😇', '😖', '😥', '😑', '😣', '😩', '😛', '😯', '🙁', '😱', '😕', '🙉', '😴', '😵', '😲', '😫', '😈', '😌', '😤', '🙂', '😎', '😨', '😻', '😒', '😰', '😋', '🙈', '😶', '😓', '🙅', '😼', '😧', '😪', '😟', '😘', '🙊', '😡', '😔', '🙀', '🙇', '😠', '🙍', '😬', '😾', '😳', '🙆', '😗', '😽', '😸', '😚', '🙏', '😞', '🙌', '😏', '😉', '😅', '😜', '😄', '😙', '😝', '😁', '😍', '🙎', '😦', '😊', '😀', '😺', '😢', '😿', '😭', '😆', '😃', '😹']
for emoji in emoticon:
    vector = e2v[emoji]
    emoticon_vector.append(vector)

cosine_dist = cosine_similarity(emoticon_vector)
sim_matrix = torch.from_numpy(cosine_dist)
sim_matrix = (sim_matrix - lower_bound) / (1 - lower_bound)
sim_matrix[sim_matrix < 0.0] = 0.0



