# %% [markdown]
# ## doc2vec training excercise

# %% [markdown]
# In this excercise, you will train a Paragraph Vectors / doc2vec model using gensim. You can find information on the gensim doc2vec api here: https://radimrehurek.com/gensim/models/doc2vec.html
# 
# N.B. You should be using Python 3 for this.

# %% [markdown]
# The data folder contains a train and test set with small sets of documents from the "20 newsgroups" dataset.

# %% [markdown]
# What we're going to do is the following:
# * Read a dataset with documents
# * Transform each document into a list of tokens (words)
# * Train a doc2vec model (DM)
# * Train a second model (DBOW)
# * Inspect the outcomes a bit

# %%
import os
from gensim.models import doc2vec
from gensim.utils import simple_preprocess

# %%
# generic settings
HOMEDIR = './'
CORPUS_FILE = os.path.join(HOMEDIR, "data/corpus_train.txt")

# file names for the models we'll be creating
MODEL_FILE_DM = os.path.join(HOMEDIR, "models/doc2vec_DM_v20171229.bin")
MODEL_FILE_DBOW = os.path.join(HOMEDIR, "models/doc2vec_DBOW_v20171229.bin")

# %% [markdown]
# **Read the corpus. Each line is a document / paragraph. Optionally preprocess it first.**

# %%
flg_preprocess = False

if flg_preprocess:
    # quick & simple approach
    docs = doc2vec.TaggedLineDocument(CORPUS_FILE)
else:
    # with pre-processing
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        docs = [simple_preprocess(line, deacc=False, min_len=1) for line in lines]
        docs = [doc2vec.TaggedDocument(doc, tags=[i]) for i, doc in enumerate(docs)]

# %%
# have a look at the data
docs[0]

# %% [markdown]
# ## Training a DM (Distributed Memory) model

# %%
# train DM model
model_dm = doc2vec.Doc2Vec(docs, 
                           vector_size=200, # vector size, should be the same size as pre-trained embedding size when not using dm_concat
                           window=10, # window size for word context, on each side
                           min_count=1, # minimum nr. of occurrences of a word
                           sample=1e-5, # threshold for undersampling high-frequency words
                           workers=4, # for multicore processing
                           hs=0, # if 1, use hierarchical softmax; if 0, use negative sampling
                           dm=1, # if 1 use PV-DM, if 0 use PM-DBOW
                           negative=5, # how many words to use for negative sampling
                           dbow_words=1, # train word vectors
                           dm_concat=1, # concatenate vectors or sum/average them?
                           epochs=100 # nr of epochs to train
                          )

# %%
# save it for later use
model_dm.save(MODEL_FILE_DM)

# %% [markdown]
# ## Training a DBOW (Distributed Bag Of Words) model

# %% [markdown]
# **_Excercise 1: Train a DBOW model_**
# 
# It's very similar to the previous command. What should you change?

# %%
# train DBOW model
# ...enter your code here...
model_dbow = doc2vec.Doc2Vec(docs, 
                            vector_size=200, # vector size, should be the same size as pre-trained embedding size when not using dm_concat
                            window=10, # window size for word context, on each side
                            min_count=1, # minimum nr. of occurrences of a word
                            sample=1e-5, # threshold for undersampling high-frequency words
                            workers=4, # for multicore processing
                            hs=0, # if 1, use hierarchical softmax; if 0, use negative sampling
                            dm=0, # if 1 use PV-DM, if 0 use PM-DBOW
                            negative=5, # how many words to use for negative sampling
                            dbow_words=1, # train word vectors
                            epochs=100 # nr of epochs to train
                            )

# %%
# solution *This is code of an older version. do not run*
model_dbow = doc2vec.Doc2Vec(docs, 
                            size=200, # vector size, should be the same size as pre-trained embedding size when not using dm_concat
                            window=10, # window size for word context, on each side
                            min_count=1, # minimum nr. of occurrences of a word
                            sample=1e-5, # threshold for undersampling high-frequency words
                            workers=4, # for multicore processing
                            hs=0, # if 1, use hierarchical softmax; if 0, use negative sampling
                            dm=0, # if 1 use PV-DM, if 0 use PM-DBOW
                            negative=5, # how many words to use for negative sampling
                            dbow_words=1, # train word vectors
                            iter=100 # nr of epochs to train
                            )

# %% [markdown]
# ========= END OF EXERCISE ============

# %%
# also save this one
model_dbow.save(MODEL_FILE_DBOW)

# %% [markdown]
# ## **Question: Look at the model files that are now created in the models directory. Can you explain why there are 2 files for the DM model, but only 1 for the DBOW model?**

# %%
def show_most_similar(model, docs, ref_doc_id):
    """
    For a given document, display the most similar ones in the corpus
    """
    def print_doc(doc_id):
        doc_txt = ' '.join(docs[doc_id].words)
        print("[Doc {}]: {}".format(doc_id, doc_txt))
        
    print("[Original document]")
    print_doc(ref_doc_id)
    print("\n[Most similar documents]")
    for doc_id, similarity in model.docvecs.most_similar(ref_doc_id, topn=3):
        print("-----------------")
        print("similarity: {}".format(similarity))
        print_doc(doc_id)


# %%
show_most_similar(model_dbow, list(docs), 200)

# %% [markdown]
# ## Prediction phase

# %%
test_data_file = os.path.join(HOMEDIR, "data/corpus_total.txt")

# %%
# read test data: each line into a list of tokens
with open(test_data_file, "r") as f:
    test_docs = [ x.strip().split() for x in f.readlines() ]

# %%
# inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

# %% [markdown]
# Create the embeddings for the test documents. Remember: this is an inference step that actually trains a network.

# %%
# test_docvecs = [model_dm.infer_vector(d, alpha=start_alpha, steps=infer_epoch) for d in test_docs]

infer_epoch = 20  # Number of inference epochs

test_docvecs = [model_dm.infer_vector(d, alpha=start_alpha, epochs=infer_epoch) for d in test_docs]


# %%
# see what one document embedding looks like
test_docvecs[0]

# %% [markdown]
# ### The excercise continues in the next notebook!


