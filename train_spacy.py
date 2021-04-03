import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
from spacy.util import decaying
from thinc.neural.optimizers import Adam
import random
from matplotlib import pyplot as plt

def custom_optimizer(optimizer, learn_rate=0.0001, beta1=0.9, beta2=0.999, eps=1e-8, L2=1e-6, max_grad_norm=1.0):
    """
    Function to customizer spaCy default optimizer
    """
    
    optimizer.learn_rate = learn_rate
    optimizer.beta1 = beta1
    optimizer.beta2 = beta2
    optimizer.eps = eps
    optimizer.L2 = L2
    optimizer.max_grad_norm = max_grad_norm
    
    return optimizer

def train_spacy(data, 
                iterations, 
                learn_rate=0.001, 
                beta1=0.9, 
                beta2=0.999, 
                eps=1e-8, 
                L2=1e-4, 
                max_grad_norm=1.0):
    """Load the model, set up the pipeline and train the entity recognizer."""
    
    # pada dasarnya, steps yang harus dilakukan untuk mentraining model NER di spaCy adalah sbb:
    # 1. instantiate blank model. blank model ini isi tasknya bisa bermacam2. tapi kita hanya butuh task untuk NER
    # 2. karena kita hanya butuh task NER, kita define task (pipe) bernama 'ner'
    # 3. setelah itu, labels yang sudah dipersiapkan di training data diadd ke dalam pipe ner tsb
    # 4. seperti yg disebutkan di poin 1 di atas, task model itu bisa bermacam2. tapi karena kita hanya butuh task NER, makanya task/pipe lain harus kita disable agar tidak ditraining
    # 5. definisikan grad descent optimizer, yaitu menggunakan Adam optimizer. pada method begin_training(), kita bisa mengganti hyperparameters pada model menggunakan arg "component_cfg" yang inputnya berbentuk dict of dicts {'<jenis pipe>': {'<hyperparameter>': value}}. kalau mau tau keys apa saja yg tersedia untuk hyperparameters tsb, setelah model selesai ditraining lalu disave, coba buka folder tempat model disave. lalu di dalam folder tsb, buka folder "ner". di dalam folder "ner", ada file bernama "cfg". file tsb melist semua hyperparameters yg tersedia pada model dan valuesnya.
    # 6. jika kita masih ingin tuning optimizernya, kita bisa menggunakan func custom_optimizer
    # 7. kalau kita mau menggunakan mekanisme dropout, kita bisa menggunakan func "decaying"
    # 8. mulai lakukan iterasi untuk training. iteration = epoch. kita melakukan training sejumlah n epochs. lalu tiap 1 epoch, kita melakukan training sebanyak m minibatches
    # 9. kita menggunakan func minibatch untuk mensplit training data sebanyak x size. func minibatch menerima input arg "size" dalam bentuk iterator, contohnya func compounding yg berfungsi untuk membuat iterator dengan args "start" sebagai nilai awal, "end" sebagai nilai max, dan "compound" sebagai kelipatan untuk dikali dengan nilai pada "Start". baca di sini https://spacy.io/api/top-level#util.minibatch
    # 10. func minibatch akan menghasilkan generator. dari generator tsb, kita melakukan update (forward propagation dan back propagation) pada model menggunakan method update()
    # 11. setelah seluruh minibatches pada 1 epoch/iteration selesai, proses pada epoch berikutnya akan dilakukan. terus seperti itu hingga mencapai epoch terakhir
        
    TRAIN_DATA = data
    nlp = spacy.blank('id')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        
        # additional lines
        optimizer = nlp.begin_training(component_cfg={"ner": {"conv_window": 3}})
        optimizer = custom_optimizer(optimizer, learn_rate=learn_rate)
        # Define decaying dropout
#         dropout = decaying(0.8, 0.2, 1e-6)
        
#         optimizer = nlp.begin_training()
        loss_list = []
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 64.0, 1.001))
            
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
#                     drop=next(dropout),  # dropout - make it harder to memorise data
                    sgd= optimizer,
                    losses=losses)
            
#             for text, annotations in TRAIN_DATA:
#                 nlp.update(
#                     [text],  # batch of texts
#                     [annotations],  # batch of annotations
#                     drop=next(dropout),  # dropout - make it harder to memorise data
#                     sgd=optimizer,  # callable to update weights
#                     losses=losses)

            print(losses)
            loss_list.append(losses)
        
    return nlp, loss_list
