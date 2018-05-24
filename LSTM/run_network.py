#from download_data import download_files
from create_training_sets import create_trainingsets
#from use_lstm_model import use_lstm_model
#from train_lstm_model import train_lstm_model
from preprocess_data import parse_xml
from preprocess_data import preprocess_sentences

if __name__ == "__main__":
    #download_files()
    #parse_xml()
    #preprocess_sentences()
    create_trainingsets()
    #train_lstm_model()
    #use_lstm_model()
