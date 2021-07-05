DEBUG = 1  # debug model if DEBUG=1 else mute debug=0

JSON_FILE_NAME = "./resnet50.json"
WEIGHTS_DIR = "./weights/"
INPUT_C = 3
INPUT_H = 256
INPUT_W = 256
ENG_PATH = "resnet50.eng"
ONNXPATH = "resnet50.onnx"
FP16 = True
INT8 = False
CALI_TXT = "cali.txt"
CALI_TABLE = "cali.table"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUTBLOBNAME = "data"
OUTPUTBLOBNAME = "linear_1_1" # TODO: maybe conflict? different with before.
MAXBATCHSIZE = 10
OUTPUTSIZE = 1000