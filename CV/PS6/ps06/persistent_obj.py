import pickle


def save(obj):
    with open('obj', 'wb') as obj_file:
        pickle.dump(obj, obj_file)


def load():
    with open('obj', 'rb') as obj_file:
        py_obj = pickle.load(obj_file)
        return py_obj
