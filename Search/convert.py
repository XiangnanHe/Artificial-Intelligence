import pickle
import cPickle

def convert(file):
    print "Converting: " + file
    with open(file, 'rb') as infile:
        indata = pickle.load(infile)
    with open(file, 'wb') as outfile:
        cPickle.dump(indata, outfile, pickle.HIGHEST_PROTOCOL)

convert("romania_graph.pickle")
convert("atlanta_osm.pickle")