IN_PREDICTION="../predictionsULMFiT_200_noLM.csv"

IN_GOLD="../data/sentiment2/tgt-test.txt"
# pred-sentiment.txt Accuracy: 0.9025
# pred-sentiment-trans.txt 0.9008333333333334
# ../pred-sentiment2.txt" 0.9462672372800761
# ../pred-sentiment2-gru.txt 0.948644793152639
# ../pred-sentiment2-large.txt" 0.9481692819781264
# pred-sentiment2-gru-small.txt 0.9498335710889206
#pred-sentiment-trans.txt 0.14621968616262482
# pred-sentiment2-gru-smallest.txt 0.9465049928673324

#Accuracy = 0.9954033919797115
#tp = 4175 tn = 8385 fp = 27 fn = 31
#precision = 0.9935744883388863  recall = 0.9926295767950547  f = 0.9932593180015861

#with new data
# ../pred-sentiment2-gru.txt Accuracy: 0.9353304802662863

# Accuracy: 0.8487874465049928

#predictionsULMFiT2.csv Accuracy: 0.9931050879695673
#predictionsULMFiT2-noTL Accuracy: 0.990489776509748

# predictionsULMFiT_200.csv Accuracy: 0.76
# ../predictionsULMFiT_200_noTL.csvAccuracy: 0.71
# "../predictionsULMFiT_200_noLM.csv" 0.73 - ale przy eval musialem podac model wiec moze jednak byl


def accuracy(gold, predictions):
    correct = 0

    if len(predictions)<len(gold):
        gold = gold[:len(predictions)]
    all = len(gold)
    for g, p in zip(gold, predictions):
        if g == p:
            correct += 1
    return (correct / all)

def read_file(path):
    lines =[]
    with open(path) as f:
        lines = f.read().splitlines()
    return lines

if __name__ == "__main__":
    gold = read_file(IN_GOLD)
    preds = read_file(IN_PREDICTION)
    print("Accuracy: "+str(accuracy(gold,preds)))