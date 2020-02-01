import monkdata
import dtree
import numpy as np
import random
import matplotlib.pyplot as plt


def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def info_gain(monk, attributes):
    return [dtree.averageGain(monk,attrib) for attrib in attributes]

def oneprune(tree,valset):
    tree_list = [ tr for tr in dtree.allPruned(tree) if dtree.check(tr,valset) > dtree.check(tree,valset)]
    if len(tree_list) == 0:
        return [tree]
    return [tree  for tr in tree_list for tree in oneprune(tr,valset)]

def prune(tree, valset):
    new_tree_list = oneprune(tree,valset)
    #print("HEY",len(new_tree_list))
    return max(new_tree_list, key=lambda x: dtree.check(x,valset))

def evaluate_fraction(data, fraction, monktest):
        #data = monkdata.monk1
    res = [None]*2000
    for i in range(2000):
        monktrain, monkval = partition(data, fraction)
        t=dtree.buildTree(monktrain, monkdata.attributes)
        res[i]=1-dtree.check(prune(t, monkval),monktest)
    return res

def main():
    # Assignement 1
    print("Assignement 1")
    monks = [monkdata.monk1, monkdata.monk2, monkdata.monk3]
    monk_tests = [monkdata.monk1test, monkdata.monk2test, monkdata.monk3test]
    entropies = [dtree.entropy(monk) for monk in monks]
    print("*** Monk1 entropy: ",entropies[0])
    print("*** Monk2 entropy: ",entropies[1])
    print("*** Monk3 entropy: ",entropies[2])


    # Assignement 3
    print(" ")
    print("Assignement 3")
    attributes = monkdata.attributes
    info_gain1 = info_gain(monks[0], attributes) 
    info_gain2 = info_gain(monks[1], attributes) 
    info_gain3 = info_gain(monks[2], attributes)  
    print("*** Monk1 information gain for attribute:",['%.5f'%x for x in info_gain1])
    print("*** Monk2 information gain for attribute:",['%.5f'%x for x in info_gain2])
    print("*** Monk3 information gain for attribute:",['%.5f'%x for x in info_gain3])


   # Assignement 5
    print("")
    print("Assignement 5")
    print("*** Attribute:",np.argmax(info_gain1)+1,"maximizes info gain for MONK1 dataset")
    print("*** Attribute:",np.argmax(info_gain2)+1,"maximizes info gain for MONK2 dataset")
    print("*** Attribute:",np.argmax(info_gain3)+1,"maximizes info gain for MONK3 dataset")
    print("***")
    max0 = np.argmax(info_gain1) # attribute of first split
    attributes_left = [attrib for attrib in attributes if attrib != attributes[max0] ]
    print("*** 1) Attributes the next nodes should be tested on: ",attributes_left)

    # Attributes to split on in second step
    splits = [np.argmax(info_gain(dtree.select(monks[0],attributes[max0],value) ,attributes))+1 for value in attributes[max0].values]
    print("*** 2) Second split is on the attriburtes: ", splits)
    
    # Decision after second split
    subsets = [dtree.select(monks[0],attributes[max0],split) for split in splits]
    print("*** 3) Assignement after second split: ",[dtree.mostCommon(subset) for subset in subsets])
    print("***")

    print("*** Train and test set errors")
    t1=dtree.buildTree(monkdata.monk1, monkdata.attributes)
    print("*** Monk1:", "Etrain=",1-dtree.check(t1, monkdata.monk1), " Etest=",1-dtree.check(t1, monkdata.monk1test))
    t2=dtree.buildTree(monkdata.monk2, monkdata.attributes)
    print("*** Monk2:", "Etrain=",1-dtree.check(t2, monkdata.monk2), " Etest=",1-dtree.check(t2, monkdata.monk2test))
    t3=dtree.buildTree(monkdata.monk3, monkdata.attributes)
    print("*** Monk3:", "Etrain=",1-dtree.check(t3, monkdata.monk3), " Etest=",1-dtree.check(t3, monkdata.monk3test))
    
    import drawtree_qt5
    #print(t1) # tree in text form(weird)
    #drawtree_qt5.drawTree(t1) # uncoment to visualize the decision tree


   # Assignement 7
    print("")
    print("Assignement 7")

    # The prunning for the exanple of monk1
    monk1train, monk1val = partition(monkdata.monk1, 0.9)
    t1=dtree.buildTree(monk1train, monkdata.attributes) # tree trained from monk1train
    t11 = prune(t1, monk1val) # prunned tree
    print("*** Monk1:", "Etrain=",1-dtree.check(t1, monk1val), " Etest=",1-dtree.check(t1, monkdata.monk1test))
    print("*** Monk1:", "Etrain=",1-dtree.check(t11, monk1val), " Etest=",1-dtree.check(t11, monkdata.monk1test))


    # Statistic information for different fraction for monk1 and monk3
    fraction = [0.3,0.4,0.5,0.6,0.7,0.8]

    # Evaluation of Monk1
    eval1 = [evaluate_fraction(monkdata.monk1, frac, monkdata.monk1test) for frac in fraction]
    means1 = [np.mean(x) for x in eval1]
    vars1 = [np.var(x) for x in eval1]
 
    plt.figure(1)
    plt.subplot(121)
    plt.plot(fraction,means1,'ro')
    plt.xlabel(r'$\lambda$')
    plt.title("Mean of error for different "+r'$\lambda$s')
    plt.subplot(122)
    plt.plot(fraction,vars1,'ro')
    plt.xlabel(r'$\lambda$')
    plt.title("Variance of error for different "+r'$\lambda$s' )
    plt.suptitle('Monk1')

    # Evaluation of Monk2
    eval3 = [evaluate_fraction(monkdata.monk3, frac, monkdata.monk3test) for frac in fraction]
    means3 = [np.mean(x) for x in eval3]
    vars3 = [np.var(x) for x in eval3]
    
    plt.figure(2)
    plt.subplot(121)
    plt.plot(fraction,means3,'ro')
    plt.xlabel(r'$\lambda$')
    plt.title("Mean of error for different "+r'$\lambda$s')
    plt.subplot(122)
    plt.plot(fraction,vars3,'ro')
    plt.xlabel(r'$\lambda$')
    plt.title("Variance of error for different "+r'$\lambda$s' )
    plt.suptitle('Monk2')
    plt.show()
if __name__ == '__main__':
    main()
