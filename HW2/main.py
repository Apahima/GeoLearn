import matplotlib.pyplot as plt
from HW2 import Q3, Q5

if __name__ == '__main__':
    # --- Question 3 --- ##
    Q3.SectionOne()
    Q3.SectionTwo()
    Q3.SectionThree()

    # --- Question 5 --- ##
    Q5.ISOMAPEmbbeding(nei=[20,50,500], DataSet={'Turos'})
    Q5.LLEEmbedding(nei=[20,50,500], DataSet={'Turos'})
    Q5.ISOMAPEmbbeding(nei=[5,20,50], DataSet={'Digits'})
    Q5.LLEEmbedding(nei=[5,20,50], DataSet={'Digits'})
    Q5.ClassificationAccuracy(Class=[10],folds = 20, neighbours = 20)
    Q5.SwissRollWithConstrain(nei = [5,25,35])
    # plt.show()
    print('Finish')