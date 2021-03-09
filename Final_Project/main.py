from Final_Project import BickleyJet
from Final_Project import DoubleGyreFlow

if __name__ == '__main__':
    Data_name = r'DoubleGyreFlowCorArr'
    #### ------ ####
    ### Important note - For the first time running the load flag should be false ###
    # It takes around 20min to calculate the raw data, I know there is faster way for implementation BUT I tried without sucess
    #### ------ ####
    DoubleGyreFlow.DoubleGyreFlow(Data_name, load=False)
    print('Finish Double Gyre Flow')

    BickleyJetPath = r'BickleyJetDS\BickleyJet'  # Path to (bickley_x.csv, bickley_y.csv) CSV files
    #### ------ ####
    ### Important note - For the first time running the load flag should be false ###
    #### ------ ####
    BickleyJet.BickleyJet(BickleyJetPath, load=True)
    print('Finish BickleyJet')

    print('Finish to create all the paper results \ figures')



