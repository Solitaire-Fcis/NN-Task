from tkinter import *
from matplotlib.pyplot import cla, text
import Model

# Read Data from specific file
path = 'IrisData.txt'
dataset = Model.read_data("./" + path)

master = Tk()
master.geometry("600x250")
master.configure(bg="#EEE")


# Begin Perceptron Model
def Begin_Model_Perceptron(feat1, feat2, class1, class2, l_rate, epochs, bias):
    if (feat1 == "X1" and feat2 == "X2") or (feat1 == "X2" and feat2 == "X1"):
        feat1 = "X1"
        feat2 = "X2"
    elif (feat1 == "X1" and feat2 == "X3") or (feat1 == "X3" and feat2 == "X1"):
        feat1 = "X1"
        feat2 = "X3"
    elif (feat1 == "X1" and feat2 == "X4") or (feat1 == "X4" and feat2 == "X1"):
        feat1 = "X1"
        feat2 = "X4"
    elif (feat1 == "X2" and feat2 == "X3") or (feat1 == "X3" and feat2 == "X2"):
        feat1 = "X2"
        feat2 = "X3"
    elif (feat1 == "X2" and feat2 == "X4") or (feat1 == "X4" and feat2 == "X2"):
        feat1 = "X2"
        feat2 = "X4"
    elif (feat1 == "X3" and feat2 == "X4") or (feat1 == "X4" and feat2 == "X3"):
        feat1 = "X3"
        feat2 = "X4"
    W, bias = Model.Perc_ALG(dataset, feat1, feat2, class1, class2, l_rate, epochs, bias)
    Model.plot_data(dataset, feat1, feat2, class1, class2, W, bias)


# Begin Adaline Model
def Begin_Model_Adaline(feat1, feat2, class1, class2, l_rate, epochs, bias, threshold):
    if (feat1 == "X1" and feat2 == "X2") or (feat1 == "X2" and feat2 == "X1"):
        feat1 = "X1"
        feat2 = "X2"
    elif (feat1 == "X1" and feat2 == "X3") or (feat1 == "X3" and feat2 == "X1"):
        feat1 = "X1"
        feat2 = "X3"
    elif (feat1 == "X1" and feat2 == "X4") or (feat1 == "X4" and feat2 == "X1"):
        feat1 = "X1"
        feat2 = "X4"
    elif (feat1 == "X2" and feat2 == "X3") or (feat1 == "X3" and feat2 == "X2"):
        feat1 = "X2"
        feat2 = "X3"
    elif (feat1 == "X2" and feat2 == "X4") or (feat1 == "X4" and feat2 == "X2"):
        feat1 = "X2"
        feat2 = "X4"
    elif (feat1 == "X3" and feat2 == "X4") or (feat1 == "X4" and feat2 == "X3"):
        feat1 = "X3"
        feat2 = "X4"
    W, bias = Model.adaline_algo(dataset, feat1, feat2, class1, class2, l_rate, epochs, bias, threshold)
    Model.plot_data(dataset, feat1, feat2, class1, class2, W, bias)


# Begin Back_Propagation Model
def Begin_Model_Back_Propagation(options_list_features, options_list_classes, l_rate, epochs, bias,
                                 hidden_layers, neurons, choosenFunction):
    W, bias = Model.backPropagation_algo(dataset, options_list_features, options_list_classes, l_rate, epochs, bias,
                                         hidden_layers, neurons, choosenFunction)
    # Model.plot_data(dataset, feat1, feat2, class1, class2, W, bias)


# Function options menu for BP model
string_function = StringVar(master)
string_function.set("Function")
options_list_function = ["Sigmoid", "TanH"]
options_menu_function = OptionMenu(master, string_function, *options_list_function)
options_menu_function.config(bg="gray")
options_menu_function.place(x=400, y=140)

# Feature options menu for Perceptron and adaline models
string_feat1 = StringVar(master)
string_feat1.set("X1")
string_feat2 = StringVar(master)
string_feat2.set("X2")
options_list_feat = ["X1", "X2", "X3", "X4"]
options_menu1_feat = OptionMenu(master, string_feat1, *options_list_feat)
options_menu2_feat = OptionMenu(master, string_feat2, *options_list_feat)
options_menu1_feat.config(bg="gray")
options_menu2_feat.config(bg="gray")
options_menu1_feat.place(x=10, y=25)
options_menu2_feat.place(x=130, y=25)

# Classes option menu for Perceptron and adaline models
string_class1 = StringVar(master)
string_class1.set("C1")
string_class2 = StringVar(master)
string_class2.set("C2")
options_menu1_feat_label = Label(text="Feature 1")
options_menu2_feat_label = Label(text="Feature 2")
options_menu1_class_label = Label(text="Class 1")
options_menu2_class_label = Label(text="Class 2")
options_menu_Function_label = Label(text="Choose a Function")
options_list_class = ["C1", "C2", "C3"]
options_menu1_class = OptionMenu(master, string_class1, *options_list_class)
options_menu2_class = OptionMenu(master, string_class2, *options_list_class)
options_menu1_class.config(bg="gray")
options_menu2_class.config(bg="gray")
options_menu1_class.place(x=10, y=80)
options_menu2_class.place(x=130, y=80)

# Labels placements for all options menu
options_menu1_feat_label.place(x=10, y=10)
options_menu2_feat_label.place(x=130, y=10)
options_menu1_class_label.place(x=10, y=60)
options_menu2_class_label.place(x=130, y=60)
options_menu_Function_label.place(x=400, y=110)

# Text initializations and their labels
learn_rate_label = Label(text="Learning Rate")
learn_rate = Text(master, height=1, width=15, bg="#BBB")
epochs_nom_label = Label(text="Number of Epochs")
epochs_nom = Text(master, height=1, width=15, bg="#BBB")
threshold_label = Label(text="Threshold")
threshold = Text(master, height=1, width=15, bg="#BBB")
hiddenLayers_label = Label(text="Number of Hidden Layers")
hiddenLayers = Text(master, height=1, width=15, bg="#BBB")
neurons_label = Label(text="Number of Neurons/Hidden Layer")
neurons = Text(master, height=1, width=15, bg="#BBB")

# Text placements and their labels
learn_rate_label.place(x=230, y=10)
epochs_nom_label.place(x=230, y=60)
threshold_label.place(x=230, y=110)
hiddenLayers_label.place(x=400, y=10)
neurons_label.place(x=400, y=60)
learn_rate.place(x=230, y=30)
epochs_nom.place(x=230, y=85)
threshold.place(x=230, y=140)
hiddenLayers.place(x=400, y=30)
neurons.place(x=400, y=85)

# Add bias or not options menu
string_bias = StringVar(master)
string_bias.set("Yes")
add_bias_label = Label(text="Add Bias?")
options_list_bool = ["Yes", "No"]
add_bias_options_menu = OptionMenu(master, string_bias, *options_list_bool)
add_bias_options_menu.config(bg="gray")
add_bias_label.place(x=10, y=120)
add_bias_options_menu.place(x=10, y=140)

# Button commands and initializations

begin_button_perceptron = Button(master, bg="gray", height=2, width=14, text="Begin Perceptron",
                                 command=lambda: Begin_Model_Perceptron(string_feat1.get(), string_feat2.get(),
                                                                        string_class1.get(),
                                                                        string_class2.get(),
                                                                        learn_rate.get("1.0", 'end-1c'),
                                                                        epochs_nom.get("1.0", 'end-1c'),
                                                                        string_bias.get()))
begin_button_Adaline = Button(master, bg="gray", height=2, width=14, text="Begin Adaline",
                              command=lambda: Begin_Model_Adaline(string_feat1.get(), string_feat2.get(),
                                                                  string_class1.get(),
                                                                  string_class2.get(), learn_rate.get("1.0", 'end-1c'),
                                                                  epochs_nom.get("1.0", 'end-1c'), string_bias.get(),
                                                                  threshold.get("1.0", 'end-1c')))
begin_button_BackPropagation = Button(master, bg="gray", height=2, width=18, text="Begin Back Propagation",
                                      command=lambda: Begin_Model_Back_Propagation(options_list_feat,
                                                                                   options_list_class,
                                                                                   learn_rate.get("1.0", 'end-1c'),
                                                                                   epochs_nom.get("1.0", 'end-1c'),
                                                                                   string_bias.get(),
                                                                                   hiddenLayers.get("1.0", "end-1c"),
                                                                                   neurons.get("1.0", "end-1c"),
                                                                                   string_function.get()))
# Buttons placements
begin_button_perceptron.place(x=10, y=200)
begin_button_Adaline.place(x=130, y=200)
begin_button_BackPropagation.place(x=250, y=200)

master.mainloop()
