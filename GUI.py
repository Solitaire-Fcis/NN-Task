from tkinter import *
from matplotlib.pyplot import cla, text
import Model

# Read Data from specific file
path = 'IrisData.txt'
dataset = Model.read_data("./" + path)

master = Tk()
master.geometry("750x500")

def Begin_Model(feat1, feat2, class1, class2, l_rate, epochs, bias):
    Model.Perc_ALG(dataset, feat1, feat2, class1, class2, l_rate, epochs, bias)

def select_and_plot_data(feat2, class1, class2):
    feat1 = string_feat1.get()
    feat2 = string_feat2.get()
    class1 = string_class1.get()
    class2 = string_class2.get()
    Model.plot_data(dataset, feat1, feat2, class1, class2)
    Model.Perc_ALG(dataset, feat1, feat2, class1, class2)


string_feat1 = StringVar(master)
string_feat1.set("X1")
string_feat2 = StringVar(master)
string_feat2.set("X2")
options_list_feat = ["X1", "X2", "X3", "X4"]
options_menu1_feat = OptionMenu(master, string_feat1, *options_list_feat)
options_menu2_feat = OptionMenu(master, string_feat2, *options_list_feat)
options_menu1_feat.place(x=10, y=25)
options_menu2_feat.place(x=130, y=25)

string_class1 = StringVar(master)
string_class1.set("C1")
string_class2 = StringVar(master)
string_class2.set("C2")
options_menu1_feat_label = Label(text="Feature 1")
options_menu2_feat_label = Label(text="Feature 2")
options_menu1_class_label = Label(text="Class 1")
options_menu2_class_label = Label(text="Class 2")
options_list_class = ["C1", "C2", "C3"]
options_menu1_class = OptionMenu(master, string_class1, *options_list_class)
options_menu2_class = OptionMenu(master, string_class2, *options_list_class)
options_menu1_class.place(x=10, y=80)
options_menu2_class.place(x=130, y=80)

options_menu1_feat_label.place(x=10,y=10)
options_menu2_feat_label.place(x=130,y=10)
options_menu1_class_label.place(x=10,y=60)
options_menu2_class_label.place(x=130,y=60)

learn_rate_label = Label(text= "Learning Rate")
learn_rate = Text(master, height=1, width=15, bg="white")
epochs_nom_label = Label(text= "Number of Epochs")
epochs_nom = Text(master, height=1, width=15, bg="white")

learn_rate_label.place(x=230, y=10)
epochs_nom_label.place(x=230, y=60)
learn_rate.place(x=230, y=30)
epochs_nom.place(x=230, y=85)

string_bias = StringVar(master)
string_bias.set("Yes")
add_bias_label = Label(text= "Add Bias?")
options_list_bool = ["Yes", "No"]
add_bias_options_menu = OptionMenu(master, string_bias, *options_list_bool)
add_bias_label.place(x=10,y=120)
add_bias_options_menu.place(x=10,y=140)


begin_button = Button(master, height = 2,width = 10, text ="Begin",command = lambda:Begin_Model(string_feat1.get(), string_feat2.get(), string_class1.get(),
string_class2.get(),learn_rate.get("1.0", 'end-1c'),epochs_nom.get("1.0", 'end-1c'),string_bias.get()))
begin_button.place(x=10,y=300)

master.mainloop()
