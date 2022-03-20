from tkinter import *
import Model

# Read Data from specific file
path = 'IrisData.txt'
dataset = Model.read_data("./" + path)
print(dataset)

master = Tk()
master.geometry("750x500")


def select_and_plot_data(choice2):
    choice1 = string_class1.get()
    choice2 = string_class2.get()
    Model.plot_data(dataset, choice1, choice2)


string_class1 = StringVar(master)
string_class1.set("X1")
string_class2 = StringVar(master)
string_class2.set("X2")
options_list = ["X1", "X2", "X3", "X4"]
options_menu1 = OptionMenu(master, string_class1, *options_list)
options_menu2 = OptionMenu(master, string_class2, *options_list, command=select_and_plot_data)
options_menu1.place(x=10, y=10)
options_menu2.place(x=130, y=10)

master.mainloop()
