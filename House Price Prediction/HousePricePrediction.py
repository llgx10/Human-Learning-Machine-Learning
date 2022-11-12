from tkinter import *
from tkinter import ttk
import banglore_home_prices as data
import tkinter.messagebox as Messagebox

def estimate():
    #print(l.get())
    #print(bhk_var.get())
    #print(bath_var.get())
    if(bhk_var.get()+2<bath_var.get()):
        Messagebox.showwarning('Error','Bathrooms cannot be 2 more than BHK')
    elif (not sqft.get().isdigit()):
        Messagebox.showwarning('Error', 'Area value invalid')
    elif(int(sqft.get())<300):
        Messagebox.showwarning('Error', 'Area cannot be less than 300')
    else:
        output=data.predict_price(l.get(),sqft.get(),bhk_var.get(),bath_var.get())
        Messagebox.showinfo('Price(Lakhs)',round(output,2))

root = Tk()
root.geometry('500x500')
style = ttk.Style(root)
root.tk.call('source', 'azure/azure.tcl')
style.theme_use('azure')
style.configure("Accentbutton", foreground='white')
style.configure("Togglebutton", foreground='white')

l1 = Label(root, text="Area (Square Feet)")
l1.place(x=20, y=20)

sqft=StringVar()

area = Entry(root, textvariable=sqft)
area.place(x=200, y=20)

bhk_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bhk_var = IntVar()

l2 = Label(root, text="BHK")
l2.place(x=20, y=80)

bhk = ttk.Combobox(root, textvariable=bhk_var, values=bhk_list)
bhk.place(x=200, y=80)

bathrooms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
bath_var = IntVar()

l3 = Label(root, text="Bathroom")
l3.place(x=20, y=140)

bath = ttk.Combobox(root, textvariable=bath_var, values=bathrooms)
bath.place(x=200, y=140)

locations=list(data.List)
l=StringVar()

l4 = Label(root, text="Location")
l4.place(x=20, y=200)

loc = ttk.Combobox(root, textvariable=l, values=locations)
loc.place(x=200, y=200)

btn = ttk.Button(root, text="Estimate price", command=lambda: estimate(), style="Accentbutton")
btn.place(x=20, y=260)

root.mainloop()
