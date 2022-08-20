import tkinter
import tkinter.messagebox

top = tkinter.Tk()
top.title("Logo Generator")

def helloCallBack():
    tkinter.messagebox.showinfo("Hello Python", "Hello World")


B = tkinter.Button(top, text="Generate", command=helloCallBack)

B.pack()
top.mainloop()
