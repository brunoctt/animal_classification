from tkinter import filedialog, Tk, Button
import os


class Interface:
    def __init__(self) -> None:
        self.folder_path = ""
        
        self.window = Tk(className="Classificador de Animais")
        self.window.geometry("300x300")
        self.button = Button(text="Escolher pasta", height=50, width=50, command=self.folder_selector)
        self.button.pack(side='top')

    def folder_selector(self):
        self.folder_path = filedialog.askdirectory(initialdir=os.getcwd(), title="Selecione uma pasta com imagens para classificar")
        self.window.destroy()
    
    def open_selector(self):
        self.window.mainloop()
