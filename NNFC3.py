import os

import customtkinter as ctk


from tkinter import filedialog

from tkinter import ttk


import pandas as pd

from pandastable import Table, TableModel

class InstallWizard:
    def __init__(self, master):
        self.master = master
        self.master.title("Установка программы")
        self.master.geometry("300x200")

        self.page_index = 0
        self.pages = [
            self.create_file_selection_page(),
            # self.create_page("Второе окно", "Далее", self.next_page),
            # self.create_page("Третье окно", "Далее", self.next_page),
            # self.create_page("Четвертое окно", "Завершить", self.finish_installation)
        ]


        self.data = None
        self.show_current_page()

    def create_file_selection_page(self):
        page = ctk.CTkFrame(self.master)
        page.title = "Загрузка данных"

        page.columnconfigure(0, weight=1)

        label = ctk.CTkLabel(page, text="Выберите файл:")
        label.grid(row=0, column=0, sticky="ew")

        file_load_button = ctk.CTkButton(page, text="Загрузить данные", command=self.load_file)
        file_load_button.grid(row=1, column=0, sticky="ew")

        self.data_model = TableModel()
        self.data_table = Table(page, model=self.data_model, showtoolbar=True, showstatusbar=True)
        self.data_table.grid(row=2, column=0, sticky="ew")

        # self.table.show()

        # button = ctk.CTkButton(page, text="Далее", command=self.next_page)
        # button.pack()

        return page
    
    def show_current_page(self):
        current_page = self.pages[self.page_index]
        current_page.grid(row=0, column=0, sticky="ew")

        

    # def create_page(self, title, button_text, button_command):
    #     page = ctk.CTkFrame(self.master)
    #     page.title = title

    #     label = ctk.CTkLabel(page, text=f"Это {title}")
    #     label.pack(pady=10)

    #     button = ctk.CTkButton(page, text=button_text, command=button_command)
    #     button.pack()

    #     return page
    


    # def next_page(self):
    #     if self.page_index < len(self.pages) - 1:
    #         self.pages[self.page_index].pack_forget()
    #         self.page_index += 1
    #         self.show_current_page()

    # def finish_installation(self):
    #     self.master.destroy()

    def load_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            print("Data file:", os.path.basename(file_path))

            self.data_model.df = self.data
            self.data_table.redraw()







    # def populate_table(self, df):
    #     # Очистим предыдущие данные
    #     self.tree.delete(*self.tree.get_children())

    #     columns = df.columns

    #     # Если таблица еще не создана, создаем ее с заголовками столбцов
    #     if not self.tree["columns"]:
    #         self.tree["columns"] = columns
    #         for col_name in columns:
    #             self.tree.heading(col_name, text=col_name)

    #     # Заполняем таблицу данными из DataFrame
    #     for _, row in df.iterrows():
    #         values = tuple(row[col] for col in columns)
    #         self.tree.insert("", "end", values=values)




if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.columnconfigure(0, weight=1)
    wizard = InstallWizard(root)
    root.mainloop()

    