import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import pandas as pd


class CSVProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("CSV Data Processor")
        self.master.geometry("1000x700")

        self.canvas = tk.Canvas(self.master)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        self.container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        self.open_button = tk.Button(self.container, text="Open CSV", command=self.open_file)
        self.open_button.grid(row=0, column=0, pady=10, sticky=tk.EW)

        self.task_type = tk.StringVar()
        self.task_type.set("classification")  # Default value

        self.task_frame = tk.Frame(self.container)
        self.task_frame.grid(row=1, column=0, pady=10, sticky=tk.EW)

        self.df = None
        self.data_window = None
        self.modified_df = None
        self.second_window = None
        self.third_window = None
        self.var_types = {}

        self.radio_frame = tk.Frame(self.container)
        self.radio_frame.grid(row=3, column=0, pady=10, sticky=tk.EW)

        self.master.bind("<Configure>", self.on_window_resize)

        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(0, weight=1)

        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(1, weight=1)  # Task frame row
        self.container.grid_rowconfigure(2, weight=1)  # Data window row
        self.container.grid_rowconfigure(3, weight=1)  # Radio frame row

        self.task_type = tk.StringVar(value="classification")

        self.numeric_variables = []
        self.categorical_variables = []
        self.target_variable = None  # Only one target variable is allowed

    def on_window_resize(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.show_data_window()
                self.open_button.destroy()
            except Exception as e:
                self.show_error(f"Error: {str(e)}")

    def update_variable_lists(self):
        # Clear the lists
        self.numeric_variables = []
        self.categorical_variables = []
        self.target_variable = None

        # Update the lists based on user selections
        for col, var_type in self.var_types.items():
            if var_type.get() == "numeric":
                self.numeric_variables.append(col)
            elif var_type.get() == "categorical":
                self.categorical_variables.append(col)
            elif var_type.get() == "target":
                self.target_variable = col

    def show_data_window(self):
        if self.data_window:
            self.data_window.destroy()

        self.data_window = tk.Frame(self.container)
        self.data_window.grid(row=1, column=0, pady=10)

        tree = ttk.Treeview(self.data_window)
        tree["columns"] = list(self.df.columns)

        for column in self.df.columns:
            tree.heading(column, text=column)
            tree.column(column, width=100)

        for index, row in self.df.iterrows():
            tree.insert("", index, values=list(row))

        tree.pack()

        task_label = tk.Label(self.radio_frame, text="Выберите тип входных данных:")
        task_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        header_label = tk.Label(self.radio_frame, text="Variable Name:")
        header_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        header_label_numeric = tk.Label(self.radio_frame, text="Numeric:")
        header_label_numeric.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        header_label_categorical = tk.Label(self.radio_frame, text="Categorical:")
        header_label_categorical.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)

        header_label_target = tk.Label(self.radio_frame, text="Target:")
        header_label_target.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)

        for i, col in enumerate(self.df.columns):
            var_type = tk.StringVar()
            var_type.set("numeric")
            self.var_types[col] = var_type

            label = tk.Label(self.radio_frame, text=col)
            label.grid(row=i + 2, column=0, padx=5, pady=5, sticky=tk.W)

            numeric_radio = tk.Radiobutton(self.radio_frame, text="", variable=var_type, value="numeric")
            numeric_radio.grid(row=i + 2, column=1, padx=5, pady=5, sticky=tk.W)

            categorical_radio = tk.Radiobutton(self.radio_frame, text="", variable=var_type, value="categorical")
            categorical_radio.grid(row=i + 2, column=2, padx=5, pady=5, sticky=tk.W)

            target_radio = tk.Radiobutton(self.radio_frame, text="", variable=var_type, value="target")
            target_radio.grid(row=i + 2, column=3, padx=5, pady=5, sticky=tk.W)


        task_label = tk.Label(self.radio_frame, text="Выберите тип задачи:")
        task_label.grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)

        classification_radio = tk.Radiobutton(self.radio_frame, text="Классификация", variable=self.task_type, value="classification")
        classification_radio.grid(row=1, column=4, padx=5, pady=5, sticky=tk.W)

        regression_radio = tk.Radiobutton(self.radio_frame, text="Регрессия", variable=self.task_type, value="regression")
        regression_radio.grid(row=1, column=5, padx=5, pady=5, sticky=tk.W)

        next_button = tk.Button(self.container, text="Далее", command=self.show_next_window)
        next_button.grid(sticky=tk.EW)

    def show_next_window(self):
        target_count = sum(value.get() == "target" for value in self.var_types.values())

        if target_count == 1:
            self.update_variable_lists()
            self.modify_dataframe()
            self.show_second_window()
        else:
            self.show_error("Выберите одну и только одну целевую переменную.")

    def modify_dataframe(self):
        if len(self.categorical_variables) > 0:
            self.modified_df = pd.get_dummies(self.df, columns=self.categorical_variables)
        else:
            self.modified_df = self.df.copy()

        # Modify the DataFrame if necessary

    def show_second_window(self):
        if self.second_window:
            self.second_window.destroy()

        self.second_window = tk.Toplevel(self.master)  # Create a new top-level window
        self.second_window.title("Modified Data")

        # Display the modified DataFrame in the second window (treeview)
        tree = ttk.Treeview(self.second_window)
        tree["columns"] = list(self.modified_df.columns)

        for column in self.modified_df.columns:
            tree.heading(column, text=column)
            tree.column(column, width=100)

        for index, row in self.modified_df.iterrows():
            tree.insert("", index, values=list(row))

        tree.pack(pady=10)

        # Добавляем кнопку для возврата в первое окно
        back_button = tk.Button(self.second_window, text="Назад", command=self.show_first_window)
        back_button.pack(fill=tk.X, expand=True)  # Растягиваем кнопку по ширине окна

        next_button = tk.Button(self.second_window, text="Далее", command=self.show_third_window)
        next_button.pack(fill=tk.X, expand=True)  # Растягиваем кнопку по ширине окна

        # Добавляем кнопку для сохранения измененного DataFrame
        save_button = tk.Button(self.second_window, text="Сохранить набор данных", command=self.save_modified_df)
        save_button.pack(fill=tk.X, expand=True)  # Растягиваем кнопку по ширине окна

        # Скрываем первое окно
        self.master.withdraw()

    def save_modified_df(self):
        # Ask the user where to save the file
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.modified_df.to_csv(file_path, index=False)

    def show_first_window(self):
        # Destroy the second window
        self.second_window.destroy()

        # Show the first window
        self.master.deiconify()

    def show_third_window(self):
        if self.third_window:
            self.third_window.destroy()

        self.third_window = tk.Toplevel(self.master)
        self.third_window.title("Neural Network Parameters")

        # Создаем entry объекты для ввода параметров нейронной сети
        self.iteration_var = tk.StringVar(value="35")
        iteration_label = tk.Label(self.third_window, text="Количество итераций:")
        iteration_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.iteration_entry = tk.Entry(self.third_window, textvariable=self.iteration_var)
        self.iteration_entry.grid(row=0, column=1, padx=5, pady=5)

        self.population_size_var = tk.StringVar(value="50")
        population_size_label = tk.Label(self.third_window, text="Размер популяции:")
        population_size_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.population_size_entry = tk.Entry(self.third_window, textvariable=self.population_size_var)
        self.population_size_entry.grid(row=1, column=1, padx=5, pady=5)

        self.tournament_size_var = tk.StringVar(value="5")
        tournament_size_label = tk.Label(self.third_window, text="Размер турнира:")
        tournament_size_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.tournament_size_entry = tk.Entry(self.third_window, textvariable=self.tournament_size_var)
        self.tournament_size_entry.grid(row=2, column=1, padx=5, pady=5)

        self.weight_iteration_var = tk.StringVar(value="100")
        weight_iteration_label = tk.Label(self.third_window, text="Количество итераций для настройки весов:")
        weight_iteration_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.weight_iteration_entry = tk.Entry(self.third_window, textvariable=self.weight_iteration_var)
        self.weight_iteration_entry.grid(row=3, column=1, padx=5, pady=5)

        self.weight_population_size_var = tk.StringVar(value="100")
        weight_population_size_label = tk.Label(self.third_window, text="Размер популяции для настройки весов:")
        weight_population_size_label.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.weight_population_size_entry = tk.Entry(self.third_window, textvariable=self.weight_population_size_var)
        self.weight_population_size_entry.grid(row=4, column=1, padx=5, pady=5)

        # Добавляем кнопку "Готово" для завершения работы
        done_button = tk.Button(self.third_window, text="Готово", command=self.finish_processing)
        done_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.second_window.destroy()

    def finish_processing(self):
        # Здесь вы можете получить значения параметров
        iteration_value = self.get_iteration_value()
        population_size_value = self.get_population_size_value()
        tournament_size_value = self.get_tournament_size_value()
        weight_iteration_value = self.get_weight_iteration_value()
        weight_population_size_value = self.get_weight_population_size_value()
        print(iteration_value)

        # Здесь можно выполнить необходимые действия с параметрами
        # Например, передать их в вашу нейронную сеть или сохранить в файл

        # Закрываем третье окно
        self.third_window.destroy()

    def get_iteration_value(self):
        return self.iteration_entry.get()

    def get_population_size_value(self):
        return self.population_size_entry.get()

    def get_tournament_size_value(self):
        return self.tournament_size_entry.get()

    def get_weight_iteration_value(self):
        return self.weight_iteration_entry.get()

    def get_weight_population_size_value(self):
        return self.weight_population_size_entry.get()

    def show_error(self, message):
        messagebox.showerror("Error", message)


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVProcessorApp(root)
    root.mainloop()
