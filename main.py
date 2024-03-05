import os
from datetime import datetime
import pickle

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.regressors import GeneticProgrammingNeuralNetRegressor
from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SelfCGA
from thefittest.tools.print import print_net

from FL import FuzzyClassifier
from FL import FuzzyRegressor

from threading import Thread
from collections import defaultdict

import matplotlib.pyplot as plt

from warnings import filterwarnings

filterwarnings("ignore")


def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def create_folder(prefix, base_path):
    current_time = datetime.now().strftime("%d_%m_%y_%H_%M_%S")
    folder_name = f"{prefix}_{current_time}"
    folder_path = os.path.join(base_path, folder_name)
    
    # Проверяем, существует ли папка по указанному пути
    if os.path.exists(folder_path):
        # Если папка существует, добавляем числовой суффикс
        suffix = 1
        while os.path.exists(f"{folder_path} ({suffix})"):
            suffix += 1
        folder_path = f"{folder_path} ({suffix})"
    
    # Создаем папку
    os.makedirs(folder_path)
    
    return folder_path

class FFNNApp:
    def __init__(self, master):
        self.master = master
        self.master.title("FFNNApp")
        self.master.geometry("1000x700")

        self.canvas = tk.Canvas(self.master)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        self.container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        self.open_button = tk.Button(self.container, text="Open CSV", command=self.open_file)
        self.open_button.grid(row=0, column=0, pady=10, sticky=tk.EW)

        self.task_type = tk.StringVar(value="classification")

        self.df = None
        self.modified_df = None
        self.var_types = {}

        self.numeric_variables = []
        self.categorical_variables = []
        self.categorical_after_modified = []
        self.target_variable = None

        self.data_window = None
        self.second_window = None
        self.third_window = None
        self.fourth_window = None
        self.fifth_window = None

        self.master.bind("<Configure>", self.on_window_resize)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

    def on_window_resize(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def open_file(self):
        self.data_file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.data_file_path:
            try:
                self.df = pd.read_csv(self.data_file_path)
                self.show_data_window()
                self.open_button.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")

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

        self.radio_frame = tk.Frame(self.container)
        self.radio_frame.grid(row=3, column=0, pady=10, sticky=tk.EW)
        self.create_variable_type_selection()

        next_button = tk.Button(self.container, text="Далее", command=self.show_next_window)
        next_button.grid(row=4, column=0, sticky=tk.EW)

    def create_variable_type_selection(self):
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
        self.categorical_after_modified = [col for col in self.modified_df.columns if col not in self.numeric_variables and col not in self.target_variable]

    def show_second_window(self):
        if not self.second_window:
            self.second_window = tk.Toplevel(self.master)
            self.second_window.title("Modified Data")
            self.create_second_window_content()
        self.master.withdraw()
        self.second_window.deiconify()

    def create_second_window_content(self):
        tree = ttk.Treeview(self.second_window)
        tree["columns"] = list(self.modified_df.columns)
        for column in self.modified_df.columns:
            tree.heading(column, text=column)
            tree.column(column, width=100)
        for index, row in self.modified_df.iterrows():
            tree.insert("", index, values=list(row))
        tree.pack(pady=10)

        back_button = tk.Button(self.second_window, text="Назад", command=self.back_to_first_window)
        back_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        next_button = tk.Button(self.second_window, text="Далее", command=self.show_third_window)
        next_button.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Добавляем кнопку для сохранения измененного DataFrame
        save_button = tk.Button(self.second_window, text="Сохранить набор данных", command=self.save_modified_df)
        save_button.pack(fill=tk.X, expand=True)  # Растягиваем кнопку по ширине окна

    def back_to_first_window(self):
        self.second_window.withdraw()
        self.master.deiconify()

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
        if not self.third_window:
            self.third_window = tk.Toplevel(self.master)
            self.third_window.title("Neural Network Parameters")
            self.create_third_window_content()
        if self.second_window:
            self.second_window.withdraw()  # Скрыть второе окно
        self.third_window.deiconify()  # Показать третье окно

    def create_third_window_content(self):
        # Создаем entry объекты для ввода параметров нейронной сети
        self.iteration_var = tk.StringVar(value="50")
        iteration_label = tk.Label(self.third_window, text="Количество итераций:")
        iteration_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.iteration_entry = tk.Entry(self.third_window, textvariable=self.iteration_var)
        self.iteration_entry.grid(row=0, column=1, padx=5, pady=5)

        self.population_size_var = tk.StringVar(value="25")
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

        # Добавляем кнопку "Назад" для возвращения ко второму окну
        back_button = tk.Button(self.third_window, text="Назад", command=self.back_to_second_window)
        back_button.grid(row=6, column=0, pady=10, sticky=tk.EW)

        # Добавляем кнопку "Далее" для перехода к четвертому окну
        next_button = tk.Button(self.third_window, text="Далее", command=self.show_fourth_window)
        next_button.grid(row=6, column=1, pady=10, sticky=tk.EW)

    def back_to_second_window(self):
        self.third_window.withdraw()  # Скрыть третье окно
        if self.second_window:
            self.second_window.deiconify()  # Показать второе окно

    def show_fourth_window(self):
        if self.fourth_window is None:
            self.fourth_window = tk.Toplevel(self.master)
            self.fourth_window.title("Fuzzy System Settings")
        else:
            # Очистите предыдущее содержимое окна, если оно существует
            for widget in self.fourth_window.winfo_children():
                widget.destroy()

        self.create_fourth_window_content()  # Пересоздайте содержимое окна

        if self.third_window:
            self.third_window.withdraw()  # Скрыть третье окно
        self.fourth_window.deiconify()  # Показать четвертое окно

    def create_fourth_window_content(self):
        # a. Определение количества итераций (по стандарту 100)
        self.fs_iteration_var = tk.StringVar(value="200")
        fs_iteration_label = tk.Label(self.fourth_window, text="Количество итераций:")
        fs_iteration_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.fs_iteration_entry = tk.Entry(self.fourth_window, textvariable=self.fs_iteration_var)
        self.fs_iteration_entry.grid(row=0, column=1, padx=5, pady=5)

        # b. Установка размера популяции (по стандарту 100)
        self.fs_population_size_var = tk.StringVar(value="200")
        fs_population_size_label = tk.Label(self.fourth_window, text="Размер популяции:")
        fs_population_size_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.fs_population_size_entry = tk.Entry(self.fourth_window, textvariable=self.fs_population_size_var)
        self.fs_population_size_entry.grid(row=1, column=1, padx=5, pady=5)

        # c. Определение параметров турнира (по стандарту 5)
        self.fs_tournament_size_var = tk.StringVar(value="5")
        fs_tournament_size_label = tk.Label(self.fourth_window, text="Размер турнира:")
        fs_tournament_size_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.fs_tournament_size_entry = tk.Entry(self.fourth_window, textvariable=self.fs_tournament_size_var)
        self.fs_tournament_size_entry.grid(row=2, column=1, padx=5, pady=5)

        # d. Максимальное количество правил (по стандарту 15)
        self.fs_max_rules_var = tk.StringVar(value="30")
        fs_max_rules_label = tk.Label(self.fourth_window, text="Максимальное количество правил:")
        fs_max_rules_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.fs_max_rules_entry = tk.Entry(self.fourth_window, textvariable=self.fs_max_rules_var)
        self.fs_max_rules_entry.grid(row=2, column=1, padx=5, pady=5)

        # Создаем Text виджет для отображения нечетких переменных
        self.fuzzy_variables_text = tk.Text(self.fourth_window, height=15, width=50)
        self.fuzzy_variables_text.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # Генерируем и отображаем список нечетких переменных
        self.generate_fuzzy_variables_list()

        # Добавляем кнопку "Назад" для возврата к третьему окну
        back_button = tk.Button(self.fourth_window, text="Назад", command=self.back_to_third_window)
        back_button.grid(row=4, column=0, pady=10, sticky=tk.EW)

        # Добавляем кнопку "Далее" для перехода к пятому окну
        next_button = tk.Button(self.fourth_window, text="Далее", command=self.show_fifth_window)
        next_button.grid(row=4, column=1, pady=10, sticky=tk.EW)

    def back_to_third_window(self):
        self.fourth_window.withdraw()  # Скрыть четвертое окно
        if self.third_window:
            self.third_window.deiconify()  # Показать третье окно

    def generate_fuzzy_variables_list(self):
        fuzzy_text = ""
        for variable in self.numeric_variables:
            fuzzy_text += f"{variable}: Низкое, Среднее, Высокое\n"
        for variable in self.categorical_after_modified:
            fuzzy_text += f"{variable}: Да, Нет\n"
        if self.target_variable:
            if self.task_type.get() == "classification":
                unique_values = self.df[self.target_variable].unique()
                unique_values_str = ", ".join([str(value) for value in unique_values])
                fuzzy_text += f"{self.target_variable}: {unique_values_str}\n"
            else:
                fuzzy_text += f"{self.target_variable}: Низкое, Среднее, Высокое\n"
        
        self.fuzzy_variables_text.insert(tk.END, fuzzy_text)

    def show_fifth_window(self):
        if not self.fifth_window:
            self.fifth_window = tk.Toplevel(self.master)
            self.fifth_window.title("Experiment Parameters")
            self.create_fifth_window_content()
        
        # Скрыть четвертое окно, если оно открыто
        if self.fourth_window:
            self.fourth_window.withdraw()

        self.fifth_window.deiconify()  # Показать пятое окно

    def create_fifth_window_content(self):

        test_size_label = tk.Label(self.fifth_window, text="Доля тестовой выборки:")
        test_size_label.pack(pady=(10, 0))
        self.test_size_var = tk.StringVar(value="0.33")
        self.test_size_entry = tk.Entry(self.fifth_window, textvariable=self.test_size_var)
        self.test_size_entry.pack()

        # random_seed_label = tk.Label(self.fifth_window, text="Random Seed:")
        # random_seed_label.pack(pady=(10, 0))
        # self.random_seed_var = tk.StringVar(value="42")
        # self.random_seed_entry = tk.Entry(self.fifth_window, textvariable=self.random_seed_var)
        # self.random_seed_entry.pack()

        num_runs_label = tk.Label(self.fifth_window, text="Количество прогонов:")
        num_runs_label.pack(pady=(10, 0))
        self.num_runs_var = tk.StringVar(value="30")
        self.num_runs_entry = tk.Entry(self.fifth_window, textvariable=self.num_runs_var)
        self.num_runs_entry.pack()

        choose_folder_button = tk.Button(self.fifth_window, text="Выбрать папку сохранения статистики", command=self.choose_save_folder)
        choose_folder_button.pack(pady=(10, 0))

        self.apply_button = tk.Button(self.fifth_window, text="Запуск", command=self.run_processing, state=tk.DISABLED)
        self.apply_button.pack(pady=(10, 20))

        # Добавляем кнопку "Назад" для возврата к третьему окну
        back_button = tk.Button(self.fifth_window, text="Назад", command=self.back_to_fourth_window)
        back_button.pack(pady=10)

    def choose_save_folder(self):
        self.save_folder_path = filedialog.askdirectory()
        self.apply_button.config(state=tk.NORMAL)
        print(f"Выбранная папка для сохранения: {self.save_folder_path}")

    def back_to_fourth_window(self):
        self.fifth_window.withdraw()  # Скрыть четвертое окно
        if self.fourth_window:
            self.fourth_window.deiconify()  # Показать третье окно

    def parse_fuzzy_sets(self):
        text = self.fuzzy_variables_text.get("1.0", "end-1c")
        lines = text.strip().split("\n")
        fuzzy_sets = {}

        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                variable = parts[0].strip()
                sets = parts[1].split(",")
                sets = [s.strip() for s in sets]
                fuzzy_sets[variable] = sets

        return fuzzy_sets

    def run_processing(self):
        # Здесь вы можете получить значения параметров

        data_name = os.path.basename(self.data_file_path)

        self.apply_button.config(state=tk.DISABLED)
        folder_path = create_folder(data_name+"_запуск", self.save_folder_path)
        

        print(folder_path)

        folder_nets_png_path = create_folder("nets_png", folder_path)
        folder_nets_pkl_path = create_folder("nets_pkl", folder_path)
        folder_fuzzy_systems_pkl_path = create_folder("fuzzy_systems_pkl", folder_path)
        folder_fuzzy_systems_base_path = create_folder("fuzzy_systems_base", folder_path)
        folder_fuzzy_systems_Xnn_path = create_folder("fuzzy_systems_Xnn_pkl", folder_path)
        folder_fuzzy_systems_Xnn_base_path = create_folder("fuzzy_systems_Xnn_base", folder_path)

        def run():
            self.nn_iteration_value = int(self.get_iteration_value())
            self.nn_population_size_value = int(self.get_population_size_value())
            self.nn_tournament_size_value = int(self.get_tournament_size_value())
            self.nn_weight_iteration_value = int(self.get_weight_iteration_value())
            self.nn_weight_population_size_value = int(self.get_weight_population_size_value())

            self.fc_iteration_value = int(self.fs_iteration_entry.get())
            self.fc_population_size_value = int(self.fs_population_size_entry.get())
            self.fc_tournament_size_value = int(self.fs_tournament_size_entry.get())
            self.fc_max_rules_value = int(self.fs_max_rules_entry.get())

            test_size_entry = float(self.test_size_entry.get())
            # random_seed_var = float(self.random_seed_var.get())
            num_runs_var = int(self.num_runs_var.get())

            text = self.parse_fuzzy_sets()

            # variables = {key: value for key, value in text.items() if key in self.categorical_after_modified or key in self.numeric_variables}
            variables = {}
            for key in self.modified_df.columns:
                if key not in self.target_variable:
                    variables[key] = text[key]
            print("variables", variables)

            target = {key: value for key, value in text.items() if key in self.target_variable}

            X = self.modified_df.loc[:,self.numeric_variables + self.categorical_after_modified].values
            y = self.modified_df.loc[:,self.target_variable].values

            n_vars = X.shape[1]

            print("Запуск")
            
            stats = defaultdict(list)

            stats_fs = defaultdict(list)

            predict_train_stats = defaultdict(list)
            predict_test_stats = defaultdict(list)

            predict_train_stats = pd.DataFrame()
            predict_test_stats = pd.DataFrame()

            for i in range(num_runs_var):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size_entry)
                
                var_scaler = MinMaxScaler()
                label_encoder = LabelEncoder()
                target_scaler = MinMaxScaler()

                X_train_scaled = var_scaler.fit_transform(X_train)
                X_test_scaled = var_scaler.transform(X_test)

                if self.task_type.get() == "classification":
                    y_train_prep = label_encoder.fit_transform(y_train)
                    y_test_pred = label_encoder.transform(y_test)

                    predict_train_stats[f"y_train_{i}"] = y_train_prep
                    predict_test_stats[f"y_test_{i}"] = y_test_pred

                    nn_model = GeneticProgrammingNeuralNetClassifier(iters=self.nn_iteration_value,
                                                            pop_size=self.nn_population_size_value,
                                                            optimizer=SelfCGP,
                                                            optimizer_args={"tour_size": self.nn_tournament_size_value,
                                                                            # "show_progress_each": 3
                                                                            },
                                                                            weights_optimizer=SelfCGA,
                                                                            weights_optimizer_args={"iters": self.nn_weight_iteration_value,
                                                                            "pop_size": self.nn_weight_population_size_value}, cache=False)
                    nn_model.fit(X_train_scaled, y_train_prep)

                    y_NN_train = nn_model.predict(X_train_scaled)
                    y_NN_test = nn_model.predict(X_test_scaled)

                    f1_train_y_train_y_NN_train = f1_score(y_train_prep, y_NN_train, average="macro")
                    f1_test_y_test_y_NN_test = f1_score(y_test_pred, y_NN_test, average="macro")

                    stats["f1(y_train, y_NN_train)"].append(f1_train_y_train_y_NN_train)
                    stats["f1(y_test, y_NN_test)"].append(f1_test_y_test_y_NN_test)

                    predict_train_stats[f"y_NN_train_{i}"] = y_NN_train
                    predict_test_stats[f"y_NN_test_{i}"] = y_NN_test


                    fs_model = FuzzyClassifier(iters=self.fc_iteration_value,
                                            pop_size=self.fc_population_size_value,
                                            n_features_fuzzy_sets=[len(value) for key, value in variables.items()],
                                            max_rules_in_base=self.fc_max_rules_value)

                    feature_names = list(variables.keys())
                    fs_model.define_sets(X_train_scaled, y_train_prep, set_names=variables, feature_names=feature_names, target_names=list(target.values())[0])

                    fs_model.fit(X_train_scaled, y_train_prep)

                    with open(folder_fuzzy_systems_pkl_path + f"/fs_model_{i}.pkl", "wb") as file:  
                        pickle.dump(fs_model, file)

                    write_to_file(folder_fuzzy_systems_base_path + f"/fs_base_{i}.txt", fs_model.get_text_rules())

                    y_FS_train = fs_model.predict(X_train_scaled)
                    y_FS_test = fs_model.predict(X_test_scaled)

                    f1_train_y_train_y_FS_train = f1_score(y_train_prep, y_FS_train, average="macro")
                    f1_test_y_test_y_FS_test = f1_score(y_test_pred, y_FS_test, average="macro")

                    stats["f1(y_train, y_FS_train)"].append(f1_train_y_train_y_FS_train)
                    stats["f1(y_test, y_FS_test)"].append(f1_test_y_test_y_FS_test)

                    predict_train_stats[f"y_FS_train_{i}"] = y_FS_train
                    predict_test_stats[f"y_FS_test_{i}"] = y_FS_test

                    net = nn_model.get_net()

                    print_net(net)
                    plt.savefig(folder_nets_png_path + f'/net_{i}.png')
                    plt.close()

                    with open(folder_nets_pkl_path + f"/net_{i}.pkl", "wb") as file:  
                        pickle.dump(net, file)
                    
                    mask = [inp for inp in net._inputs if inp < n_vars]
                    mask.sort()

                    X_NN_train = X_train_scaled[:, mask]
                    X_NN_test = X_test_scaled[:, mask]
                    feature_names_NN = [feature_names[mask_i] for feature_name, mask_i in zip(feature_names, mask)]
                    
                    variables_NN = {}


                    for key, value in variables.items():
                        if key in feature_names_NN:
                            variables_NN[key] = value

                    fsnn_model = FuzzyClassifier(iters=self.fc_iteration_value,
                                                 pop_size=self.fc_population_size_value,
                                                 n_features_fuzzy_sets=[len(value) for key, value in variables_NN.items()],
                                                 max_rules_in_base=self.fc_max_rules_value)
                    fsnn_model.define_sets(X_NN_train, y_NN_train, set_names=variables_NN, feature_names=feature_names_NN, target_names=list(target.values())[0])

                    fsnn_model.fit(X_NN_train, y_NN_train)

                    with open(folder_fuzzy_systems_Xnn_path + f"/fsnn_model_{i}.pkl", "wb") as file:  
                        pickle.dump(fsnn_model, file)

                    write_to_file(folder_fuzzy_systems_Xnn_base_path + f"/fsnn_base_{i}.txt", fsnn_model.get_text_rules())

                    y_NN_FS_train = fsnn_model.predict(X_NN_train)
                    y_NN_FSNN_test = fsnn_model.predict(X_NN_test)

                    f1_train_y_NN_train_y_NN_FS_train = f1_score(y_NN_train, y_NN_FS_train, average="macro")
                    f1_test_y_NN_test_y_NN_FSNN_test = f1_score(y_NN_test, y_NN_FSNN_test, average="macro")

                    stats["f1(y_NN_train, y_NN_FS_train)"].append(f1_train_y_NN_train_y_NN_FS_train)
                    stats["f1(y_NN_test, y_NN_FSNN_test)"].append(f1_test_y_NN_test_y_NN_FSNN_test)
                    
                    predict_train_stats[f"y_NN_FS_train{i}"] = y_NN_FS_train
                    predict_test_stats[f"y_NN_FSNN_test{i}"] = y_NN_FSNN_test

                    f1_train_y_train_y_NN_FS_train = f1_score(y_train_prep, y_NN_FS_train, average="macro")
                    f1_test_y_test_y_NN_FSNN_test = f1_score(y_test_pred, y_NN_FSNN_test, average="macro")

                    stats["f1(y_train, y_NN_FS_train)"].append(f1_train_y_train_y_NN_FS_train)
                    stats["f1(y_test, y_NN_FSNN_test)"].append(f1_test_y_test_y_NN_FSNN_test)
                else:
                    y_train_prep = target_scaler.fit_transform(y_train.reshape(-1, 1))[:,0]
                    y_test_pred = target_scaler.transform(y_test.reshape(-1, 1))[:,0]

                    predict_train_stats[f"y_train_{i}"] = y_train_prep
                    predict_test_stats[f"y_test_{i}"] = y_test_pred

                    nn_model = GeneticProgrammingNeuralNetRegressor(iters=self.nn_iteration_value,
                                                            pop_size=self.nn_population_size_value,
                                                            optimizer=SelfCGP,
                                                            optimizer_args={"tour_size": self.nn_tournament_size_value,
                                                                            # "show_progress_each": 1
                                                                            },
                                                                            weights_optimizer=SelfCGA,
                                                                            weights_optimizer_args={"iters": self.nn_weight_iteration_value,
                                                                            "pop_size": self.nn_weight_population_size_value}, cache=False)
                    
                    nn_model.fit(X_train_scaled, y_train_prep)

                    y_NN_train = nn_model.predict(X_train_scaled)
                    y_NN_test = nn_model.predict(X_test_scaled)

                    r2_train_y_train_y_NN_train = r2_score(y_train_prep, y_NN_train)
                    r2_test_y_test_y_NN_test = r2_score(y_test_pred, y_NN_test)

                    stats["r2(y_train, y_NN_train)"].append(r2_train_y_train_y_NN_train)
                    stats["r2(y_test, y_NN_test)"].append(r2_test_y_test_y_NN_test)

                    predict_train_stats[f"y_NN_train_{i}"] = y_NN_train
                    predict_test_stats[f"y_NN_test_{i}"] = y_NN_test


                    feature_names = list(variables.keys())
                    target_names=list(target.values())[0]

                    fs_model = FuzzyRegressor(iters=self.fc_iteration_value,
                                            pop_size=self.fc_population_size_value,
                                            n_features_fuzzy_sets=[len(value) for key, value in variables.items()],
                                            n_target_fuzzy_sets=len(target_names),
                                            max_rules_in_base=self.fc_max_rules_value,
                                            target_grid_volume=100)

                    
                    fs_model.define_sets(X_train_scaled, y_train_prep, set_names=variables, feature_names=feature_names, target_names=target_names)

                    fs_model.fit(X_train_scaled, y_train_prep)

                    with open(folder_fuzzy_systems_pkl_path + f"/fs_model_{i}.pkl", "wb") as file:  
                        pickle.dump(fs_model, file)

                    write_to_file(folder_fuzzy_systems_base_path + f"/fs_base_{i}.txt", fs_model.get_text_rules())

                    y_FS_train = fs_model.predict(X_train_scaled)
                    y_FS_test = fs_model.predict(X_test_scaled)

                    r2_train_y_train_y_FS_train = r2_score(y_train_prep, y_FS_train)
                    r2_test_y_test_y_FS_test = r2_score(y_test_pred, y_FS_test)

                    stats["r2(y_train, y_FS_train)"].append(r2_train_y_train_y_FS_train)
                    stats["r2(y_test, y_FS_test)"].append(r2_test_y_test_y_FS_test)

                    predict_train_stats[f"y_FS_train_{i}"] = y_FS_train
                    predict_test_stats[f"y_FS_test_{i}"] = y_FS_test

                    net = nn_model.get_net()

                    print_net(net)
                    plt.savefig(folder_nets_png_path + f'/net_{i}.png')
                    plt.close()

                    with open(folder_nets_pkl_path + f"/net_{i}.pkl", "wb") as file:  
                        pickle.dump(net, file)

                    mask = [inp for inp in net._inputs if inp < n_vars]
                    mask.sort()

                    X_NN_train = X_train_scaled[:, mask]
                    X_NN_test = X_test_scaled[:, mask]
                    feature_names_NN = [feature_names[mask_i] for feature_name, mask_i in zip(feature_names, mask)]
                    
                    variables_NN = {}

                    for key, value in variables.items():
                        if key in feature_names_NN:
                            variables_NN[key] = value

                    fsnn_model = FuzzyRegressor(iters=self.fc_iteration_value,
                                                pop_size=self.fc_population_size_value,
                                                n_features_fuzzy_sets=[len(value) for key, value in variables_NN.items()],
                                                n_target_fuzzy_sets=len(target_names),
                                                max_rules_in_base=self.fc_max_rules_value)
                    fsnn_model.define_sets(X_NN_train, y_NN_train, set_names=variables_NN, feature_names=feature_names_NN, target_names=list(target.values())[0])

                    fsnn_model.fit(X_NN_train, y_NN_train)

                    with open(folder_fuzzy_systems_Xnn_path + f"/fsnn_model_{i}.pkl", "wb") as file:  
                        pickle.dump(fsnn_model, file)

                    write_to_file(folder_fuzzy_systems_Xnn_base_path + f"/fsnn_base_{i}.txt", fsnn_model.get_text_rules())

                    y_NN_FS_train = fsnn_model.predict(X_NN_train)
                    y_NN_FSNN_test = fsnn_model.predict(X_NN_test)

                    r2_train_y_NN_train_y_NN_FS_train = r2_score(y_NN_train, y_NN_FS_train)
                    r2_test_y_NN_test_y_NN_FSNN_test = r2_score(y_NN_test, y_NN_FSNN_test)

                    stats["r2(y_NN_train, y_NN_FS_train)"].append(r2_train_y_NN_train_y_NN_FS_train)
                    stats["r2(y_NN_test, y_NN_FSNN_test)"].append(r2_test_y_NN_test_y_NN_FSNN_test)
                    
                    predict_train_stats[f"y_NN_FS_train{i}"] = y_NN_FS_train
                    predict_test_stats[f"y_NN_FSNN_test{i}"] = y_NN_FSNN_test

                    r2_train_y_train_y_NN_FS_train = r2_score(y_train_prep, y_NN_FS_train)
                    r2_test_y_test_y_NN_FSNN_test = r2_score(y_test_pred, y_NN_FSNN_test)

                    stats["f1(y_train, y_NN_FS_train)"].append(r2_train_y_train_y_NN_FS_train)
                    stats["f1(y_test, y_NN_FSNN_test)"].append(r2_test_y_test_y_NN_FSNN_test)

                print(f"Прогон {i} закончен")
                        
                stat_df = pd.DataFrame(stats)
                stats_fs[f"fs_base_len"].append(len(fs_model.base))
                stats_fs[f"fsnn_base_len"].append(len(fsnn_model.base))
                stats_fs[f"fs_base_mean_antecedents"].append(sum(fs_model.count_antecedents())/len(fs_model.count_antecedents()))
                stats_fs[f"fsnn_base_mean_antecedents"].append(sum(fsnn_model.count_antecedents())/len(fsnn_model.count_antecedents()))

            stat_df.to_excel(folder_path + f"/f1_r2_stats.xlsx")
            stats_fs_df = pd.DataFrame(stats_fs)
            stats_fs_df.to_excel(folder_path + f"/fs_fsnn_stats.xlsx")
            predict_train_stats.to_excel(folder_path + f"/predict_train_stats.xlsx")
            predict_test_stats.to_excel(folder_path + f"/predict_test_stats.xlsx")

            self.apply_button.config(state=tk.NORMAL)

        algorithm_thread = Thread(target=run)
        algorithm_thread.start()



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
    app = FFNNApp(root)
    root.mainloop()
