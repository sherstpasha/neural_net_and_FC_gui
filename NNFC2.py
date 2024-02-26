import os
import pickle

from collections import defaultdict

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from threading import Thread

from pandastable import Table

import matplotlib.pyplot as plt

from thefittest.classifiers import GeneticProgrammingNeuralNetClassifier
from thefittest.optimizers import SelfCGP
from thefittest.optimizers import SelfCGA
from thefittest.tools.print import print_net


def show_table(data_frame):
    top = tk.Toplevel(root)
    top.title("Data Table")
    table = Table(top, dataframe=data_frame, showtoolbar=True, showstatusbar=True)
    table.show()

def nn_save_all_stats():
    pass


class System:
    def __init__(self, root):
        self.root = root
        self.root.title("Software System for Designing Interpretable Artificial Intelligence Technologies Based on Hybrid Self-Configuring Genetic Algorithms with Neural Networks and Fuzzy Logic")
        self.predict_data_loaded = False
        self.predict_model_loaded = False

        self.setup_ui()

    def train_data_open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.train_data = pd.read_csv(file_path)
            self.X_train = self.train_data.loc[:, self.train_data.columns != "class"].values
            self.labels_train = self.train_data["class"].values

            self.label_encoder_train_data = LabelEncoder()
            self.y_train = self.label_encoder_train_data.fit_transform(self.labels_train)

            self.standart_scaler_train_data = StandardScaler()
            self.X_train_scaled = self.standart_scaler_train_data.fit_transform(self.X_train)

            self.columns = list(self.train_data.columns)

            print("Train data file: " + os.path.basename(file_path))
    
            self.nn_fit_start_button.config(state=tk.NORMAL)
            self.train_data_show_button.config(state=tk.NORMAL)

    def test_data_open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.test_data = pd.read_csv(file_path)
            self.X_test = self.test_data.loc[:, self.test_data.columns != "class"].values
            self.labels_test = self.test_data["class"].values

            print("Test data file: " + os.path.basename(file_path))

            self.predict_data_loaded = True
            self.predict_data_show_button.config(state=tk.NORMAL)

            if self.predict_model_loaded and self.predict_data_loaded:
                self.nn_predict_button.config(state=tk.NORMAL)

    def train_data_show_table(self):
        show_table(self.train_data)

    def predict_data_show_table(self):
        show_table(self.test_data)

    def save_trained_net(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialfile="trained_net.pkl",
        )
        if file_path:
            net_dict = {"net": self.nn_train_model.get_net(),
                        "label_encoder": self.label_encoder_train_data,
                        "data_scaler": self.standart_scaler_train_data}
            with open(file_path, "wb") as file:
                
                pickle.dump(net_dict, file)

            print("Network saved to: " + os.path.basename(file_path))

    def load_trained_net(self):
        self.file_path_trained_net = filedialog.askopenfilename(
            title="Choose model file",
            filetypes=[("Pickle files", "*.pkl")],
            )
        if self.file_path_trained_net:
            with open(self.file_path_trained_net, "rb") as file:
                self.trained_loaded_net_dict = pickle.load(file)
                self.predict_model_loaded = True

            if self.predict_model_loaded and self.predict_data_loaded:
                self.nn_predict_button.config(state=tk.NORMAL)

            print("Trained net file: " + os.path.basename(self.file_path_trained_net))
            
    def plot_net_tree(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            initialfile=f"net_f1_{round(self.f1_train, 4)}.png",
        )
        net = self.nn_train_model.get_net()

        print_net(net)
        plt.savefig(file_path)
        plt.close()

    def save_stat_nn(self):
        
        stats = self.nn_train_model.get_optimizer().get_stats()
        selectiom_proba = defaultdict(list)
        for i in range(self.nn_iters):
            for key, value in stats['s_proba'][i].items():
                selectiom_proba[key].append(value)

        selectiom_proba = defaultdict(list)
        for i in range(self.nn_iters):
            for key, value in stats['s_proba'][i].items():
                selectiom_proba[key].append(value)

        mutation_proba = defaultdict(list)
        for i in range(self.nn_iters):
            for key, value in stats['m_proba'][i].items():
                mutation_proba[key].append(value)
        
        

    def start_nn_fit(self):
        self.nn_iters = int(self.nn_iters_entry.get())
        self.nn_pop_size = int(self.nn_pop_size_entry.get())
        self.nn_tour_size = int(self.nn_tour_size_entry.get())
        self.nn_weights_iters = int(self.nn_weights_iters_entry.get())
        self.nn_weights_pop_size = int(self.nn_weights_pop_size_entry.get())

        self.nn_train_model = GeneticProgrammingNeuralNetClassifier(iters=self.nn_iters,
                                                                    pop_size=self.nn_pop_size,
                                                                    optimizer=SelfCGP,
                                                                    optimizer_args={"tour_size": self.nn_tour_size, "show_progress_each": 1},
                                                                    weights_optimizer=SelfCGA,
                                                                    weights_optimizer_args={"iters": self.nn_weights_iters,
                                                                                            "pop_size": self.nn_weights_pop_size},
                                                                    cache=False)
        def run_algorithm():
            self.nn_fit_start_button.config(state=tk.DISABLED)
            print("Building the network in progress...")
            self.nn_train_model.fit(self.X_train_scaled, self.y_train)
            y_train_predict = self.nn_train_model.predict(self.X_train_scaled)
            label_train_predict = self.label_encoder_train_data.inverse_transform(y_train_predict)

            

            self.f1_train = f1_score(self.labels_train, label_train_predict, average="macro")
            cm = confusion_matrix(self.labels_train, label_train_predict, labels=self.label_encoder_train_data.classes_)
            confusion_df = pd.DataFrame(cm, index=self.label_encoder_train_data.classes_, columns=self.label_encoder_train_data.classes_)

            print(f"Building the network completed. f1: {self.f1_train}")
            print(confusion_df)

            self.nn_fit_start_button.config(state=tk.NORMAL)
            self.nn_save_net_button.config(state=tk.NORMAL)
            self.nn_save_png_button.config(state=tk.NORMAL)
        
        # Run the algorithm in a separate thread
        algorithm_thread = Thread(target=run_algorithm)
        algorithm_thread.start()
        
    def nn_start_predict(self):

        net = self.trained_loaded_net_dict["net"]
        label_encoder = self.trained_loaded_net_dict["label_encoder"]
        data_scaler = self.trained_loaded_net_dict["data_scaler"]

        X_test_scaled = data_scaler.transform(self.X_test)

        X_test_scaled_ones = np.hstack([X_test_scaled, np.ones((X_test_scaled.shape[0], 1))])

        def run_algorithm():
            proba = net.forward(X_test_scaled_ones)[0]
            y_predict = np.argmax(proba, axis=1)

            self.labels_predict_net = label_encoder.inverse_transform(y_predict)

            f1_test = f1_score(self.labels_test, self.labels_predict_net, average="macro")
            print(f"f1 ({os.path.basename(self.file_path_trained_net)}): {f1_test}")

        # Run the algorithm in a separate thread
        algorithm_thread = Thread(target=run_algorithm)
        algorithm_thread.start()

    def setup_train_net_tab(self):
        self.train_net_tab = ttk.Frame(self.nested_notebook_train)
        self.nested_notebook_train.add(self.train_net_tab, text="Neural Network")

        # Create a control frame
        self.train_net_frame = ttk.Frame(self.train_net_tab)
        self.train_net_frame.pack(pady=10, anchor=tk.W)

        # Entry windows with default values
        self.nn_iters_label = tk.Label(self.train_net_frame, text="Number of iterations:", anchor=tk.W)
        self.nn_iters_entry = tk.Entry(self.train_net_frame)
        self.nn_iters_entry.insert(0, "15")  # Default value
        self.nn_iters_label.pack(anchor=tk.W)
        self.nn_iters_entry.pack(anchor=tk.W)

        self.nn_pop_size_label = tk.Label(self.train_net_frame, text="Population size:", anchor=tk.W)
        self.nn_pop_size_entry = tk.Entry(self.train_net_frame)
        self.nn_pop_size_entry.insert(0, "35")  # Default value
        self.nn_pop_size_label.pack(anchor=tk.W)
        self.nn_pop_size_entry.pack(anchor=tk.W)

        self.nn_tour_size_label = tk.Label(self.train_net_frame, text="Tournament size:", anchor=tk.W)
        self.nn_tour_size_entry = tk.Entry(self.train_net_frame)
        self.nn_tour_size_entry.insert(0, "5")  # Default value
        self.nn_tour_size_label.pack(anchor=tk.W)
        self.nn_tour_size_entry.pack(anchor=tk.W)

        self.nn_weights_iters_label = tk.Label(
            self.train_net_frame,
            text="Iters for training network weights:",
            anchor=tk.W,
        )
        self.nn_weights_iters_entry = tk.Entry(self.train_net_frame)
        self.nn_weights_iters_entry.insert(0, "100")  # Default value
        self.nn_weights_iters_label.pack(anchor=tk.W)
        self.nn_weights_iters_entry.pack(anchor=tk.W)

        self.nn_weights_pop_size_label = tk.Label(
            self.train_net_frame,
            text="Pop. size for training network weights:",
            anchor=tk.W,
        )
        self.nn_weights_pop_size_entry = tk.Entry(self.train_net_frame)
        self.nn_weights_pop_size_entry.insert(0, "100")  # Default value
        self.nn_weights_pop_size_label.pack(anchor=tk.W)
        self.nn_weights_pop_size_entry.pack(anchor=tk.W)

        # Frame for start and save buttons
        self.nn_start_and_safe_frame = tk.Frame(self.train_net_frame)
        self.nn_start_and_safe_frame.pack(pady=10, anchor=tk.W)

        # Start button
        self.nn_fit_start_button = tk.Button(
            self.nn_start_and_safe_frame, text="Start", command=self.start_nn_fit, state=tk.DISABLED, anchor=tk.W
        )
        self.nn_fit_start_button.pack(side=tk.LEFT, anchor=tk.W)

        # Save network button
        self.nn_save_net_button = tk.Button(
            self.nn_start_and_safe_frame,
            text="Save network",
            command=self.save_trained_net,
            state=tk.DISABLED,
        )
        self.nn_save_net_button.pack(side=tk.RIGHT, anchor=tk.W)

        # Frame for save stat and plot
        self.nn_save_state_and_plot_frame = tk.Frame(self.train_net_frame)
        self.nn_save_state_and_plot_frame.pack(pady=10, anchor=tk.W)

        # Button for saving statistics
        self.nn_save_stat_button = ttk.Button(
            self.nn_save_state_and_plot_frame,
            text="Save training statistics",
            command=nn_save_all_stats,
            state=tk.DISABLED,
        )
        self.nn_save_stat_button.pack(side=tk.LEFT, anchor=tk.W)

        # Button for saving network visualization
        self.nn_save_png_button = ttk.Button(
            self.nn_save_state_and_plot_frame,
            text="Save network visualization",
            command=self.plot_net_tree,
            state=tk.DISABLED,
        )
        self.nn_save_png_button.pack(side=tk.RIGHT, anchor=tk.W)

    def setup_train_tab(self):

        self.train_main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.train_main_tab, text="Training")

        # Frame for save stat and plot
        self.nn_load_train_data_and_show_frame = tk.Frame(self.train_main_tab)
        self.nn_load_train_data_and_show_frame.pack(pady=10, anchor=tk.W)

        # File selection button
        self.train_data_load_button = tk.Button(
            self.nn_load_train_data_and_show_frame,
            text="Choose train data file",
            command=self.train_data_open_file,
            anchor=tk.W,
        )
        self.train_data_load_button.pack(side=tk.LEFT, anchor=tk.W)

        # Show train_data table
        self.train_data_show_button = tk.Button(
            self.nn_load_train_data_and_show_frame,
            text="Show data",
            command=self.train_data_show_table,
            anchor=tk.W,
            state=tk.DISABLED,
        )
        self.train_data_show_button.pack(side=tk.RIGHT, anchor=tk.W)

        # Nested tabs
        self.nested_notebook_train = ttk.Notebook(self.train_main_tab)

        self.setup_train_net_tab()

        # Nested tabs pack
        self.nested_notebook_train.pack(expand=1, fill="both")

    def setup_predict_net_tab(self):
        self.predict_nn_tab = ttk.Frame(self.nested_notebook_predict)
        self.nested_notebook_predict.add(self.predict_nn_tab, text="Neural Network")
        
        # Load model button
        self.load_trained_net_button = tk.Button(
            self.predict_nn_tab,
            text="Load model",
            command=self.load_trained_net,
            anchor=tk.W,
        )
        self.load_trained_net_button.pack(pady=10, anchor=tk.W)

        # Test start button
        self.nn_predict_button = tk.Button(
            self.predict_nn_tab,
            text="Start test",
            command=self.nn_start_predict,
            anchor=tk.W,
            state=tk.DISABLED,
        )
        self.nn_predict_button.pack(pady=10, anchor=tk.W)

    def setup_predict_tab(self):
        self.predict_main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_main_tab, text="Training")

        # Frame for save stat and plot
        self.nn_load_predict_data_and_show_frame = tk.Frame(self.predict_main_tab)
        self.nn_load_predict_data_and_show_frame.pack(pady=10, anchor=tk.W)

        # File selection button
        self.data_button_predict = tk.Button(
            self.nn_load_predict_data_and_show_frame,
            text="Choose test data file",
            command=self.test_data_open_file,
            anchor=tk.W,
        )
        self.data_button_predict.pack(side=tk.LEFT, anchor=tk.W)

        # Show train_data table
        self.predict_data_show_button = tk.Button(
            self.nn_load_predict_data_and_show_frame,
            text="Show data",
            command=self.predict_data_show_table,
            anchor=tk.W,
            state=tk.DISABLED,
        )
        self.predict_data_show_button.pack(side=tk.RIGHT, anchor=tk.W)

        # Nested tabs inside "Test"
        self.nested_notebook_predict = ttk.Notebook(self.predict_main_tab)

        self.setup_predict_net_tab()

        # Nested tabs pack
        self.nested_notebook_predict.pack(expand=1, fill="both")

    def setup_ui(self):
        # Creating tabs
        self.notebook = ttk.Notebook(self.root)

        ############################################ "Training" Tab
        self.setup_train_tab()

        self.setup_predict_tab()


        self.notebook.pack(expand=1, fill="both")
        
        


if __name__ == "__main__":
    root = tk.Tk()
    app = System(root)
    root.mainloop()
