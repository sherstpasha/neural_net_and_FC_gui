import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import pandas as pd



def one_hot_encode_categorical(df, numerical_cols, categorical_cols):
    # Проверка наличия категориальных переменных в DataFrame


    return df_encoded

class CSVProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("CSV Data Processor")
        self.master.geometry("1000x700")

        self.canvas = tk.Canvas(self.master)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.scrollbar = ttk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky="ns")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.container = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        self.open_button = tk.Button(self.container, text="Open CSV", command=self.open_file)
        self.open_button.grid(row=0, column=0, pady=10)

        self.df = None
        self.data_window = None
        self.modified_df = None
        self.second_window = None
        self.var_types = {}

        self.radio_frame = tk.Frame(self.container)
        self.radio_frame.grid(row=2, column=0, pady=10)

        self.master.bind("<Configure>", self.on_window_resize)

        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(0, weight=1)

        self.container.grid_columnconfigure(0, weight=1)

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

        tree.pack(pady=10)

        header_label = tk.Label(self.radio_frame, text="Variable Name:")
        header_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

        header_label_numeric = tk.Label(self.radio_frame, text="Numeric:")
        header_label_numeric.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        header_label_categorical = tk.Label(self.radio_frame, text="Categorical:")
        header_label_categorical.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        header_label_target = tk.Label(self.radio_frame, text="Target:")
        header_label_target.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        for i, col in enumerate(self.df.columns):
            var_type = tk.StringVar()
            var_type.set("numeric")
            self.var_types[col] = var_type

            label = tk.Label(self.radio_frame, text=col)
            label.grid(row=i + 1, column=0, padx=5, pady=5, sticky=tk.W)

            numeric_radio = tk.Radiobutton(self.radio_frame, text="", variable=var_type, value="numeric")
            numeric_radio.grid(row=i + 1, column=1, padx=5, pady=5, sticky=tk.W)

            categorical_radio = tk.Radiobutton(self.radio_frame, text="", variable=var_type, value="categorical")
            categorical_radio.grid(row=i + 1, column=2, padx=5, pady=5, sticky=tk.W)

            target_radio = tk.Radiobutton(self.radio_frame, text="", variable=var_type, value="target")
            target_radio.grid(row=i + 1, column=3, padx=5, pady=5, sticky=tk.W)

        next_button = tk.Button(self.container, text="Далее", command=self.show_next_window)
        next_button.grid(row=3, column=0, pady=10)

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

        # Add a button to go back to the first window
        back_button = tk.Button(self.second_window, text="Назад", command=self.show_first_window)
        back_button.pack(pady=10)

        # Add any other elements you want to display in the second window

        # Hide the first window
        self.master.withdraw()

        # Show the second window
        self.second_window.deiconify()

    def show_first_window(self):
        # Destroy the second window
        self.second_window.destroy()

        # Show the first window
        self.master.deiconify()





    def show_error(self, message):
        messagebox.showerror("Error", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = CSVProcessorApp(root)
    root.mainloop()
