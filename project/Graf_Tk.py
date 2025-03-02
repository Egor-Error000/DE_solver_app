import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import filedialog
import numpy as np
from numpy import*
import pandas as pd
import csv
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA


class ParameterManager:
    matrix_instance = np.zeros((2, 2))
    file_name = "File.txt" 
    Job_file="out.txt"

    def __init__(self, root, parameters=None, operations=None,Graph=None,Give_Text=None,Func_name=None,text_mas=None,text=None ,colors=None,bg=None,fg=None,title=None):
        self.root = root 
        self.Graph = Graph or False
        self.Give_Text=Give_Text or False
        self.parameters = parameters or {}  
        self.buttons = {}  
        self.entries = {}
        self.combo={}
        self.selected=tk.StringVar()
        self.selected.set(None)
        self.text_mas=text_mas
        self.text=text or "time" "position"
        self.colors=colors
        self.fig=None
        self.ax=None
        self.x=None
        self.other_variabel=None
        self.canvases=None
        self.nv_frames=None
        self.bg=bg or "write"
        self.fg=fg or 'black'
        self.title=title or "График"

        # Функция должна возвращать двумерный массив Numpy
        self.Func_name=Func_name or False
        
        self.operations_list= {
        "set_parameters": self.set_parameters,
        "calculate_parametrs": self.calculate_parametrs,
        "draw_graph": self.draw_graph,
        "write_to_file": self.write_to_file,
        "read_from_file": self.read_from_file,
        "save_graph": self.save_graph,
        "clear_graph": self.clear_graph
         }


        self.assign_values(self.operations_list, operations)
        self.operations = operations or {} 
        

        self.create_widgets() 

    def ndarray(self,string_array):

      try:
       # Извлекаем числовые значения из строки
            values = [float(val) for val in string_array.split("[")[1].split("]")[0].split(",")]

       # Преобразуем список значений в массив numpy.ndarray
            numeric_array = np.array(values)
            print("Успешно преобразовано в numpy.ndarray:")
            print(numeric_array)
            return numeric_array
      except ValueError:
            print("Не удалось преобразовать в numpy.ndarray. Проверьте формат строки.")       

    def assign_values(self, source_dict, target_dict):
        for key, value in source_dict.items():
            if key in target_dict:
               target_dict[key]["function"] = value

    def enable_buttons(self, *button_names):
        
        button_names = list(button_names)  # Преобразовать кортеж в список
        for name in button_names:
            
            try:
              button = self.buttons.get(name)  # Получаем кнопку по имени.           
              if button:
                  button.config(state="active")  # Включаем кнопку.
                
            except:
              pass

    def disable_buttons(self, *button_names):
        
        button_names = list(button_names)  # Преобразовать кортеж в список
        for name in button_names:
            button = self.buttons.get(name)  # Получаем кнопку по имени.
            if button:
                button.config(state="disabled")  # Отключаем кнопку.

    def keys(self):
        
        keys = list(self.buttons.keys())  # Получаем ключи кнопок в виде списка
        index = 1  # Индекс элемента, который хотим получить (например, второй элемент)

        if 0 <= index < len(keys):
            key = keys[index]  # Получаем ключ (название операции) по индексу.
            value = self.buttons[key]  # Получаем объект кнопки по ключу.
            return value  # Возвращаем объект кнопки.
        else:
            print("Индекс за пределами диапазона ключей")  # Если индекс находится за пределами доступных кнопок, выдаем сообщение.      
 
    def error(self,e,F=None):
         self.entry_EROR.configure(bg="red")    
         self.entry_EROR.delete(0, tk.END)
         if F != None:
             string=F
         else:
             string=""
         self.entry_EROR.insert(0,f"{e}" )

    def right(self,e):
         self.entry_EROR.configure(bg="#004d40")
         self.entry_EROR.delete(0, tk.END)
         self.entry_EROR.insert(0,e )

    def create_widgets(self):
        # Создаем фрейм для параметров

        self.param_frame = tk.Frame(self.root,bg=self.bg)
        self.param_frame.grid(row=0, column=0, rowspan=len(self.parameters), columnspan=2)

        for i, (param_name, _) in enumerate(self.parameters.items()):
           label = tk.Label(self.param_frame, text=f"{param_name}:",height=round(23/len(self.parameters)),bg=self.bg,fg=self.fg)  
           label.grid(row=i, column=0,rowspan=1)  
           entry = tk.Entry(self.param_frame,bg=self.bg,fg=self.fg) 
           param_info = self.parameters.get(param_name)
           entry.insert(0, param_info["default_value"])
           entry.grid(row=i, column=1,rowspan=1)  
           self.entries[param_name] = entry


        self.operation_frame = tk.Frame(self.root,bg=self.bg)
        self.operation_frame.grid(row=len(self.parameters), column=0,  columnspan=1)
        for operation_name, operation_info in self.operations.items():
             button = tk.Button(self.operation_frame, text=operation_info["description"], command=operation_info["function"], width=40,bg=self.bg,fg=self.fg)  
             button.grid(row=list(self.operations.keys()).index(operation_name), column=0, columnspan=2) 

             self.buttons[operation_name] = button  # Добавляем кнопку в словарь с ключом - названием операции.
             self.disable_buttons(operation_name)  # Изначально отключаем кнопку операции.
        self.enable_buttons("set_parameters")  # Включаем кнопку операции "set_parameters".
        
        self.L=abs(6.5)
        self.entry_EROR = tk.Entry(self.root, width=int(self.L*16),bg=self.bg)
        self.entry_EROR.insert(0,"Здесь будут сообщения об ошибках.\n" )
        self.entry_EROR.grid(row=len(self.parameters)-1 , column=2) 
        
        parameters=self.chek_zeros_parametrs()
        result = self.Func_name(parameters)
        
        self.Shape = np.transpose(result[1], (0, 2, 1)).shape
        self.length = len(result[1])
        print(np.array(result[1]).shape)
        self.Len=np.array(result[1]).shape[2]
        print('l=',self.length)
        print('L=',self.Len)

        if (self.Graph is not False):
           self.Graph_Frame = tk.Frame(self.root,bg=self.bg)
           self.Graph_Frame.grid(row=0, column=2, columnspan=2,rowspan=len(self.parameters)-3)
           self.nav_frame = tk.Frame(self.Graph_Frame)
           self.nav_frame.grid(row=1, column=0, columnspan=2)
           length=self.length
           
           if self.length <= 1:
              self.fig, self.ax = plt.subplots(figsize=(self.L, 3.6))
              self.canvas = FigureCanvasTkAgg(self.fig, master=self.Graph_Frame)
              self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2,rowspan=len(self.parameters)-3)

   # Создайте панель навигации внутри Graph_Frame


              self.toolbar = NavigationToolbar2Tk(self.canvas, self.nav_frame)
              self.toolbar.update()
              self.toolbar.get_tk_widget().grid(row=0, column=0, columnspan=2)
        
        
        if (self.Give_Text is not False):
            self.Text_frame = tk.Frame(self.root,bg=self.bg)
            self.Text_frame.grid(row=0, column=4, rowspan=len(self.parameters),columnspan=2)
            self.Give_Text = tk.Text(self.Text_frame, wrap=tk.WORD, width=11*(length+1),height=27,bg=self.bg,fg=self.fg)
            self.Give_Text.grid(row=0, column=0,rowspan=5,columnspan=2)
            # Создаем виджет Scrollbar для вертикальной прокрутки
            #vertical_scrollbar = tk.Scrollbar(self.Text_frame, command=self.Give_Text.yview)
            #vertical_scrollbar.grid(row=0, column=2,rowspan=5, sticky="ns")

            # Привязываем полосу прокрутки к виджету Text
            #self.Give_Text.config(yscrollcommand=vertical_scrollbar.set)
            #self.Give_Text_button= tk.Button(self.Text_frame, text="Считать данные ", command=self.Give_Text_reader)
            #self.Give_Text_button.grid(row=5, column=0)
            self.Set_Text_button= tk.Button(self.Text_frame, text="Получить данные", command=self.save_to_text_widget,state="disabled",bg=self.bg) #self.save_to_text_widget
            self.Set_Text_button.grid(row=6, column=0)
        

        length=self.length*self.Len+2
        print(length)
        if length > 1 :
            RB_Frame=tk.Frame(self.root)
            RB_Frame.grid(row=len(self.parameters), column=2)
            
            label_RB=tk.Label(RB_Frame,text="Выберите график")
            label_RB.grid(row=0 , column=0,columnspan=length)
            
            
            canvases = {}
            nv_frames={}
            canvas_names=[]
            radio_buttons = []
            
            for i in range(length):
                canvas_names.append(f"№{i+1}")
             
            self.fig = [None] * length
            self.ax  = [None] * length
            for i, canvas_name in enumerate(canvas_names):
                if i+1<length:
                   self.fig[i], self.ax[i] = plt.subplots(figsize=(self.L, 3.9))
                else:
                   fig = Figure(figsize=(self.L, 3.9), dpi=100)
                   ax = fig.add_subplot(111, projection='3d')
                   
                   self.fig[-1] =fig
                   self.ax[-1] = ax
                   
            
                    

                canvas =  FigureCanvasTkAgg(self.fig[i], master=self.Graph_Frame)
                canvas.get_tk_widget().grid(row=2, column=0, columnspan=2)
                nv_frame = tk.Frame(self.Graph_Frame)
                nv_frame.grid(row=1, column=0, columnspan=2)
                nv_frame.grid_remove()

                toolbar = NavigationToolbar2Tk(canvas, nv_frame)
                toolbar.update()
                canvases[canvas_name] = canvas
                nv_frames[canvas_name]=nv_frame

            for i, canvas_name in enumerate(canvases):
                radio_button = tk.Radiobutton(RB_Frame, text=canvas_name, variable=self.selected, value=canvas_name, command= lambda: self.switch_canvas(canvases),bg=self.bg,fg=self.fg)
                radio_button.grid(row=0, column=i)
                radio_buttons.append(radio_button)

                if i == 0:
                    self.show_canvas(canvas)  # Отображаем первый холст по умолчанию
                    nv_frames[canvas_name].grid()
                    
            self.canvases=canvases
            self.nv_frames=nv_frames 
            self.canvas_names=canvas_names

    def show_canvas(self,canvas):
       canvas.get_tk_widget().grid(row=2, column=0, columnspan=2)
       canvas.get_tk_widget().update_idletasks()

    def hide_canvas(self,canvas):
       canvas.get_tk_widget().grid_remove()

    def switch_canvas(self,canvases):
       selected_canvas = self.selected.get()
    
       for canvas_name, canvas in canvases.items():
          if selected_canvas == canvas_name:
              self.show_canvas(canvas)
              self.nv_frames[canvas_name].grid()
          else:
              self.hide_canvas(canvas)
              self.nv_frames[canvas_name].grid_remove()

    def chek_zeros_parametrs(self):
      parameters=self.parameters
      for param_name, param_info in parameters.items():
        if "default_value" not in param_info:
            raise ValueError(f"Параметр {param_name} не имеет значения 'default_value'")

        value = param_info["default_value"]

        param_type = param_info["type"]
        default_value = param_info["default_value"]

        if not value:
            value = default_value

        if param_type == "int":
            try:
                value = int(value)
            except ValueError:
                raise ValueError(f"Недопустимое значение параметра {param_name}: {value} (ожидается int)")
        elif param_type == "float":
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"Недопустимое значение параметра {param_name}: {value} (ожидается float)")
        elif param_type == "Func":
            try:
                value = eval(value)
            except:
                raise ValueError(f"Недопустимое значение параметра {param_name}: {value} (ожидается функция)")
        else:
            raise ValueError(f"Тип параметра не поддерживается для {param_name}: {param_type}")

        param_info["default_value"] = value

      return parameters

    def set_parameters(self):
       try:
            for param_name, entry in self.entries.items():
           
               value = entry.get()

               if param_name not in self.parameters:
                   raise ValueError(f"Параметр {param_name} не определен")

               param_info = self.parameters.get(param_name)
   
               if param_info is None:
                   raise ValueError(f"Информация о параметре {param_name} не найдена")

               param_type = param_info["type"]
               default_value = param_info["default_value"]

               if not value:
                   value = default_value

               if param_type == "int":
                   try:
                       value = int(value)
                   except ValueError:
                       raise ValueError(f"Недопустимое значение параметра {param_name}: {value} (ожидается int)")
               elif param_type == "float":
                   try:
                       value = float(value)
                   except ValueError:
                      raise ValueError(f"Недопустимое значение параметра {param_name}: {value} (ожидается float)")
               elif param_type == "numpy.ndarray":
                   try:
                        value = np.array(value)
                   except ValueError:
                        raise ValueError(f"Недопустимое значение параметра {param_name}: {value} (ожидается строка)") 
               elif param_type == "Func":
                   try:
                       value = eval(value)
                   except:
                       raise ValueError(f"Недопустимое значение параметра {param_name}: {value} (ожидается функция)")
               else:
                   raise ValueError(f"Тип параметра не поддерживается для {param_name}: {param_type}")

               param_info["default_value"] = value
               

               self.enable_buttons("calculate_parametrs")
               self.right("Всё верно")
       except ValueError as e:
               self.error(e)

    def calculate_parametrs(self):
        if self.Func_name is not False :  
          try :
            x,other_variabel= self.Func_name(self.parameters)
            if other_variabel is not None:
               pass
            else:
               raise ValueError("Фукция обработки не возвращает значения зависимых переменных , исправьте код функции обработки")
            print("Мы дошли до сюда") 
            self.x=x
            self.other_variabel=other_variabel
       
            mass=[]
            mass.append(self.x)
            print("right")
            l=len(self.other_variabel)
            print(l)
            for i in range(l):
                print(np.array(self.other_variabel[i]).shape)
                mass.append(self.other_variabel[i])
                print(type(mass))
            print(type(mass))
            
            

            self.mass_MA = mass
            

            

            self.right("Всё верно")
            self.enable_buttons("draw_graph", "write_to_file", "read_from_file", "close_app")
            if self.Give_Text is not False:
             self.Set_Text_button.config(state="active")
          except ValueError as e:
            self.error(e)

    def Give_Text_reader(self):
        try:
            text = self.Give_Text.get("1.0", "end-1c")
            with open(self.Job_file, "w") as file:
                file.write(text)
            self.read_from_file
            self.mass_MA = np.loadtxt(self.Job_file)
            
            #Блок ax
            self.ax.set_title("График")
            self.ax.set_xlabel("X-ось")
            self.ax.set_ylabel("Y-ось")
            
            #Рисовать в Ax
            self.ax.plot(self.mass_MA[:, 0], self.mass_MA[:, 1], label='График', color='red', linestyle='--')
            
            self.canvas.draw()
            self.right("Всё верно")
        except ValueError as e:
            self.error(e)
 
    def save_to_text_widget(self):
       #Создайте DataFrame с данными и именованными столбцами
       data={}
       mass_MA_transposed = self.mass_MA
       for i in range(len(self.text_mas)):
           name=self.text_mas[i-1]
           if i == 0:
             data[self.text[0]]=mass_MA_transposed[i]
           else:
             words = name.split()  # Разбиваем строку на слова
             last_word = words[-1]  # Получаем последнее слово
             shortened_last_word = last_word[:8]
             data[shortened_last_word]=mass_MA_transposed[i]
           
       

       df = pd.DataFrame(data)
       table_string = df.to_string(index=False, justify='center')

       # Сохраните строку с выравниванием столбцов в файл
       with open(self.Job_file, 'w', encoding='cp1251') as file:
          file.write(table_string)
          
       self.load_file_to_text_widget()

    def load_file_to_text_widget(self):
       try:
          with open(self.Job_file, "r") as file:
            file_contents = file.read()
            self.Give_Text.delete("1.0", tk.END)  # Очищаем виджет Text
            self.Give_Text.insert("1.0", file_contents)  # Вставляем содержимое файла в виджет Text
       except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
        
        
    def draw_graph(self):
       if self.length > 1:
           canvas_mas=[]
           f=self.length+1
           for canvas_name in self.canvases:
               canvas = self.canvases[canvas_name]
               canvas_mas.append(canvas)
           VECTOR = np.transpose(np.array(self.other_variabel), (2, 0, 1))
           for j in range(self.Len):
               for k in range(self.length):
                   self.ax[j].plot(self.x, VECTOR[j][k],label=self.text_mas[k])
                   
               self.ax[j].set_xlabel(self.text[0])
               self.ax[j].set_ylabel(self.text[1])
               self.ax[j].set_title(self.title)
               self.ax[j].grid(True)
               self.ax[j].legend()
               canvas_mas[j].draw()
               for k in range(self.length-1):
                  self.ax[j+f+k].plot(VECTOR[j][k], VECTOR[j][k+1])
                  self.ax[j+f+k].set_xlabel(self.text_mas[k])
                  self.ax[j+f+k].set_ylabel(self.text_mas[k+1])
                  self.ax[j+f+k].set_title(self.title)
                  self.ax[j+f+k].grid(True)
                  self.ax[j+f+k].legend()
                  canvas_mas[j+f+k].draw()
           print(self.ax[-1])
           
           
           V= np.transpose(VECTOR , (1, 0, 2))[1]
  
           self.ax[-2].plot(self.x, LA.norm(V,axis=0),label="Модуль скорости")
           
           self.ax[-2].set_xlabel("time")
           self.ax[-2].set_ylabel("norm(v)")
           self.ax[-2].set_title(self.title)
           self.ax[-2].grid(True)
           self.ax[-2].legend()
           canvas_mas[-2].draw()
           
           # А затем в функции `draw_graph`, используйте `plot_surface` на объекте `Axes3D`:
           print(VECTOR.shape)
           self.ax[-1].plot(VECTOR[0][0], VECTOR[1][0], zs=VECTOR[2][0])
           

           self.ax[-1].set_xlabel('X')
           self.ax[-1].set_ylabel('Y')
           self.ax[-1].set_zlabel('Z')
           self.ax[-1].set_title('Траектория частицы в пространсве')
           
       else:   
          pass
           
       self.right("График построен")
       self.enable_buttons("save_graph", "clear_graph")
    
    def write_to_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("Text Files", "*.csv")])
        dictionary={}
        print(self.Shape,"shape")
        print(self.other_variabel,"masShape")
        for i in range(self.Shape[0]):
           for j in range(self.Shape[1]):
             Vec = f'Vec{i},{j}'  # Генерация ключа, например, 'key_0', 'key_1', и т.д.
             print(self.other_variabel[i][j].shape)
             
             dictionary[Vec] = np.transpose(np.array(self.other_variabel), (0, 2, 1))[i][j].tolist()  # Добавление ключа и значения в словарь
         
        dictionary[self.text[0]]=self.x
        self.df = pd.DataFrame(dictionary)
        # Если пользователь выбрал файл для сохранения
        if file_path:
          try:
       
             with open(file_path, "w") as file:
                  # Записываем в файл нужный текст (замените эту строку на ваш текст)
                   self.df.to_csv(file_path, index=False)
                   
             self.right(f"Файл успешно сохранен по пути: {file_path}")
             
          except Exception as e:
             self.error(f"Ошибка при сохранении файла: {e}")
        else:
             self.error(f"Сохранение отменено или файл не выбран")

    def read_from_file(self):
        
        file = filedialog.askopenfile(filetypes=[("Text Files", "*.txt")])
        
        try:
           
              file_contents = file.read()
              self.Give_Text.delete("1.0", tk.END)  # Очищаем виджет Text
              self.Give_Text.insert("1.0", file_contents)  # Вставляем содержимое файла в виджет Text
        except Exception as e:
             print(f"Ошибка при загрузке файла: {e}")

    def save_graph(self):
        for i in range(self.length):
             file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Files", "*.png")])
             if file_path:
               self.fig[i].savefig(file_path)

    def clear_graph(self):
       for i in range(len(self.ax)):
           self.ax[i].clear()
           self.ax[i].set_title("")
           self.ax[i].grid(True)
           self.ax[i].legend()
           self.fig[i].canvas.draw()

    def close_app(self):
        self.root.destroy()