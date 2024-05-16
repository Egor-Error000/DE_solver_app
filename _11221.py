


from re import T
from Graf_Tk import *
from Super_DE import *
import tkinter as tk
from  numpy import*
from numba import jit, float64 , njit , types, prange




# Создаем объект окна tkinter
root = tk.Tk()
root.title("График")
root.configure(bg="#004d40")

# Создаем объект ParameterManager
parameters = {
    "x0": {
        "type": "float",
        "default_value": 0
    },
    "y0": {
        "type": "float",
        "default_value": 0
    },
    "z0": {
        "type": "float",
        "default_value": 0
    },
    "vx0": {
        "type": "float",
        "default_value": 0.1
    },
    "vy0": {
        "type": "float",
        "default_value": 0.1
    },
        "vz0": {
        "type": "float",
        "default_value": 0.1
    },
    "k": {
        "type": "float",
        "default_value": 0.7
    },
    "b": {
        "type": "float",
        "default_value": 0.7
    },
    "m": {
        "type": "float",
        "default_value": 0.7
    },
    "wy": {
        "type": "float",
        "default_value": 0.7
    },
    "h": {
        "type": "float",
        "default_value": 0.0001
    },
    "t_0": {
        "type": "float",
        "default_value": 0
    },
    "t_end": {
        "type": "float",
        "default_value": 0.5
    }
}

operations = {
    "set_parameters": {
        "function": 0,
        "description": "Установить параметры"
     },
    "calculate_parametrs": {
        "function": 0, 
        "description": "Вычислить параметры"
     },
    "draw_graph": {
        "function": 0, 
        "description": "Построить график"
     },
    "write_to_file": {
        "function": 0, 
        "description": "Записать в файл"
     },
    "read_from_file": {
        "function": 0, 
        "description": "Прочитать из файла"
     },
    "save_graph": {
        "function": 0, 
        "description": "Сохранить график"
     },
    "clear_graph": {
        "function": 0, 
        "description": "Очистить график"
     }
}


    
def Func_usualy(parameters_new):
    
    def get(name):
       return parameters_new.get(name, {}).get("default_value")
    
    global B , k  ,b ,m ,w
    
    x0 = get("x0")
    y0 = get("y0")
    z0 = get("z0")

    
    vx0 = get("vx0")
    vy0 = get("vy0")
    vz0 = get("vz0")
    
    
    
    
    k = get("k")
    b = get("b")
    m = get("m")
    wy = get("wy")
    
    max_step =get("h")
    t_bound =get("t_end")
    t0 =get("t_0")
    
    w = np.array([[0.,0.,-2*wy ],[0.,0.,0. ],[2*wy,0.,0. ]])


    X0=[ x0 , y0 , z0 ]
    V0=[vx0 , vy0, vz0]

    y0=np.array(X0+V0)
    # вместо func можно подставить либо DOP853 , RK45 , RK24  всё это методы рунге кутты
    #func = DOP853
    func = RK45
    #func = RK23

    MyDE = IDE_many_particuls_scipy(equations,func)
    MyDE.set_parameters(y0, t0, t_bound, max_step,1)
    
    time_points,coordinate = MyDE.calculete_parametrs()

    
    
    MyDE.print_result(10,10)  
    print(np.array(coordinate).shape,"Размер нового массива")


    return time_points,coordinate 
     

@jit(nopython=True, fastmath=True)
def equations(t, y):
    dydt = np.zeros_like(y)


    x = y[0:3]
    v = y[3:6]

    dxdt = v

    r_3=np.sum(x**3)

    if r_3 == 0:
      K=0
    else:
      K=k/r_3


    dvdt = -(b/m *v + v@w + K*x )


    dydt[0:3] = dxdt
    dydt[3:6] = dvdt

    return dydt




n = 1
text=['время',"позиция"]
text_mas = ['координата', ' скорость', n]
colors = ['b', 'r']

# Передаем объект Graf в параметры объекта pm

mp = ParameterManager(root, parameters, operations,Graph=True,Give_Text=True,Func_name=Func_usualy,text_mas=text_mas,text=text,colors=colors,bg="#004d40",fg='#ffcdd2',title="Дифур ")
# Выводим окно tkinter
root.mainloop()

