import numpy as np
import matplotlib.pyplot as plt
from numba import jit, float64 , njit , types, prange
from scipy.integrate import DOP853 , RK45 , RK23
'''
# Класс, представляющий амплитуду внешней силы
class Force_Amplitude():
    def __init__(self, F0, omega, func):
        # Инициализация параметров
        self.F0 = F0  # Амплитуда силы
        self.omega = omega  # Частота силы


        # Проверка, что func является NumPy ufunc
        if isinstance(func, np.ufunc):
            self.func = func
        else:
            raise ValueError("func should be a NumPy universal function (ufunc)")

    def __call__(self, t):
        # Функция, вычисляющая амплитуду силы в зависимости от времени
        force_amplitude = self.F0 * self.func(self.omega * (t))
        return force_amplitude

# Класс, представляющий дифференциальное уравнение демпфированного и возбужденного осциллятора
class DampedDrivenOscillator:
    def __init__(self, Function):
        self.Function = Function  # Функция, описывающая уравнение

    def set_parameters(self, state, t_end, h,t0):
        # Установка начальных условий и параметров интегрирования
        self.state = state  # Начальные условия
        self.t_end = t_end  # Время окончания
        self.h = h  # Шаг интегрирования
        self.t0=t0

    def damped_driven_oscillator(self, t, state):
        # Функция, вычисляющая производные состояния в соответствии с уравнением
        result = self.Function(state, t)
        return result

    def runge_kutta(self):
        # Решение дифференциального уравнения методом Рунге-Кутта
        state = self.state
        
        positions = [state.copy()]  # Создаем копию начального состояния и добавляем в positions

        t = self.t0
        time_points = [t]
        while t < self.t_end:
           # Метод Рунге-Кутта для решения дифференциального уравнения
           k1 = self.h * np.array(self.damped_driven_oscillator(t, state))
           k2 = self.h * np.array(self.damped_driven_oscillator(t + 0.5 * self.h, np.add(state, 0.5 * k1)))
           k3 = self.h * np.array(self.damped_driven_oscillator(t + 0.5 * self.h, np.add(state, 0.5 * k2)))
           k4 = self.h * np.array(self.damped_driven_oscillator(t + self.h, np.add(state, k3)))
           state = np.add(state, (k1 + 2 * k2 + 2 * k3 + k4) / 6)
           t += self.h
           time_points.append(t)
           positions.append(state.copy())  # Добавляем копию состояния в список
        return time_points, positions
    
 '''
class DE_solver:
    def __init__(self, Function):
        self.Function = Function  # Функция, описывающая уравнение
        self.coordinate = None

    def set_parameters(self, state, t_0, t_end, h):
        # Установка начальных условий и параметров интегрирования
        self.state = state  # Начальные условия
        self.t_0=t_0
        self.t_end = t_end  # Время окончания
        self.h = h  # Шаг интегрирования

    def DE_Function(self, t, state):
        # Функция, вычисляющая производные состояния в соответствии с уравнением
        result = self.Function(state, t)
        return result

    def runge_kutta(self):
        # Решение дифференциального уравнения методом Рунге-Кутта
        state = self.state
        t = self.t_0
        time_points = [t]
        positions = [state.copy()]  # Создаем копию начального состояния и добавляем в positions


        while t < self.t_end:
           # Метод Рунге-Кутта для решения дифференциального уравнения
           k1 = self.h * np.array(self.DE_Function(t, state))
           k2 = self.h * np.array(self.DE_Function(t + 0.5 * self.h, np.add(state, 0.5 * k1)))
           k3 = self.h * np.array(self.DE_Function(t + 0.5 * self.h, np.add(state, 0.5 * k2)))
           k4 = self.h * np.array(self.DE_Function(t + self.h, np.add(state, k3)))
           state = np.add(state, (k1 + 2 * k2 + 2 * k3 + k4) / 6)
           t += self.h
           time_points.append(t)
           # print(state) # <--
           positions.append(state.copy())  # Добавляем копию состояния в список
        return time_points, positions


    def calculate_results(self):
      time_points, positions = self.runge_kutta()

      A=np.array([positions])

      print( A.shape )

      L=len(self.state[0])
      print(L)

      coordinate=[]

      for i in range(len(positions[0])):
            state_i = [state[i] for state in positions]
            coordinate.append(state_i)

      self.coordinate=coordinate
      self.time_points = time_points
      return coordinate , time_points 

    def print_results(self):
       if self.coordinate is None:
          self.calculate_results()
       new_coordinate=np.array(self.coordinate)


       print(new_coordinate.shape)

       print("-----------------------------------------------------------")
       print("Координаты :")
       for i , mass in enumerate(new_coordinate[0].T):
        print(f"{i}-ая координата")
        print(mass)
        print("  ")

       print("-----------------------------------------------------------")
       print("Скорости :")
       for i , mass in enumerate(new_coordinate[1].T):
        print(f"{i}-ая скорость")
        print(mass)


            
# Класс, представляющий дифференциальное уравнение
class IDE_many_particuls_scipy:
    def __init__(self, Function,methood):
        self.Function = Function  # Функция, описывающая уравнение
        self.methood = methood

    # Задание параметров ДУ
    def set_parameters(self, state, t0, t_end, h , K):
        """
        Установка параметров для решения уравнения.

        Parameters:
        - state: Начальные условия
        - t_end: Время окончания
        - h: Шаг интегрирования
        """
        self.state = state
        self.t0 = t0
        self.t_end = t_end
        self.h = h
        self.k=K

    # Метод Рунге-Кутты 4 для расчёта ДУ
    @jit(parallel=True, fastmath=True,forceobj=True)
    def mhetood_calcul(self):

      solver = self.methood(self.Function, self.t0, self.state, self.t_end,self.h )

      # Списки для хранения траекторий маятников
      x = [[] for _ in range(self.k)]
      v = [[] for _ in range(self.k)]
      times = []

      # Интегрирование системы и сохранение траекторий
      while solver.status == 'running':
         t = solver.t
         y = solver.y
         times.append(t)
         for i in range(self.k):
            idx = i * 6
            idx3=idx+3
            x[i].append(y[idx:idx + 3].copy())
            v[i].append(y[idx3:idx3 + 3].copy())
         solver.step()

      # Преобразуем список списков в массив numpy для удобства использования
      x = np.array(x)
      v = np.array(v)
      times = np.array(times)

      return times, x,v  # Преобразуем список в массив для совместимости с остальным кодом

    # Расчёт ДУ
    @jit(parallel=True, fastmath=True,forceobj=True)
    def calculete_parametrs(self):
        time_points, x ,v = self.mhetood_calcul()
        self.x = x
        self.v = v

        self.time_points = time_points
        # Создание маски, содержащей значения True для векторов, не содержащих NaN


        
        print("Размер координаты",self.x.shape)
        print("Размер скорости  ",self.v.shape)
        print("начальные параметры")
        print("X = ",self.x[:,0])
        print("V = ",self.v[:,0])

        print("конечные параметры")
        print("X = ",self.x[:,-1])
        print("V = ",self.v[:,-1])
        return time_points , np.array([x[0],v[0]])

    # Вывод результатов расчёта
    def print_result(self,N,K):
      X = self.x
      V = self.v
      print("Размер координаты",self.x.shape)
      print("Размер скорости  ",self.v.shape)



      for i in range(self.k):
        print("------------------------------------------")
        print(f"Координаты и скорости {i+1}-ой частицы ")
        x=X[i]
        v=V[i]
        print(x.shape)
        print(v.shape)
        for j in range(3):
          print(f"{j+1}-ая компонента координаты")
          print(x.T[j])
          print(" ")
          print(f"{j+1}-ая компонента скорости")
          print(v.T[j])
          print(" ")
        print("  ")
        print("  ")

   