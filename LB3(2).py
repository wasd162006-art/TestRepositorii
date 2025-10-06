import numpy as np
from colorama import Fore, Back, Style, init
import matplotlib.pyplot as plt

class f:
    """
    Класс для представления математической функции и её производной.
    
    Attributes:
        func (callable): Основная функция
        derivative_func (callable): Производная функции (опционально)
    """
    
    def __init__(self, func, derivative_func=None):
        """
        Инициализация функции и её производной.
        
        Args:
            func (callable): Основная функция
            derivative_func (callable, optional): Производная функции. Если не задана, 
                                                будет вычисляться численно
        """
        self.func = func
        self.derivative_func = derivative_func

    def __call__(self, arg: float) -> float:
        """
        Вызов функции в точке arg.
        
        Args:
            arg (float): Точка, в которой вычисляется функция
            
        Returns:
            float: Значение функции в точке arg
        """
        return self.func(arg)           
    
    def derivative(self, x: float, h: float = 1.e-8) -> float:
        """
        Вычисление производной функции в точке x.
        
        Args:
            x (float): Точка, в которой вычисляется производная
            h (float, optional): Шаг для численного дифференцирования
            
        Returns:
            float: Значение производной в точке x
        """
        if self.derivative_func is None:
            return (self(x + h) - self(x - h)) / (2 * h)
        else:
            return self.derivative_func(x)

#-----------------------------------------------------------------------------------#
#                               Глобальные константы                                #
#-----------------------------------------------------------------------------------#

INITIAL_STEP = 1.0                        # Начальный шаг для градиентного спуска
Beta = 0.5                                # Изменения шага
TOLERANCE = 1e-6                          # Точность
MAX_ITERATIONS = 1000                     # Максимальное кол-во итераций

#-----------------------------------------------------------------------------------#
#                                   Задание функции                                 #
#-----------------------------------------------------------------------------------#

def func1(x: float) -> float:
    """Квадратичная функция: f(x) = x²"""
    return x**2

def func1_df(x: float) -> float:
    """Производная квадратичной функции: f'(x) = 2x"""
    return 2*x

def func2(x: float) -> float:
    """Розенброк-подобная функция в 1D"""
    return 100 * (x**2 - x) + (1 - x)**2

def func2_df(x: float) -> float:
    """Производная Розенброк-подобной функции: f'(x) = 202x - 102"""
    return 202 * x - 102

def func3(x: float) -> float:
    """Функция с плато в области [-1, 1]"""
    if isinstance(x, (int, float)):
        if abs(x) < 1:
            return 0.1
        else:
            return (abs(x) - 1)**2
    else:
        # Для массивов (для построения графиков)
        return np.where(np.abs(x) < 1, 0.1, (np.abs(x) - 1)**2)

def func3_df(x: float) -> float:
    """Производная функции с плато"""
    if abs(x) < 1:
        return 0.0
    elif x > 0:
        return 2 * (x - 1)
    else:
        return 2 * (x + 1)

FUNC1C = f(func1)           # Квадратичная (численная производная)
FUNC1  = f(func1, func1_df)  # Квадратичная (аналитическая производная)       
FUNC2C = f(func2)           # Розенброк-подобная (численная производная)
FUNC2  = f(func2, func2_df)  # Розенброк-подобная (аналитическая производная)
FUNC3C = f(func3)           # Плато-функция (численная производная)
FUNC3  = f(func3, func3_df)  # Плато-функция (аналитическая производная)

#-----------------------------------------------------------------------------------#
#                                  Реализация метода                                #
#-----------------------------------------------------------------------------------#

def GradientDescent(f: f, FirstPoint: float, A: float, B: float) -> float:
    """
    Выполняет один шаг градиентного спуска с подбором шага.
    
    Args:
        f (f): Функция для оптимизации
        FirstPoint (float): Текущая точка
        A (float): Левая граница области поиска
        B (float): Правая граница области поиска
        
    Returns:
        float: Новая точка после шага градиентного спуска
    """
    step = INITIAL_STEP  # Используем константу как начальное значение
    
    # Добавляем ограничение на максимальное количество итераций для поиска шага
    max_step_iterations = 100
    step_iterations = 0
    
    while step_iterations < max_step_iterations:
        grad = f.derivative(FirstPoint)
        new_point = FirstPoint - step * grad
        
        # Проверяем условие достаточного убывания (Armijo condition)
        if f(new_point) <= f(FirstPoint) - Beta * step * (grad ** 2):
            # Добавляем проверку границ
            if new_point > B:
                return B
            elif new_point < A:
                return A
            return new_point
        else:
            step = Beta * step
            step_iterations += 1
    
    # Если не нашли подходящий шаг, возвращаем текущую точку
    return FirstPoint

def Metod(func: f, A: float, B: float, Sp: float) -> tuple:
    """
    Реализация метода градиентного спуска для минимизации функции.
    
    Args:
        func (f): Функция для минимизации
        A (float): Левая граница области поиска
        B (float): Правая граница области поиска
        Sp (float): Начальная точка
        
    Returns:
        tuple: (x, k, Sp, arr_x) - конечная точка, количество итераций, 
                начальная точка, массив точек траектории
    """
    k = 0  # Кол-во итераций
    arr_x = [Sp]
    x = arr_x[0]
    
    while k < MAX_ITERATIONS and abs(func.derivative(x)) > TOLERANCE:
        x_prev = x
        x = GradientDescent(func, x, A, B)
        arr_x.append(x)
        if abs(x - x_prev) < 1e-8:
            break
        k += 1 

    return x, k, arr_x 

def FunctionСompare(name: str, FUNC: f, F: float, Fk: int, FSp: float, 
                   FUNCC: f, FC: float, FkC: int) -> None:
    """
    Сравнивает результаты оптимизации с аналитической и численной производными.
    
    Args:
        name (str): Название функции
        FUNC (f): Функция с аналитической производной
        F (float): Конечная точка для аналитической производной
        Fk (int): Количество итераций для аналитической производной
        FSp (float): Начальная точка
        FUNCC (f): Функция с численной производной
        FC (float): Конечная точка для численной производной
        FkC (int): Количество итераций для численной производной
        FCSp (float): Начальная точка для численной производной
    """
    print()
    print()
    print(Fore.BLACK + Back.WHITE + f"{name} - FUNC или FUNCC" + Style.RESET_ALL)
    print()
    print(Fore.BLUE + f"Начальная точка FUNC и FUNCC : {FSp}")
    print(Fore.GREEN + "Сравнение аналитической и численной производной")
    print()
    print(Fore.RED + "Аналитическая - FUNC")
    print(Fore.WHITE + f"По конечной точке:                     FUNC: {F}")
    print(f"По конечной производной:               FUNC: {FUNC.derivative(F)}")
    print(f"Кол-во итераций:                       FUNC: {Fk}")
    print()
    print(Fore.RED + "Численная - FUNCC")
    print(Fore.WHITE + f"По конечной точке:                     FUNCC: {FC}")
    
    # Для плато-функции используем меньший шаг для численной производной
    if name == "Плато-функция":
        deriv_FC = FUNCC.derivative(FC, 1e-12)  # Уменьшаем шаг для повышения точности
    else:
        deriv_FC = FUNCC.derivative(FC)
        
    print(f"По конечной производной:               FUNCC: {deriv_FC}")
    print(f"Кол-во итераций:                       FUNCC: {FkC}")

def FunctionSchedule(FUNC: f, name: str, Farr: list, FCarr: list, A: int, B: int) -> None:
    """
    Строит график функции и траекторий градиентного спуска.
    
    Args:
        FUNC (f): Функция для построения графика
        name (str): Название функции
        Farr (list): Траектория точек для аналитической производной
        FCarr (list): Траектория точек для численной производной
    """
    x = np.linspace(A, B, 1000)
    plt.figure(figsize=(8, 6))

    # Рисуем график функции
    y = FUNC(x)
    plt.plot(x, y, linewidth=2, label='Функция')

    # Отображаем градиентный спуск в виде точек-крестиков
    Fy = [FUNC(x_point) for x_point in Farr]
    FCy = [FUNC(x_point) for x_point in FCarr]
    
    plt.scatter(Farr, Fy, marker='o', color='red', s=50, label='Аналитическая производная')
    plt.scatter(FCarr, FCy, marker='x', color='blue', s=50, label='Численная производная')

    plt.title(f"График функции {name}(x)")  # Динамический заголовок
    plt.xlabel("Ось X")
    plt.ylabel("Ось Y")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{name}_plot.png", dpi=300, bbox_inches='tight')
    plt.close()# Для плато-функции используем меньший шаг для численной производной

def plot_convergence_comparison(FUNC: f, name: str, Farr: list, FCarr: list):
    """
    Строит сравнительный график сходимости для разных типов производных
    """
    plt.figure(figsize=(12, 8))

    # Вычисляем значения функции в точках траектории для аналитической и численной производных
    values_analytical = [FUNC(x) for x in Farr]
    values_numerical = [FUNC(x) for x in FCarr]

    # Вычисляем разности между последовательными значениями
    # Для аналитической производной
    diff_analytical = [abs(values_analytical[i] - values_analytical[i+1]) for i in range(len(values_analytical)-1)]
    # Для численной производной
    diff_numerical = [abs(values_numerical[i] - values_numerical[i+1]) for i in range(len(values_numerical)-1)]

    # Номера итераций для разностей (на одну меньше)
    iterations_analytical_diff = list(range(len(diff_analytical)))
    iterations_numerical_diff = list(range(len(diff_numerical)))

    # Строим графики разностей функции от номера итерации
    plt.plot(iterations_analytical_diff, diff_analytical, marker='o', color='red', linewidth=2, label='Аналитическая производная')
    plt.plot(iterations_numerical_diff, diff_numerical, marker='x', color='blue', linewidth=2, label='Численная производная')

    plt.xlabel('Итерация')
    plt.ylabel('|f(x_i) - f(x_{i+1})|')
    plt.title(f'Сравнение сходимости метода градиентного спуска: {name}')
    plt.grid(True)
    plt.yscale('log')  # Логарифмическая шкала по Y для лучшей видимости сходимости
    plt.legend()
    plt.savefig(f"{name}_convergence.png", dpi=300, bbox_inches='tight')
    plt.show()

    
if __name__ == '__main__':
    A1, B1 = -2, 3                   # Задание начальной и конечной точки по отрезку
    A2, B2 = -2, 2                   # Задание начальной и конечной точки по отрезку
    A3, B3 = -3, 3                   # Задание начальной и конечной точки по отрезку
    Sp1 = np.random.uniform(A1, B1)  # Задание стартовой позиции по оси абцисс
    Sp2 = np.random.uniform(A2, B2)  # Задание стартовой позиции по оси абцисс
    Sp3 = np.random.uniform(A3, B3)  # Задание стартовой позиции по оси абцисс

    # Оптимизация для разных функций
    F1, F1k, F1arr = Metod(FUNC1, A1, B1, Sp1)  
    F1C, F1kC, F1Carr = Metod(FUNC1C, A1, B1, Sp1)  
    F2, F2k, F2arr = Metod(FUNC2, A2, B2, Sp2) 
    F2C, F2kC, F2Carr = Metod(FUNC2C, A2, B2, Sp2)
    F3, F3k, F3arr = Metod(FUNC3, A3, B3, Sp3) 
    F3C, F3kC, F3Carr = Metod(FUNC3C, A3, B3, Sp3)

    # Сравнение результатов
    FunctionСompare("Квадратичная", FUNC1, F1, F1k, Sp1, FUNC1C, F1C, F1kC)
    FunctionСompare("Розенброк-подобная(1D)", FUNC2, F2, F2k, Sp2, FUNC2C, F2C, F2kC)
    FunctionСompare("Плато-функция", FUNC3, F3, F3k, Sp3, FUNC3C, F3C, F3kC)
    
    # Построение графиков
    FunctionSchedule(FUNC1, "Квадратичная", F1arr, F1Carr, A1, B1)
    FunctionSchedule(FUNC2, "Розенброк-подобная(1D)", F2arr, F2Carr, A2, B2)
    FunctionSchedule(FUNC3, "Плато-функция", F3arr, F3Carr, A3, B3)

    # Построение графиков сходимости
    plot_convergence_comparison(FUNC2, "Розенброк-подобная(1D)", F2arr, F2Carr)