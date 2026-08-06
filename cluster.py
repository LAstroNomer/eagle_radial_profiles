from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
# Настройка отображения pandas в терминале
pd.set_option('display.max_columns', None)  # Показывать все колонки
pd.set_option('display.max_rows', None)     # Показывать все строки
pd.set_option('display.width', None)        # Автоматическая ширина
pd.set_option('display.max_colwidth', None) # Не обрезать содержимое колонок
pd.set_option('display.expand_frame_repr', False)  # Не переносить на новую строку
import os
from pathlib import Path
import re
import numpy as np
import pyimfit
from visualisation import ShowResults
from astropy.io import fits
from bin.common_functions import AB_mag

def parse_config_fixed( lines: List[str], labels=None) -> ModelDescription:
    """
    Parse an Imfit model configuration from a list of strings.

    Parameters
    ----------
    lines : list of str
        String representantion of Imfit model configuration.

    Returns
    -------
    model : :class:`~imfit.ModelDescription`
        A model description object.

    See also
    --------
    parse_config_file
    """
    lines = clean_lines(lines)

    model = ModelDescription()

    block_start = 0
    functionBlock_id = 0   # number of current function set
    for i in range(block_start, len(lines)):
        if lines[i].startswith(x0_str):
            if block_start == 0:
                options = read_options(lines[block_start:i])
                model.options.update(options)
            else:
                # possible auto-label-generation code
                funcSetLabel = "fs{0:d}".format(functionBlock_id)
                funcSetLabel = ""
                model.addFunctionSet(read_function_set(funcSetLabel, lines[block_start:i]))
                functionBlock_id += 1
            block_start = i
    funcSetLabel = "fs{0:d}".format(functionBlock_id)
    funcSetLabel = ""
    model.addFunctionSet(read_function_set(funcSetLabel, lines[block_start:i + 1]))
    return model


def cluster_results(
    results: pd.DataFrame,
    features=None,
    eps=0.5,
    min_samples=3,
):
    """
    Кластеризация результатов multistart.

    Parameters
    ----------
    results : DataFrame
        Таблица результатов.

    features : list
        Какие параметры использовать для кластеризации.
        Если None, используются все столбцы, кроме статистик.

    eps : float
        Радиус DBSCAN после стандартизации.

    min_samples : int
        Минимальное число точек в кластере.

    Returns
    -------
    clustered : DataFrame
        Исходная таблица с новым столбцом 'cluster'.

    summary : DataFrame
        Статистика по каждому кластеру.

    best : DataFrame
        Лучший фит (минимальный chi2) в каждом кластере.
    """

    if features is None:
        excluded = {
            "chi2",
            "AIC",
            "BIC",
            "cluster",
            "fitConverged",
            
        }

        features = [
            c for c in results.columns
            if c not in excluded
        ]

    X = results[features].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = db.fit_predict(X)

    clustered = results.copy()
    clustered["cluster"] = labels

    summary = (
        clustered
        .groupby("cluster")
        .agg(
            size=("cluster", "size"),
            chi2=("chi2", "min"),
            BIC=("BIC", "min"),
            r_break=("disk_r_break", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
            h1=("disk_h1", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
            h2=("disk_h2", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
            r_e=("bulge_r_e", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
            n=("bulge_n", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
        )
        .sort_values("size", ascending=False)
    )

    best = (
        clustered
        .sort_values("chi2")
        .groupby("cluster")
        .first()
    )

    return clustered, summary, best

def parse_break_file(filepath):
    """
    Парсинг одного файла break_*.dat от pyimfit.
    
    Parameters
    ----------
    filepath : str
        Путь к файлу.
    
    Returns
    -------
    dict : словарь с параметрами и статистиками.
    """
    params = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    current_function = None
    current_label = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Определяем текущую функцию и её метку
        if line.upper().startswith('FUNCTION'):
            func_match = re.search(r'FUNCTION\s+(\w+)', line, re.IGNORECASE)
            label_match = re.search(r'LABEL\s+(\w+)', line, re.IGNORECASE)
            
            if func_match:
                current_function = func_match.group(1)
            if label_match:
                current_label = label_match.group(1)
            continue
        
        # Парсим параметры функций (с погрешностями в комментариях)
        # Формат: "I_e  8.861445525221225  # +/- 0.5660627957355975"
        param_match = re.match(r'(\w+)\s+([\d.eE+\-]+)\s*#\s*\+\/-\s*([\d.eE+\-]+)', line)
        if param_match and current_function and current_label:
            param_name = param_match.group(1)
            param_value = float(param_match.group(2))
            param_error = float(param_match.group(3))
            
            # Формируем имя параметра: label_param
            full_param_name = f"{current_label}_{param_name}"
            params[full_param_name] = param_value
            params[f"{full_param_name}_err"] = param_error
            continue
        
        # Парсим общие параметры модели (X0, Y0) с погрешностями
        common_match = re.match(r'(X0|Y0)\s+([\d.eE+\-]+)\s*#\s*\+\/-\s*([\d.eE+\-]+)', line, re.IGNORECASE)
        if common_match:
            param_name = common_match.group(1)
            param_value = float(common_match.group(2))
            param_error = float(common_match.group(3))
            params[param_name] = param_value
            params[f"{param_name}_err"] = param_error
            continue
        
        # Парсим статистики фита (строки вида "solverName LM", "fitConverged True")
        # Ищем строки без комментариев с ключом и значением
        stat_match = re.match(r'(\w+)\s+([\w.+\-]+)\s*$', line)
        if stat_match and '#' not in line:
            key = stat_match.group(1)
            value = stat_match.group(2)
            
            try:
                # Пробуем преобразовать в число
                if '.' in value or 'e' in value.lower():
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                # Для булевых значений и строк
                if value == 'True':
                    params[key] = True
                elif value == 'False':
                    params[key] = False
                else:
                    params[key] = value
            continue
        
        # Парсим статистики с числами (fitStat, AIC, BIC)
        stat_num_match = re.match(r'(\w+)\s+([\d.eE+\-]+)\s*$', line)
        if stat_num_match and '#' not in line:
            key = stat_num_match.group(1)
            try:
                value = float(stat_num_match.group(2))
                params[key] = value
            except ValueError:
                pass
    
    # Добавляем имя файла для идентификации
    params['filename'] = os.path.basename(filepath)
    
    # Переименовываем статистики в стандартные названия
    stats_mapping = {
        'fitStat': 'fitStat',
        'fitStatReduced': 'chi2',
        'aic': 'AIC',
        'bic': 'BIC',
        'fitConverged': 'fitConverged',
        'nIter': 'nIter',
        'nFuncEvals': 'nFuncEvals',
        'solverName': 'solverName',
    }
    
    # Создаем копии с правильными именами
    for old_name, new_name in stats_mapping.items():
        if old_name in params:
            params[new_name] = params[old_name]
    
    return params


def load_break_files_from_directory(directory, pattern="break_*.dat", verbouse=True):
    """
    Чтение всех файлов break_*.dat из папки и создание DataFrame.
    
    Parameters
    ----------
    directory : str
        Путь к папке с файлами.
    pattern : str
        Шаблон имени файлов (по умолчанию "break_*.dat").
    
    Returns
    -------
    pd.DataFrame
        Таблица со всеми результатами, готовая для cluster_results().
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Папка не найдена: {directory}")
    
    # Собираем все файлы по шаблону
    file_list = sorted(directory.glob(pattern))
    
    if not file_list:
        raise FileNotFoundError(f"Файлы {pattern} не найдены в {directory}")
    
    if verbouse:
        print(f"Найдено файлов: {len(file_list)}")
    
    # Парсим каждый файл
    results_list = []
    for filepath in file_list:
        try:
            params = parse_break_file(filepath)
            results_list.append(params)
        except Exception as e:
            print(f"Ошибка при чтении {filepath}: {e}")
            continue
    
    # Создаем DataFrame
    df = pd.DataFrame(results_list)
    
    # Переименовываем статистики для совместимости с cluster_results
    rename_map = {
        'fitStat': 'chi2',
        'aic': 'AIC',
        'bic': 'BIC',
        'fitConverged': 'fitConverged',
    }
    
    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]
    
    # Убеждаемся, что fitConverged булевый
    if 'fitConverged' in df.columns:
        df['fitConverged'] = df['fitConverged'].astype(bool)
    
    # Отфильтровываем только сошедшиеся решения
    converged_count = df['fitConverged'].sum() if 'fitConverged' in df.columns else len(df)
    if verbouse:
        print(f"Сошедшихся решений: {converged_count} из {len(df)}")
    
    return df


def prepare_dataframe_for_clustering(df, keep_converged_only=True, verbouse=True):
    """
    Подготовка DataFrame для кластеризации.
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame от load_break_files_from_directory().
    keep_converged_only : bool
        Оставить только сошедшиеся решения.
    
    Returns
    -------
    pd.DataFrame
        Подготовленный DataFrame.
    """
    df_prepared = df.copy()
    
    # Оставляем только сошедшиеся
    if keep_converged_only and 'fitConverged' in df_prepared.columns:
        df_prepared = df_prepared[df_prepared['fitConverged']].copy()
        if verbouse:
            print(f"Оставлено {len(df_prepared)} сошедшихся решений для кластеризации")
    
    return df_prepared


def run_clustering(dir_path, features=None, verbouse=True):
    df = load_break_files_from_directory(dir_path, verbouse=verbouse)
    df_clean = prepare_dataframe_for_clustering(df, verbouse=verbouse)


    if features is None:
        features = [
            "bulge_n",
            "bulge_r_e",
            "disk_h1",
            "disk_h2",
            "disk_r_break"
        ]

    clustered, summary, best = cluster_results(df_clean, features=features)

    return clustered, summary, best


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    #plt.figure()
    for i in range(28,11,-1):
        file = f'746518_{i}'
        dir_path = f"fits/{file}"
        if file in ["746518_18"]:
            continue

        clustered, summary, best = run_clustering(dir_path, verbouse=False)
        labels = clustered['cluster'].values 
        labels = sorted(set(labels))
        k = labels[0]
        exp_best = parse_break_file(f'{dir_path}/best_exp.dat')
        print(summary)
        print(i)
        for k_ in np.append(np.arange(len(labels)-1), -1):
            if k_ in labels:
                if exp_best["BIC"]-best["BIC"][np.int64(k_)] >= 10:
                    k = np.int64(k_)
                    break
      

       
        break_best = parse_break_file(f'{dir_path}/best_break.dat')

        #print(exp_best["BIC"])
        #print(clustered)
        #print(summary)
        #print(i, best["filename"][0], best["disk_alpha"][0], best["BIC"][0], break_best["BIC"]-best["BIC"][0], exp_best["BIC"]-best["BIC"][0])
        if exp_best["BIC"]-best["BIC"][k] < 10:
            #plt.plot(i, exp_best["bulge_n"], 'or')
            file_best = f'{dir_path}/best_exp.dat'
        else:
            file_best = f'{dir_path}/{best["filename"][k]}'
            #plt.plot(i, best["disk_r_break"][k], 'ob')
            #plt.plot(i, best["disk_h2"][0]*0.2, 'og')
            #plt.plot(i, break_best["disk_r_break"], 'or')
        image = fits.getdata(f"../images_r/{file}_face.fits")
        with open(file_best, 'r') as ff:
            lines = ff.readlines()[:-8]

       
        model_desc = pyimfit.parse_config(lines)
        print('lables',model_desc.functionLabelList())
        imfit = pyimfit.Imfit(model_desc)
        # Все публичные методы (без __магических__)
        methods = [m for m in dir(model_desc) if not m.startswith('_')]
       
        print(imfit.getModelDescription())
        zp = 8.9 -2.5*np.log10(AB_mag(1))
        sr = ShowResults(image, imfit, zp = zp)
        sr.labels = ["bulge", "disk", "total"]
        sr.plot_cuts(f'best_pics/{file}.jpg')

        #print(summary)
        
    
    #plt.savefig('r_evo.jpg', format='jpg')