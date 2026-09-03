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
from visualisation import ShowResults, MakeImage
from astropy.io import fits
from bin.common_functions import AB_mag



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
    results_clean = results.copy()
    #print(results_clean.to_csv('tab.csv'))
    if features is None:
        excluded = {
            "chi2",
            "AIC",
            "BIC",
            "cluster",
            "fitConverged",
            
        }

        features = [
            c for c in results_clean.columns
            if c not in excluded
        ]
    #print(results_clean.to_csv('tab.csv'))
    X = results_clean[features].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = db.fit_predict(X)

    clustered = results_clean.copy()
    clustered["cluster"] = labels
    #if 'disk_r_break' in clustered.columns:
    #    summary = (
    #    clustered
    #    .groupby("cluster")
    #    .agg(
    #        size=("cluster", "size"),
    #        chi2=("chi2", "min"),
    #        BIC=("BIC", "min"),
    #        r_break=("disk_r_break", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        h1=("disk_h1", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        h2=("disk_h2", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        r_e=("bulge_r_e", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        n=("bulge_n", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #    )
    #    .sort_values("size", ascending=False)
    #    )
    #else:
    #    summary = (
    #    clustered
    #    .groupby("cluster")
    #    .agg(
    #        size=("cluster", "size"),
    #        chi2=("chi2", "min"),
    #        BIC=("BIC", "min"),
    #        r_break1=("disk_r_break1", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        r_break2=("disk_r_break2", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        h1=("disk_h1", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        h2=("disk_h2", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        h3=("disk_h3", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        r_e=("bulge_r_e", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #        n=("bulge_n", lambda x: x.loc[clustered.loc[x.index, "chi2"].idxmin()]),
    #    )
    #    .sort_values("size", ascending=False)
    #    )

    #best = (
    #    clustered
    #    .sort_values("chi2")
    #    .groupby("cluster")
    #    .first()
    #)

    return clustered#, summary, best

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
    #print('df', df)
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


def run_clustering(dir_path, features=None, pattern='', verbouse=True):
    df = load_break_files_from_directory(dir_path,pattern=pattern, verbouse=verbouse)
    df = df.replace('nan', np.nan)
    df = df.dropna()
    print(df.to_csv('tab.csv'))
    df_clean = prepare_dataframe_for_clustering(df, verbouse=verbouse)


    if features is None:
        if 'double' in pattern:
            features = [
            "bulge_n",
            "bulge_r_e",
            "disk_h1",
            "disk_h2",
            "disk_h3",
            "disk_r_break1",
            "disk_r_break2"
            ]
        else:
            features = [
            "bulge_n",
            "bulge_r_e",
            "disk_h1",
            "disk_h2",
            "disk_r_break"
            ]

    clustered= cluster_results(df_clean, features=features)

    return clustered


import pandas as pd


def make_candidates(clustered):

    candidates = []

    n_total = len(clustered)
    print(clustered["AIC"])
    for cluster, group in clustered.groupby("cluster"):

        #if cluster == -1:
        #    continue
        #print(group["aic"])
        best = group.loc[group["aic"].idxmin()]

        row = best.to_dict()

        row["cluster"] = cluster
        row["size"] = len(group)
        row["fraction"] = len(group) / n_total

        candidates.append(row)
    
    return (
        pd.DataFrame(candidates)
        .sort_values("fraction", ascending=False)
        .reset_index(drop=True)
    )


def filter_candidates(candidates, delta_aic_limit=6):

    if len(candidates) == 1:
        return candidates.iloc[0]

    # Кластер с минимальным AIC
    best_aic_idx = candidates["AIC"].idxmin()

    if not('fraction' in candidates):
        return candidates.loc[0]
    # Самый большой кластер
    largest_idx = candidates["fraction"].idxmax()

    delta_aic = (
        candidates.loc[largest_idx, "AIC"]
        - candidates.loc[best_aic_idx, "AIC"]
    )

    if delta_aic <= delta_aic_limit:
        return candidates.loc[largest_idx]

    return candidates.loc[best_aic_idx]


def choose_model(image, exp_best, best_break, dir_path, best_double=None):

    
    log_image = np.log10(image)
    #log_image = np.nan_to_num(log_image)
    log_image[np.isinf(log_image)] = np.nan

    with open(f'{dir_path}/best_exp.dat', 'r') as ff:
        lines = ff.readlines()[:-8]

    model_desc = pyimfit.parse_config(lines)
    imfit = pyimfit.Imfit(model_desc)
    
    exp_image = imfit.getModelImage(shape=image.shape)
    log_exp_image = np.log10(exp_image)
    #log_exp_image = np.nan_to_num(log_exp_image)
    log_exp_image[np.isinf(log_exp_image)] = np.nan

    with open(f'{dir_path}/{best_break["filename"]}', 'r') as ff:
        lines = ff.readlines()[:-8]

    model_desc = pyimfit.parse_config(lines)
    imfit = pyimfit.Imfit(model_desc)
    
    best_break_image = imfit.getModelImage(shape=image.shape)
    log_best_break_image = np.log10(best_break_image)
    #log_best_image = np.nan_to_num(log_best_image)
    log_best_break_image[np.isinf(log_best_break_image)] = np.nan

    if best_double is None:
        best_double = best_break


    with open(f'{dir_path}/{best_double["filename"]}', 'r') as ff:
        lines = ff.readlines()[:-8]

    model_desc = pyimfit.parse_config(lines)
    imfit = pyimfit.Imfit(model_desc)
    
    best_double_image = imfit.getModelImage(shape=image.shape)
    log_best_double_image = np.log10(best_double_image)
    #log_best_image = np.nan_to_num(log_best_image)
    log_best_double_image[np.isinf(log_best_double_image)] = np.nan



    Lxi_exp   = np.nansum((log_exp_image - log_image)**2)  + 2*13 # 2*(11**2 + 11)/(500**2-11-1)
    Lxi_best_break = np.nansum((log_best_break_image - log_image)**2)  + 2*15 #(13**2 + 13)/(500**2-13-1)
    Lxi_best_double = np.nansum((log_best_double_image - log_image)**2)  + 2*17 #(13**2 + 13)/(500**2-13-1)

    print('Xi exp', Lxi_exp, exp_best["AIC"])
    print('Xi break', Lxi_best_break, best_break["AIC"])
    print('Xi double', Lxi_best_double, best_double["AIC"])
    #print('best_break', best_break)    
 
    if (exp_best["AIC"]-best_break["AIC"] < 2) and (Lxi_exp < Lxi_best_break):
        if (exp_best["AIC"]-best_double["AIC"] < 2) and (Lxi_exp < Lxi_best_double):
            file_best = f'{dir_path}/best_exp.dat'
            return file_best

   
    if (best_break["AIC"]-best_double["AIC"] < 2) and (Lxi_best_break < Lxi_best_double):
        file_best = f'{dir_path}/{best_break["filename"]}'
        return file_best

    else:
        file_best = f'{dir_path}/{best_double["filename"]}'      
        return file_best



def get_best_candidate(dir_path, pattern='', image=None):
    clustered = run_clustering(dir_path, pattern=pattern, verbouse=False)
    candidates = make_candidates(clustered)
    best_clustered = filter_candidates(candidates)

    #exp_best = parse_break_file(f'{dir_path}/best_exp.dat')
    #file_best = choose_model(image, exp_best, best_clustered, dir_path)

    return best_clustered

if __name__ == '__main__':
    import subprocess as sp
    from matplotlib import pyplot as plt
    import shutil
    #plt.figure()
    gals = sorted(set([a.split('_')[0] for a in os.listdir('../images_r')]))
    for gal in gals:
        rb_ = []
        h1_ = []
        h2_ = []
        re_ = []
        n_  = []
        sn = []
        for i in range(28,11,-1):
            file = f'{gal}_{i}'
            dir_path = f"fits/{file}"
            if not(os.path.exists(dir_path)):
                continue
            #print(file)
            if os.path.exists(f'pics_new/{file}.jpg'):
                continue
            clustered, summary, best = run_clustering(dir_path, verbouse=False)
            candidates = make_candidates(clustered)
            best_clustered = filter_candidates(candidates)

            exp_best = parse_break_file(f'{dir_path}/best_exp.dat')
            
            image = fits.getdata(f"../images_r/{file}_face.fits")
            print('file', file)
            file_best = choose_model(image, exp_best, best_clustered )
            #shutil.copy2(file_best, f'{dir_path}/best_clustered.dat')
            #continue 

            image = fits.getdata(f"../images_r/{file}_face.fits")
            with open(file_best, 'r') as ff:
                lines = ff.readlines()[:-8]
            
                    
            model_desc = pyimfit.parse_config(lines)
            print('lables',model_desc.functionLabelList())
            imfit = pyimfit.Imfit(model_desc)
            # Все публичные методы (без __магических__)
            #methods = [m for m in dir(model_desc) if not m.startswith('_')]
        
            #print(imfit.getModelDescription())
            zp = 8.9 -2.5*np.log10(AB_mag(1))
            makeimage = MakeImage(imfit)
            PA = makeimage.disk['parameters']['PA'][0]
            sr = ShowResults(image, imfit, zp = zp, PA=PA, maj_axis=True, scale=0.2)
            #sr.labels = ["bulge", "disk", "total"]
            sr.plot_cuts(f'pics_new/{file}.jpg', cuts=True)
            print(makeimage.bulge)
            if 'h1' in makeimage.disk['parameters']:
                h1_.append(makeimage.disk['parameters']["h1"][0])
                h2_.append(makeimage.disk['parameters']["h2"][0])
                rb_.append(makeimage.disk['parameters']["r_break"][0])
            else:
                h1_.append(makeimage.disk['parameters']["h"][0])
                h2_.append(makeimage.disk['parameters']["h"][0])
                rb_.append(float('nan'))
            

            re_.append(makeimage.bulge['parameters']['r_e'][0])
            n_.append(makeimage.bulge['parameters']['n'][0])
            sn.append(i)
                
        plt.figure() 
        plt.subplot(2,3,1)
        plt.title('r_break')
        plt.plot(sn, np.array(rb_)*.2 , '-o')
        plt.subplot(2,3,2)
        plt.title('h1')
        plt.plot(sn, np.array(h1_)*.2, '-o')
        plt.subplot(2,3,3)
        plt.title('h2')
        plt.plot(sn, np.array(h2_)*.2, '-o')
        plt.subplot(2,3,4)
        plt.title('r_e')
        plt.plot(sn, np.array(re_)*.2, '-o')
        plt.subplot(2,3,5)
        plt.title('n')
        plt.plot(sn, n_, '-o')
        plt.subplot(2,3,6)
        plt.title('lg h1/h2')
        plt.plot(sn, np.log(np.array(h1_)/np.array(h2_)), '-o')
        plt.savefig(f'evo/{gal}.jpg', format='jpg')
        

        
        
            #sp.call(f'cp {file_best} {dir_path}/best_clustered.dat', shell=True)
            #continue 
        continue

        labels = clustered['cluster'].values 
        labels = sorted(set(labels))
        k = labels[0]
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
        #methods = [m for m in dir(model_desc) if not m.startswith('_')]
    
        #print(imfit.getModelDescription())
        zp = 8.9 -2.5*np.log10(AB_mag(1))
        sr = ShowResults(image, imfit, zp = zp)
        #sr.labels = ["bulge", "disk", "total"]
        sr.plot_cuts(f'best_pics/{file}.jpg')

        #print(summary)
            
        
        #plt.savefig('r_evo.jpg', format='jpg')