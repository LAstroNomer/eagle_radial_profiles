import matplotlib.pyplot as plt
import numpy as np
from bin.common_functions import calc_ith_isophote_radius, AB_mag, get_ell
from bin.imfit_fit import parse_imfit_output

import csv
import os
stop_processing = False

processed = set()

if os.path.exists("results.csv"):
    with open("results.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed.add(row["name"])

gals = sorted(list(set([a.split('_')[0] for a in os.listdir('../2d_results')])))
list_models = np.array(os.listdir('../2d_results'))

profiles = os.listdir('profiles')

fit_result = dict()
for gal in gals:
    fit_result[gal] = dict()
    fit_result[gal]['rb1'] = []
    fit_result[gal]['rb2'] = []
    fit_result[gal]['h1']  = []
    fit_result[gal]['h2']  = []
    fit_result[gal]['h3']  = []

    for i in range(12,29,1):
        clicked_x = []
        lines = []          # Здесь будем хранить объекты линий
        fit_success = None

        fname = f'{gal}_{i}'
        if not(os.path.exists(f'pics/{fname}.jpg')):
            fit_result[gal]['rb1'].append(float('nan'))
            fit_result[gal]['rb2'].append(float('nan'))
            fit_result[gal]['h2'].append(float('nan'))
            fit_result[gal]['h3'].append(float('nan'))
            fit_result[gal]['h1'].append(float('nan'))
            continue
        for a in list_models:
            if fname in a:
                tmp_model = a
                break

        model = parse_imfit_output(f'../2d_results/{tmp_model}')
        params = model['functions'][-1]['parameters']
        #print(params)
        if not('h1' in params):
            fit_result[gal]['rb1'].append(float('nan'))
            fit_result[gal]['rb2'].append(float('nan'))
            fit_result[gal]['h2'].append(float('nan'))
            fit_result[gal]['h3'].append(float('nan'))

            fit_result[gal]['h1'].append(params['h']['value'])
        elif 'r_break' in params:
            fit_result[gal]['rb2'].append(float('nan'))
            fit_result[gal]['h3'].append(float('nan'))

            fit_result[gal]['h1'].append(params['h1']['value'])
            fit_result[gal]['rb1'].append(params['r_break']['value'])
            fit_result[gal]['h2'].append(params['h2']['value'])
        else:


            fit_result[gal]['h1'].append(params['h1']['value'])
            fit_result[gal]['rb1'].append(params['r_break1']['value'])
            fit_result[gal]['h2'].append(params['h2']['value'])
            fit_result[gal]['rb2'].append(params['r_break2']['value'])
            fit_result[gal]['h3'].append(params['h3']['value'])

for gal in gals:
    for i in range(12,29,1):
        

        clicked_x = []
        lines = []          # Здесь будем хранить объекты линий
        fit_success = None

        fname = f'{gal}_{i}'
        if not(os.path.exists(f'pics/{fname}.jpg')):
            continue
        for a in list_models:
            if fname in a:
                tmp_model = a
                break

        model = parse_imfit_output(f'../2d_results/{tmp_model}')
        rbs = []
        params = model['functions'][-1]['parameters']
        if 'r_break' in params:
            rbs = [params['r_break']['value']]
        elif 'r_break1' in params:
            rbs =[
                 params['r_break1']['value'],
                 params['r_break2']['value']
            ]



        if fname in processed:
            print(f"Пропуск {fname}")
            continue

        
        profile_teg = f'{fname}_azim_0'
        profile_files = []
        for a in profiles:
            if profile_teg in a:
                profile_files.append('profiles/'+a)
        
        fig, axs = plt.subplots(1,3, figsize=(15,10))
        ax = axs[0]
        sma_0, flux_0, flux_err_0 = np.loadtxt(f'profiles/{fname}_azim_0.txt', skiprows=1, unpack=True)
        zp = 8.9 -2.5*np.log10(AB_mag(1))
        ax.plot(sma_0*0.2, zp-2.5*np.log10(flux_0), '-', color='grey')

        for a in profile_files:
            if '_azim_0.txt' in a:
                continue
            key = a[:-4].split('_0_')[-1]
            sma_0_model, flux_0_model, flux_err_0_model = np.loadtxt(a, skiprows=1, unpack=True)
            ax.plot(sma_0_model*0.2, zp-2.5*np.log10(flux_0_model), '-', label=key)
        for rb in rbs:
            ax.axvline(rb*0.2, ls='--', color='magenta')
        ax.invert_yaxis()
        ax.set_ylim(30,)

        ax.axhline(27, ls='--', color='red')

        def on_close(event):
            if stop_processing:
                return
            
            filename = "results.csv"

            file_exists = os.path.exists(filename)

            with open(filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(["name", "fit_success", "clicked_x"])

                writer.writerow([
                    fname,
                    fit_success,
                    ";".join(map(str, clicked_x))
                ])
            processed.add(fname)
            print("Результат сохранен.")

        def on_click(event):
            if event.inaxes != ax:
                return

            # Левая кнопка — добавить метку
            if event.button == 1:
                clicked_x.append(event.xdata)

                line = ax.axvline(event.xdata,
                                color="red",
                                linestyle="--",
                                linewidth=1.5)
                lines.append(line)

                print(f"Добавлена метка: {event.xdata:.4f}")

            # Правая кнопка — отменить последнюю
            elif event.button == 3:
                if clicked_x:
                    removed_x = clicked_x.pop()

                    line = lines.pop()
                    line.remove()

                    print(f"Удалена метка: {removed_x:.4f}")
                else:
                    print("Нет меток для удаления.")

            fig.canvas.draw_idle()

        def on_key(event):
            global fit_success, stop_processing

            if event.key == "y":
                fit_success = True
                print("Фит успешный")

            elif event.key == "n":
                fit_success = False
                print("Фит НЕ успешный")

            elif event.key == "s":
                print("\n------")
                print("Флаг:", fit_success)
                print("Метки:", clicked_x)

            elif event.key == "escape":          # аварийный выход
                stop_processing = True
                plt.close(fig)

        ax1 = axs[1]
        rb1 = fit_result[gal]['rb1']
        rb2 = fit_result[gal]['rb2']

        h1 = fit_result[gal]['h1']
        h2 = fit_result[gal]['h2']
        h3 = fit_result[gal]['h3']


        ax1.plot(np.arange(12,29,1), np.array(rb1)*0.2, '-o', color='orange')
        ax1.plot(np.arange(12,29,1), np.array(rb2)*0.2, '-o', color='blue')
        ax1.axvline(i, ls='--', color='red')
        ax1.set_xlim(12,28)

        ax2 = axs[2]
        ax2.plot(np.arange(12,29,1), np.array(h1)*0.2, '-o', label='h1')
        ax2.plot(np.arange(12,29,1), np.array(h2)*0.2, '-o', label='h2')
        ax2.plot(np.arange(12,29,1), np.array(h3)*0.2, '-o', label='h3')
        
        ax2.axvline(i, ls='--', color='red')
        ax2.set_xlim(12,28)


        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)
        plt.suptitle(fname)
        plt.title(
            "ЛКМ — добавить метку | ПКМ — отменить последнюю | Y/N — оценка | S — вывести результат"
        )
        ax.legend()
        ax2.legend()
        plt.show()
        if stop_processing:
            break

    if stop_processing:
        break

