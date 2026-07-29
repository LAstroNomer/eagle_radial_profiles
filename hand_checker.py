import matplotlib.pyplot as plt
import numpy as np
from bin.common_functions import calc_ith_isophote_radius, AB_mag, get_ell

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
profiles = os.listdir('profiles')
for gal in gals:
    for i in range(28,12,-1):


        clicked_x = []
        lines = []          # Здесь будем хранить объекты линий
        fit_success = None

        fname = f'{gal}_{i}'

        fname = f"{gal}_{i}"

        if fname in processed:
            print(f"Пропуск {fname}")
            continue

        
        profile_teg = f'{fname}_azim_0'
        profile_files = []
        for a in profiles:
            if profile_teg in a:
                profile_files.append('profiles/'+a)
        
        fig, ax = plt.subplots()
        
        sma_0, flux_0, flux_err_0 = np.loadtxt(f'profiles/{fname}_azim_0.txt', skiprows=1, unpack=True)
        zp = 8.9 -2.5*np.log10(AB_mag(1))
        ax.plot(sma_0*0.2, zp-2.5*np.log10(flux_0), '-', color='grey')

        for a in profile_files:
            if '_azim_0.txt' in a:
                continue
            key = a[:-4].split('_0_')[-1]
            sma_0_model, flux_0_model, flux_err_0_model = np.loadtxt(a, skiprows=1, unpack=True)
            ax.plot(sma_0_model*0.2, zp-2.5*np.log10(flux_0_model), '-', label=key)
        ax.invert_yaxis()
        ax.set_ylim(30,)



        def on_close(event):
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


        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)
        plt.suptitle(fname)
        plt.title(
            "ЛКМ — добавить метку | ПКМ — отменить последнюю | Y/N — оценка | S — вывести результат"
        )
        plt.legend()
        plt.show()
        if stop_processing:
            break

    if stop_processing:
        break

