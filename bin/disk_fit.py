# %load bin/disk_fit.py
from scipy.stats import chi2
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import  curve_fit
from matplotlib import pyplot as plt

def exp(r: np.ndarray, I0: float, h: float, *args):
    '''
    Диск без излома
    '''
    return I0 * np.exp(-r / h)


def one_break_cont(r: np.ndarray, I0: float, h1: float, h2: float, rb: float, alpha: float):
    k = alpha * (r - rb)
    res = np.zeros(len(r))
    ind = np.where(k >= 90)
    res[ind] = I0 * np.exp(-rb / h1) * np.exp(-(r[ind] - rb) / h2)
    ind = np.where(k < 90)
    res[ind] = I0 * np.exp(-r[ind] / h1) * ((1 + np.exp(k[ind])) ** (1 / alpha * (1 / h1 - 1 / h2)))
    return res


def multy_break_cont(r, I0, hs, rbs, alphas):
    k = []

    S = 1
    for i in range(len(rbs)):
        S *= (1 + np.exp(-alphas[i] * rbs[i])) ** (-1 / alphas[i] * (1 / hs[i] - 1 / hs[i + 1]))

    for i in range(len(rbs)):
        k.append(alphas[i] * (r - rbs[i]))
    # print(k)
    # plt.figure()
    # for i, k_ in enumerate(k):
    #    plt.plot(r, k_, label=i)
    # plt.gca().axhline(90)
    # plt.legend()
    # plt.show()

    res = np.zeros(len(r))

    ind = np.where(k[0] < 90)
    res[ind] = I0 * np.exp(-r[ind] / hs[0])
    for i in range(0, len(rbs)):
        res[ind] *= (1 + np.exp(alphas[i] * (r[ind] - rbs[i]))) ** (1 / alphas[i] * (1 / hs[i] - 1 / hs[i + 1]))

    for j in range(len(rbs)):
        # print('j', j)
        if j == (len(k) - 1):
            # print('ended', j)
            ind = np.where((k[j] >= 90))
        else:
            ind = np.where((k[j] >= 90) * (k[j + 1] < 90))

        # res[ind] = I0 * np.exp(-rbs[0]/hs[0])*np.exp(-(r[ind] - rbs[1])/hs[2])

        res[ind] = I0 * np.exp(-rbs[0] / hs[0]) * np.exp(-(r[ind] - rbs[j]) / hs[j + 1])
        for m in range(1, j + 1):
            #    print('m', m)
            res[ind] *= np.exp(-(rbs[m] - rbs[m - 1]) / hs[m])

        for i in range(j + 1, len(rbs)):
            #    print('i',i)
            res[ind] *= (1 + np.exp(alphas[i] * (r[ind] - rbs[i]))) ** (1 / alphas[i] * (1 / hs[i] - 1 / hs[i + 1]))

    return res * S


def main_fit_disk(sma: np.ndarray, I: np.ndarray, I_err: np.ndarray, rin: float = 0, rmax: float = 100, zp: float = 25.,
                  plot: bool = True, max_breaks: int = 2):
    ind = np.where(I > 0)
    sma = sma[ind]
    I = I[ind]
    I_err = I_err[ind]

    rs = main_find(sma, I, I_err, rin, zp=zp, rmax=rmax)
    rs1 = rs[1:-1]

    mu = zp - 2.5 * np.log10(I)
    mu_err = I_err / I * 2.5 / np.log(10)

    ind = np.where((sma > rin) * (sma < rmax))
    x = sma[ind]
    y = I[ind]
    yerr = I_err[ind]
    foo = interp1d(x, y)

    print('rs1', rs1)
    all_arrays = subsets(rs1)
    # print(all_arrays)
    p = None
    SSE2 = np.inf
    rs_ = None
    for rs1 in all_arrays:
        if len(rs1) > max_breaks:
            continue

        def foo_fix_rb(x, I0, *hs_alphas):
            hs = hs_alphas[0:len(rs1) + 1]
            alphas = hs_alphas[len(rs1) + 1:]
            return multy_break_cont(x, I0, hs, rs1, alphas)

        if len(rs1) > 0:
            I0, hs, rbs, alphas = disk_init_guess(x, foo, rs1)
            p0 = np.append(I0, abs(np.array(hs)))
            p0 = np.append(p0, alphas)
            #try:
            print('p0', p0)
            p_, covp_, info_dict, mesg, ier = curve_fit(foo_fix_rb, xdata=x, ydata=y, sigma=yerr, p0=p0,
                                                        bounds=(np.append([0] * (len(rs1) + 2), [0.9] * len(rs1)),
                                                                np.append([np.inf] * (len(rs1) + 2),
                                                                          [1] * len(rs1))), full_output=True)
            #except:
            #    print('Fall')
            #   continue
            hs = p_[1:len(rs1) + 2]

            if logistic_criteria_hs(hs):
                SSE_ = -2 * log_likelihood_heteroscedastic(-2.5 * np.log10(y), -2.5 * np.log10(foo_fix_rb(x, *p_)),
                                                           yerr / y * 2.5 / np.log(10)) + (len(p_) + len(rs1)) * np.log(
                    len(x))
            else:
                SSE_ = np.inf

            if SSE_ < SSE2:
                p = p_
                covp = covp_
                SSE2 = SSE_
                rs_ = rs1
        else:
            I0, hs, rbs, alphas = disk_init_guess(x, foo, rs1)
            try:
                p_, covp_, info_dict, mesg, ier = curve_fit(exp, xdata=x, ydata=y, sigma=yerr, p0=[I0, hs[0]],
                                                            full_output=True)
            except:
                continue

            SSE_ = -2 * log_likelihood_heteroscedastic(-2.5 * np.log10(y), -2.5 * np.log10(exp(x, *p_)),
                                                       yerr / y * 2.5 / np.log(10)) + (len(p_) + len(rs1)) * np.log(
                len(x))
            if SSE_ < SSE2:
                p = p_
                covp = covp_
                SSE2 = SSE_
                rs_ = []

    if rs_ is None:
        print('BAD FIT!!!')
        rs1 = []
        p = np.append(I0, hs)
        covp = np.zeros((3, 3))
    else:
        rs1 = rs_

    if len(rs1) > 0:
        def foo_fix_rb(x, I0, *hs_alphas):
            hs = hs_alphas[0:len(rs1) + 1]
            alphas = hs_alphas[len(rs1) + 1:]
            return multy_break_cont(x, I0, hs, rs1, alphas)

        I0_ = p[0]
        hs_ = p[1:len(rs1) + 2]
        alphas_ = p[len(rs1) + 2:]
        print('I0:', I0_)
        print('hs:', hs_)
        print('rbs:', rs1)
        print('alphas:', alphas_)

        if plot:
            plt.figure()
            # plt.title('%s %s' %(file, typ))

            plt.errorbar(sma, 8.9 - 2.5 * np.log10((I)), fmt='-', color='grey', alpha=0.5)
            ind = np.where((sma > rin))
            plt.errorbar(x, 8.9 - 2.5 * np.log10((y)), yerr=yerr / y, fmt='-', alpha=0.5)
            plt.plot(sma, 8.9 - 2.5 * np.log10((foo_fix_rb(sma, *p))), '-r')

            # plt.plot(r, -2.5*np.log10(exp(r,*p)))
            # plt.plot(r, -2.5*np.log10(one_break1(r,*p2)))
            for i in rs1:
                plt.gca().axvline(i)
            # print('rs', rs)
            # plt.gca().axvline(r27)
            # plt.gca().axhline(27, color='black')
            # plt.ylim(13,30)
            plt.gca().invert_yaxis()

            plt.show()
    else:
        I0_ = p[0]
        hs_ = [p[1]]

        print('I0:', I0_)
        print('hs:', hs_)
        print('rbs:', rs1)

        if plot:
            plt.figure()
            # plt.title('%s %s' %(file, typ))
            plt.errorbar(sma, zp - 2.5 * np.log10((I)), fmt='-', color='grey', alpha=0.5)
            ind = np.where((sma > rin))
            plt.errorbar(x, zp - 2.5 * np.log10((y)), yerr=yerr / abs(y), fmt='-', alpha=0.5)
            plt.plot(sma, zp - 2.5 * np.log10((exp(sma, *p))), '-r')

            # plt.plot(r, -2.5*np.log10(exp(r,*p)))
            # plt.plot(r, -2.5*np.log10(one_break1(r,*p2)))
            for i in rs1:
                plt.gca().axvline(i)
            # print('rs', rs)
            # plt.gca().axvline(r27)
            # plt.gca().axhline(27, color='black')
            # plt.ylim(25,30)
            plt.gca().invert_yaxis()

            plt.show()
    return I0_, hs_, rs1, alphas, covp


def log_likelihood_heteroscedastic(y, y_pred, sigma_vector):
    """
    Вычисляет логарифм правдоподобия для случая гетероскедастичности
    с известными стандартными отклонениями для каждого наблюдения

    Параметры:
    y - фактические значения (вектор)
    y_pred - предсказанные значения (вектор)
    sigma_vector - стандартные отклонения для каждого наблюдения (вектор)

    Возвращает:
    Логарифм правдоподобия
    """
    n = len(y)
    residuals = y - y_pred

    # Проверка размерностей
    if len(sigma_vector) != n:
        raise ValueError("Длина вектора sigma_vector должна совпадать с количеством наблюдений")

    # Компоненты логарифма правдоподобия
    constant_term = -np.sum(np.log(sigma_vector * np.sqrt(2 * np.pi)))
    exponent_term = -0.5 * np.sum((residuals / sigma_vector) ** 2)

    return constant_term + exponent_term


def one_step_disk_fit(r: np.ndarray, I: np.ndarray, I_err: np.array, alpha=0.0027, n_step=100, full_responce=False):
    '''
    The function investigate the part of galactic disk profile.
    The profile is aprroximated two different models simple and double-exponential galactic profile.
    Double-exponential profile is continuous with smooth-sharp parameter.
    This models are compared with LR-statistic. The H0 hypothesis is keep simple model.
    The H1 hypothesis is changing for alternative model.
    The critical p-vaule is 0.0027. This value correspond to 3 sigma threshold.
    If LR-statistic is less than 0.0027 the H0 hypothesis is true.

    Parametes:
    - r :: Radius of counters
    - I :: The intensivety of counters
    - I_err :: Absolute error of intensivity
    - alpha :: Critical p-value
    - n_step :: count of init gueses in brute force method

    Return:
    - p    :: parameters list
    - covp :: covariation matrix
    - key  :: key for suggestion of H0 hypothesis
    '''

    rmax = np.max(r)
    rmin = np.min(r)
    mu = -2.5 * np.log10(I)
    mu_err = I_err / I * 2.5 / np.log(10)
    foo = interp1d(r, mu)
    # Simple fit
    h0 = 2.5 * np.log10(np.e) * (rmax - rmin) / (max(mu) - min(mu))
    mu0 = np.min(mu) - 2.5 * np.log10(np.e) * rmin / h0
    I0 = 10 ** (-0.4 * mu0)

    p1, covp1, info_dict, mesg, ier = curve_fit(exp, xdata=r, ydata=I, sigma=I_err, p0=np.array([I0, h0]),
                                                full_output=True)
    llt_null = log_likelihood_heteroscedastic(mu, -2.5 * np.log10(exp(r, *p1)), mu_err)

    # Double exp fit with brute force
    p2 = None
    covp2 = None
    SSE2 = np.inf
    for i in np.linspace(rmin, rmax, n_step)[1:-1]:

        h01 = 2.5 * np.log10(np.e) * (i - rmin) / (foo(i) - min(mu))
        h02 = 2.5 * np.log10(np.e) * (rmax - i) / (max(mu - foo(i)))

        mu0 = foo(i) - 2.5 * np.log10(np.e) * i / h01
        I0 = 10 ** (-0.4 * mu0)

        try:
            p_, covp_, info_dict, mesg, ier = curve_fit(one_break_cont, xdata=r, ydata=I, sigma=I_err,
                                                        p0=np.array([I0, h01, h02, i, 0.95]),
                                                        bounds=([0, 0, 0, 0, 0.9],
                                                                [np.inf, np.inf, np.inf, np.inf, 1]), full_output=True)
        except:
            continue
        SSE_ = np.sum(info_dict['fvec'] ** 2)

        if SSE_ < SSE2:
            p2 = p_
            covp2 = covp_
            SSE2 = SSE_
    if full_responce:
        plt.figure()
        plt.errorbar(r, np.log10(I), yerr=I_err / I, alpha=0.1)
        plt.plot(r, np.log10(exp(r, *p1)), '-r')
        plt.plot(r, np.log10(one_break_cont(r, *p2)), '-', color='magenta')
        plt.show()

    try:
        llt_alt = log_likelihood_heteroscedastic(mu, -2.5 * np.log10(one_break_cont(r, *p2)), mu_err)
    except:
        print('No Fit Break')
        llt_alt = -np.inf
    LR_stat = 2 * (llt_alt - llt_null)
    try:
        df = len(p2) - len(p1)
    except:
        df = 2

    p_value = 1 - chi2.cdf(LR_stat, df)
    print('pvalue', p_value)
    if full_responce:
        if p_value < alpha:
            print("\nВывод: Отвергаем нулевую модель - альтернативная модель значимо лучше")
            if logistic_criteria(p2):
                print('\n Излом прошёл проверку')
                print(p2[1] / p2[2])
                return p2, covp2, False
            else:
                print('\n Излом нефизичен')
                return p1, covp1, True

        else:
            print("\nВывод: Нет оснований отвергать нулевую модель")
        return p1, covp1, True  # p2, covp2, LR_stat, p_value
    else:
        if p_value < alpha:
            if logistic_criteria(p2):
                return p2, covp2, False
        return p1, covp1, True


def disk_init_guess(x, foo, rbs):
    rmin = np.min(x)
    rmax = np.max(x)
    rs = np.append(rmin, rbs)
    rs = np.append(rs, rmax)
    # print('rs', rs)
    hs = []
    I0 = 0

    for i in range(len(rs) - 1):
        hs.append((rs[i + 1] - rs[i]) / (np.log(foo(rs[i])) - np.log(foo(rs[i + 1]))))

    I0 = foo(rs[1]) * np.exp(rs[1] / hs[0])
    alphas = [1] * len(rbs)
    return I0, hs, rbs, alphas


def logistic_criteria(p2):
    I0, h1, h2, rb, _ = p2

    sb = h1 / h2

    if (sb < 1.1) and (sb > 1 / 1.1):
        return False
    return True


def logistic_criteria_hs(hs):
    for i in range(len(hs) - 1):
        sb = hs[i] / hs[i + 1]

        if (sb < 1.1) and (sb > 1 / 1.1):
            return False
    return True


def subsets(new_array):
    power_set = [[]]
    for x in new_array:
        # print(power_set)
        for i in range(len(power_set)):
            tmp_list = power_set[i].copy()
            tmp_list.append(x)
            power_set.append(tmp_list)
    return power_set


def F_go(r: np.ndarray, I: np.ndarray, Ie0: np.ndarray, rs: list, step_i=5, step_pix=20, level=0, maxlevel=2):
    '''
    Iterrative find of breaks with using LR-statistic and logistic filter

    '''
    print('level', level)
    print(rs)
    if level > maxlevel:
        return rs
    level += 1
    for i in np.arange(len(rs) - 1):

        ind = np.where((r >= rs[i]) * (r <= rs[i + 1]))
        r_ = r[ind]
        I_ = I[ind]
        if Ie0 is None:
            Ie = np.sqrt(I)
        else:
            Ie_ = Ie0[ind]

        if len(r_) < step_i:
            continue
        try:
            p, covp, key = one_step_disk_fit(r_, I_, Ie_, full_responce=False)
        except:
            continue
        # print('p', p)
        if (((p[-2] - np.min(r_)) < step_pix) or ((-p[-2] + np.max(r_)) < step_pix)) and (len(rs) > 2):
            continue

        if key:

            continue
        else:
            rs.append(p[-2])
            rs = sorted(rs)

        rs = F_go(r_, I_, Ie_, rs, level=level)
    return rs


def main_find(sma: np.ndarray, I: np.ndarray, I_err: np.ndarray, rin: float = 0, zp: float = 25.0,
              max_iter: int = 2, rmax=100):
    ind = np.where((sma > rin) * (sma < rmax))
    r = sma[ind]
    I = I[ind]
    I_err = I_err[ind]

    rmin = np.min(r)
    rmax = np.max(r)
    rs = [rmin, rmax]
    print('rs', rs)
    rs = F_go(r, I, I_err, rs, level=0, maxlevel=max_iter)

    return rs