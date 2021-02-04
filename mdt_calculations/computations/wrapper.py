import subprocess


def create_params(mss, domss, egm, dogm, ts, res, trunc):
    r"""Writes parameters to parameters.txt in correct format.

    Args:
        mss (str): name of mss.
        domss (int): max d/o of mss.
        egm (str): name of egm.
        dogm (int): max d/o of gravity model.
        ts (str): tide system of gravity model.
            Possible options are 'free', 'mean', 'zero'.
        res (int): reciprocal of resolution of required MDT.
        trunc (int): max truncation (d/o) or required MDT.
    """
    text = ("{}\n"
            "{}\n"
            "{}\n"
            "{}\n"
            "{}\n"
            "{}\n"
            "{}\n".format(mss, domss, egm, dogm, ts, res, trunc))

    file = open("parameters.txt", 'w')
    file.write(text)
    file.close()


def run_mdt_shell_script():
    r"""Runs the compute_mdt.sh shell script.
    """
    subprocess.call(['mdt_calculations\\computations\\compute_mdt.bat'])

def run_cs_shell_script():
    r"""Runs the compute_mdt.sh shell script.
    """
    subprocess.call(['mdt_calculations\\computations\\compute_cs.bat'])

def mdt_wrapper(mss="dtu15", domss=2190, egm="gtim5", dogm=280, ts="free", res=4, trunc=280):
    r"""Interfaces the fortran program to automatically create an mdt.

    The mdt is written to /data/res/<mss>_<egm>_do<dogm>_rr<res>.dat.
    Both <dogm> and <res> are parsed and written to 4 s.f., e.g. {:>04d}.

    Args:
        mss (str, optional): name of mss.
        domss (int, optional): max d/o of mss.
        egm (str, optional): name of egm.
        dogm (int, optional): max d/o of gravity model.
        ts (str, optional): tide system of gravity model.
            Possible options are 'free', 'mean', 'zero'.
        res (int, optional): reciprocal of resolution of required MDT.
        trunc (int, optional): max truncation (d/o) or required MDT.
    """
    create_params(mss, domss, egm, dogm, ts, res, trunc)
    run_mdt_shell_script()


def cs_wrapper():
    run_cs_shell_script()


def main():
    mdt_wrapper()
    cs_wrapper()


if __name__ == '__main__':
    main()
