from fbs_runtime.application_context.PyQt5 import ApplicationContext
from PyQt5.QtWidgets import QMainWindow

import sys

from gui import SelWindow


class AppContext(ApplicationContext):
    """
    A Wrapper object to run the PyQt Application Instance.
    If fbs is to be used, then this class will be a subclass.
    """

    def run(self):
        self.window = SelWindow(ctx=self)
        self.window.setWindowTitle("shoecomp v0.0.0")
        self.window.show()
        return self.app.exec_()


if __name__ == "__main__":
    appctxt = AppContext()  # 1. Instantiate ApplicationContext
    exit_code = appctxt.run()  # 2. Invoke appctxt.app.exec_()
    sys.exit(exit_code)
