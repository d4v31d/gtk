
"""

  ## This script creates a Python 3.10 virtual environment in the current directory.
  py -3.10 -m venv .venv310

  ## funkar inte med python på Windows
  python -3.10 -m venv .venv310

"""

import os
import sys
import time


def create_header():
    """Create a simple header, just 80 '-'."""
    return "-" * (80)

def create_date_header():
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    header_pf = f"---- {ts} "
    return f"---- {ts} " + "-" * (80 - len(header_pf))

def create_subject_header(subject, date=False):
    if date is False:
        header_pf = f"---- {subject} "
    else:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        header_pf = f"---- {ts}  {subject} "

    header = header_pf + "-" * (80 - len(header_pf))
    return header



if __name__ == "__main__":
    if sys.version_info < (3, 10):
        print("This script requires Python 3.10 or higher.")
        sys.exit(1)
    
    venv_path = os.path.join(os.getcwd(), '.venv310')
    
    if not os.path.exists(venv_path):
        os.system(f"{sys.executable} -m venv {venv_path}")
    
    print()
    print(create_header())
    print(f"Virtual environment created at: {venv_path}")
    print("To activate the virtual environment, run:")
    print(f"source {venv_path}/bin/activate" if os.name != 'nt' else f"{venv_path}\\Scripts\\activate.bat")

    print()
    print(create_header())
    print("If you need to upgrade pip, you can run:")
    print(f"{venv_path}/bin/pip install --upgrade pip" if os.name != 'nt' else f"{venv_path}\\Scripts\\pip install --upgrade pip")
    
    print()
    print(create_header())
    print("Make sure to check the installed packages with:")
    print(f"{venv_path}/bin/pip list" if os.name != 'nt' else f"{venv_path}\\Scripts\\pip list")

    print()
    print(create_header())
    print("You can also create a requirements.txt file with:")
    print(f"{venv_path}/bin/pip freeze > requirements.txt" if os.name != 'nt' else f"{venv_path}\\Scripts\\pip freeze > requirements.txt")
    

    print()
    print(create_header())
    print("You can check your current Python version with:")
    print(f"{sys.executable} --version")
    print("If you need to install additional packages, use:")
    print(f"{venv_path}/bin/pip install <package_name>" if os.name != 'nt' else f"{venv_path}\\Scripts\\pip install <package_name>")


    print()
    print(create_header())
    print("You can do this by running:")
    print(f"{venv_path}/bin/pip list --outdated" if os.name != 'nt' else f"{venv_path}\\Scripts\\pip list --outdated")

    print()
    print(create_header())
    print("To upgrade a specific package, use:")
    print(f"{venv_path}/bin/pip install --upgrade <package_name>" if os.name != 'nt' else f"{venv_path}\\Scripts\\pip install --upgrade <package_name>")
    
    print()
    print(create_header())
    print("If you need to remove a package, you can use:")
    print(f"{venv_path}/bin/pip uninstall <package_name>" if os.name != 'nt' else f"{venv_path}\\Scripts\\pip uninstall <package_name>")

    print(create_header())
    print("You can run your Python scripts in the virtual environment using:")
    print(f"{venv_path}/bin/python <script_name>.py" if os.name != 'nt' else f"{venv_path}\\Scripts\\python <script_name>.py")
    

    print()
    print(create_header())
    print("If you need to share your project, consider including a requirements.txt file.")
    print("This will allow others to easily set up the same environment using:")
    print(f"{venv_path}/bin/pip install -r requirements.txt" if os.name != 'nt' else f"{venv_path}\\Scripts\\pip install -r requirements.txt")


    print()
    print(create_header())
    print("To deactivate the virtual environment, simply run 'deactivate' in your terminal.")


    print()
    print(create_header())
    print("To ensure your virtual environment is working correctly, you can run:")
    print(f"{venv_path}/bin/python -m venv --clear" if os.name != 'nt' else f"{venv_path}\\Scripts\\python -m venv --clear")

    print()
    print(create_header())
    print("Happy coding with Python 3.10!")

print()
print()
print()

print("create_header()")
print(create_header())
print()
print()

print("create_subject_header(\"important\")")
print(create_subject_header("important"))
print()
print()

print("create_subject_header(\"important\",True)")
print(create_subject_header("important",True))
print()
print()

print("create_date_header()")
print(create_date_header())
print()
print()



