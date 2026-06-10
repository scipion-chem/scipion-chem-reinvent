# **************************************************************************
# *
# * Authors:     Izana Alcalde (izana.alcalde@alumnos.upm.es)
# *
# * Universidad Politécnica de Madrid
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

import os
import pyworkflow.utils as pwutils
import pwem

from reinvent.constants import *

__version__ = "1.0"  # plugin version
_logo = "icon.png"
_references = ['you2019']


class Plugin(pwem.Plugin):
    _homeVar = REINVENT_HOME
    _pathVars = [REINVENT_HOME]
    _url = "https://github.com/scipion-em/scipion-chem-reinvent"
    _supportedVersions = [V1]  # binary version

    @classmethod
    def _defineVariables(cls):
        ENV_NAME = getReinventEnvName(V1)
        cls._defineVar(REINVENT_BINARY, "reinvent")
        cls._defineEmVar(REINVENT_HOME, ENV_NAME)
        cls._defineVar(REINVENT_ENV_ACT, f"conda activate {ENV_NAME}")

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch my program. """
        environ = pwutils.Environ(os.environ)

        # ...

        return environ

    @classmethod
    def getDependencies(cls):
        """ Return a list of dependencies. """
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = []
        if not condaActivationCmd:
            neededProgs.append('conda')
        return neededProgs

    @classmethod
    def defineBinaries(cls, env):
        for version in [V1]:
            cls.addReinventPackage(env, version,
                                   default=(version==REINVENT_DEF_VER))


    @classmethod
    def addReinventPackage(cls, env, version, default=False):
        ENV_NAME = getReinventEnvName(version)

        REINVENT_INSTALLED = 'reinvent_installed'
        PRIORS_INSTALLED = 'priors_installed'

        installCmd = [cls.getCondaActivationCmd(),
                      f'conda create -y -n {ENV_NAME} python=3.10 &&',
                      f'conda activate {ENV_NAME} &&',
                      ' git clone https://github.com/MolecularAI/REINVENT4.git --depth 1 . &&',
                      ' python install.py cpu &&',
                      ' pip install rdkit toml',
                      f'&& touch {REINVENT_INSTALLED}'
                      ]

        scriptPath = os.path.join(os.path.dirname(__file__), 'priors.py')
        installPriors = [cls.getCondaActivationCmd(),
                         f'conda activate {ENV_NAME} &&',
                         f'python3 {scriptPath} &&',
                         f'touch {PRIORS_INSTALLED}'
                        ]

        reinventCommands = [
            (" ".join(installCmd), REINVENT_INSTALLED),
            (" ".join(installPriors), PRIORS_INSTALLED)
        ]

        envPath = os.environ.get('PATH', "")
        installEnvVars = {'PATH': envPath} if envPath else None
        
        env.addPackage('reinvent', version=version,
                       tar='void.tgz',
                       commands=reinventCommands,
                       neededProgs=cls.getDependencies(),
                       default=default,
                       vars=installEnvVars)

    @classmethod
    def getProgram(cls, program=None):
        return cls.getVar(REINVENT_BINARY)

    @classmethod
    def getEnvActivation(cls):
        return cls.getVar(REINVENT_ENV_ACT)

    @classmethod
    def getSoftwarePath(cls):
        return cls.getVar(REINVENT_HOME)

    @classmethod
    def getPriorPath(cls, *args):
        return os.path.join(cls.getSoftwarePath(),'priors',*args)