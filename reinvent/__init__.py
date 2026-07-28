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
import pwchem

from scipion.install.funcs import InstallHelper

from reinvent.constants import *

__version__ = "1.0"  # plugin version
_logo = "icon.png"
_references = ['Loeffler2024']


class Plugin(pwchem.Plugin):
    _homeVar = REINVENT_HOME
    _pathVars = [REINVENT_HOME]
    _url = "https://github.com/MolecularAI/REINVENT4"
    _supportedVersions = [V1]

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(REINVENT_BINARY, "reinvent")
        cls._defineEmVar(REINVENT_DIC['home'], cls.getEnvName(REINVENT_DIC))

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch my program. """
        environ = pwutils.Environ(os.environ)
        return environ

    @classmethod
    def getPluginHome(cls, path=""):
        """ Return a path inside this plugin's source directory. """
        import reinvent
        return os.path.join(os.path.split(reinvent.__file__)[0], path)

    @classmethod
    def defineBinaries(cls, env):
        cls.addReinventPackage(env, default=True)

    @classmethod
    def addReinventPackage(cls, env, default=True):
        # Instantiating install helper
        installer = InstallHelper(REINVENT_DIC['name'], packageHome=cls.getVar(REINVENT_DIC['home']),
                                  packageVersion=REINVENT_DIC['version'])

        # Path to the priors download script shipped with the plugin
        priorsScript = cls.getPluginHome('priors.py')

        # REINVENT4 source directory (the release archive is extracted and renamed to it)
        repoPath = os.path.join(cls.getVar(REINVENT_DIC['home']), 'REINVENT4')

        # Installing package
        installer.addCommand(f'wget {cls.getReinventUrl()} -O reinvent4.zip && '
                             f'unzip reinvent4.zip && '
                             f'mv REINVENT4-*/ REINVENT4 && '
                             f'rm reinvent4.zip', 'REINVENT_DOWNLOADED')\
            .getCondaEnvCommand(binaryName=REINVENT_DIC['name'], binaryVersion=REINVENT_DIC['version'],
                                pythonVersion='3.11', binaryPath=repoPath,
                                extraCommands=['python install.py cpu', 'pip install rdkit toml'],
                                targetName='REINVENT_INSTALLED')\
            .addCommand(f'{cls.getEnvActivationCommand(REINVENT_DIC)} && python3 {priorsScript}',
                        'PRIORS_DOWNLOADED')\
            .addPackage(env, dependencies=['wget', 'unzip', 'conda'], default=default)

    @classmethod
    def getReinventUrl(cls):
        return f'{cls._url}/archive/refs/tags/{REINVENT_TAG}.zip'

    @classmethod
    def getProgram(cls, program=None):
        return cls.getVar(REINVENT_BINARY)

    @classmethod
    def getEnvActivation(cls):
        return cls.getEnvActivationCommand(REINVENT_DIC, condaHook=False)

    @classmethod
    def getSoftwarePath(cls):
        return cls.getVar(REINVENT_HOME)

    @classmethod
    def getPriorPath(cls, *args):
        return os.path.join(cls.getSoftwarePath(), 'priors', *args)
