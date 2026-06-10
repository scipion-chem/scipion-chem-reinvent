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
import json
from urllib.request import urlopen


def download():
    url_prior = 'https://zenodo.org/api/records/15641297'
    os.makedirs('priors', exist_ok=True)

    print(f"Connecting to Zenodo...")
    with urlopen(url_prior) as response:
        data = json.loads(response.read().decode())

    for f_info in data['files']:
        fname = os.path.join('priors', f_info['key'])
        print(f"Downloading {f_info['key']}...")
        with urlopen(f_info['links']['self']) as d, open(fname, 'wb') as f:
            f.write(d.read())


if __name__ == "__main__":
    download()