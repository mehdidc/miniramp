import os
import hashlib
from functools import partial
import importlib

from urllib.request import urlretrieve
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlopen

def base_name(full_name):
    split = full_name.split('.')
    return split[-1]


def build_func(params):
    if isinstance(params, dict):
        name, params = params['name'], params['params']
        func = import_object(name)
        return partial(func, **params)
    elif type(params) == str:
        return import_object(params)
    else:
        raise ValueError('Expect params to be either a dict or a str')


def eval_and_get(code, name):
    local_vars = {}
    exec(code, local_vars, local_vars)
    return local_vars[name]


def import_object(full_name):
    split = full_name.split('.')
    *modules, object_name = split
    if len(modules) == 0:
        return importlib.import_module(object_name)
    else:
        module = '.'.join(modules) 
        return getattr(importlib.import_module(module), object_name)


# get_file, borrowed from keras
data_folder = '.miniramp'
def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    """Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.
    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.
    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).
    # Returns
        Path to the downloaded file
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(os.path.join('~', data_folder))
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', data_folder)
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' +
                      file_hash + ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    # Example
    ```python
       >>> from keras.data_utils import _hash_file
       >>> _hash_file('/path/to/file.zip')
       'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    # Arguments
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        The file hash
    """
    if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.
    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        Whether the file is valid
    """
    if ((algorithm is 'sha256') or
            (algorithm is 'auto' and len(file_hash) is 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False

