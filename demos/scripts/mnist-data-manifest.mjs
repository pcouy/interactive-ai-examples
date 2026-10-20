/**
 * Manifeste des archives IDX MNIST attendues par la démo.
 *
 * Ce module est volontairement pur (aucun import Node) : il est importé par le
 * script de téléchargement `fetch-mnist-data.mjs` et par les tests, qui
 * vérifient que les noms, tailles et empreintes MD5 correspondent à ce que
 * documente `public/mnist/README.md`.
 */

/** Miroir S3 maintenu par PyTorch, source des quatre archives. */
export const MNIST_BASE_URL = 'https://ossci-datasets.s3.amazonaws.com/mnist/';

/** Les quatre archives attendues, avec leur taille en octets et leur MD5. */
export const MNIST_FILES = [
  {
    name: 'train-images-idx3-ubyte.gz',
    size: 9912422,
    md5: 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
  },
  {
    name: 'train-labels-idx1-ubyte.gz',
    size: 28881,
    md5: 'd53e105ee54ea40749a09fcbcd1e9432',
  },
  {
    name: 't10k-images-idx3-ubyte.gz',
    size: 1648877,
    md5: '9fb629c4189551a2d022fa330f9573f3',
  },
  {
    name: 't10k-labels-idx1-ubyte.gz',
    size: 4542,
    md5: 'ec29112dd5afa0611ce80d1b7f02629c',
  },
];
