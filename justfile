start:
    hugo server --buildDrafts --disableFastRender

update:
    hugo mod get -u
    hugo mod get -u github.com/imfing/hextra
