start:
    hugo server --buildDrafts --disableFastRender

update:
    hugo mod get -u
    hugo mod get -u github.com/imfing/hextra

new slug:
    hugo new content/{{ slug }}.md && nvim $_

til slug:
    hugo new content/til/{{ slug }}.md && nvim $_
