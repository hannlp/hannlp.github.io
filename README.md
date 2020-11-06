# Hello World! I'm hans
This is a repo of my Blog. It's a simple and fast blog with the help of Github Pages and [beautiful-jekyll](https://github.com/daattali/beautiful-jekyll) by daattali.

# Some personal settings
## a TOC system ([link](https://github.com/allejo/jekyll-toc))
> _Usage_: 
> 1.Download the latest toc.html file and add that file in your _includes folder.
> 2.Update your post.html file like this:
```
    <article role="main" class="blog-post">
        {% include toc.html html=content sanitize=true class="inline_toc" id="my_toc" %}
        {{ content }}
    </article>
```
# To do list
* Add a commit system base on github issues