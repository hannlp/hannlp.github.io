# Hello World! I'm Hans
This is a repo of my Blog. It's a simple and fast blog with the help of Github Pages and [beautiful-jekyll](https://github.com/daattali/beautiful-jekyll) by daattali.

# 1 Some personal settings

## 1.1 My Color Style
```
navbar-col: "#3686EC"
navbar-text-col: "#FFFFFF"
navbar-border-col: "#DDDDDD"
page-col: "#FFFFFF"
text-col: "#404040"
link-col: "#008AFF"
hover-col: "#0085A1"
footer-col: "#F5F5F5"
footer-text-col: "#777777"
footer-link-col: "#404040"
```
![my_style](/assets/img/my_style.png)

## 1.2 My Code Insert Style
Update your 'beautiful-jekyll.css' like this:
```
.highlight > pre {
  /*
  background-image: linear-gradient(
    rgba(0,0,0,0.03), rgba(0,0,0,0.03) 1.5em, rgba(0,0,0,0.02) 1.5em, rgba(0,0,0,0.02) 3em);
  */
  background-size: auto 3em;
  background-position-y: 0.625rem;
  border: 1px solid rgba(0,0,0,0.1);
  border-left: 0.1rem solid #00ff71;
  background: lightyellow;
}
```
and it will be like this:
![code_style](/assets/img/code_style.png)
## 1.3 A TOC(Article Directory) System ([link](https://github.com/allejo/jekyll-toc))
> _Usage_: <br/>1.Download the latest toc.html file and add that file in your _includes folder.<br/>2.Update your post.html file like this:
```
    <article role="main" class="blog-post">
        {% include toc.html html=content sanitize=true class="inline_toc" id="my_toc" %}
        {{ content }}
    </article>
```
and it will be like this:
![toc_system](/assets/img/toc_system.png)

## 1.4 Mathematical Formula Supporting
I use KaTex in my blog to show any mathematical formula. Like this: 
![formula](/assets/img/formula.png)
> _Usage_: <br/>**Only need to add these code in your 'head.html' file:**
```
<head>
  ...
  <link rel="stylesheet" href="https://cdn.bootcss.com/KaTeX/0.11.1/katex.min.css">
  <script defer src="https://cdn.bootcss.com/KaTeX/0.11.1/katex.min.js"></script>
  <script defer>
    //KaTeX settings
    var katex_config = {
      delimiters:
        [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\(", right: "\\)", display: false },
          { left: "\\[", right: "\\]", display: true }
        ]
    };
  </script>
  <script defer src="https://cdn.bootcss.com/KaTeX/0.11.1/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body,katex_config)"></script>

</head>
```
Thanks to my friend [LoliMay](https://www.lolimay.cn) and This article [JS配置KaTeX渲染LaTeX公式](https://blog.csdn.net/qq_43491040/article/details/104174730?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param), and you can get more in the [KaTex Official Documents](https://katex.org/docs/api.html)

## 1.5 Center display Tables and Pictures
Update your 'beautiful-jekyll.css' like this:
```
...
table {
  padding: 0;
  margin: 0 auto;
}
...
.blog-post img {
  display: block;
  margin: 0 auto;
  max-width: 100%;
}
```
and it will be like this:
![center_pic](/assets/img/center_pic.png)
![center_table](/assets/img/center_table.png)
# 2 To do list
* Add a commit system base on github issues
* Use some method to accelerate image loading
* Use cdn to accelerate my Github pages