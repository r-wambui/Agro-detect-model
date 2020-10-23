{% assign code = include.code %}
{% assign language = include.language %}

{% assign nanosecond = "now" | date: "%N" %}

<div id="code{{ nanosecond }}" markdown="1">

```{{ language }}
{{ code }}
```

</div>
<button class="copybutton{{ nanasecond }}" data-clipboard-target="#code{{ nanosecond }}">copy</button>

<script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>


<script src="https://cdn.jsdelivr.net/npm/clipboard@2.0.6/dist/clipboard.min.js"></script>
<script>
var copybutton = document.getElementById('copybutton{{ nanasecond }}');
var clipboard{{ nanosecond }} = new ClipboardJS('.copybutton');

clipboard{{ nanosecond }}.on('success', function(e) {
        console.log(e);
      
});

clipboard{{ nanosecond }}.on('error', function(e) {
        console.log(e);
    });

</script> 
