var script = document.createElement('script');
script.src = 'https://code.jquery.com/jquery-3.6.0.min.js';
document.getElementsByTagName('head')[0].appendChild(script);
<script type="text/javascript">
    (function($) {
        $("a").sortElements(function (a, b) {
            return $(a).attr("href") < $(b).attr("href") ? 1 : -1;
        })
    })(jQuery);
</script>