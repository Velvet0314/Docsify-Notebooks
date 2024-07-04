!function (t) {
    (typeof exports !== 'object' || typeof module === 'undefined') && typeof define === 'function' && define.amd ? define(t) : t()
}(function () {
    'use strict';

    var doc = typeof window !== 'undefined' ? window.document : null;

    function createSelectorQuery(selector, context) {
        context = context || document;
        return context.querySelector(selector);
    }

    function createSelectorQueryAll(selector, context) {
        context = context || document;
        return Array.prototype.slice.call(context.querySelectorAll(selector));
    }

    // Insert CSS styles into the document head
    if (doc) {
        var head = doc.head || doc.getElementsByTagName('head')[0];
        var style = doc.createElement('style');
        var cssText = `
            .docsify-pagination-container {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                overflow: hidden;
                margin: 5em 0 1em;
                border-top: 1px solid rgba(0,0,0,0.07);
            }
            .pagination-item {
                margin-top: 2.5em;
            }
            .pagination-item a, .pagination-item a:hover {
                text-decoration: none;
            }
            .pagination-item a {
                color: currentColor;
            }
            .pagination-item a:hover .pagination-item-title {
                text-decoration: none;
            }
            .pagination-item:not(:last-child) a .pagination-item-label, .pagination-item:not(:last-child) a .pagination-item-subtitle, .pagination-item:not(:last-child) a .pagination-item-title {
                opacity: 0.3;
                transition: all 0.2s;
            }
            .pagination-item:last-child .pagination-item-label, .pagination-item:not(:last-child) a:hover .pagination-item-label {
                opacity: 0.6;
            }
            .pagination-item:not(:last-child) a:hover .pagination-item-title {
                opacity: 1;
            }
            .pagination-item-label {
                font-size: 0.8em;
            }
            .pagination-item-label>* {
                line-height: 1;
                vertical-align: middle;
            }
            .pagination-item-label svg {
                height: 0.8em;
                width: auto;
                stroke: currentColor;
                stroke-linecap: round;
                stroke-linejoin: round;
                stroke-width: 1px;
            }
            .pagination-item--next {
                margin-left: auto;
                text-align: right;
            }
            .pagination-item--next svg {
                margin-left: 0.5em;
            }
            .pagination-item--previous svg {
                margin-right: 0.5em;
            }
            .pagination-item-title {
                font-size: 1.6em;
            }
            .pagination-item-subtitle {
                text-transform: uppercase;
                opacity: 0.3;
            }`;

        if (style.styleSheet) {
            style.styleSheet.cssText = cssText;
        } else {
            style.appendChild(doc.createTextNode(cssText));
        }
        head.appendChild(style);
    }

    var containerClass = "docsify-pagination-container";

    function updatePagination(config) {
        var container = createSelectorQuery('.' + containerClass);
        if (container) {
            try {
                var links = createSelectorQueryAll('.sidebar-nav > ul > li > a');
                var currentPath = location.hash;
                var currentIndex = links.findIndex(link => link.getAttribute('href') === currentPath);
                var prevLink = currentIndex > 0 ? links[currentIndex - 1] : null;
                var nextLink = currentIndex < links.length - 1 ? links[currentIndex + 1] : null;

                var paginationHTML = '';
                if (prevLink) {
                    paginationHTML += `
                        <div class="pagination-item pagination-item--previous">
                            <a href="${prevLink.getAttribute('href')}">
                                <div class="pagination-item-label">
                                    <svg width="10" height="16" viewBox="0 0 10 16" xmlns="http://www.w3.org/2000/svg">
                                        <polyline fill="none" vector-effect="non-scaling-stroke" points="8,2 2,8 8,14"/>
                                    </svg>
                                    <span>${config.previousText || '上一章节'}</span>
                                </div>
                                <div class="pagination-item-title">${prevLink.innerText}</div>
                            </a>
                        </div>`;
                }
                if (nextLink) {
                    paginationHTML += `
                        <div class="pagination-item pagination-item--next">
                            <a href="${nextLink.getAttribute('href')}">
                                <div class="pagination-item-label">
                                    <span>${config.nextText || '下一章节'}</span>
                                    <svg width="10" height="16" viewBox="0 0 10 16" xmlns="http://www.w3.org/2000/svg">
                                        <polyline fill="none" vector-effect="non-scaling-stroke" points="2,2 8,8 2,14"/>
                                    </svg>
                                </div>
                                <div class="pagination-item-title">${nextLink.innerText}</div>
                            </a>
                        </div>`;
                }
                container.innerHTML = paginationHTML;
            } catch (err) {
                console.error('Error updating pagination:', err);
                container.innerHTML = ''; // Clear the container on error
            }
        }
    }
    
    // Docsify plugin function
    function paginationPlugin(hook, vm) {
        hook.afterEach(function (html) {
            return html + '<div class="' + containerClass + '"></div>';
        });
        hook.doneEach(function () {
            var config = vm.config.pagination || {};
            updatePagination(config);
        });
    }

    // Register the plugin
    window.$docsify = window.$docsify || {};
    window.$docsify.plugins = (window.$docsify.plugins || []).concat(paginationPlugin);
});
