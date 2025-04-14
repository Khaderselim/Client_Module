{extends file="helpers/list/list_content.tpl"}

<!---
This template is used to display the content of the product list in the admin panel.
It includes a table that shows the product details, including the product name, price, and stock status.
The template also includes a section for displaying competitors' prices and stock status.
The competitors' section can be expanded to show more competitors. ( Only shows the first competitor by default )
-->

{block name="td_content"}
    {if $key == 'competitors'}
        <div class="competitors-inline-list">
            <table class="table competitors-table-{$tr.id_product}">
                <thead>
                <tr>
                    <th>Priority</th>
                    <th>Logo</th>
                    <th>Competitor</th>
                    <th class="fixed-width-md">Price</th>
                    <th>Change</th>
                    <th>Status</th>
                    <th>Difference</th>
                    <th>Update</th>
                    <th></th>
                </tr>
                </thead>
                <tbody>
                {assign var=competitors value=Products_relation::getProductRelations($tr.id_product)} {* Get competitors for the product *}
                {assign var=activeCount value=0}
                {if $competitors}
                    {foreach from=$competitors item=competitor name=compLoop}
                        {if $competitor.active == 1}
                            {assign var=activeCount value=$activeCount+1}
                            <tr class="competitor-row priority-{$competitor.priority}"
                                {if $activeCount > 1}style="display: none;"{/if}>
                                <td>
                                    {if $competitor.priority == 1}
                                        <span class="label label-danger">{$competitor.priority|intval}</span>
                                    {elseif $competitor.priority == 2}
                                        <span class="label label-warning">{$competitor.priority|intval}</span>
                                    {elseif $competitor.priority == 3}
                                        <span class="label label-info">{$competitor.priority|intval}</span>
                                    {elseif $competitor.priority == 4}
                                        <span class="label label-primary">{$competitor.priority|intval}</span>
                                    {else}
                                        <span class="label label-default">{$competitor.priority|intval}</span>
                                    {/if}
                                </td>
                                <td>
                                    {if $competitor.logo}
                                        <img src="{$competitor.logo|escape:'html':'UTF-8'}" alt="Logo"
                                             class="img-responsive" style="max-width: 50px;">
                                    {/if}
                                </td>
                                <td class="fixed-width-xxl"><a href="{$competitor.url|escape:'html':'UTF-8'}"
                                                               target="_blank">{$competitor.name|escape:'html':'UTF-8'}</a>
                                </td>
                                <td class="fixed-width-lg">{$competitor.price|escape:'html':'UTF-8'}</td>
                                <td class="fixed-width-sm">
                                    {if isset($competitor.new_price) && isset($competitor.old_price)}
                                        <span class="percentage-change"
                                              data-current="{$competitor.new_price|escape:'html':'UTF-8'}"
                                              data-previous="{$competitor.old_price|escape:'html':'UTF-8'}">
                                            </span>
                                    {else}
                                        <span class="label label-default">N/A</span>
                                    {/if}

                                </td>
                                <td>{if isset($competitor.stock) && $competitor.stock != ''}
                                        <span >{$competitor.stock|escape:'html':'UTF-8'}</span>
                                    {else}
                                        <span class="label label-danger">N/A</span>
                                    {/if}
                                    </td>
                                <td>
                                    {if isset($competitor.price) && isset($competitor.client_price)}
                                        <span class="percentage-change"
                                              data-current="{$competitor.price|escape:'html':'UTF-8'}"
                                              data-previous="{$competitor.client_price|escape:'html':'UTF-8'}">
                                            </span>
                                    {else}
                                        <span class="label label-default">N/A</span>
                                    {/if}
                                </td>
                                <td class="fixed-width-lg">{$competitor.date_add|escape:'html':'UTF-8'}</td>
                                <td class="text-right">
                                    {if $activeCount == 1 && $competitors|@count > 1}
                                        <button class="show-more-btn" data-product-id="{$product_id}">
                                            <i class="icon-chevron-down"></i>
                                        </button>
                                    {/if}
                                </td>
                            </tr>
                        {/if}
                    {/foreach}
                    {if $activeCount == 0}
                        <tr>
                            <td colspan="6" class="text-center">No active competitors found</td>
                        </tr>
                    {/if}
                {else}
                    <tr>
                        <td colspan="6" class="text-center">No competitors found</td>
                    </tr>
                {/if}
                </tbody>
            </table>
        </div>
        <script type="text/javascript">
            document.addEventListener('DOMContentLoaded', function () {
                // Add event listener for show more/less buttons
                document.addEventListener('click', function (e) {
                    if (e.target.closest('.show-more-btn')) {

                        e.preventDefault();
                        var btn = e.target.closest('.show-more-btn');
                        var table = btn.closest('table');
                        var rows = table.querySelectorAll('tbody tr.competitor-row');

                        rows.forEach(function (row) {
                            row.style.display = '';
                        });

                        btn.innerHTML = '<i class="icon-chevron-up"></i>';
                        btn.classList.remove('show-more-btn');
                        btn.classList.add('show-less-btn');
                    }

                    if (e.target.closest('.show-less-btn')) {
                        e.preventDefault();
                        var btn = e.target.closest('.show-less-btn');
                        var table = btn.closest('table');
                        var rows = table.querySelectorAll('tbody tr.competitor-row');

                        rows.forEach(function (row, index) {
                            if (index > 0) {
                                row.style.display = 'none';
                            }
                        });

                        btn.innerHTML = '<i class="icon-chevron-down"></i>';
                        btn.classList.remove('show-less-btn');
                        btn.classList.add('show-more-btn');
                    }
                });
            });

            {**
             * @param priceString
             * @returns {number|string}
             * Extracts numeric value from a price string and formats it to 2 decimal places.
             * Handles various formats including currency symbols and non-breaking spaces.
             *}
            function extractNumericPrice(priceString) {
                // Remove non-breaking spaces and other unexpected characters
                let cleanedString = priceString.replace(/\s/g, '').replace(/[^\d,.]/g, '');

                // Replace comma with a dot for decimal consistency
                if (cleanedString.includes(',')) {
                    cleanedString = cleanedString.replace(',', '.');
                }

                // Convert to float
                const price = parseFloat(cleanedString);

                return isNaN(price) ? 0 : price.toFixed(2); // Ensure 2 decimal places
            }

            {**
             * @param currentPrice
             * @param previousPrice
             * @returns {number}
             * Calculates the percentage change between two prices.
             *}
            function calculatePercentageChange(currentPrice, previousPrice) {
                const current = extractNumericPrice(currentPrice);
                const previous = extractNumericPrice(previousPrice);

                if (previous > 0 && current !== previous) {
                    return ((current - previous) / previous * 100).toFixed(2);
                }

                return 0;
            }

            // Calculate percentage change and update the UI
            document.addEventListener('DOMContentLoaded', function () {
                document.querySelectorAll('.percentage-change').forEach(function (element) {
                    const currentPrice = element.dataset.current;
                    const previousPrice = element.dataset.previous;

                    if (currentPrice && previousPrice) {
                        const change = calculatePercentageChange(currentPrice, previousPrice);
                        if (change !== 0) {
                            element.innerHTML = change + '%';
                            element.classList.add(change > 0 ? 'text-success' : 'text-danger');
                            // Add arrow icons
                            const arrow = change > 0 ? '↑' : '↓';
                            element.innerHTML = arrow + ' ' + element.innerHTML;
                        } else {
                            element.innerHTML = "0%";
                        }
                    }
                });
            });

            {**
             * @description Expands all competitor rows in the table.
             *}
            function expandAll() {
                // Find all show-more buttons
                const showMoreButtons = document.querySelectorAll('.show-more-btn');

                showMoreButtons.forEach(btn => {
                    // Get the table containing competitors
                    const table = btn.closest('table');
                    const rows = table.querySelectorAll('tbody tr.competitor-row');

                    // Show all competitor rows
                    rows.forEach(row => {
                        row.style.display = '';
                    });

                    // Update button appearance
                    btn.innerHTML = '<i class="icon-chevron-up"></i>';
                    btn.classList.remove('show-more-btn');
                    btn.classList.add('show-less-btn');
                });
            }


        </script>
        <style type="text/css">
            .competitors-inline-list {
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 15px;
                background-color: #f9f9f9;
            }

            .competitors-inline-list table {
                margin-bottom: 0;
                background-color: white;
            }

            .competitors-inline-list table thead {
                background-color: #f5f5f5;
            }

            .competitors-inline-list table th {
                font-weight: bold;
            }

            .competitors-inline-list .label {
                display: inline-block;
                min-width: 30px;
                text-align: center;
            }

            .show-more-btn, .show-less-btn {
                white-space: nowrap;
            }

            .show-more-btn, .show-less-btn {
                white-space: nowrap;
                background: transparent;
                border: none;
                padding: 0;
                color: #666;
            }

            .show-more-btn:hover, .show-less-btn:hover {
                color: #333;
                background: transparent;

            }
        </style>
    {else}
        {$smarty.block.parent}
    {/if}
{/block}