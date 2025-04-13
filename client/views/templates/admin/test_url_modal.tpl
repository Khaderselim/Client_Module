<!---
This template displays a modal interface for extracting patterns that can be used to extract the details of the competitors' products.
It allows the user to input a URL and extract its patterns.
It includes a form with fields for the URL, price, description, and stock options.
-->
<div class="modal fade" id="testUrlModal" tabindex="-1" role="dialog">
    <div class="modal-dialog" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal">&times;</button>
                <h4 class="modal-title">Test URL</h4>
            </div>
            <div class="modal-body">
                <form id="testUrlForm">
                    <div class="form-group">
                        <label for="test_url">URL:</label>
                        <input type="url" class="form-control" id="test_url" required>
                    </div>


                    <div id="priceResults" class="form-group">
                        <ul class="nav nav-tabs">
                            <li class="active"><a href="#prices-tab" data-toggle="tab">Prices</a></li>
                            <li><a href="#description-tab" data-toggle="tab">Description</a></li>
                            <li><a href="#stock-tab" data-toggle="tab">Stock</a></li>
                        </ul>
                        {*The price section*}
                        <div class="tab-content">
                            <div class="tab-pane active" id="prices-tab">
                                <label>Select Price:</label>
                                <div id="radioList"></div>
                            </div>
                            {*The description section*}
                            <div class="tab-pane" id="description-tab">
                                <label>Select Description:</label>
                                <div id="descriptionList"></div>
                            </div>
                            {*The stock status section*}
                            <div class="tab-pane" id="stock-tab">
                                <label>Select Stock:</label>
                                <div id="stockList"></div>
                            </div>
                        </div>

                        <div id="priceResults" class="form-group" style="display:none;">
                            <label>Select Price:</label>
                            <div id="radioList"></div>
                        </div>
                </form>

            </div>

            <div class="modal-footer">
                <button type="button" class="btn btn-default" data-dismiss="modal">Close</button>
                <button type="button" class="btn btn-primary" onclick="testUrl()">Test</button>
                <button type="button" class="btn btn-success" id="savePrice" style="display:none;"
                        onclick="saveSelectedAttributes()">Save
                </button>
            </div>
        </div>
    </div>
</div>
<script>
    $(document).ready(function () {
        $('#client_competitor_form').on('submit', function (e) {
            // Display the modal if it's a new competitor ( When the form is submitted )
            if (!{$count} && !$('#test_url').val()) {
                e.preventDefault();
                showTestUrlPopup();
                return false;
            }
            return true;
        });
        $('#test_url_button').on('click', function (e) {
            e.preventDefault();
            showTestUrlPopup();
        });
    });


    {**
     *Show the test URL modal
     *}
    function showTestUrlPopup() {
        $('#testUrlForm')[0].reset();
        $('#priceResults').hide();
        $('#savePrice').hide();
        $('#saveManual').hide();
        $('#radioList').empty();
        $('#testUrlModal').modal('show');
    }


    {**
     *Test the URL and display the results
     *}
    function testUrl() {
        const url = $('#test_url').val();
        if (!url) {
            alert('Please enter a URL');
            return;
        }

        $('#testUrlModal .modal-footer button').prop('disabled', true);
        // Ajax request to test the URL (testUrl() function in the controller)
        $.ajax({
            url: '{$current_url|escape:'javascript':'UTF-8'}&action=test_url',
            method: 'POST',
            data: {
                url: url,
            },
            success: function (response) {
                try {
                    const result = typeof response === 'string' ? JSON.parse(response) : response;

                    if (!result.success) {
                        alert(result.error || 'Test failed');
                        return;
                    }

                    if (result.data) {
                        displayOptions(
                            result.data.prices || [],
                            result.data.description || [],
                            result.data.stock || [],
                            url
                        );
                    } else {
                        alert('No data found on this page');
                    }

                } catch (e) {
                    console.error('Parse error:', e);
                    alert('Error processing response');
                }
            },
            error: function (xhr, status, error) {
                alert('Error testing URL: ' + error);
            },
            complete: function () {
                $('#testUrlModal .modal-footer button').prop('disabled', false);
            }
        });
    }

    {**
     * @param {Array} prices - Array of price objects
     * @param {Array} descriptions - Array of description objects
     * @param {Array} stocks - Array of stock objects
     * @param {string} url - The URL being tested
     * Display the price, description and stock status options in the modal
     *}
    function displayOptions(prices, descriptions, stocks, url) {
        // Price options
        var priceContainer = $('#radioList');
        priceContainer.empty();

        prices.forEach(function (item, index) {
            var attributesText = '';
            if (item.attributes) {
                attributesText = Object.entries(item.attributes)
                    .map(function (entry) {
                        return entry[0] + '="' + (Array.isArray(entry[1]) ? entry[1].join(' ') : entry[1]) + '"';
                    })
                    .join(' ');
            }
            // Create the radio button for each price option
            var radioHtml =
                '<div class="radio">' +
                '<label>' +
                '<input type="radio" name="price_option" value="' + index + '" ' +
                'data-price="' + (item.price || '') + '" ' +
                'data-tag="' + (item.tag || '') + '" ' +
                'data-attributes=\'' + JSON.stringify(item.attributes || {}) + '\'>' +
                '<span class="price-option">' +
                '<strong>' + (item.price || '') + '</strong>' +
                '<small class="text-muted">' +
                '(Tag: ' + (item.tag || '') +
                (attributesText ? ' | Attributes: ' + attributesText : '') + ')' +
                '</small>' +
                '</span>' +
                '</label>' +
                '</div>';

            priceContainer.append(radioHtml);
        });

        // Description options`
        var descContainer = $('#descriptionList');
        descContainer.empty();

        descriptions.forEach(function (item, index) {
            var attributesText = '';
            if (item.attributes) {
                attributesText = Object.entries(item.attributes)
                    .map(function (entry) {
                        return entry[0] + '="' + (Array.isArray(entry[1]) ? entry[1].join(' ') : entry[1]) + '"';
                    })
                    .join(' ');
            }
            // Create the radio button for each description option
            var descHtml =
                '<div class="radio">' +
                '<label>' +
                '<input type="radio" name="desc_option" value="' + index + '" ' +
                'data-text="' + (item.text_content || '').replace(/"/g, '&quot;') + '" ' +
                'data-tag="' + (item.tag || '') + '" ' +
                'data-attributes=\'' + JSON.stringify(item.attributes || {}) + '\'>' +
                '<span class="desc-option">' +
                '<strong>' + (item.text_content || '') + '</strong>' +
                '<small class="text-muted">' +
                '(Tag: ' + (item.tag || '') +
                (attributesText ? ' | Attributes: ' + attributesText : '') + ')' +
                '</small>' +
                '</span>' +
                '</label>' +
                '</div>';

            descContainer.append(descHtml);
        });
        // Stock options
        var stockContainer = $('#stockList');
        stockContainer.empty();
        stocks.forEach(function (item, index) {
            var attributesText = '';
            if (item.attributes) {
                attributesText = Object.entries(item.attributes)
                    .map(function (entry) {
                        return entry[0] + '="' + (Array.isArray(entry[1]) ? entry[1].join(' ') : entry[1]) + '"';
                    })
                    .join(' ');
            }

            // Create the radio button for each stock option
            var stockhtml =
                '<div class="radio">' +
                '<label>' +
                '<input type="radio" name="stock_option" value="' + index + '" ' +
                'data-stock="' + (item.stock || '').replace(/"/g, '&quot;') + '" ' +
                'data-tag="' + (item.tag || '') + '" ' +
                'data-attributes=\'' + JSON.stringify(item.attributes || {}) + '\'>' +
                '<span class="stock-option">' +
                '<strong>' + (item.stock || '') + '</strong>' +
                '<small class="text-muted">' +
                '(Tag: ' + (item.tag || '') +
                (attributesText ? ' | Attributes: ' + attributesText : '') + ')' +
                '</small>' +
                '</span>' +
                '</label>' +
                '</div>';
            stockContainer.append(stockhtml);
        });


        $('#test_url').data('current-url', url);
        $('#priceResults').show();
        $('#savePrice').show();
    }

    {**
     * Save the selected attributes
     *}
    function saveSelectedAttributes() {
        const selectedPrice = $('input[name="price_option"]:checked');
        const selectedDesc = $('input[name="desc_option"]:checked');
        const selectedStock = $('input[name="stock_option"]:checked');
        const testUrl = $('#test_url').val();

        if (!selectedPrice.length) {
            alert('Please select at least one pattern');
            return;
        }

        const priceData = selectedPrice.length ? {
            tag: selectedPrice.data('tag'),
            attributes: selectedPrice.data('attributes')
        } : null;

        const descriptionData = selectedDesc.length ? {
            tag: selectedDesc.data('tag'),
            attributes: selectedDesc.data('attributes')
        } : null;

        const stockData = selectedStock.length ? {
            tag: selectedStock.data('tag'),
            attributes: selectedStock.data('attributes')
        } : null;
        // Ajax request to save the selected attributes (saveTest() function in the controller)
        $.ajax({
            url: '{$current_url|escape:'javascript':'UTF-8'}&action=save_test',
            method: 'POST',
            data: {
                price_data: JSON.stringify(priceData),
                description_data: JSON.stringify(descriptionData),
                stock_data: JSON.stringify(stockData),
                id_competitor: {$current_id|intval}
            },
            success: function (response) {
                try {
                    const data = JSON.parse(response);
                    if (data.success) {
                        $('#testUrlModal').modal('hide');
                        showSuccessMessage(data.message);
                        // Set both URL fields

                        $('input[name="test_url"]').val(testUrl);
                        // Submit the form
                        $('#client_competitor_form').submit();
                    } else {
                        showErrorMessage(data.error || 'Error saving attributes');
                    }
                } catch (e) {
                    showErrorMessage('Invalid response from server');
                }
            },
            error: function () {
                showErrorMessage('Error communicating with server');
            }
        });
    }
</script>

<style>
    #radioList {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
        margin-top: 10px;
    }

    #radioList .radio {
        border-bottom: 1px solid #ddd;
        padding: 10px 0;
        margin: 0;
    }

    #radioList .radio:last-child {
        border-bottom: none;
    }

    #radioList .price-option {
        display: flex;
        flex-direction: column;
    }

    #radioList .price-option strong {
        font-size: 14px;
        margin-bottom: 3px;
    }

    #radioList .price-option small {
        color: #666;
    }

    /* Scrollbar styling */
    #radioList::-webkit-scrollbar {
        width: 8px;
    }

    #radioList::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    #radioList::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }

    #radioList::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    .desc-option strong {
        font-size: 14px;
        margin-bottom: 3px;
        word-break: break-word;
    }

    #descriptionList {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
    }

    #stockList {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px;
    }

    .stock-option strong {
        font-size: 14px;
        margin-bottom: 3px;
        word-break: break-word;
    }
</style>