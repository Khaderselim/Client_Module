<?php
require_once _PS_MODULE_DIR_ . 'client/classes/Competitor.php';
require_once _PS_MODULE_DIR_ . 'client/classes/ClientCatalog.php';

class AdminClientCompetitorsController extends ModuleAdminController
{
    private $apiConfig;
    /**
     * @throws PrestaShopException
     * This constructor initializes the AdminClientCompetitorsController class.
     * It sets up the necessary properties for the controller,
     * including the table name, class name, and fields for the list view.
     * It also defines the bulk actions and the fields to be displayed in the list view.
     */
    public function __construct()
    {
        $this->bootstrap = true;
        $this->table = 'client_competitor';
        $this->className = 'Competitor';
        $this->lang = false;
        $this->identifier = 'id_client_competitor';
        $this->orderBy = 'id_client_competitor';
        $this->orderWay = 'ASC';
        $this->list_no_link = true;// Disable link to edit form
        $this->apiConfig= require_once _PS_MODULE_DIR_ . 'client/config/api.php';
        parent::__construct();

        $this->bulk_actions = array(
            'delete' => array(
                'text' => $this->l('Delete selected'),
                'confirm' => $this->l('Delete selected items?'),
                'icon' => 'icon-trash'
            )
        ); // Bulk action to enable selected items

        $this->fields_list = array(
            'id_client_competitor' => array(
                'title' => $this->l('ID'),
                'align' => 'center',
                'class' => 'fixed-width-xs',
                'filter_key' => 'a!id_client_competitor'
            ),
            'name' => array(
                'title' => $this->l('Site'),
                'align' => 'left',
                'class' => 'fixed-width-md',
                'filter_key' => 'a!name',
                'callback' => 'displayProductsName' // Callback function to display the product name
            ),
            'logo' => array(
                'title' => $this->l('Logo'),
                'callback' => 'displayLogo', // Callback function to display the logo
                'align' => 'center',
                'search' => false
            ),
            'url' => array(
                'title' => $this->l('URL'),
                'align' => 'left',
                'width' => 'auto',
                'callback' => 'displayURLLink', // Callback function to display the URL
                'filter_key' => 'a!url',
                'filter_type' => 'text',
                'orderby' => true
            ),
            'priority' => array(
                'title' => $this->l('Priority'),
                'class' => 'fixed-width-xs',
                'align' => 'center',
                'type' => 'select',
                'filter_key' => 'a!priority',
                'list' => array(
                    '1' => '1',
                    '2' => '2',
                    '3' => '3',
                    '4' => '4',
                    '5' => '5'
                ),
                'filter_type' => 'int'
            ),

            'active' => array(
                'title' => $this->l('Status'),
                'align' => 'center',
                'active' => 'status',
                'type' => 'bool'
            )
        );
        // Add the CRUD actions to the list
        $this->addRowAction('edit');
        $this->addRowAction('delete');
    }

    /**
     * @return void
     * This method is used to initialize the content of the controller.
     * It handles different actions based on the request.
     * For example, it can test a URL or save pattern data of the added competitor.
     */
    public function initContent()
    {
        parent::initContent();
        $action = Tools::getValue('action');
        switch ($action) {
            case 'test_url':
                $this->testUrl();
                break;
            case 'save_test':
                $this->saveTest();
                break;
        }
    }

    /**
     * @return void
     * This method is used to remove the default toolbar buttons.
     */
    public function initToolbar()
    {
    }

    /**
     * @return void
     * This method is used to initialize the page header toolbar.
     * It adds a button to create a new competitor.
     */
    public function initPageHeaderToolbar()
    {
        parent::initPageHeaderToolbar();
        if (Tools::getValue('addclient_competitor') === false && Tools::getValue('updateclient_competitor') === false) {
            $this->page_header_toolbar_btn['new'] = [
                'href' => self::$currentIndex . '&add' . $this->table . '&token=' . $this->token,
                'desc' => $this->trans('Add new', [], 'Admin.Actions'),
                'icon' => 'process-icon-new'
            ];
        }
        $this->page_header_toolbar_title = $this->l('List of Competitors');
    }
    /**
     * @return string
     * @throws SmartyException
     * This method is used to render the form for creating or editing a competitor.
     * It defines the fields to be displayed in the form,
     */
    public function renderForm()
    {
        $this->fields_form = array(
            'legend' => array(
                'title' => $this->l('Competitor'),
                'icon' => 'icon-folder'
            ),
            'input' => array(
                array(
                    'type' => 'text',
                    'label' => $this->l('Name'),
                    'name' => 'name',
                    'required' => true
                ),

                array(
                    'type' => 'text',
                    'label' => $this->l('URL'),
                    'name' => 'url',
                    'required' => true
                ),
                array(
                    'type' => 'select',
                    'label' => $this->l('Priority'),
                    'name' => 'priority',
                    'options' => array(
                        'query' => array(
                            array('id' => 1, 'name' => '1'),
                            array('id' => 2, 'name' => '2'),
                            array('id' => 3, 'name' => '3'),
                            array('id' => 4, 'name' => '4'),
                            array('id' => 5, 'name' => '5')
                        ),
                        'id' => 'id',
                        'name' => 'name'
                    )
                ),


                array(
                    'type' => 'switch',
                    'label' => $this->l('Status'),
                    'name' => 'active',
                    'default_value' => 1,
                    'required' => true,
                    'values' => array(
                        array(
                            'id' => 'active_on',
                            'value' => 1,
                            'label' => $this->l('Enabled')
                        ),
                        array(
                            'id' => 'active_off',
                            'value' => 0,
                            'label' => $this->l('Disabled')
                        )
                    )
                )
            ),
            'submit' => array(
                'title' => $this->l('Save'),
            ),
            'buttons' => array(
                array(
                    'title' => $this->l('Update Pattern'),
                    'class' => 'btn btn-default pull-right',
                    'icon' => 'process-icon-test',
                    'id' => 'test_url_button'
                )
            ) // A button to test, add or edit the pattern
        );
        // Check if we are editing an existing competitor
        $existing_relation = Db::getInstance()->getRow('SELECT COUNT(*) AS COUNT FROM ' . _DB_PREFIX_ . 'competitor_pattern WHERE id_competitor = ' . (int)Tools::getValue('id_client_competitor'));
        // Assign the current ID and count to the template
        $this->context->smarty->assign([
            'current_id' => Tools::getValue('id_client_competitor'),
            'count' => $existing_relation['COUNT'],
            'current_url' => $this->context->link->getAdminLink('AdminClientCompetitors')
        ]);

        return parent::renderForm() .
            $this->context->smarty->fetch(_PS_MODULE_DIR_ . 'client/views/templates/admin/test_url_modal.tpl'); // Render the test URL modal

    }

    /**
     * @return false|ObjectModel
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This method is used to process the addition of a new competitor.
     * It handles the URL input, logo generation,
     * and saving patterns from the session.
     */
    public function processAdd()
    {
        // Store the URL before processing
        $url = Tools::getValue('url');
        // Generate the logo URL based on the input URL
        if ($url) {
            $_POST['logo'] = $this->getFaviconUrl($url);
        }

        // Process the add operation
        $object = parent::processAdd();

        if ($object) {
            // Update logo
            $object->logo = $this->getFaviconUrl($url);
            $object->update();

            // Save patterns from session
            if ($this->context->cookie->price_pattern && $this->context->cookie->description_pattern) {
                $priceData = json_decode($this->context->cookie->price_pattern, true);
                $descriptionData = json_decode($this->context->cookie->description_pattern, true);
                $stockData = json_decode($this->context->cookie->stock_pattern, true);

                Db::getInstance()->insert('competitor_pattern', [
                    'id_competitor' => (int)$object->id, // Use correct ID field
                    'price_tag' => pSQL($priceData['tag']),
                    'price_attribute' => pSQL(json_encode($priceData['attributes'])),
                    'description_tag' => pSQL($descriptionData['tag']),
                    'description_attribute' => pSQL(json_encode($descriptionData['attributes'])),
                    'stock_tag' => pSQL($stockData['tag']),
                    'stock_attribute' => pSQL(json_encode($stockData['attributes']))
                ]);

                // Clear session data
                unset($this->context->cookie->price_pattern);
                unset($this->context->cookie->description_pattern);
                unset($this->context->cookie->stock_pattern);
                $this->context->cookie->write();
            }

            Tools::redirectAdmin($this->context->link->getAdminLink('AdminClientCompetitors')); // Redirect to the competitors list
        }

        return $object;
    }

    /**
     * @return void
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This method is used to process the update of an existing competitor.
     * It handles the URL input and logo generation.
     */
    public function processUpdate()
    {
        // Generate the logo URL based on the input URL
        if ($url = Tools::getValue('url')) {
            $_POST['logo'] = $this->getFaviconUrl($url);
        }
        $object = parent::processUpdate(); // TODO: Change the autogenerated stub
        // Update the logo in the database
        if ($object) {
            $object->logo = $this->getFaviconUrl($url);
            $object->update();
        }
    }

    /**
     * @param string $value The URL of the logo
     * @param array $row The row data
     * @return string HTML for displaying the logo
     * This method is used to display the logo in the list view.
     */

    public function displayLogo($value, $row)
    {
        if ($value) {
            return '<img src="' . $value . '" alt="' . $row['name'] . '" style="max-width: 32px; max-height: 32px; border-radius: 4px;">';
        }
        return '-';
    }

    /**
     * @param $url
     * @return string
     * This method generates a favicon URL based on the input URL.
     */
    private function getFaviconUrl($url)
    {
        $domain = parse_url($url, PHP_URL_HOST);
        if (!$domain) {
            $domain = $url;
        }
        return "https://www.google.com/s2/favicons?sz=128&domain=" . $domain;
    }

    /**
     * @param string $value The URL to be displayed
     * @param array $row The row data
     * @return string HTML for displaying the URL link
     * This method is used to display the URL link in the list view.
     */
    public function displayURLLink($value, $row)
    {
        return '<a href="' . Tools::safeOutput($value) . '" target="_blank" class="btn btn-link">' . Tools::truncate($value, 40) . '</a>';
    }

    /**
     * @param string $value The name of the product
     * @param array $row The row data
     * @return string The formatted product name
     * This method is used to format the product name for display.
     */
    public function displayProductsName($value, $row)
    {
        return ucwords(strtolower($value));
    }

    /**
     * @return void
     * @throws PrestaShopException
     * This method is used to test a URL by calling an external API.
     * It retrieves the price , description and stock status extraction patterns from the API
     */
    private function testUrl()
    {
        try {
            // Get the URL from the request
            $url = Tools::getValue('url');

            // Call pattern extraction API
            $apiUrl = $this->apiConfig['api_base_url'].'/api/extract-patterns?url=' . urlencode($url);
            // Get the response from the API
            $response = $this->callApi($apiUrl);
            $data = json_decode($response, true);

            if (!$data) {
                die(json_encode([
                    'success' => false,
                    'error' => 'No data returned from API'
                ]));
            }

            die(json_encode([
                'success' => true,
                'data' => $data,
                'debug' => [
                    'url' => $url
                ]
            ]));

        } catch (Exception $e) {
            die(json_encode([
                'success' => false,
                'error' => $e->getMessage()
            ]));
        }
    }

    /**
     * @param string $url The URL to be called
     * @return string The response from the API
     * This method is used to call an external API using cURL.
     */
    private function callApi($url)
    {
        if (empty($url)) {
            return 'URL is required';
        }

        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, 1);
        $output = curl_exec($ch);

        if (curl_errno($ch)) {
            $output = 'Curl error: ' . curl_error($ch);
        }

        curl_close($ch);

        return $output;
    }

    /**
     * @return void
     * @throws PrestaShopException
     * This method is used to save the patterns.
     * It handles the input data and saves it to the database.
     */
    private function saveTest()
    {
        try {
            // Get the input data
            $priceData = json_decode(Tools::getValue('price_data'), true);
            $descriptionData = json_decode(Tools::getValue('description_data'), true);
            $stockData = json_decode(Tools::getValue('stock_data'), true);
            $id_competitor = (int)Tools::getValue('id_competitor');


            // Store patterns in session if no competitor ID yet
            if (!$id_competitor) {
                $this->context->cookie->price_pattern = json_encode($priceData);
                $this->context->cookie->description_pattern = json_encode($descriptionData);
                $this->context->cookie->stock_pattern = json_encode($stockData);
                $this->context->cookie->write();

                die(json_encode([
                    'success' => true,
                    'message' => 'Patterns saved temporarily'
                ]));
            }

            // Check the input data
            if (!$priceData || !$descriptionData) {
                throw new Exception('Missing price or description data');
            }

            if (!isset($priceData['tag']) || !isset($priceData['attributes'])) {
                throw new Exception('Missing price tag or attributes');
            }

            if (!isset($descriptionData['tag']) || !isset($descriptionData['attributes'])) {
                throw new Exception('Missing description tag or attributes');
            }


            // Delete existing record
            Db::getInstance()->execute('
                DELETE FROM ' . _DB_PREFIX_ . 'competitor_pattern
                WHERE id_competitor = ' . (int)$id_competitor
            );

            // Insert new record
            $result = Db::getInstance()->insert('competitor_pattern', [
                'id_competitor' => (int)$id_competitor,
                'price_tag' => pSQL($priceData['tag']),
                'price_attribute' => pSQL(json_encode($priceData['attributes'])),
                'description_tag' => pSQL($descriptionData['tag']),
                'description_attribute' => pSQL(json_encode($descriptionData['attributes'])),
                'stock_tag' => pSQL(json_encode($stockData['tag'])),
                'stock_attribute' => pSQL(json_encode($stockData['attributes']))
            ]);

            if (!$result) {
                throw new Exception('Failed to save attributes: ' . Db::getInstance()->getMsgError());
            }

            die(json_encode([
                'success' => true,
                'message' => 'Attributes saved successfully'
            ]));

        } catch (Exception $e) {
            die(json_encode([
                'success' => false,
                'error' => $e->getMessage()
            ]));
        }
    }


}