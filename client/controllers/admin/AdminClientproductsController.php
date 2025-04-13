<?php
require_once _PS_MODULE_DIR_ . 'client/classes/Competitor.php';
require_once _PS_MODULE_DIR_ . 'client/classes/ClientProduct.php';
require_once _PS_MODULE_DIR_ . 'client/classes/Products_relation.php';
require_once _PS_MODULE_DIR_ . 'client/classes/CompetitorProduct.php';
require_once _PS_MODULE_DIR_ . 'client/classes/ClientCatalog.php';

class AdminClientproductsController extends ModuleAdminController
{

    /**
     * @throws PrestaShopException
     * This constructor initializes the AdminClientproductsController class.
     * It sets up the table, class name, identifier, and other properties for the controller.
     * It also defines the fields for the list of products and the bulk actions available.
     */
    public function __construct()
    {
        $this->bootstrap = true;
        $this->table = 'client_product';
        $this->className = 'ClientProduct';
        $this->lang = false;
        $this->identifier = 'id_product';

        $this->list_no_link = true; // Disable default link to edit product

        parent::__construct();


        // Declaration of the bulk actions
        $this->bulk_actions = array(
            'delete' => array(
                'text' => $this->l('Delete selected'),
                'confirm' => $this->l('Delete selected items?'),
                'icon' => 'icon-trash'
            ),
            'update' => array(
                'text' => $this->l('Update selected'),
                'confirm' => $this->l('Update selected items?'),
                'icon' => 'icon-refresh',
            )
        );
        // Declaration of the fields for the list
        $this->fields_list = array(
            'id_product' => array(
                'title' => $this->l('ID'),
                'align' => 'center',
                'class' => 'fixed-width-xs',
                'filter_key' => 'a!id_product'
            ),
            'name' => array(
                'title' => $this->l('Product'),
                'align' => 'left',
                'filter_key' => 'a!name',
                'class' => 'fixed-width-xxl',
                'callback' => 'displayNameLink' // Display the name of the product with a link to the product's page
            ),

            'price' => array(
                'title' => $this->l('Price'),
                'align' => 'left',
                'class' => 'fixed-width-md',
                'filter_key' => 'a!price'
            ),
            'competitors' => array(
                'title' => $this->l('Competitors'),
                'align' => 'left',
                'search' => false,
                'orderby' => false
            ) // It will be used to display the competitors of the product

        );
        // Add the CRUD actions buttons
        $this->addRowAction('edit');
        $this->addRowAction('delete');
        $this->addRowAction('update');
    }

    /**
     * @return void
     * This method initializes the content of the page.
     * It adds custom JavaScript and CSS files for the admin interface,
     * and processes the action based on the request.
     */
    public function initContent()
    {
        parent::initContent();
        // Add custom JavaScript and CSS files
        $this->context->controller->addJS(_PS_MODULE_DIR_ . 'client/views/js/admin.js');
        $this->context->controller->addCSS(_PS_MODULE_DIR_ . 'client/views/css/admin.css');
        // Process the action based on the request
        $action = Tools::getValue('action');
        switch ($action) {
            case 'updateproduct':
                $this->updateproduct();
                break;
            case 'update_all':
                $this->processUpdateAll();
                break;
        }

    }

    /**
     * @return void
     * @throws SmartyException
     * This method overrides the initToolbar method to add a custom filter for catalogs,
     * and removes the defaults buttons of list.
     */
    public function initToolbar()
    {
        //Prepare the variables for the catalog filter
        $this->context->smarty->assign(array(
            'catalogs' => ClientCatalog::getAllCatalogs(false),
            'selected_catalog' => Tools::getValue('catalog_filter', ''),
            'admin_products_url' => $this->context->link->getAdminLink('AdminClientproducts')
        ));
        $this->context->smarty->assign(array(
            'catalog_filter_html' => $this->context->smarty->fetch(_PS_MODULE_DIR_ . 'client/views/templates/admin/catalog_filter_fields.tpl')
        ));
    }

    /**
     * @return void
     * This method initializes the page header toolbar.
     * It adds custom buttons for expanding all products, adding a new product,
     * and updating all products.
     */
    public function initPageHeaderToolbar()
    {
        parent::initPageHeaderToolbar();

        // Only show buttons when not in form view
        if (Tools::getValue('addclient_product') === false && Tools::getValue('updateclient_product') === false) {
            $this->page_header_toolbar_btn['expand_all'] = [
                'href' => '',
                'desc' => $this->l('Expand All'),
                'icon' => 'process-icon-expand',
                'class' => 'expand-all-btn',
                'js' => 'expandAll(); return false;"'
            ];

            $this->page_header_toolbar_btn['new'] = [
                'href' => self::$currentIndex . '&add' . $this->table . '&token=' . $this->token,
                'desc' => $this->trans('Add new', [], 'Admin.Actions'),
                'icon' => 'process-icon-new'
            ];

            $this->page_header_toolbar_btn['update_all'] = [
                'href' => $this->context->link->getAdminLink('AdminClientproducts') . '&action=update_all',
                'desc' => $this->l('Update All'),
                'icon' => 'process-icon-refresh'
            ];
        }

        $this->page_header_toolbar_title = $this->l('Products'); // Set the title of the page header toolbar
    }

    /**
     * @return string
     * This method is used to render the form for adding or editing a product.
     * It sets up the fields for the form, including the catalog selection and competitors.
     */
    public function renderForm()
    {
        // Load the object if it exists
        $object = $this->loadObject(true);
        // Prepare the competitors array
        $competitors = array();

        if ($object->id) {
            $competitors = Products_relation::getProductRelations($object->id);
        } // Get the competitors of the product

        $this->fields_form = array(
            'legend' => array(
                'title' => $this->l('Target Website'),
                'icon' => 'icon-link'
            ),
            'input' => array(

                array(
                    'type' => 'search',
                    'name' => 'search',
                    'form_group_class' => 'less-width-label'
                ),
                array(
                    'type' => 'select',
                    'label' => $this->l('Catalog'),
                    'name' => 'id_client_catalog',
                    'required' => true,
                    'form_group_class' => 'less-width-label',
                    'options' => array(
                        'query' => ClientCatalog::getAllCatalogs(), // Get all catalogs
                        'id' => 'id_client_catalog',
                        'name' => 'name'
                    )
                ),


                array(
                    'type' => 'competitors',
                    'name' => 'competitors',
                    'form_group_class' => 'competitors-group less-width-label'
                ), // Prepare the competitors field

            ),
            'submit' => array(
                'title' => $this->l('Save')
            ),

        );

        // Add CSS to page header
        $this->context->controller->addCSS($this->module->getPathUri() . 'views/css/admin.css');
        // Send the current object to the template
        $this->fields_value['search'] = $object->name;

        // Send the competitors to the template
        $this->fields_value['competitors'] = $competitors;
        $this->tpl_form_vars['competitors'] = $competitors;


        return parent::renderForm();
    }

    /**
     * @param $name
     * @param $row
     * @return string
     * This method is used in the list to display the name of the product with a link to the product's page.
     */
    public function displayNameLink($name, $row)
    {
        return '<a href="' . $row['url'] . '" target="_blank">' . Tools::safeOutput($name) . '</a>';
    }


    /**
     * @param $token
     * @param $id
     * @param $name
     * @return false|string
     * @throws SmartyException
     * This function is used to display the update link in the update button inside the action list of the product.
     */
    public function displayUpdateLink($token, $id, $name)
    {
        // Create a template for the update link
        $tpl = $this->createTemplate('Clientproducts/helpers/list/list_action_update.tpl');
        // Assign the variables to the template
        $tpl->assign([
            'href' => $this->context->link->getAdminLink('AdminClientproducts'),
            'action' => $this->l('Update'),
            'id' => $id
        ]);
        // Return the rendered template
        return $tpl->fetch();
    }

    /**
     * @param $id_lang
     * @param null $order_by
     * @param null $order_way
     * @param int $start
     * @param null $limit
     * @param bool $id_lang_shop
     * @return void
     * This method is used to get the list of products.
     * It applies a filter for the selected catalog if one is set.
     */
    public function getList($id_lang, $order_by = null, $order_way = null, $start = 0, $limit = null, $id_lang_shop = false)
    {
        // Apply catalog filter if selected
        $catalog_filter = Tools::getValue('catalog_filter');
        if ($catalog_filter) {
            $this->_where .= ' AND a.id_client_catalog = ' . (int)$catalog_filter;
        }

        parent::getList($id_lang, $order_by, $order_way, $start, $limit, $id_lang_shop);
    }

    /**
     * @return string
     * This method is used to render the list of products.
     * It adds a custom filter for catalogs and then calls the parent renderList method.
     */
    public function renderList()
    {
        return $this->context->smarty->fetch(_PS_MODULE_DIR_ . 'client/views/templates/admin/catalog_filter.tpl')
            . parent::renderList();
    }

    /**
     * @return ClientProduct|false
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This method is used to process the addition of a new product.
     * It checks if the form is submitted,
     * retrieves the product name from the form,
     * fetches the product data from the database,
     * creates API requests for competitors,
     * executes the requests in parallel,
     * and processes the results.
     * It also creates the main product and competitor products,
     * and establishes the relations between them.
     */
    public function processAdd()
    {
        if (!Tools::isSubmit('submitAdd' . $this->table)) {
            return false;
        }
        //Get the product name from the form
        $product_name = Tools::getValue('search');
        //Get the product data from the database
        $db_product = Db::getInstance()->getRow("SELECT * FROM " . _DB_PREFIX_ . "target_product WHERE name = '" . pSQL($product_name) . "'"); // Change the table name to your target product table

        // Create API requests array
        $apiRequests = [];
        // Get the id of the product category
        $maincategory = Tools::getValue('id_client_catalog');


        // Add competitor requests
        $all_competitors = array_merge(
            Tools::getValue('competitors', array()),
            Tools::getValue('new_competitors', array())
        );
        // Structure competitors array
        foreach ($all_competitors as $key => $competitor) {
            if (!empty($competitor['url'])) {
                $id_competitor = Db::getInstance()->getValue(
                    "SELECT id_client_competitor FROM " . _DB_PREFIX_ . "client_competitor
                    WHERE url LIKE '%" . pSQL($this->get_domain($competitor['url'])) . "%'"
                );
                // Get the competitor search pattern and prepare the api request
                if ($id_competitor) {
                    $priceAttributes = Db::getInstance()->getRow('SELECT price_tag AS tag, price_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$id_competitor);
                    $descriptionAttributes = Db::getInstance()->getRow('SELECT description_tag AS tag, description_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$id_competitor);
                    $stockAttributes = Db::getInstance()->getRow('SELECT stock_tag AS tag, stock_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$id_competitor);

                    $apiUrl = 'http://localhost:8000/api/extract-price?url=' . urlencode($competitor['url']);
                    if ($priceAttributes) {
                        $apiUrl .= '&param=' . urlencode(json_encode($priceAttributes));
                    }
                    if ($descriptionAttributes) {
                        $apiUrl .= '&descr_param=' . urlencode(json_encode($descriptionAttributes));
                    }
                    if ($stockAttributes && $stockAttributes['tag'] !== 'null' && $stockAttributes['attributes'] !== 'null') {
                        $apiUrl .= '&stock_param=' . urlencode(json_encode($stockAttributes));
                    }

                    $apiRequests['comp_' . $key] = $apiUrl;
                }
            }
        }

        // Execute all requests in parallel
        $results = $this->executeParallelRequests($apiRequests);

        // Process main product result

        $main_product = new ClientProduct();
        $main_product->name = $db_product['name']; // Get the name from the database
        $main_product->url = $db_product['url']; // Get the URL from the database
        $main_product->price = $db_product['price']; // Get the price from the database
        $main_product->description = ''; // Get the description from the database
        $main_product->id_client_catalog = (int)$maincategory;
        if (!$main_product->add()) {
            $this->errors[] = $this->l('Failed to save client product');
            return false;
        }

        // Process competitor results
        foreach ($all_competitors as $key => $competitor) {
            if (!empty($competitor['url'])) {
                $id_competitor = Db::getInstance()->getValue(
                    "SELECT id_client_competitor FROM " . _DB_PREFIX_ . "client_competitor
                    WHERE url LIKE '%" . pSQL($this->get_domain($competitor['url'])) . "%'"
                );

                if ($id_competitor && isset($results['comp_' . $key])) {
                    $competitor_product = json_decode($results['comp_' . $key], true);

                    $comp = new CompetitorProduct();
                    $comp->name = isset($competitor_product['title']) ? $competitor_product['title'] : '';
                    $comp->url = $competitor['url'];
                    $comp->price = isset($competitor_product['price']) ? $competitor_product['price'] : '';
                    $comp->id_competitor = $id_competitor;
                    $comp->description = isset($competitor_product['description']) ? $competitor_product['description'] : '';
                    $comp->stock = isset($competitor_product['stock']) ? $competitor_product['stock'] : '';

                    if (!$comp->add()) {
                        $this->errors[] = $this->l('Failed to save competitor product');
                        continue;
                    }

                    // Create relation
                    $relation = new Products_relation();
                    $relation->id_product_client = (int)$main_product->id;
                    $relation->id_product_competitor = (int)$comp->id;

                    if (!$relation->add()) {
                        $this->errors[] = $this->l('Failed to save relation');
                        continue;
                    }
                }
            }
        }

        Tools::redirectAdmin(self::$currentIndex . '&conf=3&token=' . $this->token); // Redirect to the list with a success message
        return $main_product;
    }

    /**
     * @return false|void
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This method is used to process the update of a product.
     * It checks if the form is submitted,
     * retrieves the product name from the form,
     * fetches the product data from the database,
     * creates API requests for competitors,
     * executes the requests in parallel,
     * and processes the results.
     * It also updates the main product and competitor products,
     * and establishes the relations between them.
     */
    public function processUpdate()
    {
        // Load the main product
        $main_product = $this->loadObject();
        // Prepare the API requests array
        $apiRequests = [];


        // Prepare competitor requests
        $competitors = Tools::getValue('competitors');
        $new_competitors = Tools::getValue('new_competitors');
        $deleted_competitors = Tools::getValue('delete_competitors', array());

        // Structure competitors array
        $competitors_ = [];
        $currentIndex = -1;
        foreach ($competitors as $item) {
            if (isset($item['url'])) {
                $currentIndex++;
                $competitors_[$currentIndex] = ['url' => $item['url']];
            } elseif (isset($item['id_product']) && $currentIndex >= 0) {
                $competitors_[$currentIndex]['id_product'] = $item['id_product'];
            }
        }

        // Add existing competitor requests
        foreach ($competitors_ as $key => $competitor) {
            $existing_url = Db::getInstance()->getValue('
                SELECT count(*)
                FROM `' . _DB_PREFIX_ . 'competitor_product`
                WHERE url = "' . pSQL($competitor['url']) . '"'
            );

            if (!$existing_url) {
                $existing_product = Db::getInstance()->getRow('
                    SELECT *
                    FROM `' . _DB_PREFIX_ . 'competitor_product`
                    WHERE id_product = ' . (int)$competitor['id_product']
                );
                // Get the competitor search pattern and prepare the api request
                if ($existing_product) {
                    $priceAttributes = Db::getInstance()->getRow('SELECT price_tag AS tag, price_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$existing_product['id_competitor']);
                    $descriptionAttributes = Db::getInstance()->getRow('SELECT description_tag AS tag, description_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$existing_product['id_competitor']);
                    $stockAttributes = Db::getInstance()->getRow('SELECT stock_tag AS tag, stock_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$existing_product['id_competitor']);

                    $apiUrl = 'http://localhost:8000/api/extract-price?url=' . urlencode($competitor['url']);
                    if ($priceAttributes) {
                        $apiUrl .= '&param=' . urlencode(json_encode($priceAttributes));
                    }
                    if ($descriptionAttributes) {
                        $apiUrl .= '&descr_param=' . urlencode(json_encode($descriptionAttributes));
                    }
                    if ($stockAttributes && $stockAttributes['tag'] !== 'null' && $stockAttributes['attributes'] !== 'null') {
                        $apiUrl .= '&stock_param=' . urlencode(json_encode($stockAttributes));
                    }

                    $apiRequests['comp_' . $key] = $apiUrl;
                }
            }
        }

        // Add new competitor requests
        foreach ($new_competitors as $key => $competitor) {
            if (!empty($competitor['url'])) {
                $id_competitor = Db::getInstance()->getValue(
                    "SELECT id_client_competitor FROM " . _DB_PREFIX_ . "client_competitor
                    WHERE url LIKE '%" . pSQL($this->get_domain($competitor['url'])) . "%'"
                );
                // Get the competitor search pattern and prepare the api request
                if ($id_competitor) {
                    $priceAttributes = Db::getInstance()->getRow('SELECT price_tag AS tag, price_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$id_competitor);
                    $descriptionAttributes = Db::getInstance()->getRow('SELECT description_tag AS tag, description_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$id_competitor);
                    $stockAttributes = Db::getInstance()->getRow('SELECT stock_tag AS tag, stock_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$id_competitor);

                    $apiUrl = 'http://localhost:8000/api/extract-price?url=' . urlencode($competitor['url']);
                    if ($priceAttributes) {
                        $apiUrl .= '&param=' . urlencode(json_encode($priceAttributes));
                    }
                    if ($descriptionAttributes) {
                        $apiUrl .= '&descr_param=' . urlencode(json_encode($descriptionAttributes));
                    }
                    if ($stockAttributes && $stockAttributes['tag'] !== 'null' && $stockAttributes['attributes'] !== 'null') {
                        $apiUrl .= '&stock_param=' . urlencode(json_encode($stockAttributes));
                    }

                    $apiRequests['new_' . $key] = $apiUrl;
                }
            }
        }

        // Execute all requests in parallel
        $results = $this->executeParallelRequests($apiRequests);

        // Process main product result
        if ($main_product->name != Tools::getValue('search')) {
            $main_product->name = Tools::getValue('search');
            $db_product = Db::getInstance()->getRow("SELECT * FROM " . _DB_PREFIX_ . "target_product WHERE name = '" . pSQL($main_product->name) . "'"); // Change the table name to your target product table
            $main_product->url = $db_product['url']; // Get the URL from the database
            $main_product->price = $db_product['price']; // Get the price from the database
            $main_product->description = ''; //Get the description from the database
            if (!$main_product->update()) {
                $this->errors[] = $this->l('Failed to update client product');
                return false;
            }
        }


        // Process existing competitor results
        foreach ($competitors_ as $key => $competitor) {
            if (isset($results['comp_' . $key])) {
                $comp = new CompetitorProduct($competitor['id_product']);
                $competitor_product = json_decode($results['comp_' . $key], true);

                $comp->name = !empty($competitor_product['title']) ? $competitor_product['title'] : $comp->name;
                $comp->url = $competitor['url'];
                $comp->price = !empty($competitor_product['price']) ? $competitor_product['price'] : $comp->price;
                $comp->description = !empty($competitor_product['description']) ? $competitor_product['description'] : $comp->description;
                $comp->stock = !empty($competitor_product['stock']) ? $competitor_product['stock'] : $comp->stock;

                if (!$comp->update()) {
                    $this->errors[] = $this->l('Failed to update competitor product');
                }
            }
        }

        // Process new competitor results
        foreach ($new_competitors as $key => $competitor) {
            if (isset($results['new_' . $key])) {
                $id_competitor = Db::getInstance()->getValue(
                    "SELECT id_client_competitor FROM " . _DB_PREFIX_ . "client_competitor
                    WHERE url LIKE '%" . pSQL($this->get_domain($competitor['url'])) . "%'"
                );

                if ($id_competitor) {
                    $competitor_product = json_decode($results['new_' . $key], true);
                    $comp = new CompetitorProduct();
                    $comp->name = !empty($competitor_product['title']) ? $competitor_product['title'] : '';
                    $comp->url = $competitor['url'];
                    $comp->price = !empty($competitor_product['price']) ? $competitor_product['price'] : '';
                    $comp->id_competitor = $id_competitor;
                    $comp->description = !empty($competitor_product['description']) ? $competitor_product['description'] : '';
                    $comp->stock = !empty($competitor_product['stock']) ? $competitor_product['stock'] : '';

                    if (!$comp->add()) {
                        $this->errors[] = $this->l('Failed to save competitor product');
                        continue;
                    }
                    // Create relation
                    $relation = new Products_relation();
                    $relation->id_product_client = (int)$main_product->id;
                    $relation->id_product_competitor = (int)$comp->id;

                    if (!$relation->add()) {
                        $this->errors[] = $this->l('Failed to save relation');
                    }
                }
            }
        }
        // Update the main product's category if it has changed
        if ($main_product->id_client_catalog != (int)Tools::getValue('id_client_catalog')) {
            $main_product->id_client_catalog = (int)Tools::getValue('id_client_catalog');
            if (!$main_product->update()) {
                $this->errors[] = $this->l('Failed to update client product');
                return false;
            }
        }

        // Handle deleted competitors
        if (!empty($deleted_competitors)) {
            foreach ($deleted_competitors as $id_product) {
                Db::getInstance()->delete('products_relation', 'id_product_competitor = ' . (int)$id_product);
                $comp = new CompetitorProduct((int)$id_product);
                if (Validate::isLoadedObject($comp)) {
                    $comp->delete();
                }
            }
        }

        Tools::redirectAdmin($this->context->link->getAdminLink('AdminClientproducts', true, [], ['conf' => 4])); // Redirect to the list with a success message
    }

    /**
     * @return bool
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This method is used to process the deletion of a product.
     * It checks if the product can be loaded,
     * retrieves the related competitor products,
     * deletes them,
     * and then deletes the main product.
     */
    public function processDelete()
    {
        // Load the main product
        $object = $this->loadObject();
        if (!Validate::isLoadedObject($object)) {
            $this->errors[] = $this->trans('Object cannot be loaded (or found)', [], 'Admin.Notifications.Error');
            return false;
        }

        // Get related competitor products
        $competitor_products = Db::getInstance()->executeS('
                SELECT cp.id_product 
                FROM `' . _DB_PREFIX_ . 'products_relation` pr
                LEFT JOIN `' . _DB_PREFIX_ . 'competitor_product` cp 
                ON pr.id_product_competitor = cp.id_product
                WHERE pr.id_product_client = ' . (int)$object->id
        );

        // Delete competitor products
        if ($competitor_products) {
            foreach ($competitor_products as $cp) {
                $competitor_product = new CompetitorProduct((int)$cp['id_product']);
                if (Validate::isLoadedObject($competitor_product)) {
                    $competitor_product->delete();
                }
            }
        }

        // Delete client product (this will also delete relations due to FK constraint)
        if (!$object->delete()) {
            $this->errors[] = $this->trans('An error occurred during deletion.', [], 'Admin.Notifications.Error');
            return false;
        }

        Tools::redirectAdmin(self::$currentIndex . '&conf=1&token=' . $this->token); // Redirect to the list with a success message
        return true;
    }

    /**
     * @return false|void
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This function is used to process the bulk update of selected products.
     * It checks if any products are selected,
     * prepares API requests for each selected product,
     * executes the requests in parallel,
     * and updates the competitors' products with the extracted data.
     */
    public function processBulkUpdate()
    {
        if (empty($this->boxes) || !is_array($this->boxes)) {
            $this->errors[] = $this->l('You must select at least one item to update.');
            return false;
        }

        $apiRequests = [];
        $products = [];

        // Prepare API requests for selected products
        foreach ($this->boxes as $id_product) {
            $main_product = new ClientProduct((int)$id_product);
            if (!Validate::isLoadedObject($main_product)) {
                continue;
            }

            $products[$id_product] = $main_product;


            // Get competitor products
            $competitor_products = Db::getInstance()->executeS('
            SELECT cp.*
            FROM `' . _DB_PREFIX_ . 'competitor_product` cp
            INNER JOIN `' . _DB_PREFIX_ . 'products_relation` pr
            ON cp.id_product = pr.id_product_competitor
            WHERE pr.id_product_client = ' . (int)$id_product
            );

            // Get the competitor search pattern and prepare the api requests
            foreach ($competitor_products as $cp) {
                $priceAttributes = Db::getInstance()->getRow('SELECT price_tag AS tag, price_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);
                $descriptionAttributes = Db::getInstance()->getRow('SELECT description_tag AS tag, description_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);
                $stockAttributes = Db::getInstance()->getRow('SELECT stock_tag AS tag, stock_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);

                $apiUrl = 'http://localhost:8000/api/extract-price?url=' . urlencode($cp['url']);
                if ($priceAttributes) {
                    $apiUrl .= '&param=' . urlencode(json_encode($priceAttributes));
                }
                if ($descriptionAttributes) {
                    $apiUrl .= '&descr_param=' . urlencode(json_encode($descriptionAttributes));
                }
                if ($stockAttributes && $stockAttributes['tag'] !== 'null' && $stockAttributes['attributes'] !== 'null') {
                    $apiUrl .= '&stock_param=' . urlencode(json_encode($stockAttributes));
                }

                $apiRequests['comp_' . $id_product . '_' . $cp['id_product']] = $apiUrl;
            }
        }

        // Execute all requests in parallel
        $results = $this->executeParallelRequests($apiRequests);

        $success = 0;
        $errors = [];

        // Process results
        foreach ($products as $id_product => $main_product) {


            // Update competitors
            foreach ($results as $key => $result) {
                if (strpos($key, 'comp_' . $id_product . '_') === 0) {
                    $comp_id = substr($key, strrpos($key, '_') + 1);
                    $comp = new CompetitorProduct($comp_id);

                    if (Validate::isLoadedObject($comp)) {
                        $competitor_product = json_decode($result, true);
                        if ($competitor_product) {
                            $comp->name = !empty($competitor_product['title']) ? $competitor_product['title'] : $comp->name;
                            $comp->price = !empty($competitor_product['price']) ? $competitor_product['price'] : $comp->price;
                            $comp->description = !empty($competitor_product['description']) ? $competitor_product['description'] : $comp->description;
                            $comp->stock = !empty($competitor_product['stock']) ? $competitor_product['stock'] : $comp->stock;

                            if (!$comp->update()) {
                                $errors[] = sprintf($this->l('Failed to update competitor product %d'), $comp_id);
                            }
                        }
                    }
                }
            }
        }

        if ($success > 0) {
            $this->confirmations[] = sprintf($this->l('Successfully updated %d products and their competitors'), $success);
        }
        if (!empty($errors)) {
            $this->errors = array_merge($this->errors, $errors);
        }

        if (empty($this->errors)) {
            Tools::redirectAdmin(self::$currentIndex . '&token=' . $this->token . '&conf=4');
        } else {
            Tools::redirectAdmin(self::$currentIndex . '&token=' . $this->token);
        }
    }

    /**
     * @return void
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This function is used to execute parallel requests to the API for extracting product data.
     * It takes the product ID from the request, fetches the competitor products,
     * and updates the competitors' products with the extracted data.
     */
    public function updateproduct()
    {
        // Get the product ID from the request
        $id_product = (int)Tools::getValue('id_product');
        $apiRequests = [];

        if (!$id_product) {
            die(json_encode([
                'success' => false,
                'error' => $this->l('Invalid product ID')
            ]));
        }

        // Load client product
        $main_product = new ClientProduct($id_product);
        if (!Validate::isLoadedObject($main_product)) {
            die(json_encode([
                'success' => false,
                'error' => $this->l('Product not found')
            ]));
        }


        // Get competitor products
        $competitor_products = Db::getInstance()->executeS('
        SELECT cp.*
        FROM `' . _DB_PREFIX_ . 'competitor_product` cp
        INNER JOIN `' . _DB_PREFIX_ . 'products_relation` pr
        ON cp.id_product = pr.id_product_competitor
        WHERE pr.id_product_client = ' . (int)$id_product
        );

        // Add competitor requests
        foreach ($competitor_products as $key => $cp) {
            $priceAttributes = Db::getInstance()->getRow('SELECT price_tag AS tag, price_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);
            $descriptionAttributes = Db::getInstance()->getRow('SELECT description_tag AS tag, description_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);
            $stockAttributes = Db::getInstance()->getRow('SELECT stock_tag AS tag, stock_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);

            $apiUrl = 'http://localhost:8000/api/extract-price?url=' . urlencode($cp['url']);
            if ($priceAttributes) {
                $apiUrl .= '&param=' . urlencode(json_encode($priceAttributes));
            }
            if ($descriptionAttributes) {
                $apiUrl .= '&descr_param=' . urlencode(json_encode($descriptionAttributes));
            }
            if ($stockAttributes && $stockAttributes['tag'] !== 'null' && $stockAttributes['attributes'] !== 'null') {
                $apiUrl .= '&stock_param=' . urlencode(json_encode($stockAttributes));
            }

            $apiRequests['comp_' . $key] = $apiUrl;
        }

        // Execute all requests in parallel
        $results = $this->executeParallelRequests($apiRequests);


        // Process competitor results
        $update_errors = [];
        foreach ($competitor_products as $key => $cp) {
            if (isset($results['comp_' . $key])) {
                $comp = new CompetitorProduct($cp['id_product']);
                if (Validate::isLoadedObject($comp)) {
                    $competitor_product = json_decode($results['comp_' . $key], true);
                    if ($competitor_product) {
                        $current_price = $comp->price;
                        $current_stock = $comp->stock;

                        if (!isset($competitor_product['price']) || $current_price != $competitor_product['price'] || $current_stock != $competitor_product['stock']) {
                            $comp->name = isset($competitor_product['title']) ? $competitor_product['title'] : $comp->name;
                            $comp->price = isset($competitor_product['price']) ? $competitor_product['price'] : $comp->price;
                            $comp->description = isset($competitor_product['description']) ? $competitor_product['description'] : $comp->description;
                            $comp->stock = isset($competitor_product['stock']) ? $competitor_product['stock'] : $comp->stock;

                            if (!$comp->update()) {
                                $update_errors[] = sprintf($this->l('Failed to update competitor product %d'), $cp['id_product']);
                            }
                        }
                    }
                }
            }
        }

        die(json_encode([
            'success' => empty($update_errors),
            'errors' => $update_errors,
            'message' => empty($update_errors) ?
                $this->l('Products updated successfully') :
                $this->l('Some products failed to update')
        ]));
    }

    /**
     * @return void
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This method is used to process the update of all products.
     * It retrieves all client products from the database,
     * fetches their related competitor products,
     * creates API requests for each competitor,
     * executes the requests in parallel,
     * and updates the competitor products with the results.
     */
    public function processUpdateAll()
    {
        // Get all client products
        $products = Db::getInstance()->executeS('
            SELECT id_product 
            FROM `' . _DB_PREFIX_ . 'client_product`
        ');

        $errors = [];
        $updated = 0;

        // Process each client product
        foreach ($products as $product) {
            // Get competitor products for this client product
            $competitor_products = Db::getInstance()->executeS('
                SELECT cp.*
                FROM `' . _DB_PREFIX_ . 'competitor_product` cp
                INNER JOIN `' . _DB_PREFIX_ . 'products_relation` pr
                ON cp.id_product = pr.id_product_competitor
                WHERE pr.id_product_client = ' . (int)$product['id_product']
            );

            // Prepare competitor API requests
            $competitorRequests = [];
            foreach ($competitor_products as $cp) {
                $priceAttributes = Db::getInstance()->getRow('SELECT price_tag AS tag, price_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);
                $descriptionAttributes = Db::getInstance()->getRow('SELECT description_tag AS tag, description_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);
                $stockAttributes = Db::getInstance()->getRow('SELECT stock_tag AS tag, stock_attribute AS attributes FROM `' . _DB_PREFIX_ . 'competitor_pattern` WHERE id_competitor = ' . (int)$cp['id_competitor']);

                $apiUrl = 'http://localhost:8000/api/extract-price?url=' . urlencode($cp['url']);
                if ($priceAttributes) {
                    $apiUrl .= '&param=' . urlencode(json_encode($priceAttributes));
                }
                if ($descriptionAttributes) {
                    $apiUrl .= '&descr_param=' . urlencode(json_encode($descriptionAttributes));
                }
                if ($stockAttributes && $stockAttributes['tag'] !== 'null' && $stockAttributes['attributes'] !== 'null') {
                    $apiUrl .= '&stock_param=' . urlencode(json_encode($stockAttributes));
                }

                $competitorRequests[$cp['id_product']] = $apiUrl;
            }

            // Execute parallel requests for competitors
            $competitorResults = $this->executeParallelRequests($competitorRequests);

            // Update competitor products
            foreach ($competitorResults as $comp_id => $comp_response) {
                $comp_result = json_decode($comp_response, true);
                $comp = new CompetitorProduct($comp_id);

                if (Validate::isLoadedObject($comp) && $comp_result) {
                    $comp->name = isset($comp_result['title']) ? $comp_result['title'] : $comp->name;
                    $comp->price = isset($comp_result['price']) ? $comp_result['price'] : $comp->price;
                    $comp->description = isset($comp_result['description']) ? $comp_result['description'] : $comp->description;
                    $comp->stock = isset($comp_result['stock']) ? $comp_result['stock'] : $comp->stock;

                    if ($comp->update()) {
                        $updated++;
                    } else {
                        $errors[] = sprintf($this->l('Failed to update competitor product %d'), $comp_id);
                    }
                }
            }
        }

        if (empty($errors)) {
            $this->confirmations[] = sprintf($this->l('Successfully updated %d competitor products'), $updated);
        } else {
            $this->errors = array_merge($this->errors, $errors);
        }

        Tools::redirectAdmin($this->context->link->getAdminLink('AdminClientproducts', true, [], ['conf' => empty($errors) ? 4 : 1]));
    }

    /**
     * @param $url
     * @return bool|string
     * This method is used to call an API endpoint.
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
     * @param $urls
     * @return array
     * This method is used to execute multiple cURL requests in parallel.
     */
    private function executeParallelRequests($urls)
    {
        $mh = curl_multi_init();
        $handles = [];

        // Initialize cURL handles for each URL
        foreach ($urls as $key => $url) {
            $handles[$key] = curl_init();
            curl_setopt($handles[$key], CURLOPT_URL, $url);
            curl_setopt($handles[$key], CURLOPT_RETURNTRANSFER, 1);
            curl_multi_add_handle($mh, $handles[$key]);
        }

        // Execute requests in parallel
        $running = null;
        do {
            curl_multi_exec($mh, $running);
        } while ($running);

        // Get results
        $results = [];
        foreach ($handles as $key => $ch) {
            $results[$key] = curl_multi_getcontent($ch);
            curl_multi_remove_handle($mh, $ch);
            curl_close($ch);
        }

        curl_multi_close($mh);
        return $results;
    }


    /**
     * @param $url
     * @return array|false|string|string[]|null
     * This method is used to extract the domain from a URL.
     */
    private function get_domain($url)
    {
        if (!preg_match("~^(?:f|ht)tps?://~i", $url)) {
            $url = "http://" . $url;
        }
        $pieces = parse_url($url);
        if (!isset($pieces['host'])) {
            return false;
        }
        $domain = $pieces['host'];
        return preg_replace('/^www\./', '', strtolower($domain));
    }


    /**
     * @return void
     * This method is used in AJAX calls to check if a competitor already exists in the database.
     * It retrieves the URL from the request,
     * queries the database for a matching competitor,
     * and returns the result as a JSON response.
     */
    public function ajaxProcessCheckCompetitor()
    {
        $url = Tools::getValue('url');

        $competitor = Db::getInstance()->getRow('
            SELECT id_competitor, url 
            FROM `' . _DB_PREFIX_ . 'competitor_product` 
            WHERE url = "' . pSQL($url) . '"'
        );

        die(json_encode(array(
            'exists' => !empty($competitor),
            'id_competitor' => $competitor ? $competitor['id_competitor'] : null,
            'name' => $competitor ? $competitor['name'] : null
        )));
    }


    /**
     * @return void
     * @throws PrestaShopDatabaseException
     * @throws PrestaShopException
     * This method is used to process the AJAX request for searching target products.
     * It retrieves the search query from the request,
     * queries the database for matching products,
     * and returns the results as a JSON response.
     * !!!Change the query to match your database structure!!!!
     */
    public function ajaxProcessSearchTargetProducts()
    {
        $query = Tools::getValue('q');
        $html = '';

        if (strlen($query) >= 3) {
            $products = Db::getInstance()->executeS('
            SELECT * 
            FROM `' . _DB_PREFIX_ . 'target_product` 
            WHERE name LIKE "%' . pSQL($query) . '%" 
            LIMIT 10
        ');

            if ($products) {
                foreach ($products as $product) {
                    $html .= '<li class="search-item" data-url="' . htmlspecialchars($product['url']) . '">'
                        . '<span class="product-name">' . htmlspecialchars($product['name']) . '</span>'
                        . '<small class="text-muted"><strong>URL: </strong>' . htmlspecialchars($product['url']) . '</small><br>'
                        . ' <small class="text-muted"><strong>Price: </strong>' . htmlspecialchars($product['price']) . '</small>'
                        . '</li>';
                }
            } else {
                $html = '<li class="no-results">' . $this->l('No products found') . '</li>';
            }
        }

        die(json_encode([
            'success' => true,
            'html' => $html,
        ]));
    }

}