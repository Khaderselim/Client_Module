<?php
require_once _PS_MODULE_DIR_ . 'client_comparing/classes/ComparingProducts.php';
require_once _PS_MODULE_DIR_ . 'client_comparing/classes/SuggestionProducts.php';

class AdminCompareproductsController extends ModuleAdminController{
    private $apiConfig;
    private $suggestion_fields_list; // Second table fields list

    public function __construct()
    {
        $this->bootstrap = true;
        $this->table = 'client_comparing_product';
        $this->className = 'ComparingProducts';
        $this->lang = false;
        $this->identifier = 'id_comparing_product';
        $this->orderBy = 'id_comparing_product';
        $this->orderWay = 'ASC';
        $this->list_no_link = true; // Disable default link to edit targets
        $this->_group="GROUP BY id_product";
        $this->apiConfig = require_once _PS_MODULE_DIR_.'client_comparing/config/api.php';


        parent::__construct();
        // Declare the fields for the list of target websites
        $this->fields_list = array(
            'id_comparing_product' => array(
                'title' => $this->l('ID'),
                'align' => 'center',
                'width' => 'auto'
            ),
            'image' => array(
                'title' => $this->l('Image'),
                'width' => 'auto',
                'align' => 'left',
            ),
            'name' => array(
                'title' => $this->l('Name'),
                'width' => 'auto',
                'align' => 'left',
            ),
            'product_brands' => array(
                'title' => $this->l('Brands'),
                'width' => 'auto',
                'align' => 'left',
            ),
            'price' => array(
                'title' => $this->l('Price'),
                'width' => 'auto',
                'align' => 'left',
            ),
            'competitor_product' => array(
                'title' => $this->l('Competitor Product'),
                'width' => 'auto',
                'align' => 'left',
            ),

        );
        $this->addRowAction('Details');


        $this->suggestion_fields_list = array(
            'id_suggestion_product' => array(
                'title' => $this->l('ID'),
                'align' => 'center',
                'width' => 'auto'
            ),
            'image' => array(
                'title' => $this->l('Image'),
                'width' => 'auto',
                'align' => 'left',
            ),
            'name' => array(
                'title' => $this->l('Name'),
                'width' => 'auto',
                'align' => 'left',
            ),
            'product_brands' => array(
                'title' => $this->l('Brands'),
                'width' => 'auto',
                'align' => 'left',
            ),
            'price' => array(
                'title' => $this->l('Price'),
                'width' => 'auto',
                'align' => 'left',
            ),
            'suggestion_competitor_product' => array(
                'title' => $this->l('Competitor Product'),
                'width' => 'auto',
                'align' => 'left',
            ),


        );

    }
    public function initContent()
    {
        parent::initContent();
        $client_catalog = Db::getInstance()->executeS('
        SELECT id_client_catalog, name 
        FROM `'._DB_PREFIX_.'client_catalog` 
        WHERE active = 1
    ');

        // Assign to template
        $this->context->smarty->assign([
            'client_catalogs' => $client_catalog
        ]);
        $action = Tools::getValue('action');
        if($action == 'compare') {
            $this->ProcessCompare();
        }
        elseif($action=='addCompare'){
            $this->addCompare();
        }

        // Main table configuration
        $this->getList($this->context->language->id);
        $helper = new HelperList();
        $this->setHelperDisplay($helper);
        $helper->title = $this->l('Tracking products');
        $helper->identifier = $this->identifier;
        $helper->table = $this->table;
        $helper->token = $this->token;
        $helper->currentIndex = self::$currentIndex;
        $list = $helper->generateList($this->_list, $this->fields_list);

        // Suggestion table configuration - copy same setup as main table
        $suggestion_helper = new HelperList();
        $this->setHelperDisplay($suggestion_helper); // Add this line to copy display settings
        $suggestion_helper->title = $this->l('Suggestion products');
        $suggestion_helper->table = 'client_suggestion_product';
        $suggestion_helper->identifier = 'id_suggestion_product';
        $suggestion_helper->token = $this->token;
        $suggestion_helper->currentIndex = self::$currentIndex;

        // Get suggestion products using same structure as main table
        $suggestionQuery = new DbQuery();
        $suggestionQuery->select('*');
        $suggestionQuery->from('client_suggestion_product');
        $suggestionQuery->groupBy('id_product');
        $suggestion_items = Db::getInstance()->executeS($suggestionQuery);

        $suggestion_list = $suggestion_helper->generateList($suggestion_items, $this->suggestion_fields_list);

        // Assign both lists to template
        $this->context->smarty->assign(array(
            'products_list' => $list,
            'competitor_list' => $suggestion_list,
            'title' => $this->l('Product Comparison'),
            'toolbar_btn' => $this->toolbar_btn,
            'show_toolbar' => $this->show_toolbar,
            'toolbar_scroll' => $this->toolbar_scroll,
            'current_index' => self::$currentIndex,
            'token' => $this->token,
        ));

        $this->content = $this->createTemplate('product_comparison.tpl')->fetch();
        $this->context->smarty->assign(array(
            'content' => $this->content,
        ));
    }
    public function initToolbar()
    {
    }

    public function renderDetails()
    {
        $object = $this->loadObject(true);
        $product_link = Context::getContext()->link->getProductLink($object->id_product);
        Tools::redirectAdmin($product_link);

    }

    public function initPageHeaderToolbar(){
        parent::initPageHeaderToolbar();
        $this->page_header_toolbar_btn['Compare'] = [
            'href' => self::$currentIndex.'&action=compare&token='.$this->token,
            'desc' => $this->l('Compare'),
            'icon' => 'process-icon-refresh'
        ];
    }
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
    public function ProcessCompare()
    {

        try {
            $apiUrl = $this->apiConfig['api_base_url'].'/api/client_compare?host='._DB_SERVER_.'&user='._DB_USER_.'&passwd='._DB_PASSWD_.'&database='._DB_NAME_.'&database_prefix='._DB_PREFIX_;
            $response = $this->callApi($apiUrl);

            $result = json_decode($response, true);
            if (!$result || !isset($result['success'])) {
                throw new Exception('Invalid API response');
            }

            if (!$result['success']) {
                throw new Exception($result['error'] ?? 'API error occurred');
            }


        } catch (Exception $e) {

        }
        Db::getInstance()->execute('
            DELETE sp FROM '._DB_PREFIX_.'client_suggestion_product sp
            INNER JOIN '._DB_PREFIX_.'client_comparing_product cp
            ON sp.id_product = cp.id_product
            AND sp.id_competitor_product = cp.id_competitor_product
        ');

        Tools::redirectAdmin(self::$currentIndex.'&token='.$this->token);

    }

    public function addCompare()
    {
        $response = array('success' => false, 'message' => '');

        try {

            // Get and validate input values
            $id_product = (int)Tools::getValue('id_product');
            $id_competitor_product = (int)Tools::getValue('id_competitor_product');
            $id_client_catalog = (int)Tools::getValue('id_client_catalog');

            if (!$id_product || !$id_competitor_product) {
                throw new Exception('Missing required parameters'.' ID Product: ' . $id_product . ' Competitor Product ID: ' . $id_competitor_product);
            }
            $suggestion_value = Db::getInstance()->getRow('SELECT * FROM '._DB_PREFIX_.'client_suggestion_product WHERE id_product = '.(int)$id_product .' AND id_competitor_product = '.(int)$id_competitor_product);
            // Debug log
            $id_target_product = Db::getInstance()->getValue('SELECT id_client_product FROM '._DB_PREFIX_.'client_product WHERE id_product = '.(int)$id_product .' AND id_client_catalog = '.(int)$id_client_catalog);
            if(!$id_target_product){
                $product = Db::getInstance()->getRow('
                SELECT 
                    prl.name,
                    prl.description,
                    pr.price,
                    pr.id_product,
                    i.id_image 
                FROM `' . _DB_PREFIX_ . 'product` pr
                LEFT JOIN `' . _DB_PREFIX_ . 'product_lang` prl ON prl.id_product = pr.id_product
                LEFT JOIN `' . _DB_PREFIX_ . 'image` i ON i.id_product = pr.id_product AND i.cover = 1
                WHERE prl.id_product = '.(int)$id_product.'
            ');

                if($product){
                    $currency = Currency::getDefaultCurrency();
                    $iso_code = $currency->iso_code;
                    Db::getInstance()->insert('client_product', array(
                        'id_product' => (int)$id_product,
                        'id_client_catalog' => (int)$id_client_catalog,
                        'name' => pSQL($product['name']),
                        'price' =>  number_format((float)$product['price'],3,',',' ').' '.$iso_code,
                        'img_url' => Context::getContext()->link->getImageLink($product['name'],$product['id_image']),
                        'description' => pSQL($product['description']),
                    ));
                }

                $id_target_product = Db::getInstance()->getValue('SELECT id_client_product FROM '._DB_PREFIX_.'client_product WHERE id_product = '.(int)$id_product .' AND id_client_catalog = '.(int)$id_client_catalog);
            }

            Db::getInstance()->insert('client_comparing_product', array(
                'id_product'=>$suggestion_value['id_product'],
                'product_brands'=>$suggestion_value['product_brands'],
                'competitor_product_brands'=>$suggestion_value['competitor_product_brands'],
                'id_competitor_product'=>$suggestion_value['id_competitor_product'],
                'similarity'=>$suggestion_value['similarity'],
            ));
            Db::getInstance()->insert('products_relation', array(
                'id_product_client' => (int)$id_target_product,
                'id_product_competitor' => (int)$id_competitor_product,
            ));
            // Send response
            $response['success'] = true;
            $response['message'] = 'Product added successfully' . ' with ID: ' . $id_product . ' and Competitor Product ID: ' . $id_competitor_product . ' and Target Website ID: ' . $id_client_catalog;

        } catch (Exception $e) {
            $response['message'] = $e->getMessage();
        }

        header('Content-Type: application/json');
        die(json_encode($response));
    }




}