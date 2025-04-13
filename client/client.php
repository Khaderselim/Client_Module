<?php
/**
* 2007-2025 PrestaShop
*
* NOTICE OF LICENSE
*
* This source file is subject to the Academic Free License (AFL 3.0)
* that is bundled with this package in the file LICENSE.txt.
* It is also available through the world-wide-web at this URL:
* http://opensource.org/licenses/afl-3.0.php
* If you did not receive a copy of the license and are unable to
* obtain it through the world-wide-web, please send an email
* to license@prestashop.com so we can send you a copy immediately.
*
* DISCLAIMER
*
* Do not edit or add to this file if you wish to upgrade PrestaShop to newer
* versions in the future. If you wish to customize PrestaShop for your
* needs please refer to http://www.prestashop.com for more information.
*
*  @author    PrestaShop SA <contact@prestashop.com>
*  @copyright 2007-2025 PrestaShop SA
*  @license   http://opensource.org/licenses/afl-3.0.php  Academic Free License (AFL 3.0)
*  International Registered Trademark & Property of PrestaShop SA
*/

if (!defined('_PS_VERSION_')) {
    exit;
}

class Client extends Module
{

    public function __construct()
    {
        $this->name = 'client';
        $this->tab = 'SEO';
        $this->version = '1.0.0';
        $this->author = 'Mohamed Selim Khader';
        $this->need_instance = 1;

        /**
         * Set $this->bootstrap to true if your module is compliant with bootstrap (PrestaShop 1.6)
         */
        $this->bootstrap = true;

        parent::__construct();

        $this->displayName = $this->l('Product Tracking');
        $this->description = $this->l('Tracking similar products');

        $this->ps_versions_compliancy = array('min' => '1.6', 'max' => '8.0');
    }
    private function installTab(){
        $parentTab = new Tab();
        $parentTab->active = 1;
        $parentTab->class_name = 'AdminClient';
        $parentTab->name = array();
        foreach (Language::getLanguages(true) as $lang)
            $parentTab->name[$lang['id_lang']] = 'SEO Target';
        $parentTab->id_parent = 0;
        $parentTab->module = $this->name;
        $parentTab->add();
        Db::getInstance()->execute('
        UPDATE `'._DB_PREFIX_.'tab`
        SET position = 1
        WHERE class_name = "AdminClient"
    ');

        $tab = new Tab();
        $tab->active = 1;
        $tab->class_name = 'AdminClientCompetitors';
        $tab->name = array();
        foreach (Language::getLanguages(true) as $lang)
            $tab->name[$lang['id_lang']] = 'Competitors';
        $tab->id_parent = (int)Tab::getIdFromClassName('AdminClient');
        $tab->module = $this->name;
        $tab->icon = 'business';
        $tab->add();

        $productTab = new Tab();
        $productTab->active = 1;
        $productTab->class_name = 'AdminClientproducts';
        $productTab->name = array();
        foreach (Language::getLanguages(true) as $lang)
            $productTab->name[$lang['id_lang']] = 'Products';
        $productTab->id_parent = (int)Tab::getIdFromClassName('AdminClient');
        $productTab->module = $this->name;
        $productTab->icon = 'category';
        $productTab->add();

        $catalogTab = new Tab();
        $catalogTab->active = 1;
        $catalogTab->class_name = 'AdminClientcatalog';
        $catalogTab->name = array();
        foreach (Language::getLanguages(true) as $lang)
            $catalogTab->name[$lang['id_lang']] = 'Catalog';
        $catalogTab->id_parent = (int)Tab::getIdFromClassName('AdminClient');
        $catalogTab->module = $this->name;
        $catalogTab->icon = 'category';
        $catalogTab->add();
        return true;
    }

    /**
     * Don't forget to create update methods if needed:
     * http://doc.prestashop.com/display/PS16/Enabling+the+Auto-Update
     */
    public function install()
    {
        include(dirname(__FILE__).'/sql/install.php');

        return parent::install() && $this->installTab()&&
            $this->registerHook('header') &&
            $this->registerHook('displayBackOfficeHeader');
    }

    public function uninstall()
    {
        include(dirname(__FILE__).'/sql/uninstall.php');

        return parent::uninstall() && $this->uninstallTab();
    }

    public function uninstallTab(){
        $id_tab = (int)Tab::getIdFromClassName('AdminClientCompetitors');
        $tab = new Tab($id_tab);
        $tab->delete();
        $id_tab = (int)Tab::getIdFromClassName('AdminClientproducts');
        $tab = new Tab($id_tab);
        $tab->delete();
        return true;
    }

    public function hookHeader()
    {
        $this->context->controller->addJS($this->_path.'/views/js/front.js');
        $this->context->controller->addCSS($this->_path.'/views/css/front.css');
    }
}
