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

class client_comparing extends Module
{
    public function __construct()
    {
        $this->name = 'client_comparing';
        $this->tab = 'SEO';
        $this->version = '1.0.0';
        $this->author = 'Mohamed Selim Khader';
        $this->need_instance = 1;

        /**
         * Set $this->bootstrap to true if your module is compliant with bootstrap (PrestaShop 1.6)
         */
        $this->bootstrap = true;

        parent::__construct();

        $this->displayName = $this->l('Client Compare');
        $this->description = $this->l('Determine Similar Products');

        $this->ps_versions_compliancy = array('min' => '1.6', 'max' => '8.0');
    }

    private function installTab()
    {
        $parentTab = new Tab();
        $parentTab->active = 1;
        $parentTab->class_name = 'AdminCompare';
        $parentTab->name = array();
        foreach (Language::getLanguages(true) as $lang) {
            $parentTab->name[$lang['id_lang']] = 'Client Compare';
        }
        $parentTab->id_parent = 0;
        $parentTab->module = $this->name;
        $parentTab->add();
        Db::getInstance()->execute('
        UPDATE `'._DB_PREFIX_.'tab`
        SET position = 2
        WHERE class_name = "AdminCompare"');
        $tab = new Tab();
        $tab->active = 1;
        $tab->class_name = 'AdminCompareproducts';
        $tab->name = array();
        foreach (Language::getLanguages(true) as $lang) {
            $tab->name[$lang['id_lang']] = 'Compare Products';
        }
        $tab->id_parent = (int)Tab::getIdFromClassName('AdminCompare');
        $tab->module = $this->name;
        $tab->icon = 'business';
        $tab->add();

        return true;

    }

    public function uninstallTab(){
        $id_tab = (int)Tab::getIdFromClassName('AdminCompareproducts');
        $tab = new Tab($id_tab);
        $tab->delete();

        $id_tab = (int)Tab::getIdFromClassName('AdminCompare');
        $tab = new Tab($id_tab);
        $tab->delete();

        return true;
    }
    public function install()
    {
        include(dirname(__FILE__).'/sql/install.php');

        return parent::install() &&
            $this->installTab() &&
            $this->registerHook('header') &&
            $this->registerHook('displayBackOfficeHeader');
    }

    public function uninstall()
    {
        include(dirname(__FILE__).'/sql/uninstall.php');
        return parent::uninstall() && $this->uninstallTab();
    }


}
