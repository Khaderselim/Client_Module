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
$sql = array();

$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'client_comparing_product` (
    `id_comparing_product` int(11) NOT NULL AUTO_INCREMENT,
    `id_product` int(11) NOT NULL,
    `product_brands` varchar(255) NOT NULL,
    `competitor_product_brands` varchar(255) NOT NULL,
    `id_competitor_product` int(11) NOT NULL,
    `similarity` float NOT NULL,
    PRIMARY KEY (`id_comparing_product`),
    UNIQUE KEY `unique_comparison` (`id_product`, `id_competitor_product`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;';

$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'client_suggestion_product` (
    `id_suggestion_product` int(11) NOT NULL AUTO_INCREMENT,
    `id_product` int(11) NOT NULL,
    `product_brands` varchar(255) NOT NULL,
    `competitor_product_brands` varchar(255) NOT NULL,
    `id_competitor_product` int(11) NOT NULL,
    `similarity` float NOT NULL,
    PRIMARY KEY (`id_suggestion_product`),
    UNIQUE KEY `unique_comparison` (`id_product`, `id_competitor_product`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;';

$sql[] = 'DROP TRIGGER IF EXISTS `'._DB_PREFIX_.'delete_client_comparing_product_competitor`;';
$sql[] = 'CREATE TRIGGER `'._DB_PREFIX_.'delete_client_comparing_product_competitor` 
    AFTER DELETE ON `'._DB_PREFIX_.'competitor_product`
    FOR EACH ROW 
    BEGIN
        DELETE FROM `'._DB_PREFIX_.'comparing_product` 
        WHERE id_competitor_product = OLD.id_product;
        DELETE FROM `'._DB_PREFIX_.'suggestion_product` 
        WHERE id_competitor_product = OLD.id_product;
    END;';

$sql[] = 'DROP TRIGGER IF EXISTS `'._DB_PREFIX_.'delete_client_comparing_product_main`;';
$sql[] = 'CREATE TRIGGER `'._DB_PREFIX_.'delete_client_comparing_product_main` 
    AFTER DELETE ON `'._DB_PREFIX_.'product`
    FOR EACH ROW 
    BEGIN
        DELETE FROM `'._DB_PREFIX_.'client_comparing_product` 
        WHERE id_product = OLD.id_product;
        DELETE FROM `'._DB_PREFIX_.'client_suggestion_product` 
        WHERE id_product = OLD.id_product;
    END;';

$sql[] = 'CREATE TRIGGER '._DB_PREFIX_.'delete_from_client_suggestion_after_compare_insert
    AFTER INSERT ON `'._DB_PREFIX_.'client_comparing_product`
    FOR EACH ROW
BEGIN
    DELETE FROM `'._DB_PREFIX_.'client_suggestion_product`
    WHERE id_product = NEW.id_product
      AND id_competitor_product = NEW.id_competitor_product;
END;';

foreach ($sql as $query) {
    if (Db::getInstance()->execute($query) == false) {
        return false;
    }
}
