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

$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'client_competitor` (
    `id_client_competitor` int(11) NOT NULL AUTO_INCREMENT,
    `name` varchar(255) NOT NULL,
    `logo` varchar(255) NOT NULL,
    `url` varchar(255) NOT NULL,
    `priority` int(1) NOT NULL DEFAULT 1,
    `active` tinyint(1) NOT NULL DEFAULT 1,
    PRIMARY KEY (`id_client_competitor`)
) ENGINE='._MYSQL_ENGINE_.' DEFAULT CHARSET=utf8;';

$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'client_catalog` (
    `id_client_catalog` int(11) NOT NULL AUTO_INCREMENT,
    `name` varchar(255) NOT NULL,
    `active` tinyint(1) NOT NULL DEFAULT 1,
    PRIMARY KEY (`id_client_catalog`),
    UNIQUE KEY `name` (`name`)
) ENGINE='._MYSQL_ENGINE_.' DEFAULT CHARSET=utf8;';

$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'competitor_pattern` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `id_competitor` int(11) NOT NULL,
    `price_tag` varchar(50) NOT NULL,
    `price_attribute` text NOT NULL,
    `description_tag` varchar(50) NOT NULL,
    `description_attribute` text,
    `stock_tag` varchar(50),
    `stock_attribute` text,
    PRIMARY KEY (`id`),
    KEY `id_competitor` (`id_competitor`),
    CONSTRAINT `fk_competitor_price` FOREIGN KEY (`id_competitor`) 
    REFERENCES `'._DB_PREFIX_.'client_competitor` (`id_client_competitor`) ON DELETE CASCADE
) ENGINE='._MYSQL_ENGINE_.' DEFAULT CHARSET=utf8;';


$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'client_product` (
    `id_client_product` int(11) NOT NULL AUTO_INCREMENT,
    `id_product` int(11) NOT NULL,
    `name` varchar(255) NOT NULL,
    `img_url` varchar(255) NOT NULL,
    `price` varchar(255) NOT NULL,
    `description` text NOT NULL,
    `id_client_catalog` int(11) NULL,
    `date_add` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id_client_product`),
    FOREIGN KEY (`id_client_catalog`) 
        REFERENCES `'._DB_PREFIX_.'client_catalog` (`id_client_catalog`) 
        ON DELETE SET NULL
) ENGINE='._MYSQL_ENGINE_.' DEFAULT CHARSET=utf8;';

$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'competitor_product` (
    `id_product` int(11) NOT NULL AUTO_INCREMENT,
    `id_competitor` int(11) NOT NULL,
    `name` varchar(255) NOT NULL,
    `url` varchar(255) NOT NULL UNIQUE,
    `price` varchar(255) NOT NULL,
    `description` text NOT NULL,
    `stock` varchar(255) NOT NULL,
    `date_add` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id_product`),
    CONSTRAINT `fk_competitor_product` FOREIGN KEY (`id_competitor`) 
    REFERENCES `'._DB_PREFIX_.'client_competitor` (`id_client_competitor`) ON DELETE CASCADE
) ENGINE='._MYSQL_ENGINE_.' DEFAULT CHARSET=utf8;';

$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'products_relation` (
    `id` int(11) NOT NULL AUTO_INCREMENT,
    `id_product_competitor` int(11) NOT NULL,
    `id_product_client` int(11) NOT NULL,
    PRIMARY KEY (`id`),
    CONSTRAINT `fk_competitor_relation` FOREIGN KEY (`id_product_competitor`) 
    REFERENCES `'._DB_PREFIX_.'competitor_product` (`id_product`) ON DELETE CASCADE,
    CONSTRAINT `fk_client_relation` FOREIGN KEY (`id_product_client`) 
    REFERENCES `'._DB_PREFIX_.'client_product` (`id_client_product`) ON DELETE CASCADE
) ENGINE='._MYSQL_ENGINE_.' DEFAULT CHARSET=utf8;';

$sql[] = 'CREATE TABLE IF NOT EXISTS `'._DB_PREFIX_.'competitor_products_history` (
    `id_history` int(11) NOT NULL AUTO_INCREMENT,
    `id_product` int(11) NOT NULL,
    `old_price` varchar(255) NOT NULL,
    `new_price` varchar(255) NOT NULL,
    `date_update` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id_history`),
    KEY `id_product` (`id_product`),
    CONSTRAINT FOREIGN KEY (`id_product`) REFERENCES `'._DB_PREFIX_.'competitor_product` (`id_product`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE='._MYSQL_ENGINE_.' DEFAULT CHARSET=utf8;';

$sql[] = 'CREATE TRIGGER competitor_product_price_history
BEFORE UPDATE ON `'._DB_PREFIX_.'competitor_product`
FOR EACH ROW
BEGIN
    IF NEW.price != OLD.price THEN
        INSERT INTO `'._DB_PREFIX_.'competitor_products_history` (id_product, old_price, new_price)
        VALUES (OLD.id_product, OLD.price, NEW.price);
        
        DELETE FROM `'._DB_PREFIX_.'competitor_products_history`
        WHERE id_product = OLD.id_product
        AND id_history NOT IN (
            SELECT id_history FROM (
                SELECT id_history
                FROM `'._DB_PREFIX_.'competitor_products_history`
                WHERE id_product = OLD.id_product
                ORDER BY date_update DESC
                LIMIT 3
            ) tmp
        );
    END IF; 
END;';

foreach ($sql as $query) {
    if (Db::getInstance()->execute($query) == false) {
        return false;
    }
}
