<?php
/**
 * This file is part of the Client module for PrestaShop.
 * It manages the competitor functionality.
 */
class Competitor extends ObjectModel
{
    /** @var int $id_client_competitor The unique identifier for the competitor

     * @var string $name The name of the competitor
     * @var string $logo The logo of the competitor
     * @var string $url The URL of the competitor
     * @var int $priority The priority level of the competitor
     * @var bool $active Indicates if the competitor is active
     */
    public $id_client_competitor;
    public $name;
    public $logo;
    public $url;
    public $priority;
    public $active;

    /**
     * Definition of the database table and fields for the Competitors model.
     */
    public static $definition = array(
        'table' => 'client_competitor',
        'primary' => 'id_client_competitor',
        'fields' => array(
            'name' => array('type' => self::TYPE_STRING, 'validate' => 'isGenericName', 'required' => true),
            'logo' => array('type' => self::TYPE_STRING),
            'url' => array('type' => self::TYPE_STRING, 'validate' => 'isUrl', 'required' => true),
            'priority' => array('type' => self::TYPE_INT, 'validate' => 'isInt'),
            'active' => array('type' => self::TYPE_BOOL, 'validate' => 'isBool')
        )
    );
}
