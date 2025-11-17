"""
Demographic mapping configurations for WVS data processing.

This module contains all the static mapping data for converting demographic
values across different years and countries.
"""

# Townsize mappings for 2012 data
TOWNSIZE_MAP_2012 = {
    "all": {
        "under 5,000": ["rural"],
        "5000-20000": ["rural", "urban"],
        "20000-100000": ["rural", "urban"],
        "100000-500000": ["rural", "urban"],
        "500000 and more": ["urban"]
    },
    "japan": {
        "jp: rural districts": "rural",
        "jp:cities with populations less than 100,000": "rural",
        "jp:cities with populations from 100,000 to under 200,000": "urban",
        "jp:cities with populations of 200,000 or more": "urban",
        "jp:18 major large cities": "urban"
    }
}

# Townsize mappings for 2006 data
TOWNSIZE_MAP_2006 = {
    "all": {
        "under 5,000": ["rural"],
        "5000-20000": ["rural", "urban"],
        "20000-100000": ["rural", "urban"],
        "100000-500000": ["rural", "urban"],
        "500000 and more": ["urban"]
    },
    "japan": {
        "jp: rural districts": "rural",
        "jp: up to 50,000 residents cities": "rural",
        "jp: 50,000 to 150,000  residents cities": "urban",
        "jp: 150,000 more residents cities": "urban",
        "jp: 12 major large cities(i.e.tokyo,osaka,etc.)": "urban"
    },
    "russia": {
        "ru: rural population": ["rural"],
        "ru: pgt (rural township)": ["rural"],
        "ru: less than 50 tsd": ["rural", "urban"],
        "ru: 50-99,9 tsd": ["rural", "urban"],
        "ru: 100-249,9 tsd": ["rural", "urban"],
        "ru: 250-499,9 tsd": ["rural", "urban"],
        'ru: 500-999,9 tsd': ["rural", "urban"],
        "ru: 1mln. and more": ["urban"]
    }
}

# Region to prefecture mappings for Japan 2012
JAPAN_REGION_TO_PREFECTURE_2012 = {
    'jp: hokkaido region': ['jp-01 hokkaido'],
    'jp: tohoku region': [
        'jp-02 aomori', 'jp-03 iwate', 'jp-04 miyagi', 'jp-05 akita',
        'jp-06 yamagata', 'jp-07 fukushima'
    ],
    'jp: kita-kanto region': [
        'jp-08 ibaraki', 'jp-09 tochigi', 'jp-10 gunma', 'jp-11 saitama'
    ],
    'jp: minami-kanto region': [
        'jp-12 chiba', 'jp-13 tokyo', 'jp-14 kanagawa'
    ],
    'jp: tokai region': [
        'jp-22 shizuoka', 'jp-23 aichi', 'jp-24 mie', 'jp-21 gifu'
    ],
    'jp: kinki region': [
        'jp-25 shiga', 'jp-26 kyoto', 'jp-27 osaka', 'jp-28 hyogo', 
        'jp-29 nara', 'jp-30 wakayama'
    ],
    'jp: hokuriku, shinetsu region': [
        'jp-15 niigata', 'jp-16 toyama', 'jp-17 ishikawa', 'jp-18 fukui', 'jp-20 nagano'
    ],
    'jp: shikoku region': [
        'jp-31 tokushima', 'jp-32 kagawa', 'jp-33 ehime', 'jp-34 kochi'
    ],
    'jp: kyushu region': [
        'jp-40 fukuoka', 'jp-41 saga', 'jp-42 nagasaki', 'jp-43 kumamoto', 
        'jp-44 oita', 'jp-45 miyazaki', 'jp-46 kagoshima', 'jp-47 okinawa'
    ]
}

# Region to prefecture mappings for Japan 2006
JAPAN_REGION_TO_PREFECTURE_2006 = {
    'jp: hokkaido/tohoku': [
        'jp-01 hokkaido', 'jp-02 aomori', 'jp-03 iwate', 'jp-04 miyagi', 
        'jp-05 akita', 'jp-06 yamagata', 'jp-07 fukushima'
    ],
    'jp: kanto': [
        'jp-08 ibaraki', 'jp-09 tochigi', 'jp-10 gunma', 'jp-11 saitama', 
        'jp-12 chiba', 'jp-13 tokyo', 'jp-14 kanagawa'
    ],
    'jp: chubu,hokuriku': [
        'jp-21 gifu', 'jp-22 shizuoka', 'jp-23 aichi', 'jp-24 mie', 
        'jp-15 niigata', 'jp-16 toyama', 'jp-17 ishikawa', 'jp-18 fukui', 'jp-20 nagano'
    ],
    'jp: kinki': [
        'jp-25 shiga', 'jp-26 kyoto', 'jp-27 osaka', 'jp-28 hyogo', 
        'jp-29 nara', 'jp-30 wakayama'
    ],
    'jp: chugoku,shikoku,kyushu,okinawa': [
        'jp-31 tottori', 'jp-32 shimane', 'jp-33 okayama', 'jp-34 hiroshima', 'jp-35 yamaguchi',
        'jp-31 tokushima', 'jp-32 kagawa', 'jp-33 ehime', 'jp-34 kochi',
        'jp-40 fukuoka', 'jp-41 saga', 'jp-42 nagasaki', 'jp-43 kumamoto', 'jp-44 oita', 
        'jp-45 miyazaki', 'jp-46 kagoshima', 'jp-47 okinawa'
    ]
}

# Region to prefecture mappings for Japan 2012
EGYPT_REGION_TO_REGION_2012 = {
    'eg: cairo': 'eg-c cairo',
    'eg: giza': 'eg-gz giza',
    'eg: luxor': 'eg-lu luxor',
    'eg: aswan': 'eg-asn aswan',
    'eg: alexandria': 'eg-alex alexandria',
    'eg: asyut': 'eg-ast asyut',
    'eg: behaira': 'eg-bh behaira',
    'eg: dakahlia': 'eg-dk dakahlia',
    'eg: damietta': 'eg-dn damietta',
    'eg: gharbia': 'eg-gh gharbia',
    'eg: menya': 'eg-myn menya',
    'eg: monufia': 'eg-mnf monufia',
    'eg: qalubia': 'eg-qa qalubia',
    'eg: qena': 'eg-kn qena',
    'eg: sharkia': 'eg-ka sharkia',
    'eg: sohag': 'eg-shg sohag',
    'eg: faiyum': 'eg-fym faiyum'
}

# Region to state mappings for US 2006
US_REGION_TO_STATE_2006 = {
    'us: new england': [
        'us-me maine', 'us-nh new hampshire', 'us-vt vermont', 'us-ma massachusetts', 
        'us-ri rhode island', 'us-ct connecticut'
    ],
    'us: middle atlantic states': [
        'us-ny new york', 'us-nj new jersey', 'us-pa pennsylvania'
    ],
    'us: east north central': [
        'us-oh ohio', 'us-in indiana', 'us-il illinois', 'us-mi michigan', 'us-wi wisconsin'
    ],
    'us: west north central': [
        'us-mn minnesota', 'us-ia iowa', 'us-mo missouri', 'us-nd north dakota', 
        'us-sd south dakota', 'us-ne nebraska', 'us-ks kansas'
    ],
    'us: south atlantic': [
        'us-de delaware', 'us-md maryland', 'us-dc district of columbia', 'us-va virginia',
        'us-wv west virginia', 'us-nc north carolina', 'us-sc south carolina', 
        'us-ga georgia', 'us-fl florida'
    ],
    'us: east south central': [
        'us-ky kentucky', 'us-tn tennessee', 'us-ms mississippi', 'us-al alabama'
    ],
    'us: west south central': [
        'us-ok oklahoma', 'us-tx texas', 'us-ar arkansas', 'us-la louisiana'
    ],
    'us: mountain': [
        'us-id idaho', 'us-mt montana', 'us-wy wyoming', 'us-nv nevada', 
        'us-ut utah', 'us-co colorado', 'us-az arizona', 'us-nm new mexico'
    ],
    'us: pacific': [
        'us-wa washington', 'us-or oregon', 'us-ca california', 'us-ak alaska', 'us-hi hawaii'
    ]
}

# Social class mappings for Russia 2006
RUSSIA_SOCIAL_CLASS_2006 = {
    "low": ["lower class", "lower middle class", "middle class", "working class"],
    "middle": ["lower class", "lower middle class", "middle class", "upper middle class", "working class"],
    "high": ["middle class", "upper middle class", "upper class"]
}

# Default regions for Colombia 2006
COLOMBIA_REGIONS_2006 = [
    'co-ant antioquia', 
    'co-dc distrito capital de bogotá', 
    'co-vac valle del cauca'
]

# Default regions for Russia (for gemma model in 2006 and 2012)
RUSSIA_REGIONS_GEMMA = [
    'ru-bel belgorodskaya area',
    'ru-mow moskva autonomous city',
    'ru-pri primorskiy kray',
    'ru-ros rostovskaya area',
    'ru-sve sverdlovskaya area'
]
