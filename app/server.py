import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1M8d1gwAwKgD83u3tdkz2x-trmVMT5I4_'
export_file_name = 'pkmn_img_classifier.pkl'

classes = ['001_bulbasaur',
 '002_ivysaur',
 '003_venusaur',
 '004_charmander',
 '005_charmeleon',
 '006_charizard',
 '008_wartortle',
 '009_blastoise',
 '010_caterpie',
 '011_metapod',
 '012_butterfree',
 '013_weedle',
 '014_kakuna',
 '015_beedrill',
 '017_pidgeotto',
 '018_pidgeot',
 '019_rattata',
 '020_ratticate',
 '021_spearow',
 '022_fearow',
 '023_ekans',
 '024_arbok',
 '025_pikachu',
 '026_raichu',
 '027_sandshrew',
 '028_sandslash',
 '029_nidoran_f',
 '030_nidorina',
 '031_nidoqueen',
 '032_nidoran_m',
 '033_nidorino',
 '034_nidoking',
 '035_clefairy',
 '036_clefable',
 '037_vulpix',
 '038_ninetales',
 '039_jigglypuff',
 '040_wigglytuff',
 '041_zubat',
 '042_golbat',
 '043_oddish',
 '044_gloom',
 '045_vileplume',
 '046_paras',
 '047_parasect',
 '048_venonat',
 '049_venomoth',
 '050_diglett',
 '051_dugtrio',
 '052_meowth',
 '053_persian',
 '054_psyduck',
 '055_golduck',
 '056_mankey',
 '057_primeape',
 '058_growlithe',
 '059_arcanine',
 '060_poliwag',
 '061_poliwhirl',
 '062_poliwrath',
 '063_abra',
 '064_kadabra',
 '065_alakazam',
 '066_machop',
 '067_machoke',
 '068_machamp',
 '069_bellsprout',
 '070_weepinbell',
 '071_victreebel',
 '072_tentacool',
 '073_tentacruel',
 '074_geodude',
 '075_graveler',
 '076_golem',
 '077_ponyta',
 '078_rapidash',
 '079_slowpoke',
 '080_slowbro',
 '081_magnemite',
 '082_magneton',
 '083_farfetch_d',
 '084_doduo',
 '085_dodrio',
 '086_seel',
 '087_dewgong',
 '088_grimer',
 '089_muk',
 '090_shellder',
 '091_cloyster',
 '092_gastly',
 '093_haunter',
 '094_gengar',
 '095_onix',
 '096_drowzee',
 '097_hypno',
 '098_krabby',
 '099_kingler',
 '100_voltorb',
 '101_electrode',
 '102_exeggcute',
 '103_exeggutor',
 '104_cubone',
 '105_marowak',
 '106_hitmonlee',
 '107_hitmonchan',
 '108_lickitung',
 '109_koffing',
 '110_weezing',
 '111_rhyhorn',
 '112_rhydon',
 '113_chansey',
 '114_tangela',
 '115_kangaskhan',
 '116_horsea',
 '117_seadra',
 '118_goldeen',
 '119_seaking',
 '120_staryu',
 '121_starmie',
 '122_mr_mime',
 '123_scyther',
 '124_jynx',
 '125_electabuzz',
 '126_magmar',
 '127_pinsir',
 '128_tauros',
 '129_magikarp',
 '130_gyarados',
 '131_lapras',
 '132_ditto',
 '133_eevee',
 '134_vaporeon',
 '135_jolteon',
 '136_flareon',
 '137_porygon',
 '138_omanyte',
 '139_omastar',
 '140_kabuto',
 '141_kabutops',
 '142_aerodactyl',
 '143_snorlax',
 '144_articuno',
 '145_zapdos',
 '146_moltres',
 '147_dratini',
 '148_dragonair',
 '149_dragonite',
 '150_mewtwo',
 '151_mew']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
