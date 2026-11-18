// Source on top of libavformat / libavcodec: any container and codec FFmpeg reads.
// Decoding is in software unless hardware decoding was asked for (set_hw_decode()). Then the source is decoded by the GPU, but only after
// its first frames have been checked against a software decode of the same file and found identical; anything else falls back to software.
#include "source.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

#include "encoder.h"
#include "ff.h"
#include "gpu.h"
#include "vk_compute.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#ifndef VES_NO_VULKAN
#include <libavutil/hwcontext_vulkan.h>
#endif
}

namespace {

// The deprecated "J" formats are plain planar YUV with full range; the range tag says so instead and the data is identical.
AVPixelFormat unj(AVPixelFormat f, bool& full) {
	full = true;
	switch (f) {
	case AV_PIX_FMT_YUVJ420P: return AV_PIX_FMT_YUV420P;
	case AV_PIX_FMT_YUVJ422P: return AV_PIX_FMT_YUV422P;
	case AV_PIX_FMT_YUVJ444P: return AV_PIX_FMT_YUV444P;
	case AV_PIX_FMT_YUVJ440P: return AV_PIX_FMT_YUV440P;
	default: full = false; return f;
	}
}

constexpr int VERIFY_FRAMES = 120;

class FfSource : public Source {
	AVFormatContext* ctx = nullptr;
	AVCodecContext* dec = nullptr;
	AVPacket* pkt = av_packet_alloc();
	AVFrame* frame = av_frame_alloc();
	AVFrame* sw_frame = av_frame_alloc(); // where a hardware frame is downloaded to
	AVFrame* planar_frame = av_frame_alloc(); // and where it is unpacked to if the download has interleaved chroma
	AVBufferRef* hw_dev = nullptr;
	AVPixelFormat hw_fmt = AV_PIX_FMT_NONE;
	bool hw_used = false;   // a frame really came from the hardware decoder
	bool fmt_known = true;  // false for hardware decoding until the first frame shows what the download looks like
	Picture pending;        // the first frame, read early to learn the format
	bool has_pending = false;
	int vi = -1;
	bool flushed = false;

	// Reads every video packet's timestamp (no decoding) to learn the real frame times.
	void scan(const std::string& path) {
		AVFormatContext* sc = nullptr;
		if (avformat_open_input(&sc, path.c_str(), nullptr, nullptr) < 0) return;
		AVPacket* pk = av_packet_alloc();
		bool missing = false;
		while (av_read_frame(sc, pk) >= 0) {
			if (pk->stream_index == vi) {
				int64_t ts = pk->pts != AV_NOPTS_VALUE ? pk->pts : pk->dts;
				if (ts == AV_NOPTS_VALUE) missing = true; else stamps.push_back(ts);
			}
			av_packet_unref(pk);
		}
		av_packet_free(&pk);
		avformat_close_input(&sc);
		if (missing || stamps.size() < 2) { stamps.clear(); return; } // raw streams and the like: assume constant rate
		std::sort(stamps.begin(), stamps.end());
		const int64_t first = stamps[0];
		for (auto& t : stamps) t -= first;

		// Constant frame rate if every frame sits within half a frame of where `framerate` puts it.
		const double fps = framerate.value(), unit = tb.value();
		double worst = 0;
		for (size_t i = 0; i < stamps.size(); i++) worst = std::max(worst, std::abs(stamps[i] * unit - i / fps));
		if (worst < 0.5 / fps) return;
		vfr = true;
		framerate = from_av(av_d2q((stamps.size() - 1) / (stamps.back() * unit), 1000000));
	}

	static AVPixelFormat pick_format(AVCodecContext* c, const AVPixelFormat* fmts) {
		const AVPixelFormat want = (AVPixelFormat)(intptr_t)c->opaque;
		for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; p++) if (*p == want) return *p;
		return fmts[0]; // the hardware cannot do this stream: decode in software after all
	}

	// Fills `out` from the next decoded frame, downloading it first if the GPU decoded it.
	int pull(Picture& out) {
		for (;;) {
			int r = avcodec_receive_frame(dec, frame);
			if (r >= 0) {
				AVFrame* f = frame;
				if (hw_fmt != AV_PIX_FMT_NONE && frame->format == hw_fmt) {
					av_frame_unref(sw_frame);
					if (const int t = av_hwframe_transfer_data(sw_frame, frame, 0); t < 0) {
						// A frame cannot be read back after others could: the GPU's context is broken (CUDA_ERROR_ILLEGAL_ADDRESS after a GPU reset), and so is every
						// later frame of every decoder that shares it. (The first frame failing only means this decoder does not work here: software decodes instead.)
						char b[AV_ERROR_MAX_STRING_SIZE];
						if (hw_used) gpu_lost(std::string("reading a hardware decoded frame back failed: ") + av_make_error_string(b, sizeof b, t));
						return AVERROR_EXTERNAL;
					}
					av_frame_copy_props(sw_frame, frame);
					av_frame_unref(frame);
					f = sw_frame;
					hw_used = true;
				}
				f = unpack_semi_planar(f, planar_frame);
				bool full = false;
				f->format = unj((AVPixelFormat)f->format, full);
				if (full) f->color_range = AVCOL_RANGE_JPEG;
				if (!fmt_known) {
					if (!describe_native((AVPixelFormat)f->format, fmt)) { fprintf(stderr, "unsupported source pixel format %s\n", av_get_pix_fmt_name((AVPixelFormat)f->format)); return AVERROR_INVALIDDATA; }
					fmt_known = true;
				}
				f->pts = read++;
				f->duration = 1;
				f->pict_type = AV_PICTURE_TYPE_NONE;
				out = picture_from_frame(f, fmt);
				return 1;
			}
			if (r == AVERROR_EOF) return 0;
			if (r != AVERROR(EAGAIN)) return r;
			if (flushed) return 0;
			if (av_read_frame(ctx, pkt) < 0) { avcodec_send_packet(dec, nullptr); flushed = true; continue; }
			if (pkt->stream_index == vi) avcodec_send_packet(dec, pkt);
			av_packet_unref(pkt);
		}
	}

public:
	std::string decoder_name;

	~FfSource() override {
		av_frame_free(&frame);
		av_frame_free(&sw_frame);
		av_frame_free(&planar_frame);
		av_packet_free(&pkt);
		avcodec_free_context(&dec);
		av_buffer_unref(&hw_dev);
		avformat_close_input(&ctx);
	}

	bool hardware() const { return hw_used; }

	// hw_type: the FFmpeg name of a hardware device type ("cuda", "d3d11va", ...) or of a hardware decoder ("hevc_cuvid"), or empty for software.
	// device: which GPU, in the device type's own terms (a CUDA ordinal, a DXGI adapter index, a render node, a Vulkan device); empty: the type's default.
	bool open(const std::string& path, const std::string& hw_type, bool do_scan, const std::string& device = {}) {
		char b[AV_ERROR_MAX_STRING_SIZE];
		int r = avformat_open_input(&ctx, path.c_str(), nullptr, nullptr);
		if (r < 0) { fprintf(stderr, "open '%s': %s\n", path.c_str(), av_make_error_string(b, sizeof b, r)); return false; }
		if ((r = avformat_find_stream_info(ctx, nullptr)) < 0) { fprintf(stderr, "stream info: %s\n", av_make_error_string(b, sizeof b, r)); return false; }
		vi = av_find_best_stream(ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
		if (vi < 0) { fprintf(stderr, "'%s' has no video stream\n", path.c_str()); return false; }
		AVStream* st = ctx->streams[vi];
		const AVCodec* c = software_decoder(st->codecpar->codec_id);
		if (!hw_type.empty() && !avcodec_find_decoder_by_name(hw_type.c_str())) c = hwaccel_decoder(st->codecpar->codec_id, hw_type); // libdav1d has no hardware path
		if (!hw_type.empty() && !c) return false;
		const AVCodec* dedicated = hw_type.empty() ? nullptr : avcodec_find_decoder_by_name(hw_type.c_str());
		if (dedicated) {
			if (dedicated->id != st->codecpar->codec_id) return false; // a decoder for some other codec
			c = dedicated;
		}
		if (!c) { fprintf(stderr, "no software decoder for this video\n"); return false; }
		dec = avcodec_alloc_context3(c);
		avcodec_parameters_to_context(dec, st->codecpar);
		if (dedicated) {
			dec->thread_count = 1; // the hardware does the work
			fmt_known = false;
		} else if (!hw_type.empty()) {
			const AVHWDeviceType type = av_hwdevice_find_type_by_name(hw_type.c_str());
			if (type == AV_HWDEVICE_TYPE_NONE) return false;
			for (int i = 0;; i++) { // the pixel format this decoder gives for that device type
				const AVCodecHWConfig* cfg = avcodec_get_hw_config(c, i);
				if (!cfg) return false;
				if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) && cfg->device_type == type) { hw_fmt = cfg->pix_fmt; break; }
			}
			if (!(hw_dev = shared_hw_device(hw_type, device))) return false;
			dec->hw_device_ctx = av_buffer_ref(hw_dev);
			dec->opaque = (void*)(intptr_t)hw_fmt;
			dec->get_format = &FfSource::pick_format;
			dec->thread_count = 1; // the GPU does the work
			fmt_known = false;
		} else {
			dec->thread_count = decode_threads(); // the library decodes on one thread unless told otherwise
			dec->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
		}
		AVDictionary* dec_opts = nullptr;
		if (dedicated && !device.empty()) av_dict_set(&dec_opts, "gpu", device.c_str(), 0); // the CUDA ordinal of a *_cuvid decoder
		const int opened = avcodec_open2(dec, c, &dec_opts);
		av_dict_free(&dec_opts);
		if (opened < 0) { if (hw_type.empty()) fprintf(stderr, "decoder open failed\n"); return false; }
		width = dec->width; height = dec->height;
		framerate = from_av(av_guess_frame_rate(ctx, st, nullptr));
		tb = from_av(st->time_base);
		color = {(int)st->codecpar->color_range, (int)st->codecpar->color_space, (int)st->codecpar->color_primaries, (int)st->codecpar->color_trc};
		if (hw_type.empty()) {
			bool full = false;
			AVPixelFormat native = unj(dec->pix_fmt, full);
			if (const SemiPlanar* sp = semi_planar(native)) native = sp->to;
			if (full) color.range = AVCOL_RANGE_JPEG;
			if (!describe_native(native, fmt)) { fprintf(stderr, "unsupported source pixel format %s\n", av_get_pix_fmt_name(native)); return false; }
		} else {
			// The first frame tells what the download looks like (NV12, planar 4:4:4, ...) and whether the GPU took the stream at all.
			if (pull(pending) != 1) return false;
			if (dedicated) hw_used = true;
			if (!hw_used) return false;
			has_pending = true;
			if (pending.color.range == AVCOL_RANGE_JPEG) color.range = AVCOL_RANGE_JPEG;
		}
		if (do_scan) scan(path);
		decoder_name = std::string(c->name) + (hw_type.empty() ? " (software)" : dedicated ? " (hardware" : " (" + hw_type + " hardware");
		if (!hw_type.empty()) decoder_name += (device.empty() ? "" : ", device " + device) + ")";
		decoder = decoder_name;
		return true;
	}

	bool open(const std::string& path) override { return open(path, "", true); }

	int next(Picture& out) override {
		if (has_pending) {
			out = std::move(pending);
			has_pending = false;
			return 1;
		}
		return pull(out);
	}
};

// Decodes the first frames both ways and compares them.
bool verify_hardware(const std::string& path, const std::string& type, const std::string& device) {
	FfSource sw, hw;
	if (!sw.open(path, "", false) || !hw.open(path, type, false, device)) return false;
	for (int i = 0; i < VERIFY_FRAMES; i++) {
		Picture a, b;
		const int ra = sw.next(a), rb = hw.next(b);
		if (ra != rb) return false;
		if (ra != 1) break;
		if (!same_pictures(a, b)) {
			fprintf(stderr, "hardware decoding (%s) differs from software at frame %d of '%s'\n", type.c_str(), i, path.c_str());
			describe_difference(a, b, i);
			return false;
		}
	}
	return true;
}

// The codec of the file's video stream, found once per file.
AVCodecID codec_id_of(const std::string& path) {
	static std::mutex m;
	static std::map<std::string, AVCodecID> known;
	std::lock_guard lock(m);
	if (auto it = known.find(path); it != known.end()) return it->second;
	AVCodecID id = AV_CODEC_ID_NONE;
	AVFormatContext* c = nullptr;
	if (avformat_open_input(&c, path.c_str(), nullptr, nullptr) >= 0) {
		if (avformat_find_stream_info(c, nullptr) >= 0) {
			const int v = av_find_best_stream(c, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
			if (v >= 0) id = c->streams[v]->codecpar->codec_id;
		}
		avformat_close_input(&c);
	}
	return known[path] = id;
}

std::mutex g_mutex;
std::string g_hw;                  // "", "auto" or a device type
std::map<std::string, bool> g_verdict; // path|type|device -> identical to software
unsigned g_vendor = 0;             // set_hw_decode_gpu(): decode on this GPU only
int g_index = -1;

// The hardware decoders of the stream's codec whose names end in `suffix` (_cuvid, _qsv, _amf).
std::vector<std::string> dedicated_decoders(AVCodecID id, const std::string& suffix) {
	std::vector<std::string> out;
	void* it = nullptr;
	while (const AVCodec* c = av_codec_iterate(&it)) {
		const std::string name = c->name;
		if (av_codec_is_decoder(c) && c->id == id && (c->capabilities & AV_CODEC_CAP_HARDWARE) && name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
			out.push_back(name);
	}
	return out;
}

// What hardware decoding can try for a codec, in order: a device type or a hardware decoder, and the GPU in that type's terms (empty: its default).
std::vector<HwChoice> hw_choices(AVCodecID id, const std::string& want) {
	std::vector<HwChoice> out;
	if (!g_vendor) { // any GPU: each type's default device
		if (want != "auto") return {{want, ""}};
		// The normal decoders on a hardware device first, then the dedicated hardware decoders of the stream's codec.
		// (OpenCL is left out: it breaks more often than it works on systems with several GPUs.)
		for (const char* t : {"cuda", "d3d11va", "d3d12va", "dxva2", "qsv", "amf", "vaapi", "vulkan"}) out.push_back({t, ""});
		for (const char* suffix : {"_cuvid", "_qsv", "_amf"}) for (auto& d : dedicated_decoders(id, suffix)) out.push_back({d, ""});
		return out;
	}
	// One GPU (an encoder process): every type is pointed at that GPU, and types that cannot be pointed at it are left out, so the frames are decoded where
	// they are encoded and scored. DXVA2 numbers adapters its own way, and FFmpeg's AMF device takes no adapter; both are left out.
	struct Where { int cuda, dxgi, vk; std::string node; };
	static const Where where{g_vendor == VENDOR_NVIDIA ? nvenc_cuda_ordinal(g_index) : -1, dxgi_adapter_index(g_vendor, g_index), vulkan_device_of_gpu(g_vendor, g_index), render_node(g_vendor, g_index)}; // asked once: the GPU does not change
	const int cuda = where.cuda, dxgi = where.dxgi, vk = where.vk;
	const std::string& node = where.node;
	auto add = [&](const std::string& type, const std::string& device) { if (want == "auto" || want == type) out.push_back({type, device}); };
	if (cuda >= 0) add("cuda", std::to_string(cuda));
	if (dxgi >= 0) { add("d3d11va", std::to_string(dxgi)); add("d3d12va", std::to_string(dxgi)); }
	if (g_vendor == VENDOR_INTEL && g_index == 0) add("qsv", "");
	if (!node.empty()) add("vaapi", node);
	if (vk >= 0) add("vulkan", std::to_string(vk));
	if (cuda >= 0) for (auto& d : dedicated_decoders(id, "_cuvid")) add(d, std::to_string(cuda));
	if (g_index == 0 && g_vendor == VENDOR_INTEL) for (auto& d : dedicated_decoders(id, "_qsv")) add(d, "");
	if (g_index == 0 && g_vendor == VENDOR_AMD) for (auto& d : dedicated_decoders(id, "_amf")) add(d, "");
	return out;
}

} // namespace

AVBufferRef* shared_hw_device(const std::string& type, const std::string& device) {
	// one device per type for the whole run: creating it is slow, and decoders can share it
	static std::mutex devices_mutex;
	static std::map<std::string, AVBufferRef*> devices;
	const AVHWDeviceType t = av_hwdevice_find_type_by_name(type.c_str());
	if (t == AV_HWDEVICE_TYPE_NONE) return nullptr;
	std::lock_guard lock(devices_mutex);
	AVBufferRef*& shared = devices[type + "|" + device];
	if (!shared && av_hwdevice_ctx_create(&shared, t, device.empty() ? nullptr : device.c_str(), nullptr, 0) < 0) shared = nullptr;
	return shared ? av_buffer_ref(shared) : nullptr;
}

#ifndef VES_NO_VULKAN
// FFmpeg's Vulkan device of a GPU is also the scorer's (vk_compute.h, VkExternal), so that what its decoder makes is read on the GPU where it is.
std::mutex g_vk_mutex;
std::map<int, uint32_t> g_vk_compute_family; // per Vulkan device given out: the family the scorer submits to

// lock_queue/unlock_queue are deprecated in favour of VK_KHR_internally_synchronized_queues, which drivers do not have yet; FFmpeg itself still uses them.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
bool vulkan_provider(int index, uint32_t vendor_id, uint32_t device_id, VkExternal& out) {
	AVBufferRef* ref = shared_hw_device("vulkan", std::to_string(index)); // kept for the whole run (the shared map holds it)
	if (!ref) return false;
	AVHWDeviceContext* hw = (AVHWDeviceContext*)ref->data;
	AVVulkanDeviceContext* vk = (AVVulkanDeviceContext*)hw->hwctx;
	av_buffer_unref(&ref);
	auto props = (PFN_vkGetPhysicalDeviceProperties)vk->get_proc_addr(vk->inst, "vkGetPhysicalDeviceProperties");
	if (!props) return false;
	VkPhysicalDeviceProperties p;
	props(vk->phys_dev, &p);
	if (p.vendorID != vendor_id || p.deviceID != device_id) return false; // FFmpeg counted the devices differently
	// The scorer needs 64 bit integers and floats; reading the decoder's frames needs timeline semaphores.
	if (!vk->device_features.features.shaderInt64 || !vk->device_features.features.shaderFloat64) return false;
	bool timeline = false;
	for (auto* n = (const VkBaseInStructure*)vk->device_features.pNext; n; n = n->pNext)
		if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES) timeline = ((const VkPhysicalDeviceVulkan12Features*)n)->timelineSemaphore;
		else if (n->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) timeline = timeline || ((const VkPhysicalDeviceTimelineSemaphoreFeatures*)n)->timelineSemaphore;
	if (!timeline || !vk->lock_queue || !vk->unlock_queue) return false;
	bool compute = false;
	for (int i = 0; i < vk->nb_qf; i++) {
		const AVVulkanDeviceQueueFamily& q = vk->qf[i];
		if (!compute && (q.flags & VK_QUEUE_COMPUTE_BIT) && q.num > 0) { out.compute_family = (uint32_t)q.idx; out.compute_queues = (uint32_t)q.num; compute = true; }
		if (!out.has_transfer && (q.flags & VK_QUEUE_TRANSFER_BIT) && !(q.flags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) && q.num > 0) { out.transfer_family = (uint32_t)q.idx; out.has_transfer = true; }
	}
	if (!compute) return false;
	out.phys = vk->phys_dev;
	out.dev = vk->act_dev;
	out.ctx = hw;
	out.lock = [](void* c, uint32_t fam, uint32_t i) { auto* h = (AVHWDeviceContext*)c; ((AVVulkanDeviceContext*)h->hwctx)->lock_queue(h, fam, i); };
	out.unlock = [](void* c, uint32_t fam, uint32_t i) { auto* h = (AVHWDeviceContext*)c; ((AVVulkanDeviceContext*)h->hwctx)->unlock_queue(h, fam, i); };
	std::lock_guard lock(g_vk_mutex);
	g_vk_compute_family[index] = out.compute_family;
	fprintf(stderr, "vulkan device %d (%s): shared by FFmpeg's decoder and the scorer\n", index, p.deviceName);
	return true;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

std::string hw_decode_gpu_identity() {
	std::lock_guard lock(g_mutex);
	return g_vendor ? gpu_description(g_vendor, g_index) : std::string("the default GPU");
}

std::vector<HwChoice> hw_decode_choices(AVCodecID id) {
	std::string want;
	{
		std::lock_guard lock(g_mutex);
		want = g_hw;
	}
	if (want.empty()) return {};
	std::vector<HwChoice> out;
	for (auto& c : hw_choices(id, want)) if (av_hwdevice_find_type_by_name(c.type.c_str()) != AV_HWDEVICE_TYPE_NONE) out.push_back(c); // device types only (see ff_decode.cpp)
	// Vulkan first: its frames can be scored on the GPU where they are, the others' have to come back to memory.
	std::stable_partition(out.begin(), out.end(), [](const HwChoice& c) { return c.type == "vulkan"; });
	return out;
}

void set_hw_decode_gpu(unsigned vendor, int index) {
	std::lock_guard lock(g_mutex);
	g_vendor = vendor;
	g_index = index;
}

void set_hw_decode(const std::string& type) {
	std::lock_guard lock(g_mutex);
	g_hw = type == "off" || type == "none" ? "" : type;
#ifndef VES_NO_VULKAN
	// Decoding on the GPU: the scorer uses FFmpeg's Vulkan device where it can, so that frames decoded there can be scored without leaving it.
	if (!g_hw.empty() && (g_hw == "auto" || g_hw == "vulkan")) vk_set_external_provider(&vulkan_provider);
#endif
}

#ifndef VES_NO_VULKAN
bool vulkan_frame(const AVFrame* f, int device, VkDecodedFrame& out) {
	if (!f || f->format != AV_PIX_FMT_VULKAN || !f->hw_frames_ctx) return false;
	uint32_t family;
	{
		std::lock_guard lock(g_vk_mutex);
		auto it = g_vk_compute_family.find(device);
		if (it == g_vk_compute_family.end()) return false; // the scorer is not on FFmpeg's device
		family = it->second;
	}
	AVHWFramesContext* fc = (AVHWFramesContext*)f->hw_frames_ctx->data;
	AVVulkanFramesContext* vfc = (AVVulkanFramesContext*)fc->hwctx;
	AVVkFrame* vf = (AVVkFrame*)f->data[0];
	const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(fc->sw_format);
	if (!d || (d->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_BE)) || d->nb_components != 3 || !(d->flags & AV_PIX_FMT_FLAG_PLANAR)) return false;
	const bool interleaved = d->comp[1].plane == d->comp[2].plane;
	out.nplanes = interleaved ? 2 : 3;
	out.interleaved = interleaved;
	out.bytes = d->comp[0].step > 1 || d->comp[0].depth > 8 ? 2 : 1;
	if (interleaved) out.bytes = d->comp[1].step / 2;
	out.shift = d->comp[0].shift;
	if (d->comp[1].shift != out.shift || d->comp[2].shift != out.shift) return false;
	int nimg = 0;
	while (nimg < AV_NUM_DATA_POINTERS && vf->img[nimg]) nimg++;
	if (nimg != 1 && nimg != out.nplanes) return false;
	for (int i = 0; i < nimg; i++) if (vf->queue_family[i] != VK_QUEUE_FAMILY_IGNORED && vf->queue_family[i] != family) return false;
	const int cw = AV_CEIL_RSHIFT(f->width, d->log2_chroma_w), ch = AV_CEIL_RSHIFT(f->height, d->log2_chroma_h);
	for (int p = 0; p < out.nplanes; p++) {
		out.planes[p].img = vf->img[nimg == 1 ? 0 : p];
		out.planes[p].aspect = nimg == 1 ? (p == 0 ? VK_IMAGE_ASPECT_PLANE_0_BIT : p == 1 ? VK_IMAGE_ASPECT_PLANE_1_BIT : VK_IMAGE_ASPECT_PLANE_2_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
		out.planes[p].w = (uint32_t)(p ? cw : f->width);
		out.planes[p].h = (uint32_t)(p ? ch : f->height);
	}
	out.nimages = nimg;
	for (int i = 0; i < nimg; i++) {
		out.images[i] = vf->img[i];
		out.layout[i] = &vf->layout[i];
		out.access[i] = (uint32_t*)&vf->access[i];
		out.sem[i] = vf->sem[i];
		out.sem_value[i] = &vf->sem_value[i];
	}
	struct Lock { AVHWFramesContext* fc; AVVulkanFramesContext* vfc; AVVkFrame* vf; };
	auto* l = new Lock{fc, vfc, vf};
	AVFrame* keep = av_frame_clone(f);
	out.keep = std::shared_ptr<void>(keep, [l](void* p) { AVFrame* a = (AVFrame*)p; av_frame_free(&a); delete l; });
	out.ctx = l;
	out.lock = [](void* c) { auto* k = (Lock*)c; if (k->vfc->lock_frame) k->vfc->lock_frame(k->fc, k->vf); };
	out.unlock = [](void* c) { auto* k = (Lock*)c; if (k->vfc->unlock_frame) k->vfc->unlock_frame(k->fc, k->vf); };
	out.device = device;
	return true;
}
#endif

std::unique_ptr<Source> open_source(const std::string& path) {
	av_log_set_level(AV_LOG_ERROR); // warnings such as "UDTA parsing failed" repeat for every run that opens the file
	std::string want;
	{
		std::lock_guard lock(g_mutex);
		want = g_hw;
	}
	if (!want.empty()) {
		for (const auto& [type, device] : hw_choices(codec_id_of(path), want)) {
			const std::string key = path + "|" + type + "|" + device;
			bool known = false, good = false;
			{
				std::lock_guard lock(g_mutex);
				if (auto it = g_verdict.find(key); it != g_verdict.end()) { known = true; good = it->second; }
			}
			if (known && !good) continue;
			if (!known) {
				good = verify_hardware(path, type, device);
				if (good) fprintf(stderr, "hardware decoding (%s%s) matched software on the first %d frames of '%s'\n", type.c_str(), device.empty() ? "" : (", device " + device).c_str(), VERIFY_FRAMES, path.c_str());
				std::lock_guard lock(g_mutex);
				g_verdict[key] = good;
				if (!good) continue;
			}
			auto s = std::make_unique<FfSource>();
			if (s->open(path, type, true, device)) return s;
		}
		static std::mutex once;
		static std::map<std::string, bool> told;
		std::lock_guard lock(once);
		if (!told[path]) { told[path] = true; fprintf(stderr, "hardware decoding is not available for '%s': decoding in software\n", path.c_str()); }
	}
	auto s = std::make_unique<FfSource>();
	return s->open(path, "", true) ? std::unique_ptr<Source>(std::move(s)) : nullptr;
}
