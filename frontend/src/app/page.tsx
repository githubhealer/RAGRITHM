import Silk from "@/components/SilkBackground";

export default function Home() {
  return (
    <div>
      <div className="absolute w-screen h-screen z-[-1]">
        <Silk
          speed={5}
          scale={1}
          color="#7B7481"
          noiseIntensity={1.5}
          rotation={0}
        />
      </div>

      <div className="flex flex-col justify-center items-center h-screen">
        <h1 className="text-4xl">RAGRITHM</h1>
        <input
          type="text"
          placeholder="Enter Document Url"
          className="border-3 border-[#ffffff55] p-2 pl-5 rounded-full backdrop-blur-xl"
        />
      </div>
    </div>
  );
}
