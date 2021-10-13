ThisBuild / scalaVersion := "3.0.0"

val breezeVersion = "2.0"
val scalaTestVersion = "3.2.10"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % breezeVersion,
  "org.scalanlp" %% "breeze-viz" % breezeVersion,
  "org.scalactic" %% "scalactic" % scalaTestVersion,
  "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.4",
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "com.github.tototoshi" %% "scala-csv" % "1.3.8"
)

// will flag errors in your ScalaTest (and Scalactic) code at compile time
resolvers += "Artima Maven Repository" at "https://repo.artima.com/releases"
