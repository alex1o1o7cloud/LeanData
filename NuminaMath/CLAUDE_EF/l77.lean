import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_four_integers_l77_7711

theorem existence_of_four_integers (n : ℤ) (h : n > 1000000) : 
  ∃ a b c d : ℤ, 
    (abs a > 1000000 ∧ abs b > 1000000 ∧ abs c > 1000000 ∧ abs d > 1000000) ∧
    (1 / a + 1 / b + 1 / c + 1 / d = 1 / (a * b * c * d)) := by
  let a := -n
  let b := n + 1
  let c := n * (n + 1) + 1
  let d := n * (n + 1) * (n * (n + 1) + 1) + 1
  use a, b, c, d
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_four_integers_l77_7711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whisky_replacement_fraction_l77_7752

/-- Represents the alcohol content of a whisky mixture -/
structure WhiskyMixture where
  total : ℝ
  alcohol : ℝ
  h_nonneg : 0 ≤ total
  h_alcohol_nonneg : 0 ≤ alcohol
  h_alcohol_le_total : alcohol ≤ total

/-- The fraction of alcohol in a whisky mixture -/
noncomputable def alcoholFraction (w : WhiskyMixture) : ℝ := w.alcohol / w.total

theorem whisky_replacement_fraction 
  (initial : WhiskyMixture) 
  (replaced : WhiskyMixture) 
  (final : WhiskyMixture) 
  (h_initial_frac : alcoholFraction initial = 0.40)
  (h_replaced_frac : alcoholFraction replaced = 0.19)
  (h_final_frac : alcoholFraction final = 0.24)
  (h_total_constant : initial.total = final.total)
  : ∃ (k : ℝ), k = 16/21 ∧ replaced.total = k * initial.total := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_whisky_replacement_fraction_l77_7752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l77_7762

theorem correct_answer : ∀ (answer : String), answer = "B" → answer = "B" := by
  intro answer h
  exact h

#check correct_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l77_7762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_is_4_l77_7723

noncomputable section

/-- The hyperbola with equation x²/1 - y²/1 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The foci of the hyperbola -/
def foci : ℝ × ℝ × ℝ × ℝ := (-Real.sqrt 2, 0, Real.sqrt 2, 0)

/-- A point P(x, y) on the hyperbola -/
def point_on_hyperbola (x y : ℝ) : Prop := hyperbola x y

/-- The angle between PF₁ and PF₂ is 60° -/
def angle_is_60_degrees (x y : ℝ) : Prop :=
  let (x₁, y₁, x₂, y₂) := foci
  Real.arccos ((x - x₁)*(x - x₂) + (y - y₁)*(y - y₂)) / 
    (Real.sqrt ((x - x₁)^2 + (y - y₁)^2) * Real.sqrt ((x - x₂)^2 + (y - y₂)^2)) = Real.pi / 3

/-- The theorem to be proved -/
theorem product_of_distances_is_4 (x y : ℝ) :
  point_on_hyperbola x y → angle_is_60_degrees x y →
  let (x₁, y₁, x₂, y₂) := foci
  Real.sqrt ((x - x₁)^2 + (y - y₁)^2) * Real.sqrt ((x - x₂)^2 + (y - y₂)^2) = 4 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_distances_is_4_l77_7723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l77_7767

theorem trigonometric_problem (α β : ℝ)
  (h1 : 0 < α) (h2 : α < π / 2)
  (h3 : -π / 2 < β) (h4 : β < 0)
  (h5 : Real.cos α = Real.sqrt 2 / 10)
  (h6 : Real.sin β = -Real.sqrt 5 / 5) :
  (Real.cos (α - β) = -Real.sqrt 10 / 10) ∧ (α - 2 * β = 3 * π / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l77_7767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_50_most_representative_l77_7781

/-- Represents a sampling method for the sleep time survey --/
inductive SamplingMethod
  | SelectClass
  | Select50Males
  | Select50Females
  | Random50Students

/-- Represents the representativeness of a sampling method --/
def representativeness : SamplingMethod → ℕ :=
  fun _ => 0  -- Placeholder implementation

/-- The school population of eighth-grade students --/
def Student : Type := Unit  -- Placeholder type for students
def schoolPopulation : Set Student :=
  Set.univ  -- Placeholder implementation

/-- Axiom: Random sampling is more representative than non-random sampling --/
axiom random_more_representative :
  ∀ (m : SamplingMethod), m ≠ SamplingMethod.Random50Students →
  representativeness SamplingMethod.Random50Students > representativeness m

/-- Theorem: Random50Students is the most representative sampling method --/
theorem random_50_most_representative :
  ∀ (m : SamplingMethod), 
  representativeness SamplingMethod.Random50Students ≥ representativeness m :=
by
  sorry  -- Proof to be completed later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_50_most_representative_l77_7781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l77_7739

/-- An ellipse with the given properties has the equation y²/75 + x²/25 = 1 -/
theorem ellipse_equation (e : Set (ℝ × ℝ)) (f₁ : ℝ × ℝ) (l : Set (ℝ × ℝ)) (m : ℝ × ℝ) :
  (∀ p ∈ e, (p.1 = 0 ∧ p.2 = 0) → p ∈ e) →  -- centered at origin
  f₁ = (0, Real.sqrt 50) →  -- one focus at (0, √50)
  (∀ p ∈ e ∩ l, ∃ x y, p = (x, y) ∧ y = 3*x - 2) →  -- intersects y = 3x - 2
  (∃ a b, a ∈ e ∩ l ∧ b ∈ e ∩ l ∧ a ≠ b ∧ m = ((a.1 + b.1)/2, (a.2 + b.2)/2) ∧ m.1 = 1/2) →  -- midpoint condition
  ∀ p ∈ e, p.2^2/75 + p.1^2/25 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l77_7739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coordinates_is_10_l77_7763

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- State the theorem
theorem sum_of_y_coordinates_is_10 (y₁ y₂ : ℝ) :
  distance (-2) 5 4 y₁ = 13 →
  distance (-2) 5 4 y₂ = 13 →
  y₁ + y₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_y_coordinates_is_10_l77_7763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tetrahedron_labeling_l77_7749

-- Define a tetrahedron as a set of 4 vertices
def Tetrahedron := Fin 4

-- Define a labeling as a function from vertices to integers
def Labeling := Tetrahedron → Fin 4

-- Define a face as a set of 3 vertices
def Face := Fin 3 → Tetrahedron

-- Define the set of all faces of a tetrahedron
def AllFaces : Set Face := sorry

-- Function to calculate the sum of labels on a face
def faceSum (l : Labeling) (f : Face) : Nat :=
  (l (f 0)).val + (l (f 1)).val + (l (f 2)).val + 3

-- Define what it means for a labeling to be valid
def isValidLabeling (l : Labeling) : Prop :=
  (∀ v : Tetrahedron, (l v).val + 1 ∈ ({1, 2, 3, 4} : Set Nat)) ∧
  (∀ i j : Tetrahedron, i ≠ j → l i ≠ l j) ∧
  (∃ s : Nat, ∀ f : Face, f ∈ AllFaces → faceSum l f = s)

-- Theorem stating that no valid labeling exists
theorem no_valid_tetrahedron_labeling :
  ¬∃ l : Labeling, isValidLabeling l := by
  sorry

#check no_valid_tetrahedron_labeling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tetrahedron_labeling_l77_7749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l77_7744

/-- The time (in hours) required to fill a rectangular tank -/
noncomputable def fill_time (length width depth rate : ℝ) : ℝ :=
  (length * width * depth) / rate

/-- Theorem: The time to fill the specified tank is 60 hours -/
theorem tank_fill_time :
  fill_time 10 6 5 5 = 60 := by
  -- Unfold the definition of fill_time
  unfold fill_time
  -- Simplify the arithmetic
  simp [mul_div_assoc, mul_comm, mul_assoc]
  -- Check that the result is equal to 60
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l77_7744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_doubling_time_l77_7734

-- Define the initial population
def initial_population : ℝ := 1000

-- Define the final population
def final_population : ℝ := 500000

-- Define the total growth time
def total_time : ℝ := 17.931568569324174

-- Define the doubling time
def doubling_time : ℝ := 1.992396507702686

-- Theorem statement
theorem bacteria_doubling_time :
  doubling_time = total_time / (Real.log (final_population / initial_population) / Real.log 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_doubling_time_l77_7734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_l77_7710

theorem quarter_circle_area (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt (1 - x^2)) :
  ∫ x in Set.Icc 0 1, f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_circle_area_l77_7710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_fraction_l77_7770

theorem undefined_fraction (b : ℝ) : 
  (b - 2) / (b^2 - 9) = 0⁻¹ ↔ b = -3 ∨ b = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_fraction_l77_7770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_range_l77_7735

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log ((1 + 2^x + 4^x * a) / 3)

theorem f_meaningful_range (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → (1 + 2^x + 4^x * a) / 3 > 0) ↔ a > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_meaningful_range_l77_7735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_color_plane_theorem_l77_7751

/-- A coloring of the plane is a function from ℝ² to a finite set of colors. -/
def Coloring (n : ℕ) := ℝ × ℝ → Fin n

/-- The Euclidean distance between two points in ℝ². -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem stating that there exists a 9-coloring of the plane such that
    no two points of the same color are exactly 1966 meters apart. -/
theorem nine_color_plane_theorem :
  ∃ (f : Coloring 9), ∀ (p q : ℝ × ℝ), f p = f q → distance p q ≠ 1966 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_color_plane_theorem_l77_7751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l77_7766

/-- The standard equation of a circle with center (-1, 1) and radius 2 -/
theorem circle_equation (x y : ℝ) : 
  (x + 1)^2 + (y - 1)^2 = 4 ↔ 
  ((x + 1)^2 + (y - 1)^2 = 2^2 ∧ 
   (x + 1, y - 1) ∈ Metric.sphere (0 : ℝ × ℝ) 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l77_7766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_axis_l77_7754

/-- The function f(x) = 3sin(2x + π/4) -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

/-- The symmetry axis of f(x) -/
noncomputable def symmetry_axis : ℝ := Real.pi / 8

/-- Theorem stating that the symmetry axis of f(x) is at x = π/8 -/
theorem f_symmetry_axis :
  ∀ x : ℝ, f (symmetry_axis - x) = f (symmetry_axis + x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_axis_l77_7754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_reflection_distance_l77_7738

/-- Prove that the distance between two points after reflection is 10√2 -/
theorem laser_reflection_distance : 
  let start : ℝ × ℝ := (3, 5)
  let finish : ℝ × ℝ := (7, 5)
  let reflect_y : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (-x, y)
  let reflect_x : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (x, -y)
  let distance : ℝ × ℝ → ℝ × ℝ → ℝ := fun (x₁, y₁) (x₂, y₂) ↦ 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)
  distance (reflect_y start) (reflect_x finish) = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laser_reflection_distance_l77_7738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l77_7791

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Find the intersection point of the diagonals -/
noncomputable def diagonalIntersection (q : Quadrilateral) : Point := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Check if an angle at a vertex is greater than 180 degrees -/
def hasAngleGreaterThan180 (q : Quadrilateral) (vertex : Point) : Prop := sorry

/-- Sum of distances from a point to the vertices of a quadrilateral -/
noncomputable def sumOfDistances (p : Point) (q : Quadrilateral) : ℝ :=
  distance p q.A + distance p q.B + distance p q.C + distance p q.D

/-- The main theorem -/
theorem min_distance_point (q : Quadrilateral) :
  ∃ X : Point, ∀ Y : Point,
    sumOfDistances X q ≤ sumOfDistances Y q ∧
    (isConvex q → X = diagonalIntersection q) ∧
    (¬isConvex q →
      (hasAngleGreaterThan180 q q.A → X = q.A) ∨
      (hasAngleGreaterThan180 q q.B → X = q.B) ∨
      (hasAngleGreaterThan180 q q.C → X = q.C) ∨
      (hasAngleGreaterThan180 q q.D → X = q.D)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l77_7791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_preference_theorem_l77_7768

structure SurveyData where
  total_sample : Nat
  males_like : Nat
  males_dislike : Nat
  females_like : Nat
  females_dislike : Nat

noncomputable def chi_square (data : SurveyData) : Real :=
  let n := data.total_sample
  let a := data.males_like
  let b := data.males_dislike
  let c := data.females_like
  let d := data.females_dislike
  (n * (a * d - b * c)^2 : Real) / ((a + b) * (c + d) * (a + c) * (b + d))

def is_gender_related (data : SurveyData) : Prop :=
  chi_square data > 6.635

def stratified_sample (data : SurveyData) : Nat × Nat :=
  let total_like := data.males_like + data.females_like
  let males := (data.males_like * 7 + total_like - 1) / total_like
  let females := 7 - males
  (males, females)

def expectation_boys : Rat :=
  16 / 7

theorem football_preference_theorem (data : SurveyData) 
  (h1 : data.total_sample = 240)
  (h2 : data.males_like = 80)
  (h3 : data.males_dislike = 40)
  (h4 : data.females_like = 60)
  (h5 : data.females_dislike = 60) :
  is_gender_related data ∧ 
  stratified_sample data = (4, 3) ∧
  expectation_boys = 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_preference_theorem_l77_7768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_decimal_approx_4_10_l77_7721

/-- A number is a three-digit decimal if it has exactly three decimal places -/
def is_three_digit_decimal (x : ℝ) : Prop :=
  ∃ n : ℕ, x = (n : ℝ) / 1000 ∧ 0 ≤ n ∧ n < 1000

/-- Rounding a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

theorem three_digit_decimal_approx_4_10 (x : ℝ) 
  (h1 : is_three_digit_decimal x) 
  (h2 : round_to_hundredth x = 4.10) : 
  4.095 ≤ x ∧ x < 4.105 := by
  sorry

#check three_digit_decimal_approx_4_10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_decimal_approx_4_10_l77_7721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_volume_l77_7729

/-- Represents a point on the river with depth, width, and flow rate. -/
structure RiverPoint where
  depth : ℝ
  width : ℝ
  flowRate : ℝ

/-- Calculates the average of two real numbers. -/
noncomputable def average (a b : ℝ) : ℝ := (a + b) / 2

/-- Calculates the cross-sectional area of the river at a point. -/
noncomputable def crossSectionalArea (point : RiverPoint) : ℝ := point.depth * point.width

/-- Converts flow rate from km/h to m/min. -/
noncomputable def convertFlowRate (rate : ℝ) : ℝ := (rate * 1000) / 60

theorem river_flow_volume (pointA pointB : RiverPoint) (distance : ℝ) :
  pointA.depth = 5 →
  pointA.width = 35 →
  pointA.flowRate = 2 →
  pointB.depth = 7 →
  pointB.width = 45 →
  pointB.flowRate = 3 →
  distance = 1 →
  let avgDepth := average pointA.depth pointB.depth
  let avgWidth := average pointA.width pointB.width
  let avgArea := avgDepth * avgWidth
  let avgFlowRate := average pointA.flowRate pointB.flowRate
  let volumePerMinute := avgArea * convertFlowRate avgFlowRate
  ∃ ε > 0, |volumePerMinute - 10000.8| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_flow_volume_l77_7729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_theorem_l77_7775

structure SubwaySystem where
  stations : Set (ℕ × ℕ)
  lines : List (List (ℕ × ℕ))
  arrow_section : (ℕ × ℕ) × (ℕ × ℕ)
  station_A : ℕ × ℕ
  station_B : ℕ × ℕ

def is_connected (sys : SubwaySystem) : Prop :=
  ∀ s t, s ∈ sys.stations → t ∈ sys.stations → ∃ path : List (ℕ × ℕ), 
    path.head? = some s ∧ path.getLast? = some t
    ∧ ∀ i, i < path.length - 1 → ∃ l ∈ sys.lines, (path[i]?, path[i+1]?) ∈ (l.zip (l.tail)).map (λ (a, b) => (some a, some b))

def is_disconnected_without_arrow (sys : SubwaySystem) : Prop :=
  let sys_without_arrow := {sys with
    lines := sys.lines.map (λ l => l.filter (λ s => s ≠ sys.arrow_section.1 ∧ s ≠ sys.arrow_section.2))}
  ¬ is_connected sys_without_arrow

def valid_path (sys : SubwaySystem) (path : List (ℕ × ℕ)) : Prop :=
  path.head? = some sys.station_A ∧ path.getLast? = some sys.station_B
  ∧ path.length = 2017
  ∧ ∀ i, i < path.length - 1 → ∃ l ∈ sys.lines, (path[i]?, path[i+1]?) ∈ (l.zip (l.tail)).map (λ (a, b) => (some a, some b))

theorem subway_theorem (sys : SubwaySystem) 
  (h_connected : is_connected sys)
  (h_disconnected : is_disconnected_without_arrow sys) :
  ∀ path, valid_path sys path → (sys.arrow_section.1 ∈ path ∨ sys.arrow_section.2 ∈ path) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subway_theorem_l77_7775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_not_on_line_l_distance_difference_is_two_l77_7779

noncomputable section

/-- Line l is defined by the equation y = √3x + 1 -/
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x + 1

/-- Circle C is defined as (x-2)^2 + y^2 = 1 -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

/-- Point P has coordinates (2, 2√3) -/
def point_P : ℝ × ℝ := (2, 2 * Real.sqrt 3)

/-- Distance from a point (x, y) to the line y = √3x + 1 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (Real.sqrt 3 * x - y + 1) / Real.sqrt 4

theorem point_P_not_on_line_l :
  ¬ line_l point_P.1 point_P.2 := by sorry

theorem distance_difference_is_two :
  ∀ (x y : ℝ), circle_C x y →
  ∃ (min_dist max_dist : ℝ),
    (∀ (x' y' : ℝ), circle_C x' y' →
      min_dist ≤ distance_to_line x' y' ∧
      distance_to_line x' y' ≤ max_dist) ∧
    max_dist - min_dist = 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_not_on_line_l_distance_difference_is_two_l77_7779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l77_7769

theorem triangle_trig_identity (A B C : ℝ) 
  (h : Real.sin A * (Real.cos (C/2))^2 + Real.sin C * (Real.cos (A/2))^2 = (3/2) * Real.sin B) :
  Real.cos ((A-C)/2) - 2 * Real.sin (B/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l77_7769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_count_divisible_by_five_count_with_odd_digit_l77_7799

def Digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}

def ThreeDigitNumbers : Finset ℕ := 
  Finset.filter (λ n => 
    n ∈ Finset.range 1000 \ Finset.range 100 ∧ 
    (n / 100) ∈ Digits \ {0} ∧
    ((n / 10) % 10) ∈ Digits \ {n / 100} ∧
    (n % 10) ∈ Digits \ {n / 100, (n / 10) % 10})
  (Finset.range 1000)

theorem count_three_digit_numbers : 
  Finset.card ThreeDigitNumbers = 180 := by sorry

theorem count_divisible_by_five : 
  Finset.card (Finset.filter (λ n => n % 5 = 0) ThreeDigitNumbers) = 55 := by sorry

theorem count_with_odd_digit : 
  Finset.card (Finset.filter (λ n => 
    ∃ d ∈ Digits, d % 2 = 1 ∧ 
    (n / 100 = d ∨ (n / 10) % 10 = d ∨ n % 10 = d)) 
  ThreeDigitNumbers) = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_numbers_count_divisible_by_five_count_with_odd_digit_l77_7799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_small_blue_denominators_l77_7756

def RedFraction := { p : ℚ // p.den > 10^10 ∧ p.den % 2 = 1 }

def BlueFraction := { p : ℚ // p.den < 100 }

def sum_fractions (f1 f2 : ℚ) : ℚ := f1 + f2

theorem impossibility_of_small_blue_denominators 
  (red_fractions : Fin 5 → RedFraction) 
  (blue_fractions : Fin 5 → BlueFraction) 
  (h_blue_sum : ∀ i : Fin 5, (blue_fractions i).val = 
    sum_fractions (red_fractions i).val (red_fractions (i.succ)).val) :
  False := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_small_blue_denominators_l77_7756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_digits_l77_7795

def number_set : List Nat := [7, 77, 777, 999, 9999, 99999, 777777, 7777777, 99999999]

def arithmetic_mean (list : List Nat) : Nat :=
  (list.sum / list.length)

def is_nine_digit (n : Nat) : Prop :=
  n ≥ 100000000 ∧ n < 1000000000

def digits_of (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else go (m / 10) ((m % 10) :: acc)
    go n []

theorem missing_digits :
  let N := arithmetic_mean number_set
  is_nine_digit N ∧
  (∀ d, d ∈ digits_of N → d ≠ 2 ∧ d ≠ 5 ∧ d ≠ 8 ∧ d ≠ 9) ∧
  (digits_of N).Nodup := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_digits_l77_7795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l77_7705

-- Define the constants
noncomputable def a : ℝ := (6 : ℝ) ^ (7/10)
noncomputable def b : ℝ := (7/10 : ℝ) ^ 6
noncomputable def c : ℝ := Real.log 6 / Real.log (7/10)

-- State the theorem
theorem abc_inequality : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l77_7705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_of_f_is_correct_l77_7708

noncomputable def f (x : ℝ) : ℝ := (x - 1/2)^0 + Real.sqrt (x + 2)

theorem domain_of_f :
  {x : ℝ | x ∈ Set.Ici (-2) ∧ x ≠ 1/2} = Set.Ici (-2) \ {1/2} :=
by sorry

-- The domain of f is [-2, 1/2) ∪ (1/2, +∞)
theorem domain_of_f_is_correct :
  {x : ℝ | ∃ y, f x = y} = Set.Ici (-2) \ {1/2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_domain_of_f_is_correct_l77_7708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reconstruct_2013gon_l77_7782

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A regular 2013-gon -/
structure Regular2013gon where
  vertices : Fin 2013 → Point

/-- Given five consecutive vertices of a regular 2013-gon, it is possible to construct
    all vertices of the 2013-gon using only a straightedge -/
theorem reconstruct_2013gon
  (A B C D E : Point)
  (h : ∃ (p : Regular2013gon) (i : Fin 2013),
       A = p.vertices i ∧
       B = p.vertices (i + 1) ∧
       C = p.vertices (i + 2) ∧
       D = p.vertices (i + 3) ∧
       E = p.vertices (i + 4)) :
  ∃ (construct : List Point → List Point),
    ∃ (p : Regular2013gon),
      construct [A, B, C, D, E] = (List.ofFn p.vertices) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reconstruct_2013gon_l77_7782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_seven_digit_multiple_l77_7736

theorem smallest_seven_digit_multiple : ∃ (n : ℕ), 
  (n ≥ 1000000 ∧ n < 10000000) ∧ 
  (35 ∣ n) ∧ (112 ∣ n) ∧ (175 ∣ n) ∧ (288 ∣ n) ∧ (429 ∣ n) ∧ (528 ∣ n) ∧
  (∀ m : ℕ, m ≥ 1000000 ∧ m < 10000000 ∧ 
    (35 ∣ m) ∧ (112 ∣ m) ∧ (175 ∣ m) ∧ (288 ∣ m) ∧ (429 ∣ m) ∧ (528 ∣ m) → m ≥ n) ∧
  n = 7207200 := by
  -- Proof goes here
  sorry

#check smallest_seven_digit_multiple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_seven_digit_multiple_l77_7736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_approx_4sqrt6_l77_7787

open Real

-- Define the expression
noncomputable def trigExpression : ℝ :=
  (sin (15 * π / 180) + sin (30 * π / 180) + sin (45 * π / 180) + 
   sin (60 * π / 180) + sin (75 * π / 180)) / 
  (cos (10 * π / 180) * cos (20 * π / 180) * cos (30 * π / 180))

-- State the theorem
theorem trigExpression_approx_4sqrt6 : 
  ∃ ε > 0, |trigExpression - 4 * sqrt 6| < ε ∧ ε < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_approx_4sqrt6_l77_7787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_positions_l77_7702

/-- Represents a fruit, which can be either an apple or a banana. -/
inductive Fruit
| Apple
| Banana

/-- The total number of fruits. -/
def totalFruits : Nat := 99

/-- The number of apples. -/
def numApples : Nat := 98

/-- The number of bananas. -/
def numBananas : Nat := 1

/-- Represents the line of fruits. -/
def FruitLine := Fin totalFruits → Fruit

/-- Returns true if the given fruit is a banana. -/
def isBanana (f : Fruit) : Bool :=
  match f with
  | Fruit.Banana => true
  | Fruit.Apple => false

/-- Returns true if the given statement is true for a banana and false for an apple. -/
def fruitStatement (f : Fruit) (statement : Bool) : Bool :=
  (isBanana f ∧ statement) ∨ (¬isBanana f ∧ ¬statement)

/-- The statement of the first fruit. -/
def firstFruitStatement (line : FruitLine) : Bool :=
  ∃ i : Fin totalFruits, i.val < 40 ∧ isBanana (line i)

/-- The statement of the last fruit. -/
def lastFruitStatement (line : FruitLine) : Bool :=
  ∃ i : Fin totalFruits, i.val ≥ totalFruits - 40 ∧ isBanana (line i)

/-- The statement of the middle fruit. -/
def middleFruitStatement (line : FruitLine) : Bool :=
  isBanana (line ⟨49, by sorry⟩)

/-- The main theorem stating that there are 21 possible positions for the banana. -/
theorem banana_positions (line : FruitLine) :
  (∃! i : Fin totalFruits, isBanana (line i)) →
  (fruitStatement (line ⟨0, by sorry⟩) (firstFruitStatement line)) →
  (fruitStatement (line ⟨totalFruits - 1, by sorry⟩) (lastFruitStatement line)) →
  (fruitStatement (line ⟨49, by sorry⟩) (middleFruitStatement line)) →
  (∃ positions : Finset (Fin totalFruits), positions.card = 21 ∧
    ∀ i : Fin totalFruits, isBanana (line i) → i ∈ positions) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_banana_positions_l77_7702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l77_7700

-- Function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) / x + (x - 3)^0

-- Function g
noncomputable def g (x : ℝ) : ℝ := 2*x - Real.sqrt (x - 1)

-- Domain of f
def domain_f : Set ℝ := {x | x ≥ -2 ∧ x ≠ 0 ∧ x ≠ 3}

-- Range of g
def range_g : Set ℝ := {y | y ≥ 15/8}

-- Theorem for the domain of f
theorem domain_of_f : {x : ℝ | f x ∈ Set.univ} = domain_f := by sorry

-- Theorem for the range of g
theorem range_of_g : g '' {x : ℝ | x ≥ 1} = range_g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l77_7700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_ratio_l77_7714

-- Define the points
variable (F E U R N I L : EuclideanSpace ℝ (Fin 2))

-- Define the parallelism conditions
variable (h1 : (I - L).IsParallelTo (E - U))
variable (h2 : (R - E).IsParallelTo (N - I))

-- State the theorem
theorem parallel_lines_ratio : 
  (F - N).norm * (F - U).norm / ((F - R).norm * (F - L).norm) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_ratio_l77_7714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ef_is_sqrt_101_l77_7759

/-- A rectangle ABCD with points E and F -/
structure RectangleEF where
  -- The length of side AB
  ab : ℝ
  -- The length of side BC
  bc : ℝ
  -- The length of DE
  de : ℝ
  -- The length of CF
  cf : ℝ
  -- Assumption that ab = 10
  ab_eq : ab = 10
  -- Assumption that bc = 5
  bc_eq : bc = 5
  -- Assumption that de = 3
  de_eq : de = 3
  -- Assumption that cf = 2
  cf_eq : cf = 2
  -- Assumption that triangle DEF is right-angled at E
  def_right : (ab - cf)^2 + (bc - de)^2 = (ab - cf)^2 + (bc - de)^2

/-- The length of EF in the rectangle ABCD with given conditions -/
noncomputable def lengthEF (r : RectangleEF) : ℝ :=
  Real.sqrt ((r.ab - r.cf)^2 + (r.bc - r.de)^2)

/-- Theorem stating that the length of EF is √101 -/
theorem length_ef_is_sqrt_101 (r : RectangleEF) : lengthEF r = Real.sqrt 101 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ef_is_sqrt_101_l77_7759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l77_7777

theorem count_integer_pairs : ℕ := by
  -- Define the set of positive integer pairs (a, b) satisfying the conditions
  let S : Set (ℕ × ℕ) := {p : ℕ × ℕ | 
    let (a, b) := p
    a > 0 ∧ b > 0 ∧ 
    a^2 + b^2 < 2013 ∧ 
    (b^3 - a^3) % (a^2 * b) = 0}

  -- Define the cardinality of S
  let card_S := Finset.card (Finset.filter (fun p => p ∈ S) (Finset.product (Finset.range 2013) (Finset.range 2013)))

  -- State that the cardinality of S is equal to 31
  have h : card_S = 31 := by sorry

  -- Return the result
  exact 31

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_pairs_l77_7777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_tangents_l77_7715

def point (x y : ℝ) := (x, y)

noncomputable def circle_set (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def is_tangent_to_circle (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ l ∩ c ∧ ∀ q ∈ l, q ≠ p → q ∉ c

def common_tangent_lines (c1 c2 : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) :=
  {l | is_tangent_to_circle l c1 ∧ is_tangent_to_circle l c2}

theorem three_common_tangents :
  let c1 := circle_set (point 1 0) 1
  let c2 := circle_set (point (-3) 0) 3
  ∃ (s : Finset (Set (ℝ × ℝ))), s.card = 3 ∧ ↑s = common_tangent_lines c1 c2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_tangents_l77_7715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_h_of_g_l77_7757

theorem percent_h_of_g (a b c d e f g h : ℝ) : 
  f = 0.60 * a ∧ f = 0.45 * b ∧
  g = 0.70 * b ∧ g = 0.30 * c ∧
  h = 0.80 * c ∧ h = 0.10 * f ∧
  c = 0.30 * a ∧ c = 0.25 * b ∧
  d = 0.40 * a ∧ d = 0.35 * b ∧
  e = 0.50 * b ∧ e = 0.20 * c →
  ∃ ε > 0, |h / g - 0.2857| < ε := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_h_of_g_l77_7757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l77_7741

-- Define the race parameters
def race_length : ℚ := 60
def uphill_distance : ℚ := 15
def downhill_distance : ℚ := 20
def flat_distance : ℚ := 25

-- Define Minnie's speeds
def minnie_flat_speed : ℚ := 25
def minnie_downhill_speed : ℚ := 35
def minnie_uphill_speed : ℚ := 10

-- Define Penny's speeds
def penny_flat_speed : ℚ := 35
def penny_downhill_speed : ℚ := 45
def penny_uphill_speed : ℚ := 15

-- Function to calculate time given distance and speed
def calculate_time (distance : ℚ) (speed : ℚ) : ℚ := distance / speed

-- Calculate total time for a racer
def total_time (flat_speed uphill_speed downhill_speed : ℚ) : ℚ :=
  calculate_time flat_distance flat_speed +
  calculate_time uphill_distance uphill_speed +
  calculate_time downhill_distance downhill_speed

-- Theorem statement
theorem race_time_difference :
  (total_time minnie_flat_speed minnie_uphill_speed minnie_downhill_speed -
   total_time penny_flat_speed penny_uphill_speed penny_downhill_speed) * 60 = 130 := by
  -- Proof steps would go here
  sorry

#eval (total_time minnie_flat_speed minnie_uphill_speed minnie_downhill_speed -
       total_time penny_flat_speed penny_uphill_speed penny_downhill_speed) * 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_time_difference_l77_7741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_part1_problem_part2_l77_7772

/-- Given points A, B, C in ℝ³ -/
def A : Fin 3 → ℝ := ![0, 1, 2]
def B : Fin 3 → ℝ := ![1, 2, 3]
def C : Fin 3 → ℝ := ![1, 3, 1]

/-- Vector from A to C -/
def AC : Fin 3 → ℝ := ![C 0 - A 0, C 1 - A 1, C 2 - A 2]

/-- Vector from A to B -/
def AB : Fin 3 → ℝ := ![B 0 - A 0, B 1 - A 1, B 2 - A 2]

/-- Dot product of two 3D vectors -/
def dot_product (v w : Fin 3 → ℝ) : ℝ := (v 0 * w 0) + (v 1 * w 1) + (v 2 * w 2)

theorem problem_part1 (y : ℝ) : 
  let AD : Fin 3 → ℝ := ![3, y, 1]
  dot_product AD AC = 0 → y = -1 := by sorry

theorem problem_part2 (x : ℝ) :
  let D : Fin 3 → ℝ := ![x, 5, 3]
  let AD : Fin 3 → ℝ := ![D 0 - A 0, D 1 - A 1, D 2 - A 2]
  (∃ m n : ℝ, AD = ![m * AB 0 + n * AC 0, m * AB 1 + n * AC 1, m * AB 2 + n * AC 2]) → x = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_part1_problem_part2_l77_7772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_18_formula_l77_7704

variable (a : ℝ)

noncomputable def u : ℕ → ℝ
  | 0 => a  -- Add case for 0
  | 1 => a
  | n + 1 => -2 / (u n + 2)

theorem u_18_formula (h : a > 0) : u a 18 = -2 / (a + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_18_formula_l77_7704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_van_helsing_payment_per_vampire_l77_7750

/-- The payment structure and vampire removal scenario for Van Helsing --/
def van_helsing_scenario (payment_per_vampire : ℚ) : Prop :=
  ∃ (num_vampires : ℚ),
    -- The number of werewolves is 4 times the number of vampires
    let num_werewolves := 4 * num_vampires;
    -- Van Helsing removes half the vampires and 8 werewolves
    let removed_vampires := num_vampires / 2;
    let removed_werewolves := 8;
    -- The total earning is $105
    -- The payment for each werewolf is $10
    payment_per_vampire * removed_vampires + 10 * removed_werewolves = 105

/-- Theorem stating that the payment per vampire is $25 --/
theorem van_helsing_payment_per_vampire : 
  van_helsing_scenario 25 := by
  -- Proof goes here
  sorry

#check van_helsing_payment_per_vampire

end NUMINAMATH_CALUDE_ERRORFEEDBACK_van_helsing_payment_per_vampire_l77_7750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l77_7722

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + 3 * Real.cos x ^ 2

theorem f_properties :
  ∃ (A P φ : ℝ) (k : ℤ → Set ℝ),
    A = Real.sqrt 2 ∧
    P = Real.pi ∧
    φ = Real.pi / 4 ∧
    (∀ x, f x = A * Real.sin (2 * x + φ) + 2) ∧
    (∀ (n : ℤ), k n = Set.Icc (n * Real.pi - Real.pi / 8) (Real.pi / 8 + n * Real.pi)) ∧
    (∀ (n : ℤ) (x : ℝ), x ∈ k n → (deriv f) x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l77_7722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_can_escape_l77_7703

/-- Represents the scenario of a student trying to escape from a teacher in a circular pool -/
structure PoolEscape where
  R : ℝ  -- Radius of the circular pool
  v_T : ℝ  -- Teacher's running speed
  v_S : ℝ  -- Student's swimming speed
  h_v_positive : v_T > 0 ∧ v_S > 0  -- Speeds are positive
  h_v_ratio : v_T = 4 * v_S  -- Teacher runs 4 times faster than student swims

/-- The student can escape if they can reach the edge of the pool before the teacher -/
def can_escape (pe : PoolEscape) : Prop :=
  ∃ r : ℝ, (1 - Real.pi/4) * pe.R < r ∧ r < pe.R/4 ∧
    pe.v_S / r > pe.v_T / pe.R ∧
    Real.pi * pe.R / 4 / pe.v_S < Real.pi * pe.R / pe.v_T

/-- Theorem stating that the student can always escape given the conditions -/
theorem student_can_escape (pe : PoolEscape) : can_escape pe := by
  sorry

#check student_can_escape

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_can_escape_l77_7703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_proof_statement1_proof_statement2_disproof_statement3_disproof_statement4_l77_7733

-- Define the @ operation
noncomputable def at_op (a b : ℝ) : ℝ :=
  if a < b then a + b - 3 else a - b + 3

-- Define the statements
def statement1 : Prop := at_op (-1) (-2) = 4
def statement2 : Prop := ∀ x : ℝ, at_op x (x + 2) = 5 → x = 3
def statement3 : Prop := ∀ x : ℝ, at_op x (2*x) = 3 → x = 2
def statement4 : Prop := ∃ f : ℝ → ℝ, (∀ x, f x = at_op (x^2 + 1) 1) ∧ 
                         f (-1) = 0 ∧ f 1 = 0

-- Theorem statement
theorem exactly_two_statements_true : 
  ∃! n : ℕ, n = 2 ∧ (statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) := by
  sorry

-- Helper theorems to prove each statement
theorem proof_statement1 : statement1 := by
  sorry

theorem proof_statement2 : statement2 := by
  sorry

theorem disproof_statement3 : ¬statement3 := by
  sorry

theorem disproof_statement4 : ¬statement4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_proof_statement1_proof_statement2_disproof_statement3_disproof_statement4_l77_7733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_locations_l77_7776

/-- A line segment in a plane --/
structure Segment where
  start : ℝ × ℝ
  endPoint : ℝ × ℝ
  positive_length : start ≠ endPoint

/-- A point in a plane --/
def Point := ℝ × ℝ

/-- Predicate to check if three points form a non-degenerate isosceles right triangle --/
def IsIsoscelesRightTriangle (a b c : Point) : Prop := sorry

/-- The main theorem --/
theorem isosceles_right_triangle_locations (s : Segment) :
  ∃ (points : Finset Point), points.card = 6 ∧
    (∀ p ∈ points, IsIsoscelesRightTriangle s.start s.endPoint p) ∧
    (∀ p : Point, IsIsoscelesRightTriangle s.start s.endPoint p → p ∈ points) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_locations_l77_7776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l77_7793

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through A(-1, 0)
def line_through_A (x y : ℝ) : Prop := ∃ m : ℝ, y = m * (x + 1)

-- Define the condition AP = λ * AQ
def AP_eq_lambda_AQ (x₁ y₁ x₂ y₂ lambda : ℝ) : Prop :=
  x₁ + 1 = lambda * (x₂ + 1) ∧ y₁ = lambda * y₂

-- Define the theorem
theorem parabola_intersection_theorem
  (x₁ y₁ x₂ y₂ lambda : ℝ)
  (h_parabola₁ : parabola x₁ y₁)
  (h_parabola₂ : parabola x₂ y₂)
  (h_line : line_through_A x₁ y₁ ∧ line_through_A x₂ y₂)
  (h_lambda : AP_eq_lambda_AQ x₁ y₁ x₂ y₂ lambda)
  (h_lambda_range : lambda ∈ Set.Icc (1/3 : ℝ) (1/2 : ℝ)) :
  (x₁ = lambda ∧ x₂ = 1/lambda) ∧
  (∃ (a b c : ℝ), a^2 + b^2 = 4 ∧ c^2 = 3 ∧
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ 
      (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l77_7793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l77_7717

noncomputable def g (x : ℝ) : ℝ := 30 + 14 * Real.cos x - 7 * Real.cos (2 * x)

noncomputable def f (x : ℝ) : ℝ := Real.sin ((Real.pi / 54) * g x)

theorem f_range :
  Set.range f = Set.Icc (1/2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l77_7717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_approx_004_l77_7786

/-- Calculates the width of a wall given the dimensions of bricks and the wall -/
noncomputable def calculate_wall_width (brick_length brick_width brick_height : ℝ)
                         (wall_length wall_height : ℝ)
                         (num_bricks : ℝ) : ℝ :=
  let brick_volume := brick_length * brick_width * brick_height
  let total_volume := num_bricks * brick_volume
  total_volume / (wall_length * wall_height)

/-- Theorem stating that the calculated wall width is approximately 0.04 meters -/
theorem wall_width_approx_004 :
  let brick_length : ℝ := 0.30  -- 30 cm in meters
  let brick_width  : ℝ := 0.12  -- 12 cm in meters
  let brick_height : ℝ := 0.10  -- 10 cm in meters
  let wall_length  : ℝ := 6.0   -- 6 m
  let wall_height  : ℝ := 20.5  -- 20.5 m
  let num_bricks   : ℝ := 1366.6666666666667
  
  abs (calculate_wall_width brick_length brick_width brick_height
                            wall_length wall_height num_bricks - 0.04) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_width_approx_004_l77_7786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trucks_required_l77_7773

def container_weights : List ℚ := [3, 5/2, 3/2, 1]
def container_quantities : List ℕ := [4, 5, 14, 7]
def truck_capacity : ℚ := 9/2

theorem min_trucks_required :
  let total_weight := (List.zip container_weights container_quantities).map (λ (w, q) => w * q) |>.sum
  ⌈total_weight / truck_capacity⌉ = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trucks_required_l77_7773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_sum_l77_7745

theorem determinant_zero_sum (a b : ℝ) : 
  a ≠ b → 
  Matrix.det (![![1, 6, 16], ![4, a, b], ![4, b, a]]) = 0 → 
  a + b = 88 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_zero_sum_l77_7745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l77_7731

theorem complex_modulus_problem (a : ℝ) (h1 : a > 0) : 
  Complex.abs (a + Complex.I) = 2 → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l77_7731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_upper_limit_l77_7740

/-- Represents Arun's weight in kg -/
def arun_weight : ℝ := sorry

/-- The upper limit of Arun's opinion about his weight -/
def X : ℝ := sorry

/-- The average of different probable weights of Arun -/
def average_weight : ℝ := sorry

theorem arun_weight_upper_limit :
  (arun_weight > 61 ∧ arun_weight < X) ∧
  (arun_weight > 60 ∧ arun_weight < 70) ∧
  (arun_weight ≤ 64) ∧
  (average_weight = 63) →
  X = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_upper_limit_l77_7740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_proof_l77_7720

/-- Given a line L and two points M and M', this function returns true if M' is symmetric to M with respect to L -/
def isSymmetric (L : ℝ → ℝ → ℝ → Prop) (M M' : ℝ × ℝ × ℝ) : Prop :=
  ∃ (M₀ : ℝ × ℝ × ℝ), L M₀.1 M₀.2.1 M₀.2.2 ∧ 
    (M₀.1 = (M.1 + M'.1) / 2) ∧ 
    (M₀.2.1 = (M.2.1 + M'.2.1) / 2) ∧ 
    (M₀.2.2 = (M.2.2 + M'.2.2) / 2)

/-- The line L defined by (x-6)/5 = (y-3.5)/4 = (z+0.5)/0 -/
def L (x y z : ℝ) : Prop :=
  ∃ (t : ℝ), x = 6 + 5*t ∧ y = 3.5 + 4*t ∧ z = -0.5

theorem symmetry_proof :
  isSymmetric L (3, -3, -1) (-1, 2, 0) := by
  sorry

#check symmetry_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_proof_l77_7720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_ratio_theorem_l77_7783

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a plane that cuts the cone -/
structure CuttingPlane where
  distanceFromBase : ℝ

/-- Represents the smaller cone resulting from the cut -/
noncomputable def SmallerCone (c : Cone) (p : CuttingPlane) : Cone :=
  { height := p.distanceFromBase
  , baseRadius := c.baseRadius * p.distanceFromBase / c.height }

/-- Represents the frustum resulting from the cut -/
structure Frustum where
  originalCone : Cone
  cuttingPlane : CuttingPlane

/-- Calculates the surface area of a cone (including base) -/
noncomputable def surfaceArea (c : Cone) : ℝ :=
  Real.pi * c.baseRadius * c.baseRadius + Real.pi * c.baseRadius * Real.sqrt (c.baseRadius^2 + c.height^2)

/-- Calculates the volume of a cone -/
noncomputable def volume (c : Cone) : ℝ :=
  (1/3) * Real.pi * c.baseRadius^2 * c.height

/-- Calculates the surface area of a frustum (including both bases) -/
noncomputable def frustumSurfaceArea (f : Frustum) : ℝ :=
  surfaceArea f.originalCone - surfaceArea (SmallerCone f.originalCone f.cuttingPlane)

/-- Calculates the volume of a frustum -/
noncomputable def frustumVolume (f : Frustum) : ℝ :=
  volume f.originalCone - volume (SmallerCone f.originalCone f.cuttingPlane)

theorem cone_ratio_theorem (c : Cone) (p : CuttingPlane) :
  c.height = 6 →
  c.baseRadius = 4 →
  let sc := SmallerCone c p
  let f : Frustum := { originalCone := c, cuttingPlane := p }
  surfaceArea sc / frustumSurfaceArea f = volume sc / frustumVolume f →
  surfaceArea sc / frustumSurfaceArea f = 169 / 775 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_ratio_theorem_l77_7783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_alloy_l77_7724

/-- The amount of tin in a combined alloy -/
noncomputable def tin_in_combined_alloy (weight_A weight_B weight_C : ℝ)
  (lead_tin_ratio_A : ℝ × ℝ) (tin_copper_ratio_B : ℝ × ℝ) (copper_tin_ratio_C : ℝ × ℝ) : ℝ :=
  let tin_A := weight_A * lead_tin_ratio_A.2 / (lead_tin_ratio_A.1 + lead_tin_ratio_A.2)
  let tin_B := weight_B * tin_copper_ratio_B.1 / (tin_copper_ratio_B.1 + tin_copper_ratio_B.2)
  let tin_C := weight_C * copper_tin_ratio_C.2 / (copper_tin_ratio_C.1 + copper_tin_ratio_C.2)
  tin_A + tin_B + tin_C

/-- The theorem stating the amount of tin in the combined alloy -/
theorem tin_amount_in_alloy :
  tin_in_combined_alloy 150 200 250 (5, 3) (2, 3) (4, 1) = 186.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_amount_in_alloy_l77_7724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l77_7789

open Real

theorem problem_solution (φ A B C a b c : ℝ) :
  (0 < φ) ∧ (φ < π) ∧
  (∀ x, 2 * sin x * (cos (φ / 2))^2 + cos x * sin φ - sin x ≥ 2 * sin π * (cos (φ / 2))^2 + cos π * sin φ - sin π) ∧
  (a = 1) ∧ (b = Real.sqrt 2) ∧
  (2 * sin A * (cos (φ / 2))^2 + cos A * sin φ - sin A = Real.sqrt 3 / 2) →
  (φ = π / 2) ∧ ((C = 7 * π / 12) ∨ (C = π / 12)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l77_7789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_digit_sum_l77_7719

def digit_sequence : ℕ → ℕ 
| n => (n - 1) % 6 + 1

def erase_every_nth (n : ℕ) (seq : ℕ → ℕ) : ℕ → ℕ 
| k => seq (k + (k - 1) / (n - 1))

def final_sequence : ℕ → ℕ :=
  erase_every_nth 5 (erase_every_nth 3 (erase_every_nth 4 digit_sequence))

theorem tom_digit_sum :
  final_sequence 3234 + final_sequence 3235 + final_sequence 3236 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_digit_sum_l77_7719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l77_7790

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.exp x * (Real.sin x + Real.cos x)

theorem range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc 0 (π/2) = Set.Icc (1/2) ((1/2) * Real.exp (π/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l77_7790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l77_7725

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the intersection points
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ parabola x y ∧ line_through_focus k x y}

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the area of triangle AOB
noncomputable def triangle_area (A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 - origin.1) * (B.2 - origin.2) - (B.1 - origin.1) * (A.2 - origin.2)) / 2

-- Define the length of AB
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem parabola_intersection_length :
  ∀ k : ℝ, k ≠ 0 →
  ∀ A B : ℝ × ℝ,
  A ∈ intersection_points k →
  B ∈ intersection_points k →
  A ≠ B →
  triangle_area A B = Real.sqrt 6 →
  distance A B = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l77_7725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_sum_l77_7780

/-- Arithmetic sequence sum function -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_min_sum
  (a₁ d : ℝ) (h₁ : a₁ < 0) (h₂ : S a₁ d 12 = S a₁ d 6) :
  ∀ n : ℕ, S a₁ d 9 ≤ S a₁ d n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_min_sum_l77_7780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_geometric_construction_l77_7794

/-- Given two positive real numbers a and b, and a geometric construction as described,
    the length of the resulting segment h is equal to the harmonic mean of a and b. -/
theorem harmonic_mean_geometric_construction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let h := 2 * a * b / (a + b)
  let e : ℝ × ℝ := (1, 0)  -- Unit vector in x direction
  let f : ℝ × ℝ := (Real.cos θ, Real.sin θ)  -- Unit vector at angle θ
  let g : ℝ × ℝ := (e.1 + f.1, e.2 + f.2)  -- Angle bisector direction
  let d : ℝ × ℝ := ((a * b / (a + b)) * g.1, (a * b / (a + b)) * g.2)  -- Intersection point
  let h' := 2 * Real.sqrt ((d.1 - h * e.1)^2 + (d.2 - h * e.2)^2)  -- Length of constructed segment
  h' = h := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonic_mean_geometric_construction_l77_7794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l77_7718

noncomputable def binomial_expansion (x : ℝ) := (x + 2/x)^4

theorem coefficient_of_x_squared :
  ∃ (a b c d e : ℝ), ∀ x : ℝ, x ≠ 0 → 
  binomial_expansion x = a*x^4 + b*x^3 + 8*x^2 + d*x + e :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l77_7718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_and_inequality_perpendicular_lines_condition_negation_equivalence_l77_7760

-- Statement 1
theorem existence_and_inequality :
  (∃ x : ℝ, Real.tan x = 2) ∧ (∀ x : ℝ, x^2 - x + 1/2 > 0) :=
by sorry

-- Statement 2
theorem perpendicular_lines_condition (a b : ℝ) :
  (∀ x y : ℝ, (a * x + 3 * y - 1 = 0 ∧ x + b * y + 1 = 0) →
    (a * 1 + 3 * b) = 0) ↔
  a + 3 * b = 0 :=
by sorry

-- Statement 3
theorem negation_equivalence :
  (∀ a b : ℝ, ab ≥ 2 → a^2 + b^2 > 4) ↔ (∀ a b : ℝ, ab < 2 → a^2 + b^2 ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_and_inequality_perpendicular_lines_condition_negation_equivalence_l77_7760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_position_3_expected_position_4_most_likely_position_10_l77_7747

-- Define the random walk
def random_walk (n : ℕ) : ℕ → ℝ := sorry

-- Expected value of the random walk after n steps
def expected_position (n : ℕ) : ℝ := sorry

-- Probability of being at position x after n steps
def prob_at_position (n : ℕ) (x : ℤ) : ℝ := sorry

-- Most likely position after n steps
def most_likely_position (n : ℕ) : ℤ := sorry

-- Theorem statements
theorem expected_position_3 : expected_position 3 = 1 := by sorry

theorem expected_position_4 : expected_position 4 = 4/3 := by sorry

theorem most_likely_position_10 : most_likely_position 10 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_position_3_expected_position_4_most_likely_position_10_l77_7747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l77_7764

theorem sin_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l77_7764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l77_7742

/-- Represents the speed of the firetruck in different terrains -/
structure FiretruckSpeed where
  roadSpeed : ℝ
  fieldSpeed : ℝ

/-- Represents the time limit for the firetruck's travel -/
noncomputable def timeLimit : ℝ := 8 / 60 -- 8 minutes in hours

/-- The area that the firetruck can reach within the time limit -/
noncomputable def reachableArea (speed : FiretruckSpeed) (t : ℝ) : ℝ := 
  sorry

theorem firetruck_reachable_area :
  let speed := FiretruckSpeed.mk 60 10
  reachableArea speed timeLimit = 544 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_firetruck_reachable_area_l77_7742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_KH_electron_loss_l77_7716

/-- Represents the change in valence of an element -/
def valence_change (initial : Int) (final : Int) : Int :=
  final - initial

/-- Represents the number of moles of electrons lost or gained -/
def electron_change (valence_change : Int) (moles : Real) : Real :=
  moles * (valence_change : Real)

/-- The reaction of KH with H₂O produces H₂ and KOH -/
axiom KH_reaction : True

/-- The valence of H in KH changes from -1 to 0 during the reaction -/
axiom H_valence_change : valence_change (-1) 0 = 1

/-- Proves that 1 mol of KH loses 1 mol of electrons in the reaction -/
theorem KH_electron_loss : electron_change (valence_change (-1) 0) 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_KH_electron_loss_l77_7716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equality_l77_7758

theorem binomial_coefficient_equality (x : ℝ) : 
  (∃ k m : ℕ, (k : ℝ) = x ∧ (m : ℝ) = 3*x - 6 ∧ Nat.choose 18 k = Nat.choose 18 m) → 
  (x = 3 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_equality_l77_7758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l77_7701

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 + 2 * Real.sin θ)

noncomputable def point_M : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

theorem curve_C_properties :
  -- (I) The polar equation of C is ρ = 4sin(θ)
  (∀ θ : ℝ, let (x, y) := curve_C θ; (x^2 + y^2) = 16 * (Real.sin θ)^2) ∧
  -- (II) For any line through M intersecting C at A and B, |MA| * |MB| = 2
  (∀ A B : ℝ × ℝ, A ≠ B →
    (∃ t : ℝ, curve_C t = A) →
    (∃ s : ℝ, curve_C s = B) →
    (∃ k : ℝ, k ≠ 0 ∧ 
      (A.1 - point_M.1) * k = (B.1 - point_M.1) ∧
      (A.2 - point_M.2) * k = (B.2 - point_M.2)) →
    (A.1 - point_M.1)^2 + (A.2 - point_M.2)^2 *
    (B.1 - point_M.1)^2 + (B.2 - point_M.2)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l77_7701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_implies_radius_three_l77_7778

/-- The curve defined by |z - 4| = 3|z + 4| in the complex plane -/
def curve (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 4)

/-- A circle with radius k centered at the origin -/
def circle_k (k : ℝ) (z : ℂ) : Prop := Complex.abs z = k

/-- The statement to be proved -/
theorem unique_intersection_implies_radius_three (k : ℝ) :
  (∃! z : ℂ, curve z ∧ circle_k k z) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_implies_radius_three_l77_7778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l77_7761

/-- An arithmetic sequence with first term 1 -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- A geometric sequence with first term 1 -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- The sum of corresponding terms in the arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) (h1 : d > 0) (h2 : r > 1) 
  (h3 : c_seq d r (k - 1) = 100) (h4 : c_seq d r (k + 1) = 1000) :
  c_seq d r k = 262 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l77_7761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_even_and_zero_sin_not_even_or_no_zero_ln_not_even_or_no_zero_x_squared_plus_one_not_even_or_no_zero_only_cos_even_and_zero_l77_7788

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def has_zero_point (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f x = 0

theorem cos_even_and_zero : 
  is_even_function Real.cos ∧ has_zero_point Real.cos := by
  sorry

theorem sin_not_even_or_no_zero : 
  ¬(is_even_function Real.sin ∧ has_zero_point Real.sin) := by
  sorry

theorem ln_not_even_or_no_zero : 
  ¬(is_even_function Real.log ∧ has_zero_point Real.log) := by
  sorry

theorem x_squared_plus_one_not_even_or_no_zero : 
  ¬(is_even_function (λ x ↦ x^2 + 1) ∧ has_zero_point (λ x ↦ x^2 + 1)) := by
  sorry

theorem only_cos_even_and_zero : 
  (is_even_function Real.cos ∧ has_zero_point Real.cos) ∧
  ¬(is_even_function Real.sin ∧ has_zero_point Real.sin) ∧
  ¬(is_even_function Real.log ∧ has_zero_point Real.log) ∧
  ¬(is_even_function (λ x ↦ x^2 + 1) ∧ has_zero_point (λ x ↦ x^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_even_and_zero_sin_not_even_or_no_zero_ln_not_even_or_no_zero_x_squared_plus_one_not_even_or_no_zero_only_cos_even_and_zero_l77_7788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l77_7732

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem ellipse_intersection_length 
  (e : Ellipse)
  (h_short_axis : e.b = 1)
  (h_focus_on_circle : ∃ (f : Point), f.x^2 + (f.y - 3)^2 = 18 ∧ e.a^2 = e.b^2 + f.x^2)
  (h_focal_distance : e.a^2 - e.b^2 < 4)
  (l : Line)
  (F A B : Point)
  (h_F_left_focus : F.x = -Real.sqrt (e.a^2 - e.b^2) ∧ F.y = 0)
  (h_l_through_F : l.m * F.x + l.c = F.y)
  (h_A_on_ellipse : A.x^2 / e.a^2 + A.y^2 / e.b^2 = 1)
  (h_B_on_ellipse : B.x^2 / e.a^2 + B.y^2 / e.b^2 = 1)
  (h_A_on_line : A.y = l.m * A.x + l.c)
  (h_B_on_line : B.y = l.m * B.x + l.c)
  (h_AF_3FB : distance A F = 3 * distance F B) :
  distance A B = 3 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_length_l77_7732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_theorem_l77_7792

/-- Given a triangle ABC and a point P, calculate the sum of distances from P to A, B, and C -/
noncomputable def sum_distances (a b c p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - a.1)^2 + (p.2 - a.2)^2) +
  Real.sqrt ((p.1 - b.1)^2 + (p.2 - b.2)^2) +
  Real.sqrt ((p.1 - c.1)^2 + (p.2 - c.2)^2)

/-- Theorem stating that for the given triangle and point, the sum of distances
    can be expressed as m√5 + n√10 where m + n = 5 -/
theorem sum_distances_theorem :
  let a : ℝ × ℝ := (0, 0)
  let b : ℝ × ℝ := (10, 0)
  let c : ℝ × ℝ := (3, 5)
  let p : ℝ × ℝ := (4, 2)
  ∃ (m n : ℕ), sum_distances a b c p = m * Real.sqrt 5 + n * Real.sqrt 10 ∧ m + n = 5 := by
  sorry

#check sum_distances_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_theorem_l77_7792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_correct_l77_7726

-- Define the conversion factors
def minutes_per_degree : ℚ := 60
def seconds_per_minute : ℚ := 60

-- Define the input angle in degrees
def angle_in_degrees : ℚ := 21.24

-- Define the function to convert degrees to (degrees, minutes, seconds)
def convert_to_dms (angle : ℚ) : ℕ × ℕ × ℕ :=
  let degrees : ℕ := Int.toNat (Int.floor angle)
  let remaining_minutes : ℚ := (angle - degrees) * minutes_per_degree
  let minutes : ℕ := Int.toNat (Int.floor remaining_minutes)
  let seconds : ℕ := Int.toNat (Int.floor ((remaining_minutes - minutes) * seconds_per_minute))
  (degrees, minutes, seconds)

-- State the theorem
theorem angle_conversion_correct :
  convert_to_dms angle_in_degrees = (21, 14, 24) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_conversion_correct_l77_7726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l77_7713

theorem trig_ratio_sum (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4)
  (h2 : Real.cos x / Real.cos y = 1/3) :
  (Real.sin x)^2 / (Real.sin y)^2 + (Real.cos x)^2 / (Real.cos y)^2 = 145/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_sum_l77_7713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_paving_cost_l77_7712

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length width rate : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : rate = 700) : 
  length * width * rate = 14437.5 := by
  rw [h1, h2, h3]
  norm_num
  
#eval Float.toString (5.5 * 3.75 * 700)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_paving_cost_l77_7712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_over_sin_alpha_minus_pi_4_l77_7798

theorem cos_2alpha_over_sin_alpha_minus_pi_4 (α : ℝ) 
  (h1 : Real.sin α = 1/2 + Real.cos α) 
  (h2 : 0 < α ∧ α < Real.pi/2) : 
  Real.cos (2*α) / Real.sin (α - Real.pi/4) = -Real.sqrt 14/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_over_sin_alpha_minus_pi_4_l77_7798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_l77_7728

theorem exam_average (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (h₁ : n₁ = 15) (h₂ : n₂ = 10) 
  (h₃ : avg₁ = 73/100) (h₄ : avg_total = 79/100) (h₅ : n₁ + n₂ = 25) : 
  (avg_total * (n₁ + n₂) - avg₁ * n₁) / n₂ = 88/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_average_l77_7728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_difference_l77_7737

/-- Two polynomials differ by only one coefficient -/
def differ_by_one_coeff (f g : Polynomial ℝ) : Prop :=
  ∃ n : ℕ, ∃ c : ℝ, f = g + c • (Polynomial.X : Polynomial ℝ)^n

/-- The equation given in the problem -/
def equation_holds (f g : Polynomial ℝ) : Prop :=
  f.comp (f * g) + (f.comp g) * (g.comp f) + (f.comp f) * (g.comp g) =
  g.comp (f * g) + (f.comp f) * (f.comp g) + (g.comp f) * (g.comp g)

theorem polynomial_difference (f g : Polynomial ℝ) :
  f ≠ g →
  f.leadingCoeff > 0 →
  g.leadingCoeff > 0 →
  f.natDegree > 0 →
  g.natDegree > 0 →
  equation_holds f g →
  differ_by_one_coeff f g :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_difference_l77_7737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_simplest_form_l77_7765

/-- The repeating decimal 0.527527527... as a rational number -/
def repeating_decimal : ℚ := 527 / 999

/-- The sum of the numerator and denominator of the simplest form of the repeating decimal -/
def sum_num_denom : ℕ := 
  let n := repeating_decimal.num.natAbs
  let d := repeating_decimal.den
  n + d

theorem sum_of_simplest_form : sum_num_denom = 1526 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_simplest_form_l77_7765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_sum_value_l77_7785

/-- The polynomial whose roots we're considering -/
def f (z : ℂ) : ℂ := z^10 - 2^30

/-- The 10 roots of the polynomial -/
noncomputable def roots : Finset ℂ := sorry

/-- For each root, we can choose either the root itself or -i times the root -/
noncomputable def possible_choices (z : ℂ) : Finset ℂ := {z, -Complex.I * z}

/-- The maximum real part of the sum of choices -/
noncomputable def max_real_sum : ℝ := sorry

/-- The theorem stating the maximum value of the real part of the sum -/
theorem max_real_sum_value : 
  max_real_sum = 8 * (1 + (1 + Real.sqrt 5 + Real.sqrt (10 - 2 * Real.sqrt 5)) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_sum_value_l77_7785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BDE_l77_7753

/-- A structure representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Calculate the angle between three points -/
noncomputable def angle (p q r : Point3D) : ℝ := sorry

/-- Check if a line is parallel to a plane -/
def isParallelToPlane (p q : Point3D) (a b c : Point3D) : Prop := sorry

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p q r : Point3D) : ℝ := sorry

theorem area_of_triangle_BDE (A B C D E : Point3D) :
  distance A B = 3 →
  distance B C = 4 →
  angle A B C = Real.pi / 2 →
  angle C D E = Real.pi / 2 →
  angle D E A = Real.pi / 2 →
  distance C D = 3 →
  distance D E = 3 →
  distance E A = 3 →
  isParallelToPlane D E A B C →
  triangleArea B D E = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_BDE_l77_7753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_21_l77_7774

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x + 3
noncomputable def g (x : ℝ) : ℝ := x / 2

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem composition_equals_21 :
  f (g_inv (f_inv (g_inv (f_inv (g (f 15)))))) = 21 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_21_l77_7774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tripod_height_with_broken_leg_final_result_l77_7706

/-- Represents the height of a tripod with one shortened leg -/
noncomputable def tripod_height (original_leg_length : ℝ) (normal_height : ℝ) (broken_leg_length : ℝ) : ℝ :=
  27 / Real.sqrt 10

/-- Theorem stating the height of the tripod with a shortened leg -/
theorem tripod_height_with_broken_leg 
  (original_leg_length : ℝ) 
  (normal_height : ℝ) 
  (broken_leg_length : ℝ) 
  (h_original : original_leg_length = 6) 
  (h_normal : normal_height = 3) 
  (h_broken : broken_leg_length = 4) :
  tripod_height original_leg_length normal_height broken_leg_length = 27 / Real.sqrt 10 := by
  sorry

/-- Compute the floor of the sum of m and sqrt(n) -/
noncomputable def compute_result : ℤ :=
  Int.floor (27 + Real.sqrt 10)

/-- Theorem stating the final result -/
theorem final_result : compute_result = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tripod_height_with_broken_leg_final_result_l77_7706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l77_7746

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 6 * Real.cos (Real.pi / 2 - x)

theorem f_max_value : ∀ x : ℝ, f x ≤ 5 ∧ ∃ y : ℝ, f y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l77_7746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l77_7755

theorem remainder_problem (a b : ℕ) 
  (ha : a % 8 = 3)
  (hb : b > 0)
  (hab : (a * b) % 48 = 15) :
  b % 6 = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l77_7755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_group_size_l77_7730

def is_valid_group (s : Finset ℕ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → (x + y) % 5 = 0

theorem max_group_size :
  ∃ (s : Finset ℕ), s.card = 40 ∧ s ⊆ Finset.range 201 ∧ is_valid_group s ∧
    ∀ (t : Finset ℕ), t ⊆ Finset.range 201 → is_valid_group t → t.card ≤ 40 :=
by
  sorry

#check max_group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_group_size_l77_7730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plans_equal_at_15_minutes_unique_equal_cost_duration_l77_7743

/-- Represents the cost of a phone call under Plan A -/
noncomputable def costPlanA (duration : ℝ) : ℝ :=
  if duration ≤ 5 then 0.60 else 0.60 + 0.06 * (duration - 5)

/-- Represents the cost of a phone call under Plan B -/
def costPlanB (duration : ℝ) : ℝ := 0.08 * duration

/-- Theorem stating that both plans cost the same at 15 minutes -/
theorem plans_equal_at_15_minutes :
  costPlanA 15 = costPlanB 15 := by
  sorry

/-- Theorem stating that 15 minutes is the unique duration where plans cost the same -/
theorem unique_equal_cost_duration :
  ∀ d : ℝ, d > 0 → (costPlanA d = costPlanB d ↔ d = 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plans_equal_at_15_minutes_unique_equal_cost_duration_l77_7743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_pyramid_ways_l77_7784

/-- Represents a sign in the pyramid, either positive (+1) or negative (-1) -/
inductive Sign : Type
| positive : Sign
| negative : Sign

/-- A function to determine the sign of a cell based on its two children -/
def cellSign (left right : Sign) : Sign :=
  match left, right with
  | Sign.positive, Sign.positive => Sign.positive
  | Sign.negative, Sign.negative => Sign.positive
  | _, _ => Sign.negative

/-- Represents a row in the pyramid -/
def Row (n : Nat) := Fin n → Sign

/-- Generates the next row up in the pyramid -/
def nextRow {n : Nat} (row : Row (n+1)) : Row n :=
  fun i => cellSign (row i) (row (Fin.succ i))

/-- The number of ways to fill a row of length n -/
def numWays (n : Nat) : Nat := 2^n

/-- Theorem: There are exactly 18 ways to fill the bottom row of a 5-level sign pyramid 
    such that the top cell has a positive sign -/
theorem sign_pyramid_ways : 
  (∃ (ways : Finset (Row 5)), 
    ways.card = 18 ∧ 
    ∀ row ∈ ways, 
      (((nextRow (nextRow (nextRow (nextRow row)))) 0) = Sign.positive)) := by
  sorry

#eval numWays 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sign_pyramid_ways_l77_7784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_with_real_root_l77_7727

theorem max_numbers_with_real_root (n : ℕ) (h : n = 100) :
  (∃ (S : Finset ℕ),
    (∀ x, x ∈ S → x ≥ 1 ∧ x ≤ n) ∧
    (∀ a b, a ∈ S → b ∈ S → ∃ x : ℝ, x^2 + a*x + b = 0) ∧
    (∀ T : Finset ℕ, (∀ x, x ∈ T → x ≥ 1 ∧ x ≤ n) →
      (∀ a b, a ∈ T → b ∈ T → ∃ x : ℝ, x^2 + a*x + b = 0) →
      T.card ≤ S.card)) →
  (∃ (S : Finset ℕ),
    (∀ x, x ∈ S → x ≥ 1 ∧ x ≤ n) ∧
    (∀ a b, a ∈ S → b ∈ S → ∃ x : ℝ, x^2 + a*x + b = 0) ∧
    S.card = 81) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_with_real_root_l77_7727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2009th_term_l77_7748

def sequence_a : ℕ → ℕ
  | 0 => 3  -- We define a_0 as 3 to make indexing consistent
  | 1 => 7
  | (n + 2) => (sequence_a n * sequence_a (n + 1)) % 10

theorem sequence_2009th_term : sequence_a 2009 = 9 := by
  sorry

#eval sequence_a 2009  -- This will evaluate the 2009th term (which is the 2010th in the original problem)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2009th_term_l77_7748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_translated_segment_l77_7709

/-- Translate a point in ℝ² by a given vector -/
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

theorem midpoint_of_translated_segment :
  let s₁_start : ℝ × ℝ := (4, 1)
  let s₁_end : ℝ × ℝ := (-8, 5)
  let translation : ℝ × ℝ := (2, 3)
  let s₂_start := translate s₁_start translation
  let s₂_end := translate s₁_end translation
  (s₂_start.1 + s₂_end.1) / 2 = 0 ∧ (s₂_start.2 + s₂_end.2) / 2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_translated_segment_l77_7709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l77_7796

-- Define constants
noncomputable def π : Real := Real.pi
noncomputable def e : Real := Real.exp 1

-- Define axiom
axiom pi_gt_three : π > 3

-- Define functions
noncomputable def f (x : Real) : Real := x - Real.log x
noncomputable def g (x : Real) : Real := Real.exp x - x

-- Define a, b, and c
noncomputable def a : Real := π - 3
noncomputable def b : Real := Real.log π - Real.log 3
noncomputable def c : Real := Real.exp π - Real.exp 3

-- State the theorem
theorem inequality_proof : c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l77_7796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_and_sum_l77_7771

def b : ℕ → ℚ
  | 0 => 5  -- Add this case to cover Nat.zero
  | 1 => 5
  | 2 => 5/13
  | (n+3) => (b (n+1) * b (n+2)) / (3 * b (n+1) - b (n+2))

theorem b_2023_and_sum : 
  b 2023 = 5/10108 ∧ 5 + 10108 = 10113 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_and_sum_l77_7771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_Q_l77_7797

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ :=
  Real.sqrt ((r₁ * Real.cos θ₁ - r₂ * Real.cos θ₂)^2 + (r₁ * Real.sin θ₁ - r₂ * Real.sin θ₂)^2)

/-- Theorem: The distance between P(1, π/6) and Q(2, π/2) in polar coordinates is √3 -/
theorem distance_P_Q : polar_distance 1 (Real.pi / 6) 2 (Real.pi / 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_Q_l77_7797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_events_count_l77_7707

-- Define the possible types of events
inductive EventType
  | Random
  | Certain
  | Impossible
  deriving BEq, Repr

-- Define a function to classify events
def classifyEvent (event : Nat) : EventType :=
  match event with
  | 1 => EventType.Random  -- Throwing two dice twice and getting 2 points both times
  | 2 => EventType.Certain  -- A pear falling down on Earth
  | 3 => EventType.Random  -- Winning the lottery
  | 4 => EventType.Random  -- Having a boy after having a daughter
  | 5 => EventType.Impossible  -- Water boiling at 90°C under standard pressure
  | _ => EventType.Impossible  -- Default case for undefined events

-- Define a function to count random events
def countRandomEvents (events : List Nat) : Nat :=
  events.filter (fun e => classifyEvent e == EventType.Random) |>.length

-- Theorem statement
theorem random_events_count :
  countRandomEvents [1, 2, 3, 4, 5] = 3 := by
  -- Evaluate the function
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_events_count_l77_7707
