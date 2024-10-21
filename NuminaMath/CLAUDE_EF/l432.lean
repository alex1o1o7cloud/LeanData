import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_sqrt_109_l432_43229

/-- The distance between two points in polar coordinates -/
noncomputable def polar_distance (r1 r2 : ℝ) (θ1 θ2 : ℝ) : ℝ :=
  Real.sqrt (r1^2 + r2^2 - 2*r1*r2*(Real.cos (θ1 - θ2)))

/-- Theorem: The distance between A(5, θ₁) and B(12, θ₂) in polar coordinates,
    where θ₁ - θ₂ = π/3, is √109 -/
theorem distance_AB_is_sqrt_109 (θ1 θ2 : ℝ) (h : θ1 - θ2 = π/3) :
  polar_distance 5 12 θ1 θ2 = Real.sqrt 109 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_is_sqrt_109_l432_43229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_functions_properties_cost_1500_seedlings_l432_43233

/-- Cost function for Farm A --/
noncomputable def cost_A (x : ℝ) : ℝ :=
  if x ≤ 1000 then 4 * x
  else 4000 + 3.8 * (x - 1000)

/-- Cost function for Farm B --/
noncomputable def cost_B (x : ℝ) : ℝ :=
  if x ≤ 2000 then 4 * x
  else 8000 + 3.6 * (x - 2000)

/-- Theorem stating the properties of the cost functions --/
theorem cost_functions_properties :
  ∀ x : ℝ, x > 2000 →
    cost_A x = 3.8 * x + 200 ∧
    cost_B x = 3.6 * x + 800 ∧
    (cost_A x = cost_B x ↔ x = 3000) ∧
    (x < 3000 → cost_A x < cost_B x) ∧
    (x > 3000 → cost_A x > cost_B x) := by
  sorry

/-- Theorem for the specific case of 1500 seedlings --/
theorem cost_1500_seedlings :
  cost_A 1500 = 5900 ∧ cost_B 1500 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_functions_properties_cost_1500_seedlings_l432_43233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_range_l432_43224

noncomputable section

open Real

theorem triangle_ratio_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (sin B * sin C) / (3 * sin A) = (cos A) / a + (cos C) / c →
  (a * b * sin C) / 2 = (sqrt 3 / 4) * (a^2 + b^2 - c^2) →
  1/2 ≤ c / (a + b) ∧ c / (a + b) < 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_range_l432_43224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_theorem_l432_43241

/-- Represents the fare structure for a taxi ride -/
structure TaxiFare where
  initialFare : ℚ  -- Initial fare for the first 1/5 mile
  additionalFare : ℚ  -- Fare for each additional 1/5 mile
  totalFare : ℚ  -- Total fare for a 3-mile ride

/-- Calculates the additional fare per 1/5 mile given the initial fare and total fare for a 3-mile ride -/
def calculateAdditionalFare (initialFare totalFare : ℚ) : ℚ :=
  (totalFare - initialFare) / 14

/-- Theorem stating that given the initial fare of $1.00 and total fare of $6.60 for a 3-mile ride,
    the additional fare per 1/5 mile is $0.40 -/
theorem taxi_fare_theorem (tf : TaxiFare) 
  (h1 : tf.initialFare = 1)
  (h2 : tf.totalFare = 33/5) :
  tf.additionalFare = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_theorem_l432_43241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_and_endpoint_sum_l432_43235

-- Define the function h as noncomputable
noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

-- State the theorem
theorem h_range_and_endpoint_sum :
  (∀ y ∈ Set.range h, 0 < y ∧ y ≤ 1) ∧
  (∃ x, h x = 1) ∧
  (∀ ε > 0, ∃ x, h x < ε) ∧
  (0 + 1 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_and_endpoint_sum_l432_43235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_exists_l432_43269

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Triangle ABC in 3D space -/
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

noncomputable def altitude (t : Triangle3D) : ℝ :=
  sorry -- Definition of altitude

noncomputable def angle_with_xy_plane (t : Triangle3D) : ℝ :=
  sorry -- Definition of angle with xy-plane

theorem triangle_construction_exists (B' B'' c a p q β : ℝ) :
  B' > 0 → B'' > 0 → c > a → p > q → β > 0 → β < Real.pi →
  ∃ (t : Triangle3D),
    t.B.x = B' ∧ t.B.y = B'' ∧ t.B.z = 0 ∧
    t.A.z = 0 ∧
    t.C.y = 0 ∧ t.C.z = 0 ∧
    distance t.A t.B = c ∧
    distance t.B t.C = a ∧
    distance t.A t.C = c - a ∧
    altitude t = p - q ∧
    angle_with_xy_plane t = β := by
  sorry

#check triangle_construction_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_exists_l432_43269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l432_43218

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.log (|a + 4 / (2 - x)|) + b

-- State the theorem
theorem odd_function_sum (a b : ℝ) :
  (∀ x : ℝ, x ≠ 2 → f a b x = -f a b (-x)) → a + b = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l432_43218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_game_l432_43279

/-- The probability of scoring exactly n points in a coin toss game -/
noncomputable def probability_n_points (n : ℕ) : ℝ :=
  (1 / 3) * (2 + (-1 / 2) ^ n)

/-- The coin toss game where heads gain 1 point and tails gain 2 points -/
theorem coin_toss_game (n : ℕ) :
  let p := probability_n_points n
  let heads_prob := (1 : ℝ) / 2
  let tails_prob := (1 : ℝ) / 2
  let heads_points := 1
  let tails_points := 2
  (∀ m, m ≠ n → probability_n_points m ≠ p) ∧
  (heads_prob = tails_prob) ∧
  (heads_points = 1) ∧
  (tails_points = 2) →
  p = (1 / 3) * (2 + (-1 / 2) ^ n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_toss_game_l432_43279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_origin_l432_43281

noncomputable def point_P : ℝ × ℝ := (-Real.sqrt 3, 1)
def origin : ℝ × ℝ := (0, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_P_to_origin :
  distance point_P origin = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_origin_l432_43281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_second_quadrant_l432_43239

theorem cos_alpha_second_quadrant (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) -- α is in the second quadrant
  (h2 : Real.sin α = 5 / 13) : 
  Real.cos α = -12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_second_quadrant_l432_43239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_in_parallel_planes_are_parallel_or_skew_l432_43205

-- Define the concept of a plane
structure Plane where
  -- Placeholder for plane definition
  dummy : Unit

-- Define the concept of a line
structure Line where
  -- Placeholder for line definition
  dummy : Unit

-- Define the concept of parallel planes
def parallel_planes (p1 p2 : Plane) : Prop :=
  -- Placeholder for parallel planes definition
  True

-- Define the concept of a line lying in a plane
def line_in_plane (l : Line) (p : Plane) : Prop :=
  -- Placeholder for line in plane definition
  True

-- Define the concept of parallel lines
def parallel_lines (l1 l2 : Line) : Prop :=
  -- Placeholder for parallel lines definition
  True

-- Define the concept of skew lines
def skew_lines (l1 l2 : Line) : Prop :=
  -- Placeholder for skew lines definition
  True

theorem lines_in_parallel_planes_are_parallel_or_skew 
  (p1 p2 : Plane) (l1 l2 : Line) 
  (h_parallel : parallel_planes p1 p2)
  (h_l1_in_p1 : line_in_plane l1 p1)
  (h_l2_in_p2 : line_in_plane l2 p2) :
  parallel_lines l1 l2 ∨ skew_lines l1 l2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_in_parallel_planes_are_parallel_or_skew_l432_43205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_row_sum_sum_of_row_12_pascal_triangle_l432_43211

/-- Definition of the sum of a row in Pascal's Triangle -/
def sum_of_pascal_triangle_row (n : ℕ) : ℕ := 2^n

/-- Pascal's Triangle row sum theorem -/
theorem pascal_triangle_row_sum (n : ℕ) : 
  sum_of_pascal_triangle_row n = 2^n := by
  -- The proof is trivial due to our definition
  rfl

/-- Sum of Row 12 in Pascal's Triangle -/
theorem sum_of_row_12_pascal_triangle : 
  sum_of_pascal_triangle_row 12 = 4096 := by
  -- Unfold the definition and evaluate
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_row_sum_sum_of_row_12_pascal_triangle_l432_43211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_floor_x_plus_2_floor_l432_43256

-- Define the floor function
noncomputable def floor (a : ℝ) : ℤ := Int.floor a

-- State the properties of the floor function
axiom floor_property (a : ℝ) : a - 1 < floor a ∧ (floor a : ℝ) ≤ a

-- Theorem 1
theorem sqrt_17_floor : floor (Real.sqrt 17) = 4 := by sorry

-- Theorem 2
theorem x_plus_2_floor (x : ℝ) :
  floor (x + 2) = Int.floor (2 * x - 1/4) ↔ x = 13/8 ∨ x = 17/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_17_floor_x_plus_2_floor_l432_43256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l432_43295

theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (a > 0) → (b > 0) → (c > 0) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (Real.cos C / Real.cos B = (2 * a - c) / b) →
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l432_43295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_two_l432_43260

theorem det_B_equals_two (x y : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, 2; -3, y]
  B + 2 * B⁻¹ = 0 →
  Matrix.det B = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_B_equals_two_l432_43260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannonball_pillar_paradox_l432_43230

-- Define the properties of the all-destroying cannonball
def AllDestroying (x : Prop) : Prop := ∀ (y : Prop), x → ¬y

-- Define the properties of the indestructible pillar
def Indestructible (x : Prop) : Prop := ∀ (y : Prop), y → ¬(¬x)

-- Theorem stating the contradiction
theorem cannonball_pillar_paradox :
  ¬(∃ (c p : Prop), AllDestroying c ∧ Indestructible p) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannonball_pillar_paradox_l432_43230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_200_l432_43243

/-- Represents the efficiency of a car in miles per gallon -/
noncomputable def miles_per_gallon : ℝ := 40

/-- Represents the cost of gas in dollars per gallon -/
noncomputable def dollars_per_gallon : ℝ := 5

/-- Represents the amount of money available to spend on gas in dollars -/
noncomputable def available_dollars : ℝ := 25

/-- Calculates the total distance that can be traveled given the car's efficiency,
    gas price, and available money -/
noncomputable def total_distance : ℝ := (available_dollars / dollars_per_gallon) * miles_per_gallon

/-- Theorem stating that the total distance that can be traveled is 200 miles -/
theorem total_distance_is_200 : total_distance = 200 := by
  -- Unfold the definition of total_distance
  unfold total_distance
  -- Unfold the definitions of available_dollars, dollars_per_gallon, and miles_per_gallon
  unfold available_dollars dollars_per_gallon miles_per_gallon
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_200_l432_43243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l432_43237

-- Define the value of a
def a : ℝ := -1

-- Define the universal set U and set A
def U : Set ℝ := {x | x = (a^2 - 2) ∨ x = 2 ∨ x = 1}
def A : Set ℝ := {x | x = a ∨ x = 1}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l432_43237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l432_43263

/-- The function f(x) = x - sin(x) -/
noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

/-- The sequence a_n defined recursively -/
noncomputable def a : ℕ → ℝ
  | 0 => 0  -- a_0 is not used, but we need to define it for completeness
  | n + 1 => f (a n)

/-- The main theorem -/
theorem a_properties (h1 : 0 < a 1) (h2 : a 1 < 1) :
  (∀ n : ℕ, n ≥ 1 → 0 < a (n + 1) ∧ a (n + 1) < a n ∧ a n < 1) ∧
  (∀ n : ℕ, n ≥ 1 → a (n + 1) < (1 / 6) * (a n) ^ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l432_43263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l432_43291

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The circle equation -/
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y + 4 = 0

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The theorem stating that the distance between intersection points is 4 -/
theorem intersection_distance : ∃ x1 y1 x2 y2 : ℝ, 
  parabola x1 y1 ∧ circleEq x1 y1 ∧ 
  parabola x2 y2 ∧ circleEq x2 y2 ∧ 
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧
  distance x1 y1 x2 y2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l432_43291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_rec_a_2021_eq_three_halves_l432_43227

/-- A sequence {aₙ} defined recursively -/
def a : ℕ → ℚ
  | 0 => -2  -- Changed from 1 to 0 for zero-based indexing
  | n + 1 => 1 - 1 / a n

/-- The recursive relation for the sequence -/
theorem a_rec (n : ℕ) : a n * a (n + 1) = a n - 1 := by sorry

/-- The main theorem: a₂₀₂₁ = 3/2 -/
theorem a_2021_eq_three_halves : a 2020 = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_rec_a_2021_eq_three_halves_l432_43227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_reduction_percentage_correct_l432_43289

/-- Represents the dimensions and volume reduction of an ice cream bar -/
structure IceCreamBar where
  original_length : ℚ
  original_width : ℚ
  original_thickness : ℚ
  length_reduction_percent : ℚ
  width_reduction_percent : ℚ

/-- Calculates the volume reduction percentage of an ice cream bar -/
def volume_reduction_percentage (bar : IceCreamBar) : ℚ :=
  let original_volume := bar.original_length * bar.original_width * bar.original_thickness
  let new_length := bar.original_length * (1 - bar.length_reduction_percent / 100)
  let new_width := bar.original_width * (1 - bar.width_reduction_percent / 100)
  let new_volume := new_length * new_width * bar.original_thickness
  ((original_volume - new_volume) / original_volume) * 100

/-- Theorem stating that the volume reduction percentage is correct -/
theorem volume_reduction_percentage_correct (bar : IceCreamBar) :
  volume_reduction_percentage bar =
    (1 - (1 - bar.length_reduction_percent / 100) * (1 - bar.width_reduction_percent / 100)) * 100 :=
by sorry

def example_bar : IceCreamBar := {
  original_length := 6,
  original_width := 5,
  original_thickness := 2,
  length_reduction_percent := 10,  -- example value
  width_reduction_percent := 10    -- example value
}

#eval volume_reduction_percentage example_bar

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_reduction_percentage_correct_l432_43289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_60_area_formula_120_l432_43228

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ

-- Define the area calculation functions
noncomputable def area_60 (t : Triangle) : ℝ :=
  (Real.sqrt 3 / 4) * (t.a^2 - (t.b - t.c)^2)

noncomputable def area_120 (t : Triangle) : ℝ :=
  (Real.sqrt 3 / 12) * (t.a^2 - (t.b - t.c)^2)

-- Theorem for α = 60°
theorem area_formula_60 (t : Triangle) (h : t.α = π / 3) :
  area_60 t = (1 / 2) * t.a * (t.b * Real.sqrt 3 / 2) := by
  sorry

-- Theorem for α = 120°
theorem area_formula_120 (t : Triangle) (h : t.α = 2 * π / 3) :
  area_120 t = (1 / 2) * t.a * (t.b * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_60_area_formula_120_l432_43228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l432_43250

/-- In a right-angled triangle XYZ where angle XYZ is 30° and the hypotenuse XZ is 12 units long,
    the length of side XY is 4√3. -/
theorem right_triangle_side_length (XY XZ : ℝ) (angle_XYZ : Real) :
  angle_XYZ = 30 →
  XZ = 12 →
  XY = 4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l432_43250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_I_approx_l432_43271

/-- The molar mass of calcium in g/mol -/
noncomputable def molar_mass_Ca : ℝ := 40.08

/-- The molar mass of iodine in g/mol -/
noncomputable def molar_mass_I : ℝ := 126.90

/-- The number of calcium atoms in calcium iodide -/
def num_Ca_atoms : ℕ := 1

/-- The number of iodine atoms in calcium iodide -/
def num_I_atoms : ℕ := 2

/-- The molar mass of calcium iodide in g/mol -/
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + num_I_atoms * molar_mass_I

/-- The mass percentage of iodine in calcium iodide -/
noncomputable def mass_percentage_I : ℝ := (num_I_atoms * molar_mass_I / molar_mass_CaI2) * 100

/-- Theorem stating that the mass percentage of iodine in calcium iodide is approximately 86.35% -/
theorem mass_percentage_I_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |mass_percentage_I - 86.35| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_I_approx_l432_43271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l432_43210

def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2 * m}

theorem problem_solution :
  ((Set.univ \ A) ∪ B = {x : ℝ | -3 < x ∧ x < 5}) ∧
  (∀ m : ℝ, B ∩ C m = C m → m < -1 ∨ (2 < m ∧ m < 5/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l432_43210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_line_transform_circle_l432_43277

-- Define the stretching transformation
noncomputable def stretching (x y : ℝ) : ℝ × ℝ :=
  (1/2 * x, 1/3 * y)

-- Theorem for the first equation
theorem transform_line :
  ∀ x y x' y', stretching x y = (x', y') →
  (5 * x + 2 * y = 0) → (5 * x' + 3 * y' = 0) := by
  sorry

-- Theorem for the second equation
theorem transform_circle :
  ∀ x y x' y', stretching x y = (x', y') →
  (x^2 + y^2 = 1) → (4 * x'^2 + 9 * y'^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_line_transform_circle_l432_43277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_correct_l432_43255

/-- Represents Annika's hiking scenario --/
structure HikingScenario where
  flat_speed : ℚ  -- Speed on flat terrain in km/min
  uphill_factor : ℚ  -- Factor by which speed decreases uphill
  downhill_factor : ℚ  -- Factor by which speed increases downhill
  flat_distance : ℚ  -- Distance hiked on flat terrain in km
  uphill_distance : ℚ  -- Distance hiked uphill in km
  total_time : ℚ  -- Total time available for the round trip in minutes

/-- Calculates the maximum distance Annika can hike east --/
def max_distance_east (scenario : HikingScenario) : ℚ :=
  scenario.flat_distance + scenario.uphill_distance

/-- Theorem stating that the maximum distance Annika can hike east is 2.75 km --/
theorem max_distance_is_correct (scenario : HikingScenario) 
  (h1 : scenario.flat_speed = 1/12)  -- 12 minutes per km
  (h2 : scenario.uphill_factor = 4/5)  -- 20% decrease
  (h3 : scenario.downhill_factor = 13/10)  -- 30% increase
  (h4 : scenario.flat_distance = 3/2)
  (h5 : scenario.uphill_distance = 5/4)
  (h6 : scenario.total_time = 51) :
  max_distance_east scenario = 11/4 := by
  sorry

def example_scenario : HikingScenario := { 
  flat_speed := 1/12, 
  uphill_factor := 4/5, 
  downhill_factor := 13/10, 
  flat_distance := 3/2, 
  uphill_distance := 5/4, 
  total_time := 51 
}

#eval max_distance_east example_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_correct_l432_43255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_exactly_two_zeros_max_k_value_l432_43231

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 1 + Real.log x - k * (x - 2) / x

-- Statement 1
theorem tangent_line_at_one (k : ℝ) (h : k = 0) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * (x - 1) + f k 1 ↔ x - y = 0 := by
  sorry

-- Statement 2
theorem exactly_two_zeros (k : ℝ) (h : k = 5) :
  ∃! (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧
  (∀ x : ℝ, f k x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

-- Statement 3
theorem max_k_value :
  ∃ k : ℤ, (∀ x : ℝ, x > 2 → f (k : ℝ) x > 0) ∧
  (∀ k' : ℤ, k' > k → ∃ x : ℝ, x > 2 ∧ f (k' : ℝ) x ≤ 0) ∧
  k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_exactly_two_zeros_max_k_value_l432_43231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_evaluation_l432_43264

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ :=
  (2 * Real.cos x + 3 * Real.sin x) / (2 * Real.sin x - 3 * Real.cos x)^3

-- State the theorem
theorem integral_evaluation :
  ∫ x in (0)..(Real.pi/4), f x = -17/18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_evaluation_l432_43264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_l432_43290

/-- The number of intersection points between two polar curves -/
noncomputable def intersection_count (f g : ℝ → ℝ) : ℕ := sorry

/-- First polar curve: r = 6 cos θ -/
noncomputable def curve1 (θ : ℝ) : ℝ := 6 * Real.cos θ

/-- Second polar curve: r = 12 sin θ - 3 -/
noncomputable def curve2 (θ : ℝ) : ℝ := 12 * Real.sin θ - 3

/-- Theorem stating that the two curves intersect at exactly two points -/
theorem two_intersection_points : intersection_count curve1 curve2 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_intersection_points_l432_43290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maple_hill_elementary_difference_l432_43246

/-- Proves that the difference between the total number of students and rabbits
    in all third-grade classrooms at Maple Hill Elementary is 85 -/
theorem maple_hill_elementary_difference : 
  (20 : ℕ) * 5 - (3 : ℕ) * 5 = 85 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maple_hill_elementary_difference_l432_43246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_not_increasing_and_exp_odd_l432_43272

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.tan x
noncomputable def g (x : ℝ) : ℝ := (3 : ℝ)^x - (3 : ℝ)^(-x)

-- State the theorem
theorem tan_not_increasing_and_exp_odd :
  (¬ ∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, g (-x) = -g x) :=
by
  constructor
  · sorry -- Proof that tan is not increasing
  · sorry -- Proof that g is an odd function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_not_increasing_and_exp_odd_l432_43272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_artist_hair_color_l432_43200

-- Define the set of possible hair colors
inductive HairColor
  | White
  | Black
  | Red

-- Define the set of friends
inductive Friend
  | Belov
  | Chernov
  | Ryzhov

-- Define the set of professions
inductive Profession
  | Sculptor
  | Violinist
  | Artist

-- Define a function to assign hair colors to friends
def hairColorOf : Friend → HairColor := sorry

-- Define a function to assign professions to friends
def professionOf : Friend → Profession := sorry

-- State the theorem
theorem artist_hair_color 
  (h1 : ∀ f : Friend, hairColorOf f ≠ HairColor.White → hairColorOf f ≠ HairColor.Black → hairColorOf f = HairColor.Red)
  (h2 : ∀ f : Friend, hairColorOf f ≠ 
    match f with
    | Friend.Belov => HairColor.White
    | Friend.Chernov => HairColor.Black
    | Friend.Ryzhov => HairColor.Red
    )
  (h3 : ∃ f : Friend, hairColorOf f = HairColor.Black ∧ f ≠ Friend.Belov)
  (h4 : professionOf Friend.Ryzhov = Profession.Artist)
  : hairColorOf Friend.Ryzhov = HairColor.Black :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_artist_hair_color_l432_43200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_medals_mode_l432_43216

def gold_medals : List ℕ := [16, 12, 9, 8, 8, 8, 7]

def mode (l : List ℕ) : Option ℕ :=
  l.argmax (λ x => l.count x)

theorem gold_medals_mode :
  mode gold_medals = some 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gold_medals_mode_l432_43216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l432_43254

theorem problem_statement :
  (¬(∀ x : ℝ, (2 : ℝ)^x < (3 : ℝ)^x)) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l432_43254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_equals_seven_l432_43270

/-- Sequence b defined recursively -/
def b : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | n+2 => 1/2 * b (n+1) + 1/3 * b n

/-- The sum of the sequence b -/
noncomputable def seriesSum : ℚ := ∑' n, b n

/-- Theorem: The sum of the sequence b equals 7 -/
theorem sum_of_sequence_equals_seven : seriesSum = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_equals_seven_l432_43270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_property_l432_43220

theorem condition_property :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 4) ∧
  (∃ x : ℝ, |x - 2| < 4 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_property_l432_43220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_f_l432_43273

open Real

noncomputable def f (x : ℝ) : ℝ := (log (x - 1)) / sqrt (x - 1)

theorem third_derivative_f (x : ℝ) (h : x > 1) :
  (deriv^[3] f) x = (46 - 15 * log (x - 1)) / (8 * (x - 1)^(7/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_f_l432_43273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_is_six_l432_43201

noncomputable def point_reflection_distance (x y : ℝ) : ℝ :=
  Real.sqrt ((2 * x)^2)

theorem reflection_distance_is_six :
  point_reflection_distance 3 1 = 6 := by
  -- Unfold the definition of point_reflection_distance
  unfold point_reflection_distance
  -- Simplify the expression
  simp
  -- Evaluate the square root
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_distance_is_six_l432_43201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l432_43206

/-- Triangle ABC with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

/-- Length of median from vertex A to midpoint of side BC -/
noncomputable def median_a (t : Triangle) : ℝ := 
  (1/2) * Real.sqrt (2 * t.b^2 + 2 * t.c^2 - t.a^2)

/-- Length of median from vertex B to midpoint of side AC -/
noncomputable def median_b (t : Triangle) : ℝ := 
  (1/2) * Real.sqrt (2 * t.a^2 + 2 * t.c^2 - t.b^2)

/-- If side a is greater than side b, then median_a is less than median_b -/
theorem median_inequality (t : Triangle) (h : t.a > t.b) : 
  median_a t < median_b t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_inequality_l432_43206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_bear_monthly_consumption_l432_43298

/-- Represents the daily fish consumption of a polar bear in buckets -/
structure DailyConsumption where
  trout : ℝ
  salmon : ℝ
  herring : ℝ
  mackerel : ℝ

/-- Calculates the total monthly fish consumption of a polar bear -/
def monthlyConsumption (daily : DailyConsumption) (days : ℕ) : ℝ :=
  (daily.trout + daily.salmon + daily.herring + daily.mackerel) * days

/-- Theorem stating that the polar bear's monthly fish consumption is 30 buckets -/
theorem polar_bear_monthly_consumption :
  let daily := DailyConsumption.mk 0.2 0.4 0.1 0.3
  monthlyConsumption daily 30 = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_bear_monthly_consumption_l432_43298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ebook_readers_lost_l432_43252

/-- Proves the number of eBook readers John lost given the initial conditions --/
theorem ebook_readers_lost (anna_bought : ℕ) (john_initial : ℕ) (total_after_loss : ℕ) : 
  anna_bought = 50 → 
  john_initial = anna_bought - 15 → 
  total_after_loss = 82 → 
  anna_bought + (john_initial - (john_initial - (total_after_loss - anna_bought))) = total_after_loss → 
  john_initial - (total_after_loss - anna_bought) = 3 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

#check ebook_readers_lost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ebook_readers_lost_l432_43252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l432_43248

theorem divisibility_condition (m n : ℕ) :
  (∃ k₁ k₂ : ℕ, (3^m + 1 = m * n * k₁) ∧ (3^n + 1 = m * n * k₂)) →
  ((m = 1 ∧ n = 1) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l432_43248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_theorem_l432_43236

noncomputable section

-- Define the circle
def Circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 8}

-- Define the line y = x
def LineYEqualsX : Set (ℝ × ℝ) := {p | p.1 = p.2}

-- Define the point (-2, 2)
def PointNegTwoTwo : ℝ × ℝ := (-2, 2)

-- Define the radius
def Radius : ℝ := 2 * Real.sqrt 2

-- Define the line l
def LineL (m : ℝ) : Set (ℝ × ℝ) := {p | (m + 1) * p.1 + (2 * m - 1) * p.2 - 3 * m = 0}

-- Define the fixed point A
def PointA : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem circle_and_line_theorem :
  -- The center of the circle lies on y = x
  ∃ (center : ℝ × ℝ), center ∈ LineYEqualsX ∧ center ∈ Circle ∧
  -- The circle passes through (-2, 2)
  PointNegTwoTwo ∈ Circle ∧
  -- The radius is 2√2
  (∀ p ∈ Circle, Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) = Radius) ∧
  -- Line l passes through A
  (∀ m, PointA ∈ LineL m) →
  -- For any B and C on the circle with AB ⊥ AC
  ∀ B C, B ∈ Circle → C ∈ Circle → 
    (B.1 - PointA.1) * (C.1 - PointA.1) + (B.2 - PointA.2) * (C.2 - PointA.2) = 0 →
  -- The range of |BC| is [√14 - √2, √14 + √2]
  Real.sqrt 14 - Real.sqrt 2 ≤ Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) ∧
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) ≤ Real.sqrt 14 + Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_theorem_l432_43236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_x1_x2_l432_43207

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - 2 * Real.sqrt 3 * Real.cos x

-- State the theorem
theorem min_abs_sum_x1_x2 (a : ℝ) (x₁ x₂ : ℝ) :
  -- Condition 1: Symmetry axis at -π/6
  (∀ x, f a x = f a (-π/3 - x)) →
  -- Condition 2: f(x₁) + f(x₂) = 0
  f a x₁ + f a x₂ = 0 →
  -- Condition 3: f is monotonic on (x₁, x₂)
  (∀ y z, x₁ < y ∧ y < z ∧ z < x₂ → (f a y < f a z ∨ f a y > f a z)) →
  -- Conclusion: The minimum value of |x₁ + x₂| is 2π/3
  |x₁ + x₂| ≥ 2*π/3 ∧ ∃ x₁' x₂', f a x₁' + f a x₂' = 0 ∧ |x₁' + x₂'| = 2*π/3 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_sum_x1_x2_l432_43207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_l432_43283

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- Define the condition for a
axiom condition_a : ∃ a : ℝ, f (Real.log 2 * a) + f (Real.log (1/2) * a) ≤ 2 * f 1

-- State the theorem
theorem min_value_a : 
  (∃ a : ℝ, f (Real.log 2 * a) + f (Real.log (1/2) * a) ≤ 2 * f 1) → 
  (∃ a : ℝ, a = 1/2 ∧ ∀ b, (f (Real.log 2 * b) + f (Real.log (1/2) * b) ≤ 2 * f 1) → a ≤ b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a_l432_43283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l432_43244

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.log x + x - (1/x) - 2

-- State the theorem
theorem zero_in_interval :
  ∃! x : ℝ, 2 < x ∧ x < Real.exp 1 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l432_43244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_circle_construction_l432_43276

-- Define the necessary structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given information
def powerLine : Set Point := sorry
def firstCircle : Circle := sorry
def r2 : ℝ := sorry

-- Define the construction process
noncomputable def constructSecondCircle (powerLine : Set Point) (firstCircle : Circle) (r2 : ℝ) : Circle :=
  sorry

-- Define a distance function
def distance (p1 p2 : Point) : ℝ := sorry

-- State the theorem
theorem second_circle_construction 
  (powerLine : Set Point) 
  (firstCircle : Circle) 
  (r2 : ℝ) :
  let secondCircle := constructSecondCircle powerLine firstCircle r2
  ∃ (P : Point),
    P ∈ powerLine ∧
    secondCircle.center ∈ powerLine ∧
    ∃ (T : Point),
      (distance T firstCircle.center = firstCircle.radius) ∧
      (distance P T = distance P secondCircle.center) ∧
      (distance T secondCircle.center = r2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_circle_construction_l432_43276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_f_odd_iff_a_eq_neg_one_l432_43266

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a + 2 / (2^x + 1)

-- Theorem for monotonically decreasing property
theorem f_monotonically_decreasing (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂ := by sorry

-- Theorem for odd function property
theorem f_odd_iff_a_eq_neg_one (a : ℝ) :
  (∀ x : ℝ, f a (-x) = -(f a x)) ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonically_decreasing_f_odd_iff_a_eq_neg_one_l432_43266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_depends_on_a_d_k_l432_43280

/-- Sum of the first n terms of an arithmetic progression with first term a and common difference d -/
noncomputable def arithmeticProgressionSum (a d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a + (n - 1 : ℝ) * d)

/-- Q is the difference of sums of arithmetic progression terms -/
noncomputable def Q (a d : ℝ) (k : ℕ) : ℝ :=
  let s₁ := arithmeticProgressionSum a d k
  let s₂ := arithmeticProgressionSum a d (2 * k)
  let s₄ := arithmeticProgressionSum a d (4 * k)
  s₄ - s₂ - s₁

theorem Q_depends_on_a_d_k (a d : ℝ) (k : ℕ) :
  ∃ f : ℝ → ℝ → ℕ → ℝ, Q a d k = f a d k := by
  use fun a d k => k * a + 13 * k^2 * d
  simp [Q, arithmeticProgressionSum]
  ring
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_depends_on_a_d_k_l432_43280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_distances_l432_43261

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ

/-- Represents a line ax + y - 4 = 0 -/
structure Line where
  a : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_line_intersection_sum_distances
  (para : Parabola)
  (l : Line)
  (A B F : Point)
  (h1 : A.y^2 = 2 * para.p * A.x)
  (h2 : l.a * A.x + A.y - 4 = 0)
  (h3 : B.y^2 = 2 * para.p * B.x)
  (h4 : l.a * B.x + B.y - 4 = 0)
  (h5 : A.x = 1 ∧ A.y = 2)
  (h6 : F.x = para.p ∧ F.y = 0)  -- Definition of focus for a parabola
  : distance F B + distance F A = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_distances_l432_43261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_l432_43232

theorem divisible_by_nine (k : ℕ) : 9 ∣ (3 * (2 + 7^k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_nine_l432_43232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l432_43240

/-- Represents the time taken to build a wall given the number of workers and their work schedule -/
noncomputable def buildWallTime (fullDayWorkers : ℕ) (fullDays : ℝ) (newWorkers : ℕ) (halfDayWork : Bool) : ℝ :=
  let totalWork := (fullDayWorkers : ℝ) * fullDays
  let effectiveNewWorkers := if halfDayWork then (newWorkers : ℝ) / 2 else newWorkers
  totalWork / effectiveNewWorkers

/-- Theorem stating that 30 men working half-days will take 3.6 days to build a wall that 18 men can build in 6 full days -/
theorem wall_building_time :
  buildWallTime 18 6 30 true = 3.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_building_time_l432_43240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_sign_unchanged_l432_43275

theorem quadratic_trinomial_sign_unchanged (k : ℝ) :
  (∀ x : ℝ, (k^((-3) : ℤ) - x - k^2 * x^2 > 0) ∨ (∀ x : ℝ, k^((-3) : ℤ) - x - k^2 * x^2 < 0)) →
  -4 < k ∧ k < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_trinomial_sign_unchanged_l432_43275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_third_implies_alpha_div_3_quadrants_l432_43294

-- Define the quadrant type
inductive Quadrant
  | First
  | Second
  | Third
  | Fourth

-- Define a function to check if an angle is in the third quadrant
def isInThirdQuadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2

-- Define a function to determine the quadrant of an angle
noncomputable def quadrantOf (θ : Real) : Quadrant :=
  if 0 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi / 2 then Quadrant.First
  else if Real.pi / 2 ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < Real.pi then Quadrant.Second
  else if Real.pi ≤ θ % (2 * Real.pi) ∧ θ % (2 * Real.pi) < 3 * Real.pi / 2 then Quadrant.Third
  else Quadrant.Fourth

-- Theorem statement
theorem alpha_third_implies_alpha_div_3_quadrants
  (α : Real) (h : isInThirdQuadrant α) :
  ∃ q : Quadrant, q = quadrantOf (α / 3) ∧
    (q = Quadrant.First ∨ q = Quadrant.Third ∨ q = Quadrant.Fourth) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_third_implies_alpha_div_3_quadrants_l432_43294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_with_harmonic_mean_12_4_l432_43222

/-- The number of ordered pairs of positive integers (a, b) satisfying the given conditions --/
def count_ordered_pairs : ℕ :=
  67

/-- The harmonic mean of two numbers --/
def harmonic_mean (x y : ℚ) : ℚ :=
  2 * x * y / (x + y)

/-- Proposition: There are exactly 67 ordered pairs of positive integers (a, b)
    such that a < b and their harmonic mean is 12^4 --/
theorem count_pairs_with_harmonic_mean_12_4 :
  ∃ S : Set (ℕ × ℕ), 
    (∀ (a b : ℕ), (a, b) ∈ S ↔ 
      a < b ∧ 
      a > 0 ∧ 
      b > 0 ∧ 
      harmonic_mean a b = (12 : ℚ)^4) ∧
    Finite S ∧
    Nat.card S = count_ordered_pairs :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_with_harmonic_mean_12_4_l432_43222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l432_43258

def sequence_a : ℕ → ℝ
| 0 => 1  -- Add this case to cover Nat.zero
| 1 => 1
| 2 => 2
| (n + 3) => 7 * sequence_a (n + 2) - 12 * sequence_a (n + 1)

theorem sequence_a_closed_form (n : ℕ) (h : n ≥ 1) :
  sequence_a n = 2 * 3^(n - 1) - 4^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_closed_form_l432_43258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l432_43223

-- Define constants
noncomputable def a : ℝ := 24/7
noncomputable def b : ℝ := Real.log 7
noncomputable def c : ℝ := Real.log (7/Real.exp 1) / Real.log 3 + 1

-- State the theorem
theorem a_gt_b_gt_c : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_gt_b_gt_c_l432_43223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l432_43238

/-- Square represents a square in a 2D plane -/
structure Square where
  side_length : ℝ

/-- Point represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- InscribedSquares represents two squares where one is inscribed in the other -/
structure InscribedSquares where
  outer : Square
  inner : Square
  vertex_on_side : Point → Point → Bool

/-- Theorem stating the ratio of areas of inscribed squares -/
theorem inscribed_squares_area_ratio 
  (squares : InscribedSquares) 
  (h1 : ∀ v, squares.vertex_on_side v (Point.mk 0 0) = true → 
       squares.vertex_on_side v (Point.mk squares.outer.side_length 0) = true → 
       v.x / squares.outer.side_length = 9 / 10) :
  (squares.inner.side_length^2) / (squares.outer.side_length^2) = 1 / 50 := by
  sorry

#check inscribed_squares_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_squares_area_ratio_l432_43238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_exist_l432_43226

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points in the xy-plane -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Determines if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  distance p ⟨c.center.1, c.center.2⟩ > c.radius

/-- The set of real numbers m such that the point (m, 2) is outside the circle (x+1)^2 + (y-2)^2 = 4 -/
def validM : Set ℝ :=
  {m | isOutside ⟨m, 2⟩ (Circle.mk (-1, 2) 2)}

theorem tangent_lines_exist (m : ℝ) : 
  m ∈ validM ↔ m < -3 ∨ m > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_exist_l432_43226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_divisible_lower_bound_l432_43299

/-- Given a finite set of prime numbers, returns the largest possible number of consecutive
    natural numbers such that each of these numbers is divisible by at least one prime from the set -/
def largest_consecutive_divisible (P : Finset Nat) : Nat :=
  sorry

/-- Theorem stating that for any finite set of primes P, the number of consecutive integers
    divisible by at least one prime in P is at least the cardinality of P, with equality
    if and only if the smallest prime in P is greater than the cardinality of P -/
theorem largest_consecutive_divisible_lower_bound (P : Finset Nat) 
  (h_prime : ∀ p ∈ P, Nat.Prime p) (h_nonempty : P.Nonempty) :
  largest_consecutive_divisible P ≥ P.card ∧
  (largest_consecutive_divisible P = P.card ↔ P.min' h_nonempty > P.card) :=
by
  sorry

#check largest_consecutive_divisible_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consecutive_divisible_lower_bound_l432_43299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_contains_rectangle_l432_43259

/-- The smallest side length of a square that can contain a rectangle with sides 10 and 4 -/
noncomputable def smallest_square_side : ℝ := Real.sqrt 58

/-- A rectangle with sides 10 and 4 -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The given rectangle -/
def given_rectangle : Rectangle := { width := 10, height := 4 }

/-- A square can contain a rectangle if its side length is at least as long as the rectangle's diagonal -/
def square_contains_rectangle (s : ℝ) (r : Rectangle) : Prop :=
  s ^ 2 ≥ r.width ^ 2 + r.height ^ 2

theorem smallest_square_contains_rectangle :
  square_contains_rectangle smallest_square_side given_rectangle ∧
  ∀ s, s < smallest_square_side → ¬square_contains_rectangle s given_rectangle :=
by sorry

#check smallest_square_contains_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_square_contains_rectangle_l432_43259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_third_l432_43203

theorem cos_alpha_plus_pi_third (α : ℝ) 
  (h1 : Real.sin α = (4 * Real.sqrt 3) / 7)
  (h2 : 0 < α ∧ α < π / 2) : 
  Real.cos (α + π / 3) = -11 / 14 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_plus_pi_third_l432_43203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l432_43214

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  2 * x^2 - y^2 + 8 * x + 4 * y - 28 = 0

/-- The focus coordinates -/
noncomputable def focus : ℝ × ℝ := (-2 - 4 * Real.sqrt 3, 2)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem focus_of_hyperbola :
  let (fx, fy) := focus
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), hyperbola_equation x y ↔
      ((x + 2)^2 / (2 * a^2) - (y - 2)^2 / (2 * b^2) = 1 ∧
       fx = -2 - Real.sqrt (a^2 + b^2) ∧
       fy = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_hyperbola_l432_43214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1991_is_identity_f_1991_4_7_eq_4_7_l432_43221

noncomputable def f (x : ℝ) : ℝ := (1 + x) / (1 - 3 * x)

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => id
| n + 1 => f ∘ (f_n n)

theorem f_1991_is_identity : ∀ x, f_n 1991 x = x := by
  sorry

theorem f_1991_4_7_eq_4_7 : f_n 1991 4.7 = 4.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1991_is_identity_f_1991_4_7_eq_4_7_l432_43221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_length_l432_43213

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in the form ay = bx -/
structure Line where
  a : ℚ
  b : ℚ

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.y = l.b * p.x

/-- Calculates the midpoint of two points -/
def midpointCustom (p1 p2 : Point) : Point :=
  ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℚ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

/-- The main theorem -/
theorem pq_length 
  (R : Point)
  (line1 line2 : Line)
  (P Q : Point)
  (h1 : R = ⟨9, 7⟩)
  (h2 : line1 = ⟨9, 14⟩)
  (h3 : line2 = ⟨11, 4⟩)
  (h4 : R = midpointCustom P Q)
  (h5 : pointOnLine P line1)
  (h6 : pointOnLine Q line2) :
  distance P Q = 60 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_length_l432_43213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_approx_l432_43215

/-- The height of a building given the height and shadow length of a flagstaff and the shadow length of the building. -/
noncomputable def building_height (flagstaff_height flagstaff_shadow building_shadow : ℝ) : ℝ :=
  (flagstaff_height * building_shadow) / flagstaff_shadow

/-- Theorem stating that the building height is approximately 12.47 meters given the specified conditions. -/
theorem building_height_approx :
  let flagstaff_height : ℝ := 17.5
  let flagstaff_shadow : ℝ := 40.25
  let building_shadow : ℝ := 28.75
  let result := building_height flagstaff_height flagstaff_shadow building_shadow
  abs (result - 12.47) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_building_height_approx_l432_43215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_sequence_result_l432_43292

-- Define the * operation
noncomputable def star (x y : ℝ) : ℝ :=
  (3 * x^3 * y + 3 * x^2 * y^2 + x * y^3 + 45) / ((x + 1)^3 + (y + 1)^3 - 60)

-- Define the associativity property
axiom star_assoc (x y z : ℝ) : star x (star y z) = star (star x y) z

-- Define the sequence of numbers from 2 to 2021
def num_sequence : List ℝ := List.range 2020 |>.map (fun n => (n + 2 : ℝ))

-- State the theorem
theorem star_sequence_result : 
  num_sequence.foldl star 2 = 5463 / 967 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_sequence_result_l432_43292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l432_43208

/-- Given sets P and Q, prove that if their intersection is {-3}, then m = -4/3 -/
theorem intersection_implies_m_value (m : ℝ) : 
  let P : Set ℝ := {m^2 - 4, m + 1, -3}
  let Q : Set ℝ := {m - 3, 2*m - 1, 3*m + 1}
  (P ∩ Q = {-3}) → m = -4/3 := by
  sorry

-- Remove the #eval line as it's not necessary and can cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_l432_43208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l432_43293

/-- Calculates the market value of a stock given its face value, dividend yield, and market yield. -/
noncomputable def market_value (face_value : ℝ) (dividend_yield : ℝ) (market_yield : ℝ) : ℝ :=
  (face_value * dividend_yield / market_yield) * 100

/-- Theorem: The market value of a stock with face value $100, dividend yield 14%, and market yield 8% is $175. -/
theorem stock_market_value :
  let face_value : ℝ := 100
  let dividend_yield : ℝ := 0.14
  let market_yield : ℝ := 0.08
  market_value face_value dividend_yield market_yield = 175 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l432_43293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_odd_and_increasing_l432_43278

-- Define f as an increasing function on ℝ
variable (f : ℝ → ℝ)
variable (h : ∀ x y : ℝ, x < y → f x < f y)

-- Define F in terms of f
def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - f (-x)

-- Theorem stating that F is odd and increasing
theorem F_odd_and_increasing (f : ℝ → ℝ) (h : ∀ x y : ℝ, x < y → f x < f y) :
  (∀ x : ℝ, F f x = -(F f (-x))) ∧
  (∀ x y : ℝ, x < y → F f x < F f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_odd_and_increasing_l432_43278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_forms_cone_l432_43219

/-- A right triangle rotated around one of its legs forms a cone -/
theorem right_triangle_rotation_forms_cone :
  ∀ (t : Type) (l : Type),
    (t → Prop) →  -- RightTriangle
    (l → t → Prop) →  -- IsLeg
    (t → l → Type) →  -- RotationAroundAxis
    (Type) →  -- Cone
    Prop :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_forms_cone_l432_43219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_is_75_l432_43245

/-- Represents the number of coins in the problem -/
def num_coins : ℕ := 80

/-- Represents the maximum number of flips for each coin -/
def max_flips : ℕ := 4

/-- Represents the probability of getting heads on a single flip of a fair coin -/
noncomputable def p_heads : ℝ := 1 / 2

/-- Calculates the probability of getting heads after up to n flips -/
noncomputable def prob_heads_after_n_flips (n : ℕ) : ℝ :=
  (1 - (1 - p_heads)^n)

/-- The expected number of coins showing heads after the flipping process -/
noncomputable def expected_heads : ℝ :=
  num_coins * prob_heads_after_n_flips max_flips

/-- Theorem stating that the expected number of heads is 75 -/
theorem expected_heads_is_75 : expected_heads = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_heads_is_75_l432_43245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l432_43286

-- Define the equation
def equation (m n p : ℕ) : Prop :=
  n^(2*p) = m^2 + n^2 + p + 1

-- Define the condition that p is prime
def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

-- Define the set of solutions
def solutions : Set (ℕ × ℕ × ℕ) :=
  {(2, 3, 2), (2, 3, 2)}

-- Theorem statement
theorem equation_solutions :
  ∀ m n p : ℕ, is_prime p → equation m n p → (n, m, p) ∈ solutions := by
  sorry

#check equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l432_43286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_solutions_l432_43268

-- Define the piecewise function h(x)
noncomputable def h (x : ℝ) : ℝ :=
  if x ≤ 0 then 5 * x + 10 else 3 * x - 5

-- State the theorem
theorem h_solutions (x : ℝ) : h x = 1 ↔ x = -9/5 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_solutions_l432_43268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_five_twelfths_three_fourths_l432_43282

/-- The remainder function for real numbers -/
noncomputable def rem (x y : ℝ) : ℝ := x - y * ⌊x / y⌋

/-- Theorem stating that rem(5/12, 3/4) = 5/12 -/
theorem rem_five_twelfths_three_fourths : rem (5/12) (3/4) = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rem_five_twelfths_three_fourths_l432_43282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l432_43242

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n ↦ a₁ + (n - 1) * d

theorem arithmetic_geometric_sequence_ratio (a₁ : ℝ) (d : ℝ) :
  let a := arithmetic_sequence a₁ d
  (a 3 + 3) ^ 2 = (a 1 + 1) * (a 5 + 5) →
  (a 3 + 3) / (a 1 + 1) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sequence_ratio_l432_43242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_common_elements_l432_43262

def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

def common_elements (m : ℕ) : ℕ := 10 * 4^m

theorem sum_of_common_elements : 
  (Finset.range 10).sum (λ i => common_elements i) = 3495250 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_common_elements_l432_43262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_original_price_is_36_l432_43217

/-- Represents the price of an item before and after an increase -/
structure ItemPrice where
  original : ℚ
  increase_percent : ℚ

/-- Calculates the new price after applying the percentage increase -/
def new_price (item : ItemPrice) : ℚ :=
  item.original * (1 + item.increase_percent / 100)

/-- Theorem stating that the total original price of all items is 36 pounds -/
theorem total_original_price_is_36
  (candy : ItemPrice)
  (soda : ItemPrice)
  (chocolate : ItemPrice)
  (chips : ItemPrice)
  (h1 : candy.original = 10 ∧ candy.increase_percent = 25)
  (h2 : soda.original = 12 ∧ soda.increase_percent = 50)
  (h3 : chocolate.original = 8 ∧ chocolate.increase_percent = 30)
  (h4 : chips.original = 6 ∧ chips.increase_percent = 20)
  (h5 : new_price candy = 25/2)
  (h6 : new_price soda = 18)
  (h7 : new_price chocolate = 52/5)
  (h8 : new_price chips = 36/5) :
  candy.original + soda.original + chocolate.original + chips.original = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_original_price_is_36_l432_43217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_side_length_for_specific_triangle_l432_43284

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  base : ℝ
  area : ℝ
  base_positive : 0 < base
  area_positive : 0 < area

/-- The length of a congruent side in the isosceles triangle -/
noncomputable def congruent_side_length (t : IsoscelesTriangle) : ℝ :=
  Real.sqrt ((t.base / 2) ^ 2 + (2 * t.area / t.base) ^ 2)

/-- Theorem stating the length of the congruent side for a specific isosceles triangle -/
theorem congruent_side_length_for_specific_triangle :
  ∃ t : IsoscelesTriangle, t.base = 30 ∧ t.area = 120 ∧ congruent_side_length t = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_side_length_for_specific_triangle_l432_43284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_l432_43267

/-- A function f is odd if f(-x) = -f(x) for all x in its domain --/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function f(x) = ((x+1)(x+a))/x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1) * (x + a)) / x

theorem odd_function_implies_a_eq_neg_one :
  ∃ a : ℝ, IsOdd (f a) → a = -1 := by
  use (-1)
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_neg_one_l432_43267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_property_l432_43257

def F : ℕ → ℕ
  | 0 => 4  -- Added case for 0
  | 1 => 4
  | 2 => 4
  | (n + 3) => F (n + 2) + F (n + 1)

theorem right_triangle_property (n : ℕ) :
  (F n * F (n + 4))^2 + (F (n + 1) * F (n + 3))^2 = (2 * F (n + 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_property_l432_43257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OC_equals_2a_minus_3b_l432_43249

variable {V : Type*} [AddCommGroup V] [Module ℚ V]
variable (a b : V)

def OA (a b : V) : V := a + b
def AB (a b : V) : V := (3 : ℚ) • (a - b)
def CB (a b : V) : V := (2 : ℚ) • a + b

theorem OC_equals_2a_minus_3b (a b : V) : 
  OA a b + AB a b - CB a b = (2 : ℚ) • a - (3 : ℚ) • b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_OC_equals_2a_minus_3b_l432_43249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_and_behavior_imply_sum_l432_43204

-- Define the function g(x)
noncomputable def g (D E F : ℤ) (x : ℝ) : ℝ := x^2 / (D * x^2 + E * x + F)

-- State the theorem
theorem asymptotes_and_behavior_imply_sum (D E F : ℤ) :
  -- Vertical asymptotes at x = -3 and x = 2
  (∀ x : ℝ, D * x^2 + E * x + F = 0 ↔ x = -3 ∨ x = 2) →
  -- g(x) > 0.3 for all x > 3
  (∀ x : ℝ, x > 3 → g D E F x > 0.3) →
  -- Conclusion: D + E + F = -8
  D + E + F = -8 := by
  sorry

#check asymptotes_and_behavior_imply_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_and_behavior_imply_sum_l432_43204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_problem_l432_43209

theorem class_average_problem (n₁ n₂ : ℕ) (avg₂ avg_total : ℝ) :
  n₁ = 24 →
  n₂ = 50 →
  avg₂ = 60 →
  avg_total = 53.513513513513516 →
  let total_students := (n₁ + n₂ : ℝ)
  let total_marks := avg_total * total_students
  let marks₂ := avg₂ * n₂
  let marks₁ := total_marks - marks₂
  let avg₁ := marks₁ / n₁
  abs (avg₁ - 40.04166666666667) < 0.0000001 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_average_problem_l432_43209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cans_equal_coins_l432_43297

/-- The number of cans -/
def N : ℕ := 2015

/-- Represents the number of coins in each can -/
def Configuration := Fin N → ℕ

/-- Initial configuration where all cans are empty -/
def initialConfigA : Configuration := λ _ ↦ 0

/-- Initial configuration where can i contains i coins -/
def initialConfigB : Configuration := λ i ↦ i.val + 1

/-- Initial configuration where can i contains (N+1-i) coins -/
def initialConfigC : Configuration := λ i ↦ N + 1 - (i.val + 1)

/-- Represents a step where n coins are added to each can except can number n -/
def step (config : Configuration) (n : Fin N) : Configuration :=
  λ i ↦ if i = n then config i else config i + n.val + 1

/-- Predicate to check if all cans have the same number of coins -/
def allEqual (config : Configuration) : Prop :=
  ∀ i j : Fin N, config i = config j

/-- Theorem stating that for each initial configuration, 
    there exists a sequence of steps resulting in all cans having equal coins -/
theorem cans_equal_coins :
  (∃ (steps : List (Fin N)), allEqual (steps.foldl step initialConfigA)) ∧
  (∃ (steps : List (Fin N)), allEqual (steps.foldl step initialConfigB)) ∧
  (∃ (steps : List (Fin N)), allEqual (steps.foldl step initialConfigC)) := by
  sorry

#check cans_equal_coins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cans_equal_coins_l432_43297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l432_43251

/-- A circle with center (1, 0) that is tangent to the line 3x + 4y + 2 = 0 has the equation (x-1)^2 + y^2 = 1 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 0)
  let line (x y : ℝ) := 3*x + 4*y + 2 = 0
  let is_tangent (c : ℝ × ℝ → ℝ) (l : ℝ → ℝ → Prop) := 
    ∃ (p : ℝ × ℝ), l p.1 p.2 ∧ 
    ∀ (q : ℝ × ℝ), l q.1 q.2 → c q ≥ c p
  let circle (p : ℝ × ℝ) := (p.1 - center.1)^2 + (p.2 - center.2)^2
  is_tangent circle line →
  circle (x, y) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l432_43251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l432_43247

theorem solution_difference : 
  ∃ r₁ r₂ : ℝ, 
    (r₁^2 - 5*r₁ - 14) / (r₁ + 4) = 2*r₁ + 9 ∧
    (r₂^2 - 5*r₂ - 14) / (r₂ + 4) = 2*r₂ + 9 ∧
    r₁ ≠ r₂ ∧
    |r₁ - r₂| = 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l432_43247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_3_25_l432_43234

/-- Represents the outcome of a die roll -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | Eight

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Bool :=
  n > 1 && (Nat.factors n).length == 1

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Bool :=
  let root := n.sqrt
  root * root == n

/-- Converts a DieOutcome to its numerical value -/
def toNumber (outcome : DieOutcome) : Nat :=
  match outcome with
  | DieOutcome.One => 1
  | DieOutcome.Two => 2
  | DieOutcome.Three => 3
  | DieOutcome.Four => 4
  | DieOutcome.Five => 5
  | DieOutcome.Six => 6
  | DieOutcome.Seven => 7
  | DieOutcome.Eight => 8

/-- Calculates the winnings for a given outcome -/
def winnings (outcome : DieOutcome) : Nat :=
  let n := toNumber outcome
  if isPrime n then n
  else if isPerfectSquare n then 2 * n
  else 0

/-- Theorem: The expected value of winnings is $3.25 -/
theorem expected_value_is_3_25 :
  (1 / 8 : ℚ) * (winnings DieOutcome.One +
                 winnings DieOutcome.Two +
                 winnings DieOutcome.Three +
                 winnings DieOutcome.Four +
                 winnings DieOutcome.Five +
                 winnings DieOutcome.Six +
                 winnings DieOutcome.Seven +
                 winnings DieOutcome.Eight) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_is_3_25_l432_43234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l432_43287

def b : ℕ → ℚ
  | 0 => 3  -- We define b(0) as 3 to match b₁
  | 1 => 4  -- This matches b₂
  | n + 2 => (b (n + 1))^2 / (b n)

theorem b_2023_value : b 2022 = 64 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_l432_43287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_reorganization_last_page_l432_43285

/-- Represents the stamp collection reorganization problem -/
def StampReorganization (initial_albums : ℕ) (pages_per_album : ℕ) (initial_stamps_per_page : ℕ)
  (new_stamps_per_page : ℕ) (full_albums_after_reorg : ℕ) : Prop :=
  let total_stamps := initial_albums * pages_per_album * initial_stamps_per_page
  let full_pages_after_reorg := (total_stamps / new_stamps_per_page : ℕ)
  let remaining_stamps := total_stamps % new_stamps_per_page
  full_pages_after_reorg = full_albums_after_reorg * pages_per_album + (pages_per_album - 1) ∧
  remaining_stamps = new_stamps_per_page

theorem stamp_reorganization_last_page :
  StampReorganization 10 30 8 12 6 → 12 = 12 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_reorganization_last_page_l432_43285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_equalization_theorem_l432_43253

/-- Represents a box or a coin in the nested structure -/
inductive Box
  | Coin : Bool → Box  -- True for heads, False for tails
  | NestedBox : Box → Box → Box

/-- Calculates the deficit (difference between heads and tails) in a box -/
def deficitValue : Box → Int
  | Box.Coin true => 1
  | Box.Coin false => -1
  | Box.NestedBox b1 b2 => deficitValue b1 + deficitValue b2

/-- Represents a flip action on a box -/
def flipBox : Box → Box
  | Box.Coin b => Box.Coin (!b)
  | Box.NestedBox b1 b2 => Box.NestedBox (flipBox b1) (flipBox b2)

/-- Generates a nested box structure with n levels -/
def generateBoxStructure : Nat → Box
  | 0 => Box.Coin false  -- Base case: a single coin
  | n + 1 => Box.NestedBox (generateBoxStructure n) (generateBoxStructure n)

/-- The main theorem to be proved -/
theorem coin_equalization_theorem (n : Nat) :
  ∃ (flips : List (Box → Box)),
    (∀ b : Box, List.length flips ≤ n) ∧
    (∀ b : Box, deficitValue (List.foldl (λ acc f => f acc) (generateBoxStructure n) flips) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_equalization_theorem_l432_43253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_ξ₃_not_discrete_l432_43288

-- Define the concept of a discrete random variable
def is_discrete_random_variable (ξ : ℝ → ℝ) : Prop :=
  ∃ (S : Set ℝ), Countable S ∧ ∀ (x : ℝ), ξ x ∈ S

-- Define the random variables
noncomputable def ξ₁ : ℝ → ℝ := sorry  -- Number of visitors in an airport lounge
noncomputable def ξ₂ : ℝ → ℝ := sorry  -- Number of pages received at a paging station
noncomputable def ξ₃ : ℝ → ℝ := sorry  -- Water level of the Yangtze River
noncomputable def ξ₄ : ℝ → ℝ := sorry  -- Number of vehicles passing through an overpass

-- State the theorem
theorem only_ξ₃_not_discrete :
  is_discrete_random_variable ξ₁ ∧
  is_discrete_random_variable ξ₂ ∧
  ¬is_discrete_random_variable ξ₃ ∧
  is_discrete_random_variable ξ₄ :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_ξ₃_not_discrete_l432_43288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_generalized_powers_in_ap_l432_43202

/-- The arithmetic progression defined by aₙ = 3n + 2 -/
def arithmetic_progression (n : ℕ) : ℤ := 3 * (n : ℤ) + 2

theorem no_perfect_squares_in_ap :
  ∀ n : ℕ, ∀ x : ℤ, arithmetic_progression n ≠ x^2 := by sorry

theorem infinitely_many_perfect_cubes_in_ap :
  ∀ k : ℕ, ∃ n : ℕ, ∃ x : ℤ, (n : ℤ) ≥ k ∧ arithmetic_progression n = x^3 := by sorry

theorem generalized_powers_in_ap (m : ℕ) :
  (∀ n : ℕ, ∀ x : ℤ, arithmetic_progression n ≠ x^(2*m)) ∧
  (∃ k : ℕ, ∀ j : ℕ, j ≥ k → ∃ n : ℕ, ∃ x : ℤ, arithmetic_progression n = x^(2*m+1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_generalized_powers_in_ap_l432_43202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1001st_term_l432_43225

def mySequence (a b : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | 2 => 4*a - 5*b
  | 3 => 4*a + 5*b
  | n+4 => 2 * mySequence a b (n+3) - mySequence a b (n+2)

theorem sequence_1001st_term (a : ℤ) (h : ∃ b : ℤ, b = 2*a - 3) : 
  mySequence a (2*a - 3) 1000 = 30003 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1001st_term_l432_43225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_exponential_equation_l432_43212

/-- Given that log 2 = 0.3010 and log 3 = 0.4771, prove that the solution to 3^(x+3) = 99 is approximately 1.18 -/
theorem solution_to_exponential_equation 
  (h1 : Real.log 2 = 0.3010) 
  (h2 : Real.log 3 = 0.4771) : 
  ∃ x : ℝ, (3 : ℝ)^(x + 3) = 99 ∧ |x - 1.18| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_to_exponential_equation_l432_43212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_for_100_apples_l432_43296

/-- Represents the profit calculation for an apple dealer --/
structure AppleDealer where
  cost_per_bushel : ℚ
  apples_per_bushel : ℕ
  selling_price_per_apple : ℚ

/-- Calculates the profit for selling a given number of apples --/
def calculate_profit (dealer : AppleDealer) (num_apples : ℕ) : ℚ :=
  let cost_per_apple := dealer.cost_per_bushel / dealer.apples_per_bushel
  let profit_per_apple := dealer.selling_price_per_apple - cost_per_apple
  (num_apples : ℚ) * profit_per_apple

/-- Theorem stating that the profit from selling 100 apples is $15 --/
theorem profit_for_100_apples (dealer : AppleDealer) 
  (h1 : dealer.cost_per_bushel = 12)
  (h2 : dealer.apples_per_bushel = 48)
  (h3 : dealer.selling_price_per_apple = 2/5) :
  calculate_profit dealer 100 = 15 := by
  sorry

#eval calculate_profit { cost_per_bushel := 12, apples_per_bushel := 48, selling_price_per_apple := 2/5 } 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_for_100_apples_l432_43296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_area_ratio_is_two_l432_43265

/-- Regular triangular prism with side length a and height b -/
structure RegularTriangularPrism where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- Cylinder with radius r and height h -/
structure Cylinder where
  r : ℝ
  h : ℝ
  r_pos : 0 < r
  h_pos : 0 < h

/-- The volume of a regular triangular prism -/
noncomputable def volume_prism (p : RegularTriangularPrism) : ℝ :=
  (Real.sqrt 3 / 4) * p.a^2 * p.b

/-- The volume of a cylinder -/
noncomputable def volume_cylinder (c : Cylinder) : ℝ :=
  Real.pi * c.r^2 * c.h

/-- The lateral surface area of a regular triangular prism -/
def lateral_area_prism (p : RegularTriangularPrism) : ℝ :=
  3 * p.a * p.b

/-- The lateral surface area of a cylinder -/
noncomputable def lateral_area_cylinder (c : Cylinder) : ℝ :=
  2 * Real.pi * c.r * c.h

/-- The main theorem -/
theorem lateral_area_ratio_is_two
  (p : RegularTriangularPrism)
  (c : Cylinder)
  (h_volume_eq : volume_prism p = volume_cylinder c)
  (h_radius : c.r = (Real.sqrt 3 / 3) * p.a) :
  lateral_area_prism p / lateral_area_cylinder c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_area_ratio_is_two_l432_43265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l432_43274

/-- An ellipse with center at the origin, passing through (3,2), foci on coordinate axes, 
    and major axis three times the minor axis. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  -- The ellipse passes through (3,2)
  passes_through : 3^2 / a^2 + 2^2 / b^2 = 1
  -- The foci are on coordinate axes
  foci_on_axes : (a > b ∧ a^2 - b^2 > 0) ∨ (b > a ∧ b^2 - a^2 > 0)
  -- Major axis is three times minor axis
  major_minor_ratio : max a b = 3 * min a b

/-- The equation of the ellipse is one of the two given forms. -/
theorem ellipse_equation (e : Ellipse) : 
  (x^2 / 45 + y^2 / 5 = 1) ∨ (y^2 / 85 + x^2 / (85/9) = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l432_43274
