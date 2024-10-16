import Mathlib

namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_is_270_l3102_310270

/-- Triangle ABC with given side lengths and parallel lines -/
structure TriangleWithParallels where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Lengths of segments formed by parallel lines
  ℓA_segment : ℝ
  ℓB_segment : ℝ
  ℓC_segment : ℝ
  -- Conditions
  AB_positive : AB > 0
  BC_positive : BC > 0
  AC_positive : AC > 0
  ℓA_segment_positive : ℓA_segment > 0
  ℓB_segment_positive : ℓB_segment > 0
  ℓC_segment_positive : ℓC_segment > 0
  AB_eq : AB = 150
  BC_eq : BC = 270
  AC_eq : AC = 210
  ℓA_segment_eq : ℓA_segment = 60
  ℓB_segment_eq : ℓB_segment = 50
  ℓC_segment_eq : ℓC_segment = 20

/-- The perimeter of the inner triangle formed by parallel lines -/
def innerTrianglePerimeter (t : TriangleWithParallels) : ℝ := sorry

/-- Theorem: The perimeter of the inner triangle is 270 -/
theorem inner_triangle_perimeter_is_270 (t : TriangleWithParallels) :
  innerTrianglePerimeter t = 270 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_is_270_l3102_310270


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3102_310264

theorem square_sum_given_product_and_sum (p q : ℝ) 
  (h1 : p * q = 12) 
  (h2 : p + q = 8) : 
  p^2 + q^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_l3102_310264


namespace NUMINAMATH_CALUDE_laptop_price_l3102_310282

theorem laptop_price (sticker_price : ℝ) : 
  (0.8 * sticker_price - 120 = 0.7 * sticker_price - 50 - 30) →
  sticker_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_l3102_310282


namespace NUMINAMATH_CALUDE_problem_solution_l3102_310222

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- Theorem statement
theorem problem_solution :
  (∀ x : ℝ, f x + 1 < |2 * x + 1| ↔ x ∈ M) ∧
  (∀ a b : ℝ, a ∈ M → b ∈ M → |a * b + 1| > |a + b|) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3102_310222


namespace NUMINAMATH_CALUDE_rectangle_length_l3102_310212

theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 12 →
  rect_width = 6 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 24 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l3102_310212


namespace NUMINAMATH_CALUDE_consecutive_even_sum_42_square_diff_l3102_310259

theorem consecutive_even_sum_42_square_diff (n m : ℤ) : 
  (Even n) → (Even m) → (m = n + 2) → (n + m = 42) → 
  (m ^ 2 - n ^ 2 = 84) := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_42_square_diff_l3102_310259


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l3102_310201

/-- 
Given an arithmetic progression with first term a₁ and common difference d,
if the product of the 3rd and 6th terms is 406,
and the 9th term divided by the 4th term gives a quotient of 2 with remainder 6,
then the first term is 4 and the common difference is 5.
-/
theorem arithmetic_progression_problem (a₁ d : ℚ) : 
  (a₁ + 2*d) * (a₁ + 5*d) = 406 →
  (a₁ + 8*d) = 2*(a₁ + 3*d) + 6 →
  a₁ = 4 ∧ d = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_progression_problem_l3102_310201


namespace NUMINAMATH_CALUDE_opposite_of_2021_l3102_310253

theorem opposite_of_2021 : -(2021 : ℤ) = -2021 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2021_l3102_310253


namespace NUMINAMATH_CALUDE_minimum_sum_geometric_sequence_l3102_310238

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = r * a n

theorem minimum_sum_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : GeometricSequence a)
    (h_positive : ∀ n, a n > 0)
    (h_product : a 3 * a 5 = 64) :
    ∃ (m : ℝ), m = 16 ∧ ∀ x y, x > 0 → y > 0 → x * y = 64 → x + y ≥ m :=
  sorry

end NUMINAMATH_CALUDE_minimum_sum_geometric_sequence_l3102_310238


namespace NUMINAMATH_CALUDE_intersection_equals_B_l3102_310294

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + 2*x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a*x = 3}

-- Define the set of possible values for a
def possible_a : Set ℝ := {0, -1, 3}

-- State the theorem
theorem intersection_equals_B (a : ℝ) :
  (A ∩ B a = B a) ↔ a ∈ possible_a :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_B_l3102_310294


namespace NUMINAMATH_CALUDE_total_eggs_collected_l3102_310255

/-- The number of dozen eggs collected by Benjamin, Carla, Trisha, and David -/
def total_eggs (benjamin carla trisha david : ℕ) : ℕ :=
  benjamin + carla + trisha + david

/-- Theorem stating the total number of dozen eggs collected -/
theorem total_eggs_collected :
  ∃ (benjamin carla trisha david : ℕ),
    benjamin = 6 ∧
    carla = 3 * benjamin ∧
    trisha = benjamin - 4 ∧
    david = 2 * trisha ∧
    total_eggs benjamin carla trisha david = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_collected_l3102_310255


namespace NUMINAMATH_CALUDE_triangle_inequality_l3102_310297

theorem triangle_inequality (a b c S : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hS : S > 0)
  (h_triangle : S = Real.sqrt (((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 16)) :
  4 * Real.sqrt 3 * S + (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ a^2 + b^2 + c^2 ∧
  a^2 + b^2 + c^2 ≤ 4 * Real.sqrt 3 * S + 3 * ((a - b)^2 + (b - c)^2 + (c - a)^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3102_310297


namespace NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_distance_l3102_310221

/-- Given a parabola y² = 4x and a line passing through its focus intersecting the parabola
    at points A(x₁, y₁) and B(x₂, y₂) with |AB| = 7, the distance from the midpoint M of AB
    to the directrix of the parabola is 7/2. -/
theorem parabola_midpoint_to_directrix_distance
  (x₁ y₁ x₂ y₂ : ℝ) :
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  (x₁ - 1)^2 + y₁^2 = (x₂ - 1)^2 + y₂^2 →
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 49 →
  (x₁ + x₂)/2 + 1 = 7/2 := by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_to_directrix_distance_l3102_310221


namespace NUMINAMATH_CALUDE_sum_of_digits_product_35_42_base8_l3102_310256

def base8_to_base10 (n : Nat) : Nat :=
  (n / 10) * 8 + n % 10

def base10_to_base8 (n : Nat) : Nat :=
  if n < 8 then n
  else (base10_to_base8 (n / 8)) * 10 + n % 8

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n
  else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_product_35_42_base8 :
  sum_of_digits (base10_to_base8 (base8_to_base10 35 * base8_to_base10 42)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_35_42_base8_l3102_310256


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l3102_310213

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 9 / Real.log 18 + 1) = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l3102_310213


namespace NUMINAMATH_CALUDE_triangle_area_l3102_310247

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos (B - C) + a * Real.cos A = 2 * Real.sqrt 3 * c * Real.sin B * Real.cos A →
  b^2 + c^2 - a^2 = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3102_310247


namespace NUMINAMATH_CALUDE_semicircle_radius_l3102_310234

/-- The radius of a semi-circle given its perimeter -/
theorem semicircle_radius (perimeter : ℝ) (h : perimeter = 113) : 
  ∃ (radius : ℝ), radius = perimeter / (Real.pi + 2) := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l3102_310234


namespace NUMINAMATH_CALUDE_strap_mask_probability_is_0_12_l3102_310280

/-- Represents a mask factory with two types of products -/
structure MaskFactory where
  regularRatio : ℝ
  surgicalRatio : ℝ
  regularStrapRatio : ℝ
  surgicalStrapRatio : ℝ

/-- The probability of selecting a strap mask from the factory -/
def strapMaskProbability (factory : MaskFactory) : ℝ :=
  factory.regularRatio * factory.regularStrapRatio +
  factory.surgicalRatio * factory.surgicalStrapRatio

/-- Theorem stating the probability of selecting a strap mask -/
theorem strap_mask_probability_is_0_12 :
  let factory : MaskFactory := {
    regularRatio := 0.8,
    surgicalRatio := 0.2,
    regularStrapRatio := 0.1,
    surgicalStrapRatio := 0.2
  }
  strapMaskProbability factory = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_strap_mask_probability_is_0_12_l3102_310280


namespace NUMINAMATH_CALUDE_cosine_inequality_l3102_310225

theorem cosine_inequality (θ : ℝ) : 5 + 8 * Real.cos θ + 4 * Real.cos (2 * θ) + Real.cos (3 * θ) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_inequality_l3102_310225


namespace NUMINAMATH_CALUDE_negative_result_l3102_310204

theorem negative_result : ∀ (x : ℝ), x = -4 →
  (-(x) > 0) ∧ (x^2 > 0) ∧ (-|x| < 0) ∧ (-(x^5) > 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_result_l3102_310204


namespace NUMINAMATH_CALUDE_demand_analysis_l3102_310258

def f (x : ℕ) : ℚ := (1 / 150) * x * (x + 1) * (35 - 2 * x)

def g (x : ℕ) : ℚ := (1 / 25) * x * (12 - x)

theorem demand_analysis (x : ℕ) (h : x ≤ 12) :
  -- 1. The demand in the x-th month
  g x = f x - f (x - 1) ∧
  -- 2. The maximum monthly demand occurs when x = 6 and is equal to 36/25
  (∀ y : ℕ, y ≤ 12 → g y ≤ g 6) ∧ g 6 = 36 / 25 ∧
  -- 3. The total demand for the first 6 months is 161/25
  f 6 = 161 / 25 :=
sorry

end NUMINAMATH_CALUDE_demand_analysis_l3102_310258


namespace NUMINAMATH_CALUDE_best_fit_r_squared_l3102_310252

def r_squared_values : List ℝ := [0.27, 0.85, 0.96, 0.5]

theorem best_fit_r_squared (best_fit : ℝ) (h : best_fit ∈ r_squared_values) :
  (∀ x ∈ r_squared_values, x ≤ best_fit) ∧ best_fit = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_best_fit_r_squared_l3102_310252


namespace NUMINAMATH_CALUDE_specific_cube_figure_surface_area_l3102_310214

/-- A three-dimensional figure composed of unit cubes -/
structure CubeFigure where
  num_cubes : ℕ
  edge_length : ℝ

/-- Calculate the surface area of a cube figure -/
def surface_area (figure : CubeFigure) : ℝ :=
  sorry

/-- Theorem: The surface area of a specific cube figure is 32 square units -/
theorem specific_cube_figure_surface_area :
  let figure : CubeFigure := { num_cubes := 9, edge_length := 1 }
  surface_area figure = 32 := by sorry

end NUMINAMATH_CALUDE_specific_cube_figure_surface_area_l3102_310214


namespace NUMINAMATH_CALUDE_white_marbles_count_l3102_310237

theorem white_marbles_count (total : ℕ) (blue : ℕ) (red : ℕ) (prob_red_or_white : ℚ) :
  total = 50 →
  blue = 5 →
  red = 9 →
  prob_red_or_white = 9/10 →
  (total - blue - red : ℚ) / total = prob_red_or_white - (red : ℚ) / total →
  total - blue - red = 36 :=
by
  sorry

#check white_marbles_count

end NUMINAMATH_CALUDE_white_marbles_count_l3102_310237


namespace NUMINAMATH_CALUDE_consecutive_squares_difference_l3102_310260

theorem consecutive_squares_difference (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_difference_l3102_310260


namespace NUMINAMATH_CALUDE_cuboid_third_edge_length_l3102_310210

/-- Given a cuboid with two known edge lengths and its volume, 
    calculate the length of the third edge. -/
theorem cuboid_third_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (volume : ℝ) (third_edge : ℝ) :
  edge1 = 2 →
  edge2 = 5 →
  volume = 30 →
  volume = edge1 * edge2 * third_edge →
  third_edge = 3 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_third_edge_length_l3102_310210


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3102_310284

-- Define the space
variable (Space : Type)

-- Define lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β : Plane)

-- State that l, m, n are different lines
variable (h_diff_lm : l ≠ m)
variable (h_diff_ln : l ≠ n)
variable (h_diff_mn : m ≠ n)

-- State that α and β are non-coincident planes
variable (h_non_coincident : α ≠ β)

-- State the theorem to be proved
theorem perpendicular_lines_from_perpendicular_planes :
  (perpendicular_plane α β ∧ perpendicular_line_plane l α ∧ perpendicular_line_plane m β) →
  perpendicular_line l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l3102_310284


namespace NUMINAMATH_CALUDE_simplify_fraction_l3102_310227

theorem simplify_fraction : 18 * (8 / 15) * (3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3102_310227


namespace NUMINAMATH_CALUDE_sum_A_B_equals_one_l3102_310289

theorem sum_A_B_equals_one (a : ℝ) (ha : a ≠ 1 ∧ a ≠ -1) :
  let x : ℝ := (1 - a) / (1 - 1/a)
  let y : ℝ := 1 - 1/x
  let A : ℝ := 1 / (1 - (1-x)/y)
  let B : ℝ := 1 / (1 - y/(1-x))
  A + B = 1 := by
sorry


end NUMINAMATH_CALUDE_sum_A_B_equals_one_l3102_310289


namespace NUMINAMATH_CALUDE_max_sum_of_product_2401_l3102_310218

theorem max_sum_of_product_2401 :
  ∀ A B C : ℕ+,
  A ≠ B → B ≠ C → A ≠ C →
  A * B * C = 2401 →
  A + B + C ≤ 351 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_product_2401_l3102_310218


namespace NUMINAMATH_CALUDE_subset_condition_implies_upper_bound_l3102_310281

theorem subset_condition_implies_upper_bound (a : ℝ) :
  let A := {x : ℝ | x > 3}
  let B := {x : ℝ | x > a}
  A ⊆ B → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_subset_condition_implies_upper_bound_l3102_310281


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_diameter_squared_l3102_310272

/-- An equiangular hexagon with specified side lengths -/
structure EquiangularHexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  equiangular : True  -- We're not proving this property, just stating it

/-- The diameter of the largest inscribed circle in an equiangular hexagon -/
def largest_inscribed_circle_diameter (h : EquiangularHexagon) : ℝ :=
  sorry  -- Definition not provided, as it's part of what needs to be proved

/-- Theorem: The square of the diameter of the largest inscribed circle in the given hexagon is 147 -/
theorem largest_inscribed_circle_diameter_squared (h : EquiangularHexagon)
  (h_AB : h.AB = 6)
  (h_BC : h.BC = 8)
  (h_CD : h.CD = 10)
  (h_DE : h.DE = 12) :
  (largest_inscribed_circle_diameter h)^2 = 147 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_diameter_squared_l3102_310272


namespace NUMINAMATH_CALUDE_nine_chapters_problem_l3102_310216

/-- Represents the problem from "The Nine Chapters on the Mathematical Art" -/
theorem nine_chapters_problem (x y : ℤ) : 
  (∀ (z : ℤ), z * x = y → (8 * x - 3 = y ↔ z = 8) ∧ (7 * x + 4 = y ↔ z = 7)) →
  (8 * x - 3 = y ∧ 7 * x + 4 = y) :=
by sorry

end NUMINAMATH_CALUDE_nine_chapters_problem_l3102_310216


namespace NUMINAMATH_CALUDE_max_S_value_l3102_310261

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality constraints
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the area function S
def S (t : Triangle) : ℝ := (t.a - t.b + t.c) * (t.a + t.b - t.c)

-- Theorem statement
theorem max_S_value (t : Triangle) (h : t.b + t.c = 8) :
  S t ≤ 64 / 17 :=
sorry

end NUMINAMATH_CALUDE_max_S_value_l3102_310261


namespace NUMINAMATH_CALUDE_inequalities_always_true_l3102_310207

theorem inequalities_always_true (x : ℝ) : 
  (x^2 + 6*x + 10 > 0) ∧ (-x^2 + x - 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l3102_310207


namespace NUMINAMATH_CALUDE_valid_subset_of_A_l3102_310269

def A : Set ℝ := {x | x ≥ 0}

theorem valid_subset_of_A : 
  ({1, 2} : Set ℝ) ⊆ A ∧ 
  ¬({x : ℝ | x ≤ 1} ⊆ A) ∧ 
  ¬({-1, 0, 1} ⊆ A) ∧ 
  ¬(Set.univ ⊆ A) :=
sorry

end NUMINAMATH_CALUDE_valid_subset_of_A_l3102_310269


namespace NUMINAMATH_CALUDE_shorter_diagonal_length_l3102_310266

/-- Represents a trapezoid EFGH with given properties -/
structure Trapezoid where
  EF : ℝ
  GH : ℝ
  side1 : ℝ
  side2 : ℝ
  acute_angles : Bool

/-- The shorter diagonal of the trapezoid -/
def shorter_diagonal (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that for a trapezoid with specific measurements, 
    the shorter diagonal has length 27 -/
theorem shorter_diagonal_length :
  ∀ t : Trapezoid, 
    t.EF = 40 ∧ 
    t.GH = 28 ∧ 
    t.side1 = 13 ∧ 
    t.side2 = 15 ∧ 
    t.acute_angles = true →
    shorter_diagonal t = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_shorter_diagonal_length_l3102_310266


namespace NUMINAMATH_CALUDE_binomial_coefficient_19_13_l3102_310257

theorem binomial_coefficient_19_13 :
  (Nat.choose 18 11 = 31824) →
  (Nat.choose 18 12 = 18564) →
  (Nat.choose 20 13 = 77520) →
  Nat.choose 19 13 = 58956 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_19_13_l3102_310257


namespace NUMINAMATH_CALUDE_gcd_245_1001_l3102_310251

theorem gcd_245_1001 : Nat.gcd 245 1001 = 7 := by sorry

end NUMINAMATH_CALUDE_gcd_245_1001_l3102_310251


namespace NUMINAMATH_CALUDE_exists_universal_source_l3102_310220

/-- A directed graph where every pair of vertices is connected by a directed edge. -/
structure CompleteDigraph (V : Type*) [Fintype V] [DecidableEq V] :=
  (edge : V → V → Prop)
  (complete : ∀ (u v : V), u ≠ v → edge u v ∨ edge v u)

/-- A path of length at most 2 between two vertices. -/
def PathOfLengthAtMostTwo {V : Type*} (edge : V → V → Prop) (u v : V) : Prop :=
  edge u v ∨ ∃ w, edge u w ∧ edge w v

/-- 
In a complete directed graph, there exists a vertex from which 
every other vertex can be reached by a path of length at most 2.
-/
theorem exists_universal_source {V : Type*} [Fintype V] [DecidableEq V] 
  (G : CompleteDigraph V) : 
  ∃ (u : V), ∀ (v : V), u ≠ v → PathOfLengthAtMostTwo G.edge u v :=
sorry

end NUMINAMATH_CALUDE_exists_universal_source_l3102_310220


namespace NUMINAMATH_CALUDE_expression_factorization_l3102_310200

theorem expression_factorization (x y z : ℤ) :
  x^2 - y^2 - z^2 + 2*y*z + x + y - z = (x + y - z) * (x - y + z + 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3102_310200


namespace NUMINAMATH_CALUDE_unique_non_representable_expression_l3102_310268

/-- Represents an algebraic expression that may or may not be
    representable as a square of a binomial or difference of squares. -/
inductive BinomialExpression
  | Representable (a b : ℤ) : BinomialExpression
  | NotRepresentable (a b : ℤ) : BinomialExpression

/-- Determines if a given expression can be represented as a 
    square of a binomial or difference of squares. -/
def is_representable (expr : BinomialExpression) : Prop :=
  match expr with
  | BinomialExpression.Representable _ _ => True
  | BinomialExpression.NotRepresentable _ _ => False

/-- The four expressions from the original problem. -/
def expr1 : BinomialExpression := BinomialExpression.Representable 1 (-2)
def expr2 : BinomialExpression := BinomialExpression.Representable 1 (-2)
def expr3 : BinomialExpression := BinomialExpression.Representable 2 (-1)
def expr4 : BinomialExpression := BinomialExpression.NotRepresentable 1 2

theorem unique_non_representable_expression :
  is_representable expr1 ∧
  is_representable expr2 ∧
  is_representable expr3 ∧
  ¬is_representable expr4 :=
sorry

end NUMINAMATH_CALUDE_unique_non_representable_expression_l3102_310268


namespace NUMINAMATH_CALUDE_log_base_range_l3102_310245

-- Define the function f(x) = log_a(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_base_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 < f a 3 → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_log_base_range_l3102_310245


namespace NUMINAMATH_CALUDE_laptop_price_proof_l3102_310224

theorem laptop_price_proof (original_price : ℝ) : 
  (0.7 * original_price - (0.8 * original_price - 70) = 20) → 
  original_price = 500 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l3102_310224


namespace NUMINAMATH_CALUDE_range_of_odd_function_l3102_310246

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f_positive (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x = 3

theorem range_of_odd_function (f : ℝ → ℝ) (h1 : is_odd f) (h2 : f_positive f) :
  Set.range f = {-3, 0, 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_odd_function_l3102_310246


namespace NUMINAMATH_CALUDE_sum_of_primes_equals_210_l3102_310202

theorem sum_of_primes_equals_210 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h : 100^2 + 1^2 = 65^2 + 76^2 ∧ 100^2 + 1^2 = p * q) : 
  p + q = 210 := by
sorry

end NUMINAMATH_CALUDE_sum_of_primes_equals_210_l3102_310202


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solve_for_y_l3102_310273

/-- Given an arithmetic sequence with the first three terms as specified,
    prove that the value of y is 5/3 -/
theorem arithmetic_sequence_solve_for_y :
  ∀ (seq : ℕ → ℚ),
  (seq 0 = 2/3) →
  (seq 1 = y + 2) →
  (seq 2 = 4*y) →
  (∀ n, seq (n+1) - seq n = seq (n+2) - seq (n+1)) →
  (y = 5/3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solve_for_y_l3102_310273


namespace NUMINAMATH_CALUDE_total_paid_correct_l3102_310240

/-- Represents the purchase details and pricing rules for fruits. -/
structure FruitPurchase where
  grapes_kg : ℝ
  grapes_price : ℝ
  mangoes_kg : ℝ
  mangoes_price : ℝ
  apples_kg : ℝ
  apples_price : ℝ
  oranges_kg : ℝ
  oranges_price : ℝ
  grapes_tax : ℝ
  apples_tax : ℝ
  oranges_discount : ℝ
  grapes_apples_discount : ℝ
  mangoes_oranges_discount : ℝ

/-- Calculates the total amount paid for the fruit purchase. -/
def calculateTotalPaid (purchase : FruitPurchase) : ℝ :=
  sorry

/-- Theorem stating that the total amount paid is correct. -/
theorem total_paid_correct (purchase : FruitPurchase) : 
  purchase.grapes_kg = 6 ∧
  purchase.grapes_price = 74 ∧
  purchase.mangoes_kg = 9 ∧
  purchase.mangoes_price = 59 ∧
  purchase.apples_kg = 4 ∧
  purchase.apples_price = 45 ∧
  purchase.oranges_kg = 12 ∧
  purchase.oranges_price = 32 ∧
  purchase.grapes_tax = 0.1 ∧
  purchase.apples_tax = 0.05 ∧
  purchase.oranges_discount = 5 ∧
  purchase.grapes_apples_discount = 0.07 ∧
  purchase.mangoes_oranges_discount = 0.05 →
  calculateTotalPaid purchase = 1494.482 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_correct_l3102_310240


namespace NUMINAMATH_CALUDE_distinct_divisors_lower_bound_l3102_310274

theorem distinct_divisors_lower_bound (n : ℕ) (A : ℕ) (factors : Finset ℕ) 
  (h1 : factors.card = n)
  (h2 : ∀ x ∈ factors, x > 1)
  (h3 : A = factors.prod id) :
  (Finset.filter (· ∣ A) (Finset.range (A + 1))).card ≥ n * (n - 1) / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_distinct_divisors_lower_bound_l3102_310274


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3102_310230

theorem quadratic_inequality_solution_range (c : ℝ) : 
  (c > 0 ∧ ∃ x : ℝ, 2 * x^2 - 8 * x + c < 0) ↔ (0 < c ∧ c < 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l3102_310230


namespace NUMINAMATH_CALUDE_cube_opposite_face_l3102_310278

/-- Represents a face of the cube -/
inductive Face : Type
| A | B | C | D | E | F

/-- Represents the adjacency relation between faces -/
def adjacent : Face → Face → Prop := sorry

/-- Represents the opposite relation between faces -/
def opposite : Face → Face → Prop := sorry

/-- The theorem stating that F is opposite to A in the given cube configuration -/
theorem cube_opposite_face :
  (adjacent Face.A Face.B) →
  (adjacent Face.A Face.C) →
  (adjacent Face.B Face.D) →
  (opposite Face.A Face.F) := by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l3102_310278


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_l3102_310279

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a box of ping-pong balls -/
structure Box where
  color : String
  count : Nat

/-- Represents the ping-pong ball problem -/
structure PingPongProblem where
  totalBalls : Nat
  boxes : List Box
  sampleSize : Nat

/-- Represents the student selection problem -/
structure StudentProblem where
  totalStudents : Nat
  selectCount : Nat

/-- Determines the optimal sampling method for a given problem -/
def optimalSamplingMethod (p : PingPongProblem ⊕ StudentProblem) : SamplingMethod :=
  match p with
  | .inl _ => SamplingMethod.Stratified
  | .inr _ => SamplingMethod.SimpleRandom

/-- The main theorem stating the optimal sampling methods for both problems -/
theorem optimal_sampling_methods 
  (pingPong : PingPongProblem)
  (student : StudentProblem)
  (h1 : pingPong.totalBalls = 1000)
  (h2 : pingPong.boxes = [
    { color := "red", count := 500 },
    { color := "blue", count := 200 },
    { color := "yellow", count := 300 }
  ])
  (h3 : pingPong.sampleSize = 100)
  (h4 : student.totalStudents = 20)
  (h5 : student.selectCount = 3) :
  (optimalSamplingMethod (.inl pingPong) = SamplingMethod.Stratified) ∧
  (optimalSamplingMethod (.inr student) = SamplingMethod.SimpleRandom) :=
sorry


end NUMINAMATH_CALUDE_optimal_sampling_methods_l3102_310279


namespace NUMINAMATH_CALUDE_five_people_four_rooms_l3102_310286

/-- The number of ways to assign n people to k rooms, where any number of people can be in a room -/
def room_assignments (n k : ℕ) : ℕ := sorry

/-- The specific case for 5 people and 4 rooms -/
theorem five_people_four_rooms : room_assignments 5 4 = 61 := by sorry

end NUMINAMATH_CALUDE_five_people_four_rooms_l3102_310286


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l3102_310236

theorem simplify_nested_expression (x : ℝ) : 1 - (1 + (1 - (1 + (1 - x)))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l3102_310236


namespace NUMINAMATH_CALUDE_house_distance_proof_l3102_310235

/-- Represents the position of a house on a straight street -/
structure HousePosition :=
  (position : ℝ)

/-- The distance between two houses -/
def distance (a b : HousePosition) : ℝ :=
  |a.position - b.position|

theorem house_distance_proof
  (A B V G : HousePosition)
  (h1 : distance A B = 600)
  (h2 : distance V G = 600)
  (h3 : distance A G = 3 * distance B V) :
  distance A G = 900 ∨ distance A G = 1800 := by
  sorry


end NUMINAMATH_CALUDE_house_distance_proof_l3102_310235


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l3102_310267

/-- Given that a + 2b + 3c + 4d = 12, prove that a^2 + b^2 + c^2 + d^2 ≥ 24/5 -/
theorem min_sum_of_squares (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) :
  a^2 + b^2 + c^2 + d^2 ≥ 24/5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l3102_310267


namespace NUMINAMATH_CALUDE_alphabet_letter_count_l3102_310205

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (line_only : ℕ) : 
  total = 40 →
  both = 11 →
  line_only = 24 →
  ∃ (dot_only : ℕ),
    dot_only = 5 ∧
    total = both + line_only + dot_only :=
by sorry

end NUMINAMATH_CALUDE_alphabet_letter_count_l3102_310205


namespace NUMINAMATH_CALUDE_cos_20_minus_cos_40_l3102_310287

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = -1 / (2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_cos_20_minus_cos_40_l3102_310287


namespace NUMINAMATH_CALUDE_shorter_container_radius_l3102_310242

-- Define the containers
structure Container where
  radius : ℝ
  height : ℝ

-- Define the problem
theorem shorter_container_radius 
  (c1 c2 : Container) -- Two containers
  (h_volume : c1.radius ^ 2 * c1.height = c2.radius ^ 2 * c2.height) -- Equal volume
  (h_height : c2.height = 2 * c1.height) -- One height is double the other
  (h_tall_radius : c2.radius = 10) -- Radius of taller container is 10
  : c1.radius = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shorter_container_radius_l3102_310242


namespace NUMINAMATH_CALUDE_intersecting_lines_regions_l3102_310208

/-- The number of regions created by n intersecting lines in a plane -/
def total_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- The number of bounded regions (polygons) created by n intersecting lines in a plane -/
def bounded_regions (n : ℕ) : ℚ :=
  (n^2 - 3*n + 2) / 2

/-- Theorem stating the number of regions and bounded regions created by n intersecting lines -/
theorem intersecting_lines_regions (n : ℕ) :
  (total_regions n = (n^2 + n + 2) / 2) ∧
  (bounded_regions n = (n^2 - 3*n + 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_lines_regions_l3102_310208


namespace NUMINAMATH_CALUDE_A_3_2_equals_6_l3102_310241

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_6 : A 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_A_3_2_equals_6_l3102_310241


namespace NUMINAMATH_CALUDE_toys_bought_after_game_purchase_l3102_310290

def initial_amount : ℕ := 57
def game_cost : ℕ := 27
def toy_cost : ℕ := 6

theorem toys_bought_after_game_purchase : 
  (initial_amount - game_cost) / toy_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_toys_bought_after_game_purchase_l3102_310290


namespace NUMINAMATH_CALUDE_tin_content_in_new_alloy_l3102_310283

theorem tin_content_in_new_alloy 
  (tin_percent_first : Real) 
  (copper_percent_second : Real)
  (zinc_percent_new : Real)
  (weight_first : Real)
  (weight_second : Real)
  (h1 : tin_percent_first = 40)
  (h2 : copper_percent_second = 26)
  (h3 : zinc_percent_new = 30)
  (h4 : weight_first = 150)
  (h5 : weight_second = 250)
  : Real :=
by
  sorry

#check tin_content_in_new_alloy

end NUMINAMATH_CALUDE_tin_content_in_new_alloy_l3102_310283


namespace NUMINAMATH_CALUDE_desk_chair_cost_l3102_310250

theorem desk_chair_cost (cost_A cost_B : ℝ) : 
  (cost_B = cost_A + 40) →
  (4 * cost_A + 5 * cost_B = 1820) →
  (cost_A = 180 ∧ cost_B = 220) := by
sorry

end NUMINAMATH_CALUDE_desk_chair_cost_l3102_310250


namespace NUMINAMATH_CALUDE_circle_center_trajectory_l3102_310275

/-- A moving circle with center (x, y) passes through (1, 0) and is tangent to x = -1 -/
def MovingCircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = (x + 1)^2

/-- The trajectory of the circle's center satisfies y^2 = 4x -/
theorem circle_center_trajectory (x y : ℝ) :
  MovingCircle x y → y^2 = 4*x := by
  sorry

end NUMINAMATH_CALUDE_circle_center_trajectory_l3102_310275


namespace NUMINAMATH_CALUDE_distributive_analogy_l3102_310226

theorem distributive_analogy (a b c : ℝ) (h : c ≠ 0) :
  ((a + b) * c = a * c + b * c) ↔ ((a + b) / c = a / c + b / c) :=
sorry

end NUMINAMATH_CALUDE_distributive_analogy_l3102_310226


namespace NUMINAMATH_CALUDE_jason_pears_l3102_310296

theorem jason_pears (mike_pears jason_pears total_pears : ℕ) 
  (h1 : mike_pears = 8)
  (h2 : total_pears = 15)
  (h3 : total_pears = mike_pears + jason_pears) :
  jason_pears = 7 := by
  sorry

end NUMINAMATH_CALUDE_jason_pears_l3102_310296


namespace NUMINAMATH_CALUDE_quadrilateral_ratio_l3102_310229

theorem quadrilateral_ratio (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + d^2 - a*d = b^2 + c^2 + b*c) (h2 : a^2 + b^2 = c^2 + d^2) :
  (a*b + c*d) / (a*d + b*c) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_ratio_l3102_310229


namespace NUMINAMATH_CALUDE_light_travel_distance_l3102_310271

/-- The distance light travels in one year in kilometers -/
def light_year : ℝ := 9460000000000

/-- The number of years we're considering -/
def years : ℕ := 120

/-- Theorem stating the distance light travels in 120 years -/
theorem light_travel_distance :
  light_year * years = 1.1352e15 := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l3102_310271


namespace NUMINAMATH_CALUDE_initial_friends_count_l3102_310276

theorem initial_friends_count (initial_group : ℕ) (additional_friends : ℕ) (total_people : ℕ) : 
  initial_group + additional_friends = total_people ∧ 
  additional_friends = 3 ∧ 
  total_people = 7 → 
  initial_group = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_friends_count_l3102_310276


namespace NUMINAMATH_CALUDE_rahul_salary_calculation_l3102_310254

def calculate_remaining_salary (initial_salary : ℕ) : ℕ :=
  let after_rent := initial_salary - initial_salary * 20 / 100
  let after_education := after_rent - after_rent * 10 / 100
  let after_clothes := after_education - after_education * 10 / 100
  after_clothes

theorem rahul_salary_calculation :
  calculate_remaining_salary 2125 = 1377 := by
  sorry

end NUMINAMATH_CALUDE_rahul_salary_calculation_l3102_310254


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3102_310265

/-- A line in the two-dimensional plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope -3 and x-intercept (7, 0), the y-intercept is (0, 21). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -3, x_intercept := (7, 0) }
  y_intercept l = (0, 21) := by sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3102_310265


namespace NUMINAMATH_CALUDE_earrings_to_necklace_ratio_l3102_310243

theorem earrings_to_necklace_ratio 
  (total_cost : ℝ) 
  (num_necklaces : ℕ) 
  (single_necklace_cost : ℝ) 
  (h1 : total_cost = 240000)
  (h2 : num_necklaces = 3)
  (h3 : single_necklace_cost = 40000) :
  (total_cost - num_necklaces * single_necklace_cost) / single_necklace_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_earrings_to_necklace_ratio_l3102_310243


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3102_310231

theorem complex_fraction_equality (z : ℂ) (h : z = 2 + I) : 
  (2 * I) / (z - 1) = 1 + I := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3102_310231


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_parallel_to_line_set_l3102_310211

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields for a 3D line

structure Plane3D where
  -- Add necessary fields for a 3D plane

-- Define parallelism between a line and a plane
def line_parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Define a set of parallel lines within a plane
def parallel_lines_in_plane (p : Plane3D) : Set Line3D :=
  sorry

-- Define parallelism between two lines
def lines_parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- The theorem to be proved
theorem line_parallel_to_plane_parallel_to_line_set 
  (a : Line3D) (α : Plane3D) 
  (h : line_parallel_to_plane a α) :
  ∃ (S : Set Line3D), S ⊆ parallel_lines_in_plane α ∧ 
    ∀ l ∈ S, lines_parallel a l :=
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_parallel_to_line_set_l3102_310211


namespace NUMINAMATH_CALUDE_sum_proper_divisors_540_l3102_310209

theorem sum_proper_divisors_540 : 
  (Finset.filter (λ x => x < 540 ∧ 540 % x = 0) (Finset.range 540)).sum id = 1140 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_540_l3102_310209


namespace NUMINAMATH_CALUDE_point_coordinates_l3102_310293

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the distance from a point to the x-axis
def distToXAxis (p : Point) : ℝ := |p.2|

-- Define the distance from a point to the y-axis
def distToYAxis (p : Point) : ℝ := |p.1|

-- Theorem statement
theorem point_coordinates (P : Point) :
  P.2 > 0 →  -- P is above the x-axis
  P.1 < 0 →  -- P is to the left of the y-axis
  distToXAxis P = 4 →  -- P is 4 units away from x-axis
  distToYAxis P = 4 →  -- P is 4 units away from y-axis
  P = (-4, 4) := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_l3102_310293


namespace NUMINAMATH_CALUDE_recipe_total_is_24_l3102_310263

/-- The total cups of ingredients required for Mary's cake recipe -/
def total_ingredients (sugar flour cocoa : ℕ) : ℕ :=
  sugar + flour + cocoa

/-- Theorem stating that the total ingredients for the recipe is 24 cups -/
theorem recipe_total_is_24 :
  total_ingredients 11 8 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_is_24_l3102_310263


namespace NUMINAMATH_CALUDE_scholarship_fund_scientific_notation_l3102_310206

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scholarship_fund_scientific_notation :
  toScientificNotation 445800000 = ScientificNotation.mk 4.458 8 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scholarship_fund_scientific_notation_l3102_310206


namespace NUMINAMATH_CALUDE_log_equation_l3102_310291

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation : (log10 5)^2 + log10 2 * log10 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_l3102_310291


namespace NUMINAMATH_CALUDE_max_min_product_l3102_310233

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (sum_prod_eq : a * b + b * c + c * a = 32) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 4 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_max_min_product_l3102_310233


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3102_310277

theorem necessary_but_not_sufficient_condition (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3102_310277


namespace NUMINAMATH_CALUDE_octal_246_equals_166_l3102_310215

/-- Converts a base-8 digit to its base-10 equivalent -/
def octal_to_decimal (digit : ℕ) : ℕ := digit

/-- Represents a base-8 number as a list of digits -/
def octal_number : List ℕ := [2, 4, 6]

/-- Converts a base-8 number to its base-10 equivalent -/
def octal_to_decimal_conversion (num : List ℕ) : ℕ :=
  List.foldl (fun acc (digit : ℕ) => acc * 8 + octal_to_decimal digit) 0 num.reverse

theorem octal_246_equals_166 :
  octal_to_decimal_conversion octal_number = 166 := by
  sorry

end NUMINAMATH_CALUDE_octal_246_equals_166_l3102_310215


namespace NUMINAMATH_CALUDE_original_triangle_area_l3102_310244

/-- Given a triangle with area A, if its dimensions are quadrupled to form a new triangle
    with an area of 144 square feet, then the area A of the original triangle is 9 square feet. -/
theorem original_triangle_area (A : ℝ) : 
  (∃ (new_triangle : ℝ), new_triangle = 144 ∧ new_triangle = 16 * A) → A = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3102_310244


namespace NUMINAMATH_CALUDE_mrs_crabapple_gift_sequences_l3102_310285

/-- Represents Mrs. Crabapple's class setup -/
structure ClassSetup where
  num_students : ℕ
  meetings_per_week : ℕ
  alternating_gifts : Bool
  starts_with_crabapple : Bool

/-- Calculates the number of different gift recipient sequences for a given class setup -/
def num_gift_sequences (setup : ClassSetup) : ℕ :=
  setup.num_students ^ setup.meetings_per_week

/-- Theorem stating the number of different gift recipient sequences for Mrs. Crabapple's class -/
theorem mrs_crabapple_gift_sequences :
  let setup : ClassSetup := {
    num_students := 11,
    meetings_per_week := 4,
    alternating_gifts := true,
    starts_with_crabapple := true
  }
  num_gift_sequences setup = 14641 := by
  sorry

end NUMINAMATH_CALUDE_mrs_crabapple_gift_sequences_l3102_310285


namespace NUMINAMATH_CALUDE_square_area_quadrupled_l3102_310298

theorem square_area_quadrupled (a : ℝ) (h : a > 0) :
  (2 * a)^2 = 4 * a^2 := by sorry

end NUMINAMATH_CALUDE_square_area_quadrupled_l3102_310298


namespace NUMINAMATH_CALUDE_soccer_team_activities_l3102_310217

/-- The number of activities required for a soccer team practice --/
def total_activities (total_players : ℕ) (goalies : ℕ) : ℕ :=
  let non_goalie_activities := goalies * (total_players - 1)
  2 * non_goalie_activities

theorem soccer_team_activities :
  total_activities 25 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_activities_l3102_310217


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_nonempty_iff_l3102_310262

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 1) = {x | 1 < x ∧ x < 3} := by sorry

-- Part 2
theorem intersection_A_B_nonempty_iff (k : ℝ) :
  (A ∩ B k).Nonempty ↔ k ≥ -1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_intersection_A_B_nonempty_iff_l3102_310262


namespace NUMINAMATH_CALUDE_certain_number_problem_l3102_310299

theorem certain_number_problem (x y z : ℝ) : 
  x + y = 15 →
  y = 7 →
  3 * x = z * y - 11 →
  z = 5 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3102_310299


namespace NUMINAMATH_CALUDE_blouse_cost_l3102_310295

/-- Given information about Jane's purchase of skirts and blouses, prove the cost of each blouse. -/
theorem blouse_cost (num_skirts : ℕ) (skirt_price : ℕ) (num_blouses : ℕ) (total_paid : ℕ) (change : ℕ) :
  num_skirts = 2 →
  skirt_price = 13 →
  num_blouses = 3 →
  total_paid = 100 →
  change = 56 →
  (total_paid - change - num_skirts * skirt_price) / num_blouses = 6 := by
  sorry

#eval (100 - 56 - 2 * 13) / 3

end NUMINAMATH_CALUDE_blouse_cost_l3102_310295


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l3102_310249

theorem sqrt_three_irrational :
  (∃ (q : ℚ), (1 : ℝ) / 3 = ↑q) ∧ 
  (∃ (q : ℚ), (3.14 : ℝ) = ↑q) ∧ 
  (∃ (q : ℚ), Real.sqrt 9 = ↑q) →
  ¬ ∃ (q : ℚ), Real.sqrt 3 = ↑q :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l3102_310249


namespace NUMINAMATH_CALUDE_factorize_ax_minus_ay_l3102_310203

theorem factorize_ax_minus_ay (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorize_ax_minus_ay_l3102_310203


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l3102_310292

-- Define proposition p
def p (x y : ℝ) : Prop := x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)

-- Define proposition q
def q (m : ℝ) : Prop := m > -2 → ∃ x : ℝ, x^2 + 2*x - m = 0

-- Theorem statement
theorem p_or_q_is_true :
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) →
  (∃ m : ℝ, m > -2 ∧ ¬(∃ x : ℝ, x^2 + 2*x - m = 0)) →
  ∀ x y m : ℝ, p x y ∨ q m :=
sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l3102_310292


namespace NUMINAMATH_CALUDE_age_of_15th_person_l3102_310223

theorem age_of_15th_person (total_persons : Nat) (avg_age_all : Nat) (group1_size : Nat) 
  (avg_age_group1 : Nat) (group2_size : Nat) (avg_age_group2 : Nat) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_size = 5 →
  avg_age_group1 = 14 →
  group2_size = 9 →
  avg_age_group2 = 16 →
  (total_persons * avg_age_all) = 
    (group1_size * avg_age_group1) + (group2_size * avg_age_group2) + 56 :=
by
  sorry

#check age_of_15th_person

end NUMINAMATH_CALUDE_age_of_15th_person_l3102_310223


namespace NUMINAMATH_CALUDE_max_sides_convex_polygon_four_obtuse_l3102_310219

/-- Represents a convex polygon with n sides and exactly four obtuse angles -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 0
  obtuse_angles : ℕ
  obtuse_count : obtuse_angles = 4

/-- The sum of interior angles of an n-sided polygon is (n-2) * 180 degrees -/
def interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- An obtuse angle is greater than 90 degrees and less than 180 degrees -/
def is_obtuse (angle : ℝ) : Prop := 90 < angle ∧ angle < 180

/-- An acute angle is greater than 0 degrees and less than 90 degrees -/
def is_acute (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

/-- The maximum number of sides for a convex polygon with exactly four obtuse angles is 7 -/
theorem max_sides_convex_polygon_four_obtuse :
  ∀ n : ℕ, ConvexPolygon n → n ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_sides_convex_polygon_four_obtuse_l3102_310219


namespace NUMINAMATH_CALUDE_equal_number_of_buyers_l3102_310239

def pencil_cost : ℕ → Prop := λ c => c > 0 ∧ c ∣ 227 ∧ c ∣ 221

theorem equal_number_of_buyers (c : ℕ) (h : pencil_cost c) :
  (221 / c : ℕ) = (227 / c : ℕ) :=
by sorry

#check equal_number_of_buyers

end NUMINAMATH_CALUDE_equal_number_of_buyers_l3102_310239


namespace NUMINAMATH_CALUDE_oatmeal_raisin_cookies_l3102_310248

/-- Given a class of students and cookie preferences, calculate the number of oatmeal raisin cookies to be made. -/
theorem oatmeal_raisin_cookies (total_students : ℕ) (cookies_per_student : ℕ) (oatmeal_raisin_percentage : ℚ) : 
  total_students = 40 → 
  cookies_per_student = 2 → 
  oatmeal_raisin_percentage = 1/10 →
  (total_students : ℚ) * oatmeal_raisin_percentage * cookies_per_student = 8 := by
  sorry

#check oatmeal_raisin_cookies

end NUMINAMATH_CALUDE_oatmeal_raisin_cookies_l3102_310248


namespace NUMINAMATH_CALUDE_truck_weight_problem_l3102_310228

theorem truck_weight_problem (truck_weight trailer_weight : ℝ) : 
  truck_weight + trailer_weight = 7000 →
  trailer_weight = 0.5 * truck_weight - 200 →
  truck_weight = 4800 := by
sorry

end NUMINAMATH_CALUDE_truck_weight_problem_l3102_310228


namespace NUMINAMATH_CALUDE_jake_money_left_jake_final_amount_l3102_310232

theorem jake_money_left (initial_amount : ℝ) (motorcycle_percent : ℝ) 
  (concert_percent : ℝ) (investment_percent : ℝ) (investment_loss_percent : ℝ) : ℝ :=
  let after_motorcycle := initial_amount * (1 - motorcycle_percent)
  let after_concert := after_motorcycle * (1 - concert_percent)
  let investment := after_concert * investment_percent
  let investment_loss := investment * investment_loss_percent
  let final_amount := after_concert - investment + (investment - investment_loss)
  final_amount

theorem jake_final_amount : 
  jake_money_left 5000 0.35 0.25 0.40 0.20 = 1462.50 := by
  sorry

end NUMINAMATH_CALUDE_jake_money_left_jake_final_amount_l3102_310232


namespace NUMINAMATH_CALUDE_final_fish_count_l3102_310288

def fish_count (initial : ℕ) (days : ℕ) : ℕ :=
  let day1 := initial
  let day2 := day1 * 2
  let day3 := day2 * 2 - (day2 * 2) / 3
  let day4 := day3 * 2
  let day5 := day4 * 2 - (day4 * 2) / 4
  let day6 := day5 * 2
  let day7 := day6 * 2 + 15
  day7

theorem final_fish_count :
  fish_count 6 7 = 207 :=
by sorry

end NUMINAMATH_CALUDE_final_fish_count_l3102_310288
