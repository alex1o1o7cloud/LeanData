import Mathlib

namespace NUMINAMATH_CALUDE_vehicle_tire_usage_l2715_271514

/-- Calculates the miles each tire is used given the total miles traveled, 
    number of tires, and number of tires used at a time. -/
def milesPerTire (totalMiles : ℕ) (numTires : ℕ) (tiresUsed : ℕ) : ℚ :=
  (totalMiles * tiresUsed : ℚ) / numTires

/-- Proves that given the conditions of the problem, each tire is used for 32,000 miles -/
theorem vehicle_tire_usage :
  let totalMiles : ℕ := 48000
  let numTires : ℕ := 6
  let tiresUsed : ℕ := 4
  milesPerTire totalMiles numTires tiresUsed = 32000 := by
  sorry

#eval milesPerTire 48000 6 4

end NUMINAMATH_CALUDE_vehicle_tire_usage_l2715_271514


namespace NUMINAMATH_CALUDE_seventeen_in_binary_l2715_271561

theorem seventeen_in_binary : 
  (17 : ℕ).digits 2 = [1, 0, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_seventeen_in_binary_l2715_271561


namespace NUMINAMATH_CALUDE_brownies_in_container_l2715_271572

/-- Represents the problem of calculating the fraction of remaining brownies in the container --/
theorem brownies_in_container (batches : ℕ) (brownies_per_batch : ℕ) 
  (bake_sale_fraction : ℚ) (given_out : ℕ) : 
  batches = 10 →
  brownies_per_batch = 20 →
  bake_sale_fraction = 3/4 →
  given_out = 20 →
  let total_brownies := batches * brownies_per_batch
  let bake_sale_brownies := (bake_sale_fraction * brownies_per_batch) * batches
  let remaining_after_bake_sale := total_brownies - bake_sale_brownies
  let remaining_after_given_out := remaining_after_bake_sale - given_out
  (remaining_after_given_out : ℚ) / remaining_after_bake_sale = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_brownies_in_container_l2715_271572


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2715_271538

theorem polynomial_divisibility : ∃ q : Polynomial ℝ, 4 * X^2 - 3 * X - 10 = (X - 2) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2715_271538


namespace NUMINAMATH_CALUDE_min_value_abc_l2715_271595

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ a₀ + 3 * b₀ + 9 * c₀ = 27 :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l2715_271595


namespace NUMINAMATH_CALUDE_min_C_for_inequality_l2715_271571

/-- The minimum value of C that satisfies the given inequality for all x and any α, β where |α| ≤ 1 and |β| ≤ 1 -/
theorem min_C_for_inequality : 
  (∃ (C : ℝ), ∀ (x α β : ℝ), |α| ≤ 1 → |β| ≤ 1 → |α * Real.sin x + β * Real.cos (4 * x)| ≤ C) ∧ 
  (∀ (C : ℝ), (∀ (x α β : ℝ), |α| ≤ 1 → |β| ≤ 1 → |α * Real.sin x + β * Real.cos (4 * x)| ≤ C) → C ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_C_for_inequality_l2715_271571


namespace NUMINAMATH_CALUDE_range_of_a_l2715_271575

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 5, a * x^2 - x - 4 > 0) → a > 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2715_271575


namespace NUMINAMATH_CALUDE_ln_sufficient_not_necessary_for_exp_l2715_271579

theorem ln_sufficient_not_necessary_for_exp (x : ℝ) :
  (∀ x, (Real.log x > 0 → Real.exp x > 1)) ∧
  (∃ x, Real.exp x > 1 ∧ ¬(Real.log x > 0)) :=
sorry

end NUMINAMATH_CALUDE_ln_sufficient_not_necessary_for_exp_l2715_271579


namespace NUMINAMATH_CALUDE_lemon_pie_angle_l2715_271570

theorem lemon_pie_angle (total : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ) 
  (h_total : total = 40)
  (h_chocolate : chocolate = 15)
  (h_apple : apple = 10)
  (h_blueberry : blueberry = 5)
  (h_remaining : total - (chocolate + apple + blueberry) = 2 * (total - (chocolate + apple + blueberry)) / 2) :
  (((total - (chocolate + apple + blueberry)) / 2) : ℚ) / total * 360 = 45 := by
sorry

end NUMINAMATH_CALUDE_lemon_pie_angle_l2715_271570


namespace NUMINAMATH_CALUDE_carrots_grown_total_l2715_271560

/-- The number of carrots grown by Joan -/
def joans_carrots : ℕ := 29

/-- The number of carrots grown by Jessica -/
def jessicas_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := joans_carrots + jessicas_carrots

theorem carrots_grown_total : total_carrots = 40 := by
  sorry

end NUMINAMATH_CALUDE_carrots_grown_total_l2715_271560


namespace NUMINAMATH_CALUDE_triangle_theorem_l2715_271541

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.c * Real.cos t.C) :
  -- Part 1: Angle C is 60 degrees (π/3 radians)
  t.C = π / 3 ∧
  -- Part 2: If c = 2, the maximum area is √3
  (t.c = 2 → ∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2715_271541


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_three_subset_of_complement_implies_m_range_l2715_271530

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Theorem 1: If A ∩ B = [1, 3], then m = 3
theorem intersection_implies_m_equals_three (m : ℝ) :
  A ∩ B m = Set.Icc 1 3 → m = 3 := by sorry

-- Theorem 2: If A ⊆ (ℝ \ B), then m > 5 or m < -3
theorem subset_of_complement_implies_m_range (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_three_subset_of_complement_implies_m_range_l2715_271530


namespace NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l2715_271599

theorem tan_value_fourth_quadrant (α : Real) :
  (α ∈ Set.Icc (3 * π / 2) (2 * π)) →  -- α is in the fourth quadrant
  (Real.sin α + Real.cos α = 1 / 5) →  -- given condition
  Real.tan α = -3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_fourth_quadrant_l2715_271599


namespace NUMINAMATH_CALUDE_hundredth_letter_is_A_l2715_271585

def pattern : ℕ → Char
| n => match n % 3 with
  | 0 => 'C'
  | 1 => 'A'
  | _ => 'B'

theorem hundredth_letter_is_A : pattern 100 = 'A' := by
  sorry

end NUMINAMATH_CALUDE_hundredth_letter_is_A_l2715_271585


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2715_271556

theorem min_sum_of_squares (x y z : ℝ) (h : x*y + y*z + x*z = 4) :
  x^2 + y^2 + z^2 ≥ 4 ∧ ∃ a b c : ℝ, a*b + b*c + a*c = 4 ∧ a^2 + b^2 + c^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2715_271556


namespace NUMINAMATH_CALUDE_statue_weight_theorem_l2715_271505

/-- Calculates the weight of a marble statue after a series of reductions --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let week1 := initial_weight * (1 - 0.35)
  let week2 := week1 * (1 - 0.20)
  let week3 := week2 * (1 - 0.05)^5
  let after_rain := week3 * (1 - 0.02)
  let week4 := after_rain * (1 - 0.08)
  let final := week4 * (1 - 0.25)
  final

/-- The weight of the final statue is approximately 136.04 kg --/
theorem statue_weight_theorem (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |final_statue_weight 500 - 136.04| < δ ∧ δ < ε :=
sorry

end NUMINAMATH_CALUDE_statue_weight_theorem_l2715_271505


namespace NUMINAMATH_CALUDE_birds_nest_building_distance_l2715_271543

/-- Calculates the total distance covered by birds making round trips to collect nest materials. -/
def total_distance_covered (num_birds : ℕ) (num_trips : ℕ) (distance_to_materials : ℕ) : ℕ :=
  num_birds * num_trips * (2 * distance_to_materials)

/-- Theorem stating that two birds making 10 round trips each to collect materials 200 miles away cover a total distance of 8000 miles. -/
theorem birds_nest_building_distance :
  total_distance_covered 2 10 200 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_birds_nest_building_distance_l2715_271543


namespace NUMINAMATH_CALUDE_oplus_inequality_l2715_271531

def oplus (x y : ℝ) : ℝ := (x - y)^2

theorem oplus_inequality : ∃ x y : ℝ, 2 * (oplus x y) ≠ oplus (2*x) (2*y) := by
  sorry

end NUMINAMATH_CALUDE_oplus_inequality_l2715_271531


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_12_points_l2715_271555

/-- The number of different convex quadrilaterals that can be drawn from n distinct points
    on the circumference of a circle, where each vertex of the quadrilateral must be one
    of these n points. -/
def convex_quadrilaterals (n : ℕ) : ℕ := Nat.choose n 4

/-- Theorem stating that the number of convex quadrilaterals from 12 points is 3960 -/
theorem convex_quadrilaterals_12_points :
  convex_quadrilaterals 12 = 3960 := by
  sorry

#eval convex_quadrilaterals 12

end NUMINAMATH_CALUDE_convex_quadrilaterals_12_points_l2715_271555


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l2715_271593

theorem product_remainder_mod_five : (1657 * 2024 * 1953 * 1865) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l2715_271593


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l2715_271518

def total_balls : ℕ := 18
def red_balls : ℕ := 4
def yellow_balls : ℕ := 5
def green_balls : ℕ := 6
def blue_balls : ℕ := 3
def drawn_balls : ℕ := 4

def favorable_outcomes : ℕ := Nat.choose green_balls 2 * Nat.choose red_balls 1 * Nat.choose blue_balls 1

def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

theorem probability_of_specific_draw :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 17 :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l2715_271518


namespace NUMINAMATH_CALUDE_percentage_of_students_liking_donuts_l2715_271544

theorem percentage_of_students_liking_donuts : 
  ∀ (total_donuts : ℕ) (total_students : ℕ) (donuts_per_student : ℕ),
    total_donuts = 4 * 12 →
    total_students = 30 →
    donuts_per_student = 2 →
    (((total_donuts / donuts_per_student) / total_students) * 100 : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_students_liking_donuts_l2715_271544


namespace NUMINAMATH_CALUDE_solve_bath_towels_problem_l2715_271573

def bath_towels_problem (kylie_towels husband_towels : ℕ) 
  (towels_per_load loads : ℕ) : Prop :=
  let total_towels := towels_per_load * loads
  let daughters_towels := total_towels - (kylie_towels + husband_towels)
  daughters_towels = 6

theorem solve_bath_towels_problem : 
  bath_towels_problem 3 3 4 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_bath_towels_problem_l2715_271573


namespace NUMINAMATH_CALUDE_certain_number_value_l2715_271589

theorem certain_number_value (x y z : ℝ) 
  (h1 : y = 1.10 * z) 
  (h2 : x = 0.90 * y) 
  (h3 : x = 123.75) : 
  z = 125 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2715_271589


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2715_271553

theorem inequality_system_solution (x : ℝ) :
  (x - 4 > (3/2) * x - 3) ∧
  ((2 + x) / 3 - 1 ≤ (1 + x) / 2) →
  -5 ≤ x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2715_271553


namespace NUMINAMATH_CALUDE_kite_side_lengths_l2715_271546

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length. -/
structure Kite where
  a : ℝ  -- First diagonal
  b : ℝ  -- Second diagonal
  k : ℝ  -- Half of the perimeter
  x : ℝ  -- Length of one side
  y : ℝ  -- Length of the other side

/-- Properties of the kite based on the given conditions -/
def kite_properties (q : Kite) : Prop :=
  q.a = 6 ∧ q.b = 25/4 ∧ q.k = 35/4 ∧ q.x + q.y = q.k

/-- The theorem stating the side lengths of the kite -/
theorem kite_side_lengths (q : Kite) (h : kite_properties q) :
  q.x = 5 ∧ q.y = 15/4 :=
sorry

end NUMINAMATH_CALUDE_kite_side_lengths_l2715_271546


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l2715_271581

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The total number of possible triangles formed by choosing 3 vertices from n vertices -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles with exactly one side being a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides being sides of the decagon (i.e., formed by three consecutive vertices) -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side being a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l2715_271581


namespace NUMINAMATH_CALUDE_max_sum_xyz_l2715_271535

theorem max_sum_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l2715_271535


namespace NUMINAMATH_CALUDE_unit_conversions_l2715_271500

-- Define unit conversion factors
def cm_to_dm : ℚ := 1 / 10
def hectare_to_km2 : ℚ := 1 / 100
def yuan_to_jiao : ℚ := 10
def yuan_to_fen : ℚ := 100
def hectare_to_m2 : ℚ := 10000
def dm_to_m : ℚ := 1 / 10
def m_to_cm : ℚ := 100

-- Theorem statement
theorem unit_conversions :
  (70000 * cm_to_dm^2 = 700) ∧
  (800 * hectare_to_km2 = 8) ∧
  (1.65 * yuan_to_jiao = 16.5) ∧
  (400 * hectare_to_m2 = 4000000) ∧
  (0.57 * yuan_to_fen = 57) ∧
  (5000 * dm_to_m^2 = 50) ∧
  (60000 / hectare_to_m2 = 6) ∧
  (9 * m_to_cm = 900) :=
by sorry

end NUMINAMATH_CALUDE_unit_conversions_l2715_271500


namespace NUMINAMATH_CALUDE_inverse_proportionality_l2715_271524

theorem inverse_proportionality (α β : ℚ) (h : α ≠ 0 ∧ β ≠ 0) :
  (∃ k : ℚ, k ≠ 0 ∧ α * β = k) →
  (α = -4 ∧ β = -8) →
  (β = 12 → α = 8/3) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l2715_271524


namespace NUMINAMATH_CALUDE_perpendicular_transitivity_l2715_271567

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)

-- Define the different relation
variable (different : ∀ {α : Type}, α → α → Prop)

theorem perpendicular_transitivity 
  (α β γ : Plane) (m n l : Line)
  (h_diff_planes : different α β ∧ different β γ ∧ different α γ)
  (h_diff_lines : different m n ∧ different n l ∧ different m l)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_transitivity_l2715_271567


namespace NUMINAMATH_CALUDE_inequality_proof_l2715_271597

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a + 1 / b > b + 1 / a := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2715_271597


namespace NUMINAMATH_CALUDE_rectangular_field_shortcut_l2715_271594

theorem rectangular_field_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_shortcut_l2715_271594


namespace NUMINAMATH_CALUDE_value_of_2x_minus_y_l2715_271523

theorem value_of_2x_minus_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 4) (hxy : x > y) :
  2 * x - y = 10 := by
sorry

end NUMINAMATH_CALUDE_value_of_2x_minus_y_l2715_271523


namespace NUMINAMATH_CALUDE_johns_share_ratio_l2715_271587

theorem johns_share_ratio (total : ℕ) (johns_share : ℕ) 
  (h1 : total = 4800) (h2 : johns_share = 1600) : 
  johns_share / total = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_johns_share_ratio_l2715_271587


namespace NUMINAMATH_CALUDE_operation_result_l2715_271532

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the operation
def op : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.four
  | Element.one, Element.four => Element.three
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.three
  | Element.two, Element.three => Element.two
  | Element.two, Element.four => Element.four
  | Element.three, Element.one => Element.four
  | Element.three, Element.two => Element.two
  | Element.three, Element.three => Element.one
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.four
  | Element.four, Element.three => Element.three
  | Element.four, Element.four => Element.two

theorem operation_result : 
  op (op Element.three Element.one) (op Element.four Element.two) = Element.two := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l2715_271532


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l2715_271536

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l2715_271536


namespace NUMINAMATH_CALUDE_side_xy_length_l2715_271542

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ) := X + Y + Z = 180

-- Define the right angle
def RightAngle (Z : ℝ) := Z = 90

-- Define the area of the triangle
def TriangleArea (A : ℝ) := A = 36

-- Define the angles of the triangle
def AngleX (X : ℝ) := X = 30
def AngleY (Y : ℝ) := Y = 60

-- Theorem statement
theorem side_xy_length 
  (X Y Z A : ℝ) 
  (tri : Triangle X Y Z) 
  (right : RightAngle Z) 
  (area : TriangleArea A) 
  (angleX : AngleX X) 
  (angleY : AngleY Y) : 
  ∃ (XY : ℝ), XY = Real.sqrt (36 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_side_xy_length_l2715_271542


namespace NUMINAMATH_CALUDE_area_enclosed_by_line_and_curve_l2715_271598

/-- Given that the binomial coefficients of the third and fourth terms 
    in the expansion of (x - 2/x)^n are equal, prove that the area enclosed 
    by the line y = nx and the curve y = x^2 is 125/6 -/
theorem area_enclosed_by_line_and_curve (n : ℕ) : 
  (Nat.choose n 2 = Nat.choose n 3) → 
  (∫ (x : ℝ) in (0)..(5), n * x - x^2) = 125 / 6 := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_by_line_and_curve_l2715_271598


namespace NUMINAMATH_CALUDE_recipe_total_ingredients_l2715_271526

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio :=
  (butter : ℕ)
  (flour : ℕ)
  (sugar : ℕ)

/-- Calculates the total cups of ingredients given a recipe ratio and cups of sugar -/
def totalIngredients (ratio : RecipeRatio) (sugarCups : ℕ) : ℕ :=
  let partSize := sugarCups / ratio.sugar
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given recipe ratio and sugar amount, the total ingredients is 28 cups -/
theorem recipe_total_ingredients :
  let ratio : RecipeRatio := ⟨1, 8, 5⟩
  totalIngredients ratio 10 = 28 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_ingredients_l2715_271526


namespace NUMINAMATH_CALUDE_rubys_math_homework_l2715_271574

/-- Ruby's math homework problem -/
theorem rubys_math_homework :
  ∀ (ruby_math ruby_reading nina_math nina_reading : ℕ),
  ruby_reading = 2 →
  nina_math = 4 * ruby_math →
  nina_reading = 8 * ruby_reading →
  nina_math + nina_reading = 48 →
  ruby_math = 6 := by
sorry

end NUMINAMATH_CALUDE_rubys_math_homework_l2715_271574


namespace NUMINAMATH_CALUDE_system_solution_l2715_271528

theorem system_solution : 
  ∀ x y : ℝ, x > 0 → y > 0 →
  (3*y - Real.sqrt (y/x) - 6*Real.sqrt (x*y) + 2 = 0 ∧ 
   x^2 + 81*x^2*y^4 = 2*y^2) →
  ((x = 1/3 ∧ y = 1/3) ∨ 
   (x = Real.sqrt (Real.sqrt 31) / 12 ∧ y = Real.sqrt (Real.sqrt 31) / 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2715_271528


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l2715_271517

theorem existence_of_special_integers :
  ∃ (A : Fin 10 → ℕ+),
    (∀ i j : Fin 10, i ≠ j → ¬(A i ∣ A j)) ∧
    (∀ i j : Fin 10, i ≠ j → (A i)^2 ∣ A j) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l2715_271517


namespace NUMINAMATH_CALUDE_find_A_l2715_271577

theorem find_A : ∃ A : ℤ, A + 19 = 47 ∧ A = 28 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l2715_271577


namespace NUMINAMATH_CALUDE_customers_stayed_behind_l2715_271545

theorem customers_stayed_behind (initial_customers : ℕ) 
  (h1 : initial_customers = 11) 
  (stayed : ℕ) 
  (left : ℕ) 
  (h2 : left = stayed + 5) 
  (h3 : stayed + left = initial_customers) : 
  stayed = 3 := by
  sorry

end NUMINAMATH_CALUDE_customers_stayed_behind_l2715_271545


namespace NUMINAMATH_CALUDE_percentage_lost_is_25_percent_l2715_271578

/-- Represents the number of kettles of hawks -/
def num_kettles : ℕ := 6

/-- Represents the average number of pregnancies per kettle -/
def pregnancies_per_kettle : ℕ := 15

/-- Represents the number of babies per pregnancy -/
def babies_per_pregnancy : ℕ := 4

/-- Represents the expected number of babies this season -/
def expected_babies : ℕ := 270

/-- Calculates the percentage of baby hawks lost -/
def percentage_lost : ℚ :=
  let total_babies := num_kettles * pregnancies_per_kettle * babies_per_pregnancy
  let lost_babies := total_babies - expected_babies
  (lost_babies : ℚ) / (total_babies : ℚ) * 100

/-- Theorem stating that the percentage of baby hawks lost is 25% -/
theorem percentage_lost_is_25_percent : percentage_lost = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_lost_is_25_percent_l2715_271578


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2715_271507

theorem arithmetic_mean_problem (m n : ℝ) 
  (h1 : (m + 2*n) / 2 = 4) 
  (h2 : (2*m + n) / 2 = 5) : 
  (m + n) / 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2715_271507


namespace NUMINAMATH_CALUDE_reading_calculation_l2715_271584

/-- Represents the daily reading rate in characters -/
def daily_rate : ℕ := 800

/-- Calculates the number of characters read in a given number of days -/
def chars_read_in_days (days : ℕ) : ℕ := daily_rate * days

/-- Calculates the number of characters read in a given number of weeks -/
def chars_read_in_weeks (weeks : ℕ) : ℕ := chars_read_in_days (7 * weeks)

/-- Approximates a number by omitting digits following the ten-thousands place -/
def approximate_to_ten_thousands (n : ℕ) : ℕ := n / 10000

theorem reading_calculation :
  (chars_read_in_days 7 = 5600) ∧
  (chars_read_in_weeks 20 = 112000) ∧
  (approximate_to_ten_thousands (chars_read_in_weeks 20) = 11) := by
  sorry

end NUMINAMATH_CALUDE_reading_calculation_l2715_271584


namespace NUMINAMATH_CALUDE_triangle_solution_l2715_271583

theorem triangle_solution (c t r : ℝ) (hc : c = 30) (ht : t = 336) (hr : r = 8) :
  ∃ (a b : ℝ),
    a + b + c = 2 * (t / r) ∧
    t = r * (a + b + c) / 2 ∧
    t^2 = (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c) ∧
    a = 26 ∧
    b = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_solution_l2715_271583


namespace NUMINAMATH_CALUDE_homothety_composition_l2715_271588

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a homothety in 3D space
structure Homothety3D where
  center : Point3D
  ratio : ℝ

-- Define a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Function to compose two homotheties
def compose_homotheties (h1 h2 : Homothety3D) : Homothety3D :=
  sorry

-- Function to check if a point lies on a line
def point_on_line (p : Point3D) (l : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem homothety_composition 
  (h1 h2 : Homothety3D) 
  (l : Line3D) :
  let h3 := compose_homotheties h1 h2
  point_on_line h3.center l ∧ 
  h3.ratio = h1.ratio * h2.ratio ∧
  point_on_line h1.center l ∧
  point_on_line h2.center l :=
sorry

end NUMINAMATH_CALUDE_homothety_composition_l2715_271588


namespace NUMINAMATH_CALUDE_tangent_square_area_l2715_271558

/-- Given a 6 by 6 square with semicircles on its sides, prove the area of the tangent square ABCD -/
theorem tangent_square_area :
  -- Original square side length
  let original_side : ℝ := 6
  -- Radius of semicircles (half of original side)
  let semicircle_radius : ℝ := original_side / 2
  -- Side length of square ABCD (original side + 2 * radius)
  let abcd_side : ℝ := original_side + 2 * semicircle_radius
  -- Area of square ABCD
  let abcd_area : ℝ := abcd_side ^ 2
  -- The area of square ABCD is 144
  abcd_area = 144 := by sorry

end NUMINAMATH_CALUDE_tangent_square_area_l2715_271558


namespace NUMINAMATH_CALUDE_spades_in_deck_l2715_271592

/-- 
Given a deck of 52 cards containing some spades, prove that if the probability 
of not drawing a spade on the first draw is 0.75, then there are 13 spades in the deck.
-/
theorem spades_in_deck (total_cards : ℕ) (prob_not_spade : ℚ) (num_spades : ℕ) : 
  total_cards = 52 →
  prob_not_spade = 3/4 →
  (total_cards - num_spades : ℚ) / total_cards = prob_not_spade →
  num_spades = 13 := by
  sorry

end NUMINAMATH_CALUDE_spades_in_deck_l2715_271592


namespace NUMINAMATH_CALUDE_square_root_equation_l2715_271537

theorem square_root_equation (x : ℝ) : 
  (Real.sqrt x / Real.sqrt 0.64) + (Real.sqrt 1.44 / Real.sqrt 0.49) = 3.0892857142857144 → x = 1.21 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l2715_271537


namespace NUMINAMATH_CALUDE_motor_pool_vehicles_l2715_271569

theorem motor_pool_vehicles (x y : ℕ) : 
  x + y < 18 →
  y < 2 * x →
  x + 4 < y →
  (x = 6 ∧ y = 11) ∨ (∀ a b : ℕ, (a + b < 18 ∧ b < 2 * a ∧ a + 4 < b) → (a ≠ x ∨ b ≠ y)) :=
by sorry

end NUMINAMATH_CALUDE_motor_pool_vehicles_l2715_271569


namespace NUMINAMATH_CALUDE_cyclist_pedestrian_meeting_point_l2715_271525

/-- Given three points A, B, C on a line, with AB = 3 km and BC = 4 km,
    a cyclist starting from A towards C, and a pedestrian starting from B towards A,
    prove that they meet at a point 2.1 km from A if they arrive at their
    destinations simultaneously. -/
theorem cyclist_pedestrian_meeting_point
  (A B C : ℝ) -- Points represented as real numbers
  (h_order : A < B ∧ B < C) -- Points are in order
  (h_AB : B - A = 3) -- Distance AB is 3 km
  (h_BC : C - B = 4) -- Distance BC is 4 km
  (cyclist_speed pedestrian_speed : ℝ) -- Speeds of cyclist and pedestrian
  (h_speeds_positive : cyclist_speed > 0 ∧ pedestrian_speed > 0) -- Speeds are positive
  (h_simultaneous_arrival : (C - A) / cyclist_speed = (B - A) / pedestrian_speed) -- Simultaneous arrival
  : ∃ (D : ℝ), D - A = 21/10 ∧ A < D ∧ D < B :=
sorry

end NUMINAMATH_CALUDE_cyclist_pedestrian_meeting_point_l2715_271525


namespace NUMINAMATH_CALUDE_mike_camera_purchase_l2715_271582

/-- Given:
  - The new camera model costs 30% more than the current model
  - The old camera costs $4000
  - Mike gets $200 off a $400 lens

  Prove that Mike paid $5400 for the camera and lens. -/
theorem mike_camera_purchase (old_camera_cost : ℝ) (lens_cost : ℝ) (lens_discount : ℝ) :
  old_camera_cost = 4000 →
  lens_cost = 400 →
  lens_discount = 200 →
  let new_camera_cost := old_camera_cost * 1.3
  let discounted_lens_cost := lens_cost - lens_discount
  new_camera_cost + discounted_lens_cost = 5400 := by
  sorry

end NUMINAMATH_CALUDE_mike_camera_purchase_l2715_271582


namespace NUMINAMATH_CALUDE_circle_triangle_count_l2715_271506

/-- The number of points on the circle's circumference -/
def n : ℕ := 10

/-- The total number of triangles that can be formed from n points -/
def total_triangles (n : ℕ) : ℕ := n.choose 3

/-- The number of triangles with consecutive vertices -/
def consecutive_triangles (n : ℕ) : ℕ := n

/-- The number of valid triangles (no consecutive vertices) -/
def valid_triangles (n : ℕ) : ℕ := total_triangles n - consecutive_triangles n

theorem circle_triangle_count :
  valid_triangles n = 110 :=
sorry

end NUMINAMATH_CALUDE_circle_triangle_count_l2715_271506


namespace NUMINAMATH_CALUDE_quadratic_roots_imaginary_l2715_271559

theorem quadratic_roots_imaginary (a b c a₁ b₁ c₁ : ℝ) : 
  let discriminant := 4 * ((a * a₁ + b * b₁ + c * c₁)^2 - (a^2 + b^2 + c^2) * (a₁^2 + b₁^2 + c₁^2))
  discriminant ≤ 0 ∧ 
  (discriminant = 0 ↔ ∃ (k : ℝ), k ≠ 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_imaginary_l2715_271559


namespace NUMINAMATH_CALUDE_arthur_purchase_cost_l2715_271576

/-- The cost of Arthur's purchases on two days -/
theorem arthur_purchase_cost
  (hamburger_cost : ℝ)
  (hot_dog_cost : ℝ)
  (day1_total : ℝ)
  (h_hot_dog_cost : hot_dog_cost = 1)
  (h_day1_equation : 3 * hamburger_cost + 4 * hot_dog_cost = day1_total)
  (h_day1_total : day1_total = 10) :
  2 * hamburger_cost + 3 * hot_dog_cost = 7 :=
by sorry

end NUMINAMATH_CALUDE_arthur_purchase_cost_l2715_271576


namespace NUMINAMATH_CALUDE_sum_and_fraction_difference_l2715_271540

theorem sum_and_fraction_difference (x y : ℝ) 
  (sum_eq : x + y = 450)
  (fraction_eq : x / y = 0.8) : 
  y - x = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_and_fraction_difference_l2715_271540


namespace NUMINAMATH_CALUDE_circle_k_range_l2715_271511

/-- Represents the equation of a potential circle -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 2*y + 5*k = 0

/-- Checks if the equation represents a valid circle -/
def is_circle (k : ℝ) : Prop :=
  ∃ (x₀ y₀ r : ℝ), r > 0 ∧ ∀ (x y : ℝ),
    circle_equation x y k ↔ (x - x₀)^2 + (y - y₀)^2 = r^2

/-- The theorem stating the range of k for which the equation represents a circle -/
theorem circle_k_range :
  ∀ k : ℝ, is_circle k ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_k_range_l2715_271511


namespace NUMINAMATH_CALUDE_find_defective_box_l2715_271568

/-- Represents the number of boxes -/
def num_boxes : ℕ := 9

/-- Represents the number of standard parts per box -/
def standard_parts_per_box : ℕ := 10

/-- Represents the number of defective parts in one box -/
def defective_parts : ℕ := 10

/-- Represents the weight of a standard part in grams -/
def standard_weight : ℕ := 100

/-- Represents the weight of a defective part in grams -/
def defective_weight : ℕ := 101

/-- Represents the total number of parts selected for weighing -/
def total_selected : ℕ := (num_boxes + 1) * num_boxes / 2

/-- Represents the expected weight if all selected parts were standard -/
def expected_weight : ℕ := total_selected * standard_weight

theorem find_defective_box (actual_weight : ℕ) :
  actual_weight > expected_weight →
  ∃ (box_number : ℕ), 
    box_number ≤ num_boxes ∧
    box_number = actual_weight - expected_weight ∧
    box_number * defective_parts = (defective_weight - standard_weight) * total_selected :=
by sorry

end NUMINAMATH_CALUDE_find_defective_box_l2715_271568


namespace NUMINAMATH_CALUDE_committee_arrangement_count_l2715_271508

/-- The number of ways to arrange n indistinguishable objects of type A
    and m indistinguishable objects of type B in a row of (n+m) positions -/
def arrangement_count (n m : ℕ) : ℕ :=
  Nat.choose (n + m) m

/-- Theorem stating that there are 120 ways to arrange 7 indistinguishable objects
    and 3 indistinguishable objects in a row of 10 positions -/
theorem committee_arrangement_count :
  arrangement_count 7 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_arrangement_count_l2715_271508


namespace NUMINAMATH_CALUDE_triangle_determinant_zero_l2715_271557

theorem triangle_determinant_zero (A B C : Real) 
  (h : A + B + C = Real.pi) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_zero_l2715_271557


namespace NUMINAMATH_CALUDE_greater_number_problem_l2715_271520

theorem greater_number_problem (x y : ℝ) (h1 : x ≥ y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x * y = 2048) (h5 : x + y - (x - y) = 64) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l2715_271520


namespace NUMINAMATH_CALUDE_foundation_cost_theorem_l2715_271566

/-- Represents the dimensions of a concrete slab -/
structure SlabDimensions where
  length : Float
  width : Float
  height : Float

/-- Calculates the volume of a concrete slab -/
def slabVolume (d : SlabDimensions) : Float :=
  d.length * d.width * d.height

/-- Calculates the weight of concrete given its volume and density -/
def concreteWeight (volume density : Float) : Float :=
  volume * density

/-- Calculates the cost of concrete given its weight and price per pound -/
def concreteCost (weight pricePerPound : Float) : Float :=
  weight * pricePerPound

theorem foundation_cost_theorem 
  (slabDim : SlabDimensions)
  (concreteDensity : Float)
  (concretePricePerPound : Float)
  (numHomes : Nat) :
  slabDim.length = 100 →
  slabDim.width = 100 →
  slabDim.height = 0.5 →
  concreteDensity = 150 →
  concretePricePerPound = 0.02 →
  numHomes = 3 →
  concreteCost 
    (concreteWeight 
      (slabVolume slabDim * numHomes.toFloat) 
      concreteDensity) 
    concretePricePerPound = 45000 := by
  sorry

end NUMINAMATH_CALUDE_foundation_cost_theorem_l2715_271566


namespace NUMINAMATH_CALUDE_september_march_ratio_is_two_to_one_l2715_271547

/-- Vacation policy and Andrew's work record --/
structure VacationRecord where
  workRatio : ℕ  -- Number of work days required for 1 vacation day
  workDays : ℕ   -- Number of days worked
  marchDays : ℕ  -- Vacation days taken in March
  remainingDays : ℕ  -- Remaining vacation days

/-- Calculate the ratio of September vacation days to March vacation days --/
def septemberToMarchRatio (record : VacationRecord) : ℚ :=
  let totalVacationDays := record.workDays / record.workRatio
  let septemberDays := totalVacationDays - record.remainingDays - record.marchDays
  septemberDays / record.marchDays

/-- Theorem stating the ratio of September to March vacation days is 2:1 --/
theorem september_march_ratio_is_two_to_one 
  (record : VacationRecord)
  (h1 : record.workRatio = 10)
  (h2 : record.workDays = 300)
  (h3 : record.marchDays = 5)
  (h4 : record.remainingDays = 15) :
  septemberToMarchRatio record = 2 := by
  sorry

#eval septemberToMarchRatio ⟨10, 300, 5, 15⟩

end NUMINAMATH_CALUDE_september_march_ratio_is_two_to_one_l2715_271547


namespace NUMINAMATH_CALUDE_simplify_fraction_l2715_271503

theorem simplify_fraction (b : ℝ) (h : b ≠ 0) : (15 * b^4) / (90 * b^3 * b^1) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2715_271503


namespace NUMINAMATH_CALUDE_new_average_weight_l2715_271552

theorem new_average_weight (n : ℕ) (w_avg : ℝ) (w_new : ℝ) :
  n = 29 →
  w_avg = 28 →
  w_new = 4 →
  (n * w_avg + w_new) / (n + 1) = 27.2 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l2715_271552


namespace NUMINAMATH_CALUDE_bus_trip_speed_l2715_271519

/-- Proves that for a trip of 880 miles, if increasing the speed by 10 mph
    reduces the trip time by 2 hours, then the original speed was 61.5 mph. -/
theorem bus_trip_speed (v : ℝ) (h : v > 0) : 
  (880 / v) - (880 / (v + 10)) = 2 → v = 61.5 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l2715_271519


namespace NUMINAMATH_CALUDE_solution_set_implies_m_range_l2715_271534

theorem solution_set_implies_m_range :
  (∀ x : ℝ, x^2 + m*x + 1 > 0) → m ∈ Set.Ioo (-2 : ℝ) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_range_l2715_271534


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l2715_271502

/-- Given a complex number z satisfying |z - 3 + 2i| = 4,
    the minimum value of |z + 1 - i|^2 + |z - 7 + 5i|^2 is 36. -/
theorem min_value_complex_expression (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 4) :
  36 ≤ Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 5*I))^2 ∧
  ∃ w : ℂ, Complex.abs (w - (3 - 2*I)) = 4 ∧
          Complex.abs (w + (1 - I))^2 + Complex.abs (w - (7 - 5*I))^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l2715_271502


namespace NUMINAMATH_CALUDE_disjoint_subsets_remainder_l2715_271515

def S : Finset Nat := Finset.range 12

def count_disjoint_subsets (S : Finset Nat) : Nat :=
  (3^S.card - 2 * 2^S.card + 1) / 2

theorem disjoint_subsets_remainder (S : Finset Nat) :
  count_disjoint_subsets S % 1000 = 625 := by
  sorry

#eval count_disjoint_subsets S % 1000

end NUMINAMATH_CALUDE_disjoint_subsets_remainder_l2715_271515


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l2715_271586

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 16) + Complex.abs (z + 3*I) = 17) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 16) + Complex.abs (w + 3*I) = 17 ∧ Complex.abs w = 768 / 265 :=
sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l2715_271586


namespace NUMINAMATH_CALUDE_carmichael_function_properties_l2715_271549

variable (a : ℕ)

theorem carmichael_function_properties (ha : a > 2) :
  (∃ n : ℕ, n > 1 ∧ ¬ Nat.Prime n ∧ a^n ≡ 1 [ZMOD n]) ∧
  (∀ p : ℕ, p > 1 → (∀ k : ℕ, 1 < k ∧ k < p → ¬(a^k ≡ 1 [ZMOD k])) → a^p ≡ 1 [ZMOD p] → Nat.Prime p) ∧
  ¬(∃ n : ℕ, n > 1 ∧ 2^n ≡ 1 [ZMOD n]) :=
by sorry

end NUMINAMATH_CALUDE_carmichael_function_properties_l2715_271549


namespace NUMINAMATH_CALUDE_equation_solution_l2715_271596

theorem equation_solution (x : ℝ) : (x + 6) / (x - 3) = 4 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2715_271596


namespace NUMINAMATH_CALUDE_square_root_sum_fractions_l2715_271590

theorem square_root_sum_fractions : 
  Real.sqrt (1/25 + 1/36 + 1/49) = Real.sqrt 7778 / 297 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_fractions_l2715_271590


namespace NUMINAMATH_CALUDE_square_sum_equals_three_l2715_271533

theorem square_sum_equals_three (a b : ℝ) (h : a^4 + b^4 = a^2 - 2*a^2*b^2 + b^2 + 6) : 
  a^2 + b^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_three_l2715_271533


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_largest_coefficient_l2715_271564

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b * c / (b + c - a)) + (a * c / (a + c - b)) + (a * b / (a + b - c)) ≥ (a + b + c) :=
sorry

theorem largest_coefficient (k : ℝ) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → c + a > b →
    (b * c / (b + c - a)) + (a * c / (a + c - b)) + (a * b / (a + b - c)) ≥ k * (a + b + c)) →
  k ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_largest_coefficient_l2715_271564


namespace NUMINAMATH_CALUDE_min_cost_for_zoo_visit_l2715_271563

/-- Represents the ticket pricing structure for the zoo --/
structure TicketPrices where
  adult : ℕ
  child : ℕ
  group : ℕ
  group_min : ℕ

/-- Calculates the total cost for a group given the pricing and number of adults and children --/
def calculate_cost (prices : TicketPrices) (adults children : ℕ) : ℕ :=
  min (prices.adult * adults + prices.child * children)
      (min (prices.group * (adults + children))
           (prices.group * prices.group_min + prices.child * (adults + children - prices.group_min)))

/-- Theorem stating the minimum cost for the given group --/
theorem min_cost_for_zoo_visit (prices : TicketPrices) 
    (h1 : prices.adult = 150)
    (h2 : prices.child = 60)
    (h3 : prices.group = 100)
    (h4 : prices.group_min = 5) :
  calculate_cost prices 4 7 = 860 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_zoo_visit_l2715_271563


namespace NUMINAMATH_CALUDE_rectangular_screen_area_l2715_271529

/-- Proves that a rectangular screen with width-to-height ratio of 3:2 and diagonal length of 65 cm has an area of 1950 cm². -/
theorem rectangular_screen_area (width height diagonal : ℝ) : 
  width / height = 3 / 2 →
  width^2 + height^2 = diagonal^2 →
  diagonal = 65 →
  width * height = 1950 := by
sorry

end NUMINAMATH_CALUDE_rectangular_screen_area_l2715_271529


namespace NUMINAMATH_CALUDE_composition_ratio_theorem_l2715_271565

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_ratio_theorem :
  (f (g (f 2))) / (g (f (g 2))) = 41 / 7 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_theorem_l2715_271565


namespace NUMINAMATH_CALUDE_triangle_type_indeterminate_l2715_271591

theorem triangle_type_indeterminate (A B C : ℝ) 
  (triangle_sum : A + B + C = π) 
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C) 
  (inequality : Real.sin A * Real.sin C > Real.cos A * Real.cos C) : 
  ¬(∀ α : ℝ, (0 < α ∧ α < π) → 
    ((A < π/2 ∧ B < π/2 ∧ C < π/2) ∨ 
     (A = π/2 ∨ B = π/2 ∨ C = π/2) ∨ 
     (A > π/2 ∨ B > π/2 ∨ C > π/2))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_type_indeterminate_l2715_271591


namespace NUMINAMATH_CALUDE_mary_nickels_l2715_271550

theorem mary_nickels (initial_nickels : ℕ) (dad_gave : ℕ) (total_now : ℕ) : 
  dad_gave = 5 → total_now = 12 → initial_nickels + dad_gave = total_now → initial_nickels = 7 := by
sorry

end NUMINAMATH_CALUDE_mary_nickels_l2715_271550


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2715_271548

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2715_271548


namespace NUMINAMATH_CALUDE_problem_statement_l2715_271551

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) :
  10 * (6 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2715_271551


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2715_271554

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (2 * x^3))^2) = x^3 / 2 + 1 / (2 * x^3) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2715_271554


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2715_271562

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, n^2 < 2^n) ↔ (∃ n₀ : ℕ, n₀^2 ≥ 2^n₀) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2715_271562


namespace NUMINAMATH_CALUDE_coverable_polyhedron_exists_l2715_271580

/-- A polyhedron that can be covered by a square and an equilateral triangle -/
structure CoverablePolyhedron where
  /-- Side length of the square -/
  s : ℝ
  /-- Side length of the equilateral triangle -/
  t : ℝ
  /-- The perimeters of the square and triangle are equal -/
  h_perimeter : 4 * s = 3 * t
  /-- The polyhedron exists and can be covered -/
  h_exists : Prop

/-- Theorem stating that there exists a polyhedron that can be covered by a square and an equilateral triangle with equal perimeters -/
theorem coverable_polyhedron_exists : ∃ (p : CoverablePolyhedron), p.h_exists := by
  sorry

end NUMINAMATH_CALUDE_coverable_polyhedron_exists_l2715_271580


namespace NUMINAMATH_CALUDE_utility_bill_amount_l2715_271539

/-- The total amount of Mrs. Brown's utility bills -/
def utility_bill_total : ℕ := 
  4 * 100 + 5 * 50 + 7 * 20 + 8 * 10

/-- Theorem stating that Mrs. Brown's utility bills amount to $870 -/
theorem utility_bill_amount : utility_bill_total = 870 := by
  sorry

end NUMINAMATH_CALUDE_utility_bill_amount_l2715_271539


namespace NUMINAMATH_CALUDE_ravenswood_gnomes_remaining_l2715_271513

/-- The number of gnomes in Westerville woods -/
def westerville_gnomes : ℕ := 20

/-- The ratio of gnomes in Ravenswood forest compared to Westerville woods -/
def ravenswood_ratio : ℕ := 4

/-- The percentage of gnomes taken by the forest owner -/
def taken_percentage : ℚ := 40 / 100

/-- The number of gnomes remaining in Ravenswood forest after some are taken -/
def remaining_ravenswood_gnomes : ℕ := 48

theorem ravenswood_gnomes_remaining :
  remaining_ravenswood_gnomes = 
    (ravenswood_ratio * westerville_gnomes) - 
    (ravenswood_ratio * westerville_gnomes * taken_percentage).floor := by
  sorry

end NUMINAMATH_CALUDE_ravenswood_gnomes_remaining_l2715_271513


namespace NUMINAMATH_CALUDE_proposition_truth_values_l2715_271527

-- Define proposition p
def p : Prop := ∀ a : ℝ, (∀ x : ℝ, (x^2 + |x - a| = (-x)^2 + |(-x) - a|)) → a = 0

-- Define proposition q
def q : Prop := ∀ m : ℝ, m > 0 → ∃ x : ℝ, m * x^2 - 2 * x + 1 = 0

-- Theorem stating the truth values of the propositions
theorem proposition_truth_values : 
  p ∧ 
  ¬q ∧ 
  (p ∨ q) ∧ 
  ¬(p ∧ q) ∧ 
  ¬((¬p) ∧ q) ∧ 
  ((¬p) ∨ (¬q)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l2715_271527


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2715_271509

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 5 ≥ -4 ∧ ∃ y : ℝ, y^2 - 6*y + 5 = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2715_271509


namespace NUMINAMATH_CALUDE_repeating_decimal_fractions_l2715_271504

def repeating_decimal_3 : ℚ := 0.333333
def repeating_decimal_56 : ℚ := 0.565656

theorem repeating_decimal_fractions :
  (repeating_decimal_3 = 1 / 3) ∧
  (repeating_decimal_56 = 56 / 99) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_fractions_l2715_271504


namespace NUMINAMATH_CALUDE_pennys_bakery_revenue_l2715_271512

/-- The revenue calculation for Penny's bakery --/
theorem pennys_bakery_revenue : 
  ∀ (price_per_slice : ℕ) (slices_per_pie : ℕ) (number_of_pies : ℕ),
    price_per_slice = 7 →
    slices_per_pie = 6 →
    number_of_pies = 7 →
    price_per_slice * slices_per_pie * number_of_pies = 294 := by
  sorry

end NUMINAMATH_CALUDE_pennys_bakery_revenue_l2715_271512


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l2715_271522

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 39 ∣ m^2) :
  39 = Nat.gcd 39 m := by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l2715_271522


namespace NUMINAMATH_CALUDE_correct_restroom_count_l2715_271516

/-- The number of students in the restroom -/
def students_in_restroom : ℕ := 2

/-- The number of absent students -/
def absent_students : ℕ := 3 * students_in_restroom - 1

/-- The total number of desks -/
def total_desks : ℕ := 4 * 6

/-- The number of occupied desks -/
def occupied_desks : ℕ := (2 * total_desks) / 3

/-- The total number of students Carla teaches -/
def total_students : ℕ := 23

theorem correct_restroom_count :
  students_in_restroom + absent_students + occupied_desks = total_students :=
sorry

end NUMINAMATH_CALUDE_correct_restroom_count_l2715_271516


namespace NUMINAMATH_CALUDE_cylinder_sphere_surface_area_l2715_271521

theorem cylinder_sphere_surface_area (r : ℝ) (h : ℝ) :
  h = 2 * r →
  (4 / 3) * Real.pi * r^3 = 4 * Real.sqrt 3 * Real.pi →
  2 * Real.pi * r^2 + 2 * Real.pi * r * h = 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_sphere_surface_area_l2715_271521


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2715_271510

theorem necessary_but_not_sufficient :
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) ∧
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2715_271510


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_is_nine_min_value_achieved_l2715_271501

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y + (1 - y) * 1 = 0 → 1/x + 4*y ≥ 1/m + 4*n :=
by sorry

theorem min_value_is_nine (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  1/m + 4*n ≥ 9 :=
by sorry

theorem min_value_achieved (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y + (1 - y) * 1 = 0 ∧ 1/x + 4*y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_is_nine_min_value_achieved_l2715_271501
