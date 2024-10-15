import Mathlib

namespace NUMINAMATH_CALUDE_expression_equals_one_l1102_110232

theorem expression_equals_one :
  (π + 2023) ^ 0 + 2 * Real.sin (π / 4) - (1 / 2)⁻¹ + |Real.sqrt 2 - 2| = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1102_110232


namespace NUMINAMATH_CALUDE_tate_education_ratio_l1102_110287

/-- Represents the duration of Tate's education -/
structure TateEducation where
  normalHighSchool : ℕ  -- Normal duration of high school
  tateHighSchool : ℕ    -- Tate's actual high school duration
  higherEd : ℕ          -- Duration of bachelor's degree and Ph.D.
  totalYears : ℕ        -- Total years spent in education

/-- Conditions of Tate's education -/
def validTateEducation (e : TateEducation) : Prop :=
  e.tateHighSchool = e.normalHighSchool - 1 ∧
  e.higherEd = e.tateHighSchool * (e.higherEd / e.tateHighSchool) ∧
  e.totalYears = e.tateHighSchool + e.higherEd ∧
  e.totalYears = 12

/-- The theorem to be proved -/
theorem tate_education_ratio (e : TateEducation) 
  (h : validTateEducation e) : 
  e.higherEd / e.tateHighSchool = 3 := by
  sorry

#check tate_education_ratio

end NUMINAMATH_CALUDE_tate_education_ratio_l1102_110287


namespace NUMINAMATH_CALUDE_simple_interest_rate_proof_l1102_110227

/-- The rate at which a sum becomes 4 times of itself in 15 years at simple interest -/
def simple_interest_rate : ℝ := 20

/-- The time period in years -/
def time_period : ℝ := 15

/-- The factor by which the sum increases -/
def growth_factor : ℝ := 4

theorem simple_interest_rate_proof : 
  (1 + simple_interest_rate * time_period / 100) = growth_factor := by
  sorry

#check simple_interest_rate_proof

end NUMINAMATH_CALUDE_simple_interest_rate_proof_l1102_110227


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1102_110243

theorem quadratic_roots_relation (A B C : ℝ) (r s : ℝ) (p q : ℝ) :
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  ((r + 3)^2 + p * (r + 3) + q = 0) →
  ((s + 3)^2 + p * (s + 3) + q = 0) →
  (A ≠ 0) →
  (p = B / A - 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1102_110243


namespace NUMINAMATH_CALUDE_percentage_commutativity_l1102_110223

theorem percentage_commutativity (x : ℝ) (h : (30 / 100) * (40 / 100) * x = 60) :
  (40 / 100) * (30 / 100) * x = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutativity_l1102_110223


namespace NUMINAMATH_CALUDE_line_segment_product_l1102_110207

/-- Given four points A, B, C, D on a line in this order, prove that AB · CD + AD · BC = 1000 -/
theorem line_segment_product (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order on the line
  (C - A = 25) →  -- AC = 25
  (D - B = 40) →  -- BD = 40
  (D - A = 57) →  -- AD = 57
  (B - A) * (D - C) + (D - A) * (C - B) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_product_l1102_110207


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l1102_110297

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 3 + a 5 = 24 →
  a 7 - a 3 = 24 →
  a 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l1102_110297


namespace NUMINAMATH_CALUDE_sum_of_solutions_equals_sixteen_l1102_110214

theorem sum_of_solutions_equals_sixteen :
  let f : ℝ → ℝ := λ x => Real.sqrt x + Real.sqrt (9 / x) + Real.sqrt (x + 9 / x)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 9 ∧ f x₂ = 9 ∧ x₁ + x₂ = 16 ∧
  ∀ (x : ℝ), f x = 9 → x = x₁ ∨ x = x₂ :=
by sorry


end NUMINAMATH_CALUDE_sum_of_solutions_equals_sixteen_l1102_110214


namespace NUMINAMATH_CALUDE_g_derivative_at_5_l1102_110267

-- Define the function g
def g (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

-- State the theorem
theorem g_derivative_at_5 : 
  (deriv g) 5 = 26 := by sorry

end NUMINAMATH_CALUDE_g_derivative_at_5_l1102_110267


namespace NUMINAMATH_CALUDE_no_valid_pairs_l1102_110264

theorem no_valid_pairs : ¬∃ (M N K : ℕ), 
  M > 0 ∧ N > 0 ∧ 
  (M : ℚ) / 5 = 5 / (N : ℚ) ∧ 
  M = 2 * K := by
  sorry

end NUMINAMATH_CALUDE_no_valid_pairs_l1102_110264


namespace NUMINAMATH_CALUDE_smallest_three_digit_geometric_sequence_l1102_110257

/-- Checks if a number is a three-digit integer -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Extracts the hundreds digit of a three-digit number -/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the tens digit of a three-digit number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Extracts the ones digit of a three-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Checks if the digits of a three-digit number are distinct -/
def hasDistinctDigits (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  h ≠ t ∧ t ≠ o ∧ h ≠ o

/-- Checks if the digits of a three-digit number form a geometric sequence -/
def formsGeometricSequence (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  ∃ r : ℚ, r > 1 ∧ t = h * r ∧ o = t * r

theorem smallest_three_digit_geometric_sequence :
  ∀ n : ℕ, isThreeDigit n → hasDistinctDigits n → formsGeometricSequence n → n ≥ 248 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_geometric_sequence_l1102_110257


namespace NUMINAMATH_CALUDE_line_intersection_intersection_point_l1102_110255

/-- Two lines intersect at a unique point -/
theorem line_intersection (s t : ℝ) : ∃! (p : ℝ × ℝ), 
  (∃ s, p = (1 + 3*s, 2 - 7*s)) ∧ 
  (∃ t, p = (-5 + 5*t, 3 - 8*t)) :=
by sorry

/-- The intersection point of the two lines is (7, -12) -/
theorem intersection_point : 
  ∃ (s t : ℝ), (1 + 3*s, 2 - 7*s) = (-5 + 5*t, 3 - 8*t) ∧ 
                (1 + 3*s, 2 - 7*s) = (7, -12) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_intersection_point_l1102_110255


namespace NUMINAMATH_CALUDE_fraction_division_and_addition_l1102_110203

theorem fraction_division_and_addition : (3 / 7 : ℚ) / 4 + 1 / 28 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_and_addition_l1102_110203


namespace NUMINAMATH_CALUDE_square_root_problem_l1102_110247

theorem square_root_problem (x y : ℝ) 
  (h1 : (5 * x - 1).sqrt = 3)
  (h2 : (4 * x + 2 * y + 1)^(1/3) = 1) :
  (4 * x - 2 * y).sqrt = 4 ∨ (4 * x - 2 * y).sqrt = -4 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l1102_110247


namespace NUMINAMATH_CALUDE_window_height_is_four_l1102_110231

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the cost of whitewashing per square foot
def cost_per_sqft : ℝ := 3

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the number of windows and their width
def num_windows : ℕ := 3
def window_width : ℝ := 3

-- Define the total cost of whitewashing
def total_cost : ℝ := 2718

-- Theorem to prove
theorem window_height_is_four :
  ∃ (h : ℝ),
    h = 4 ∧
    (2 * (room_length * room_height + room_width * room_height) -
     (door_height * door_width + num_windows * h * window_width)) * cost_per_sqft = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_window_height_is_four_l1102_110231


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l1102_110273

theorem hexagon_largest_angle :
  ∀ (a b c d e f : ℝ),
    -- The angles are consecutive integers
    (∃ (x : ℝ), a = x - 2 ∧ b = x - 1 ∧ c = x ∧ d = x + 1 ∧ e = x + 2 ∧ f = x + 3) →
    -- Sum of angles in a hexagon is 720°
    a + b + c + d + e + f = 720 →
    -- The largest angle is 122.5°
    max a (max b (max c (max d (max e f)))) = 122.5 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l1102_110273


namespace NUMINAMATH_CALUDE_max_product_sum_2020_l1102_110298

theorem max_product_sum_2020 (n : ℕ) (as : List ℕ) :
  n ≥ 1 →
  as.length = n →
  as.sum = 2020 →
  as.prod ≤ 2^2 * 3^672 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_2020_l1102_110298


namespace NUMINAMATH_CALUDE_gcd_2197_2208_l1102_110224

theorem gcd_2197_2208 : Nat.gcd 2197 2208 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2197_2208_l1102_110224


namespace NUMINAMATH_CALUDE_lcm_from_product_and_gcd_l1102_110281

theorem lcm_from_product_and_gcd (a b : ℕ+) 
  (h_product : a * b = 17820)
  (h_gcd : Nat.gcd a b = 12) :
  Nat.lcm a b = 1485 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_gcd_l1102_110281


namespace NUMINAMATH_CALUDE_square_difference_and_product_l1102_110266

theorem square_difference_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 81)
  (h2 : x * y = 15) : 
  (x - y)^2 = 21 ∧ (x + y) * (x - y) = Real.sqrt 1701 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_and_product_l1102_110266


namespace NUMINAMATH_CALUDE_M_remainder_mod_45_l1102_110284

/-- The number of digits in M -/
def num_digits : ℕ := 95

/-- The last integer in the sequence forming M -/
def last_int : ℕ := 50

/-- M is the number formed by concatenating integers from 1 to last_int -/
def M : ℕ := sorry

theorem M_remainder_mod_45 : M % 45 = 15 := by sorry

end NUMINAMATH_CALUDE_M_remainder_mod_45_l1102_110284


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1102_110256

theorem complex_equation_sum (z : ℂ) (a b : ℝ) : 
  z = a + b * I → z * (1 + I^3) = 2 + I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1102_110256


namespace NUMINAMATH_CALUDE_solve_for_y_l1102_110296

theorem solve_for_y (x y : ℝ) (h1 : 3 * x + 2 = 2) (h2 : y - x = 2) : y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1102_110296


namespace NUMINAMATH_CALUDE_total_travel_methods_eq_thirteen_l1102_110239

/-- The number of bus services from A to B -/
def bus_services : ℕ := 8

/-- The number of train services from A to B -/
def train_services : ℕ := 3

/-- The number of ship services from A to B -/
def ship_services : ℕ := 2

/-- The total number of different methods to travel from A to B -/
def total_travel_methods : ℕ := bus_services + train_services + ship_services

theorem total_travel_methods_eq_thirteen :
  total_travel_methods = 13 := by sorry

end NUMINAMATH_CALUDE_total_travel_methods_eq_thirteen_l1102_110239


namespace NUMINAMATH_CALUDE_min_value_theorem_l1102_110234

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 ∧
  (a^2 + 1 / (b * (a - b)) = 4 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1102_110234


namespace NUMINAMATH_CALUDE_combination_equation_solution_l1102_110274

def binomial (n k : ℕ) : ℕ := sorry

theorem combination_equation_solution :
  ∀ x : ℕ, (binomial 28 x = binomial 28 (3*x - 8)) ↔ (x = 4 ∨ x = 9) :=
sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l1102_110274


namespace NUMINAMATH_CALUDE_five_people_booth_arrangements_l1102_110250

/-- The number of ways to arrange n people in a booth with at most k people on each side -/
def boothArrangements (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange 5 people in a booth with at most 3 people on each side -/
theorem five_people_booth_arrangements :
  boothArrangements 5 3 = 240 := by sorry

end NUMINAMATH_CALUDE_five_people_booth_arrangements_l1102_110250


namespace NUMINAMATH_CALUDE_inequality_solution_l1102_110215

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x + 3) / ((x - 2)^2) < 0 ↔ 1 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1102_110215


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1102_110222

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-2, 4]

-- State the theorem
theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1102_110222


namespace NUMINAMATH_CALUDE_negation_of_all_politicians_are_loyal_l1102_110251

universe u

def Politician (α : Type u) := α → Prop
def Loyal (α : Type u) := α → Prop

theorem negation_of_all_politicians_are_loyal 
  {α : Type u} (politician : Politician α) (loyal : Loyal α) :
  (¬ ∀ (x : α), politician x → loyal x) ↔ (∃ (x : α), politician x ∧ ¬ loyal x) :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_politicians_are_loyal_l1102_110251


namespace NUMINAMATH_CALUDE_bounded_sequence_convergence_l1102_110216

def is_bounded (s : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, |s n| ≤ M

theorem bounded_sequence_convergence
  (a : ℕ → ℝ)
  (h_rec : ∀ n : ℕ, a (n + 1) = 3 * a n - 4)
  (h_bounded : is_bounded a) :
  ∀ n : ℕ, a n = 2 :=
sorry

end NUMINAMATH_CALUDE_bounded_sequence_convergence_l1102_110216


namespace NUMINAMATH_CALUDE_vector_orthogonality_l1102_110229

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def b : Fin 2 → ℝ := ![1, -1]

-- Define the orthogonality condition
def orthogonal (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

-- Theorem statement
theorem vector_orthogonality (x : ℝ) :
  orthogonal (λ i => a x i - b i) b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_orthogonality_l1102_110229


namespace NUMINAMATH_CALUDE_star_computation_l1102_110286

def star (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

theorem star_computation :
  star 2 (star 3 (star 4 5)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_star_computation_l1102_110286


namespace NUMINAMATH_CALUDE_cookies_in_blue_tin_l1102_110210

/-- Proves that the fraction of cookies in the blue tin is 8/27 -/
theorem cookies_in_blue_tin
  (total_cookies : ℚ)
  (blue_green_fraction : ℚ)
  (red_fraction : ℚ)
  (green_fraction_of_blue_green : ℚ)
  (h1 : blue_green_fraction = 2 / 3)
  (h2 : red_fraction = 1 - blue_green_fraction)
  (h3 : green_fraction_of_blue_green = 5 / 9)
  : (blue_green_fraction * (1 - green_fraction_of_blue_green)) = 8 / 27 := by
  sorry


end NUMINAMATH_CALUDE_cookies_in_blue_tin_l1102_110210


namespace NUMINAMATH_CALUDE_population_scientific_notation_l1102_110261

def population : ℝ := 1370000000

theorem population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), population = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.37 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_population_scientific_notation_l1102_110261


namespace NUMINAMATH_CALUDE_triangular_pyramid_surface_area_l1102_110265

/-- A triangular pyramid with given base and side areas -/
structure TriangularPyramid where
  base_area : ℝ
  side_area : ℝ

/-- The surface area of a triangular pyramid -/
def surface_area (tp : TriangularPyramid) : ℝ :=
  tp.base_area + 3 * tp.side_area

/-- Theorem: The surface area of a triangular pyramid with base area 3 and side area 6 is 21 -/
theorem triangular_pyramid_surface_area :
  ∃ (tp : TriangularPyramid), tp.base_area = 3 ∧ tp.side_area = 6 ∧ surface_area tp = 21 := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_surface_area_l1102_110265


namespace NUMINAMATH_CALUDE_linear_systems_solutions_l1102_110225

theorem linear_systems_solutions :
  -- System (1)
  let system1 : ℝ × ℝ → Prop := λ (x, y) ↦ 3 * x - 2 * y = 9 ∧ 2 * x + 3 * y = 19
  -- System (2)
  let system2 : ℝ × ℝ → Prop := λ (x, y) ↦ (2 * x + 1) / 5 - 1 = (y - 1) / 3 ∧ 2 * (y - x) - 3 * (1 - y) = 6
  -- Solutions
  let solution1 : ℝ × ℝ := (5, 3)
  let solution2 : ℝ × ℝ := (4, 17/5)
  -- Proof statements
  system1 solution1 ∧ system2 solution2 := by sorry

end NUMINAMATH_CALUDE_linear_systems_solutions_l1102_110225


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l1102_110206

/-- Definition of a point in the third quadrant -/
def is_in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- The point (-2, -3) is in the third quadrant -/
theorem point_in_third_quadrant : is_in_third_quadrant (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l1102_110206


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1102_110237

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n.choose 2) - n

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1102_110237


namespace NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l1102_110205

/-- The set of points (x, y) satisfying the polar equation θ = π/4 forms a line in the Cartesian plane. -/
theorem polar_equation_pi_over_four_is_line :
  ∀ (x y : ℝ), (∃ (r : ℝ), x = r * Real.cos (π / 4) ∧ y = r * Real.sin (π / 4)) ↔
  ∃ (m b : ℝ), y = m * x + b ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_polar_equation_pi_over_four_is_line_l1102_110205


namespace NUMINAMATH_CALUDE_function_zero_set_empty_l1102_110209

theorem function_zero_set_empty (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 3 * f (1 - x) = x^2) :
  {x : ℝ | f x = 0} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_function_zero_set_empty_l1102_110209


namespace NUMINAMATH_CALUDE_cost_price_of_article_l1102_110270

/-- 
Given an article where the profit obtained by selling it for Rs. 66 
is equal to the loss obtained by selling it for Rs. 52, 
prove that the cost price of the article is Rs. 59.
-/
theorem cost_price_of_article (cost_price : ℤ) : cost_price = 59 :=
  sorry

end NUMINAMATH_CALUDE_cost_price_of_article_l1102_110270


namespace NUMINAMATH_CALUDE_min_sum_areas_two_triangles_l1102_110235

/-- The minimum sum of areas of two equilateral triangles formed from a 12cm wire -/
theorem min_sum_areas_two_triangles : 
  ∃ (f : ℝ → ℝ), 
    (∀ x, 0 ≤ x ∧ x ≤ 12 → 
      f x = (Real.sqrt 3 / 36) * (x^2 + (12 - x)^2)) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 12 → f x ≥ 2 * Real.sqrt 3) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 12 ∧ f x = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_areas_two_triangles_l1102_110235


namespace NUMINAMATH_CALUDE_bouncy_ball_difference_l1102_110244

-- Define the given quantities
def red_packs : ℕ := 12
def yellow_packs : ℕ := 9
def balls_per_red_pack : ℕ := 24
def balls_per_yellow_pack : ℕ := 20

-- Define the theorem
theorem bouncy_ball_difference :
  red_packs * balls_per_red_pack - yellow_packs * balls_per_yellow_pack = 108 := by
  sorry

end NUMINAMATH_CALUDE_bouncy_ball_difference_l1102_110244


namespace NUMINAMATH_CALUDE_coefficient_of_x3y2z5_in_expansion_l1102_110212

/-- The coefficient of x^3y^2z^5 in the expansion of (2x+y+z)^10 -/
def coefficient : ℕ := 20160

/-- The exponent of the trinomial expression -/
def exponent : ℕ := 10

/-- Theorem stating that the coefficient of x^3y^2z^5 in (2x+y+z)^10 is 20160 -/
theorem coefficient_of_x3y2z5_in_expansion : 
  coefficient = (2^3 : ℕ) * Nat.choose exponent 3 * Nat.choose (exponent - 3) 2 * Nat.choose ((exponent - 3) - 2) 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x3y2z5_in_expansion_l1102_110212


namespace NUMINAMATH_CALUDE_correlation_identification_l1102_110218

/-- Represents a relationship between two variables -/
inductive Relationship
| AgeWealth
| CurvePoint
| AppleProduction
| TreeDiameterHeight

/-- Determines if a relationship exhibits correlation -/
def has_correlation (r : Relationship) : Prop :=
  match r with
  | Relationship.AgeWealth => true
  | Relationship.CurvePoint => false
  | Relationship.AppleProduction => true
  | Relationship.TreeDiameterHeight => true

/-- The main theorem stating which relationships have correlation -/
theorem correlation_identification :
  (has_correlation Relationship.AgeWealth) ∧
  (¬has_correlation Relationship.CurvePoint) ∧
  (has_correlation Relationship.AppleProduction) ∧
  (has_correlation Relationship.TreeDiameterHeight) :=
sorry


end NUMINAMATH_CALUDE_correlation_identification_l1102_110218


namespace NUMINAMATH_CALUDE_seating_arrangements_l1102_110211

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where a group of k people must sit together -/
def groupedArrangements (n k : ℕ) : ℕ := 
  Nat.factorial (n - k + 1) * Nat.factorial k

/-- The number of people to be seated -/
def totalPeople : ℕ := 10

/-- The number of people in the group that can't sit together -/
def groupSize : ℕ := 4

theorem seating_arrangements :
  totalArrangements totalPeople - groupedArrangements totalPeople groupSize = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1102_110211


namespace NUMINAMATH_CALUDE_homework_time_is_48_minutes_l1102_110242

def math_problems : ℕ := 15
def social_studies_problems : ℕ := 6
def science_problems : ℕ := 10

def math_time_per_problem : ℚ := 2
def social_studies_time_per_problem : ℚ := 1/2
def science_time_per_problem : ℚ := 3/2

def total_homework_time : ℚ :=
  math_problems * math_time_per_problem +
  social_studies_problems * social_studies_time_per_problem +
  science_problems * science_time_per_problem

theorem homework_time_is_48_minutes :
  total_homework_time = 48 := by sorry

end NUMINAMATH_CALUDE_homework_time_is_48_minutes_l1102_110242


namespace NUMINAMATH_CALUDE_problem_statement_l1102_110217

theorem problem_statement (x : ℝ) (h : x + 1/x = 2) : x^5 - 5*x^3 + 6*x = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1102_110217


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l1102_110253

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_symmetry 
  (f : ℝ → ℝ) 
  (hf : QuadraticFunction f) 
  (h0 : f 0 = 3)
  (h1 : f 1 = 2)
  (h2 : f 2 = 3)
  (h3 : f 3 = 6)
  (h4 : f 4 = 11)
  (hm2 : f (-2) = 11) :
  f (-1) = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l1102_110253


namespace NUMINAMATH_CALUDE_worker_a_time_l1102_110230

theorem worker_a_time (worker_b_time worker_ab_time : ℝ) 
  (hb : worker_b_time = 12)
  (hab : worker_ab_time = 4.8) : ℝ :=
  let worker_a_time := (worker_b_time * worker_ab_time) / (worker_b_time - worker_ab_time)
  8

#check worker_a_time

end NUMINAMATH_CALUDE_worker_a_time_l1102_110230


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1102_110290

theorem absolute_value_equation_solution :
  {x : ℝ | |2007*x - 2007| = 2007} = {0, 2} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1102_110290


namespace NUMINAMATH_CALUDE_sum_a_d_l1102_110260

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 48) 
  (h2 : b + c = 6) : 
  a + d = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l1102_110260


namespace NUMINAMATH_CALUDE_polynomial_unique_value_l1102_110269

theorem polynomial_unique_value (P : ℤ → ℤ) :
  (∃ x₁ x₂ x₃ : ℤ, P x₁ = 1 ∧ P x₂ = 2 ∧ P x₃ = 3 ∧ (x₁ = x₂ - 1 ∨ x₁ = x₂ + 1) ∧ (x₂ = x₃ - 1 ∨ x₂ = x₃ + 1)) →
  (∃! x : ℤ, P x = 5) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_unique_value_l1102_110269


namespace NUMINAMATH_CALUDE_union_with_complement_l1102_110254

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem union_with_complement : A ∪ (U \ B) = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_with_complement_l1102_110254


namespace NUMINAMATH_CALUDE_container_volume_increase_l1102_110201

theorem container_volume_increase (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 5 → 
  scale_factor = 4 → 
  (scale_factor ^ 3) * original_volume = 320 := by
sorry

end NUMINAMATH_CALUDE_container_volume_increase_l1102_110201


namespace NUMINAMATH_CALUDE_parrots_per_cage_l1102_110285

theorem parrots_per_cage (num_cages : ℕ) (total_birds : ℕ) : 
  num_cages = 9 →
  total_birds = 36 →
  (∃ (parrots_per_cage : ℕ), 
    parrots_per_cage * num_cages * 2 = total_birds ∧ 
    parrots_per_cage = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_parrots_per_cage_l1102_110285


namespace NUMINAMATH_CALUDE_remainder_369975_div_6_l1102_110226

theorem remainder_369975_div_6 : 369975 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_369975_div_6_l1102_110226


namespace NUMINAMATH_CALUDE_consecutive_biology_majors_probability_l1102_110248

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of biology majors -/
def biology_majors : ℕ := 4

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of physics majors -/
def physics_majors : ℕ := 2

/-- The probability of all biology majors sitting in consecutive seats -/
def consecutive_biology_prob : ℚ := 2/3

theorem consecutive_biology_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := (total_people - biology_majors) * Nat.factorial (total_people - biology_majors - 1)
  (favorable_arrangements : ℚ) / total_arrangements = consecutive_biology_prob :=
sorry

end NUMINAMATH_CALUDE_consecutive_biology_majors_probability_l1102_110248


namespace NUMINAMATH_CALUDE_seaweed_harvest_l1102_110295

theorem seaweed_harvest (total : ℝ) :
  (0.5 * total ≥ 0) →                    -- 50% used for starting fires
  (0.25 * (0.5 * total) ≥ 0) →           -- 25% of remaining for human consumption
  (0.75 * (0.5 * total) = 150) →         -- 75% of remaining (150 pounds) fed to livestock
  (total = 400) :=
by sorry

end NUMINAMATH_CALUDE_seaweed_harvest_l1102_110295


namespace NUMINAMATH_CALUDE_voldemort_remaining_calories_voldemort_specific_remaining_calories_l1102_110246

/-- Calculates the remaining calories Voldemort can consume given his intake and limit -/
theorem voldemort_remaining_calories (cake_calories : ℕ) (chips_calories : ℕ) 
  (coke_calories : ℕ) (breakfast_calories : ℕ) (lunch_calories : ℕ) 
  (daily_limit : ℕ) : ℕ :=
  by
  have dinner_calories : ℕ := cake_calories + chips_calories + coke_calories
  have breakfast_lunch_calories : ℕ := breakfast_calories + lunch_calories
  have total_consumed : ℕ := dinner_calories + breakfast_lunch_calories
  exact daily_limit - total_consumed

/-- Proves that Voldemort's remaining calories is 525 given specific intake values -/
theorem voldemort_specific_remaining_calories : 
  voldemort_remaining_calories 110 310 215 560 780 2500 = 525 :=
by
  sorry

end NUMINAMATH_CALUDE_voldemort_remaining_calories_voldemort_specific_remaining_calories_l1102_110246


namespace NUMINAMATH_CALUDE_worker_completion_time_l1102_110200

/-- Given two workers A and B, proves that A can complete a job in 14 days 
    when A and B together can complete the job in 10 days, 
    and B alone can complete the job in 35 days. -/
theorem worker_completion_time 
  (joint_completion_time : ℝ) 
  (b_alone_completion_time : ℝ) 
  (h1 : joint_completion_time = 10) 
  (h2 : b_alone_completion_time = 35) : 
  ∃ (a_alone_completion_time : ℝ), 
    a_alone_completion_time = 14 ∧ 
    (1 / a_alone_completion_time + 1 / b_alone_completion_time = 1 / joint_completion_time) :=
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l1102_110200


namespace NUMINAMATH_CALUDE_ratio_proportion_problem_l1102_110259

theorem ratio_proportion_problem (x : ℝ) :
  (2975.75 / 7873.125 = 12594.5 / x) → x = 33333.75 := by
  sorry

end NUMINAMATH_CALUDE_ratio_proportion_problem_l1102_110259


namespace NUMINAMATH_CALUDE_sum_Q_mod_500_l1102_110291

/-- The set of distinct remainders when 3^k is divided by 500, for 0 ≤ k < 200 -/
def Q : Finset ℕ :=
  (Finset.range 200).image (fun k => (3^k : ℕ) % 500)

/-- The sum of all elements in Q -/
def sum_Q : ℕ := Q.sum id

/-- The theorem to prove -/
theorem sum_Q_mod_500 :
  sum_Q % 500 = (Finset.range 200).sum (fun k => (3^k : ℕ) % 500) % 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_Q_mod_500_l1102_110291


namespace NUMINAMATH_CALUDE_statement_equivalence_l1102_110240

theorem statement_equivalence (P Q : Prop) :
  (Q → ¬P) ↔ (P → ¬Q) := by sorry

end NUMINAMATH_CALUDE_statement_equivalence_l1102_110240


namespace NUMINAMATH_CALUDE_sum_a_d_l1102_110236

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + b * c + c * a + d * b = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l1102_110236


namespace NUMINAMATH_CALUDE_theater_attendance_l1102_110228

/-- Proves that the total number of attendees is 24 given the ticket prices, revenue, and number of children --/
theorem theater_attendance
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_price = 16)
  (h2 : child_price = 9)
  (h3 : total_revenue = 258)
  (h4 : num_children = 18)
  (h5 : ∃ num_adults : ℕ, adult_price * num_adults + child_price * num_children = total_revenue) :
  num_children + (total_revenue - child_price * num_children) / adult_price = 24 :=
by sorry

end NUMINAMATH_CALUDE_theater_attendance_l1102_110228


namespace NUMINAMATH_CALUDE_f_increasing_implies_a_geq_two_l1102_110252

/-- The function f(x) = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The theorem stating that if f(x+a) is increasing on [0, +∞), then a ≥ 2 -/
theorem f_increasing_implies_a_geq_two (a : ℝ) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f (x + a) < f (y + a)) →
  a ∈ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_implies_a_geq_two_l1102_110252


namespace NUMINAMATH_CALUDE_congruence_solution_l1102_110289

theorem congruence_solution (n : ℕ) : n ∈ Finset.range 29 → (8 * n ≡ 5 [ZMOD 29]) ↔ n = 26 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1102_110289


namespace NUMINAMATH_CALUDE_intersection_point_solution_l1102_110272

/-- Given two lines y = x + b and y = ax + 2 that intersect at point (3, -1),
    prove that the solution to (a - 1)x = b - 2 is x = 3. -/
theorem intersection_point_solution (a b : ℝ) :
  (3 + b = 3 * a + 2) →  -- Intersection point condition
  (-1 = 3 + b) →         -- y-coordinate of intersection point
  ((a - 1) * 3 = b - 2)  -- Solution x = 3 satisfies the equation
  := by sorry

end NUMINAMATH_CALUDE_intersection_point_solution_l1102_110272


namespace NUMINAMATH_CALUDE_sum_of_combinations_l1102_110245

theorem sum_of_combinations : (Nat.choose 4 4) + (Nat.choose 5 4) + (Nat.choose 6 4) + 
  (Nat.choose 7 4) + (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 462 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l1102_110245


namespace NUMINAMATH_CALUDE_AX_length_l1102_110213

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the diameter of the circle
def diameter : ℝ := 1

-- Define the points on the circle
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define point X on diameter AD
def X : Point := sorry

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the angle function
def angle (p q r : Point) : ℝ := sorry

-- State the theorem
theorem AX_length (h1 : distance A D = diameter)
                  (h2 : distance B X = distance C X)
                  (h3 : 3 * angle B A C = angle B X C)
                  (h4 : angle B X C = 30 * π / 180) :
  distance A X = Real.cos (10 * π / 180) * Real.sin (20 * π / 180) * (1 / Real.sin (15 * π / 180)) :=
sorry

end NUMINAMATH_CALUDE_AX_length_l1102_110213


namespace NUMINAMATH_CALUDE_certain_number_proof_l1102_110208

theorem certain_number_proof (x : ℝ) (h : 5 * x - 28 = 232) : x = 52 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1102_110208


namespace NUMINAMATH_CALUDE_collinear_points_sum_l1102_110202

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p2 = p1 + t • (p3 - p1) ∨ p1 = p2 + t • (p3 - p2)

/-- 
If the points (2,x,y), (x,3,y), and (x,y,4) are collinear, then x + y = 6.
-/
theorem collinear_points_sum (x y : ℝ) :
  collinear (2, x, y) (x, 3, y) (x, y, 4) → x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l1102_110202


namespace NUMINAMATH_CALUDE_multiple_of_1998_l1102_110204

theorem multiple_of_1998 (a : Fin 93 → ℕ+) (h : Function.Injective a) :
  ∃ m n p q : Fin 93, m ≠ n ∧ p ≠ q ∧ 1998 ∣ (a m - a n) * (a p - a q) := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_1998_l1102_110204


namespace NUMINAMATH_CALUDE_product_of_real_parts_complex_equation_l1102_110280

theorem product_of_real_parts_complex_equation : ∃ (z₁ z₂ : ℂ),
  (z₁^2 - 2*z₁ = Complex.I) ∧
  (z₂^2 - 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (z₁.re * z₂.re = (1 - Real.sqrt 2) / 2) := by
sorry

end NUMINAMATH_CALUDE_product_of_real_parts_complex_equation_l1102_110280


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l1102_110219

/-- A function satisfying the given conditions in the problem -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∃ M : ℝ, ∀ x, |f x| ≤ M) ∧ 
  f 1 = 1 ∧
  ∀ x ≠ 0, f (x + 1/x^2) = f x + (f (1/x))^2

/-- Theorem stating that no function satisfies all the given conditions -/
theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, SatisfiesConditions f := by
  sorry


end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l1102_110219


namespace NUMINAMATH_CALUDE_min_treasures_is_15_l1102_110299

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried" -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried" -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried" -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried" -/
def signs_3 : ℕ := 3

/-- Predicate that checks if a given number of treasures satisfies all conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n ≤ total_trees ∧
  (n ≠ 15 ∨ signs_15 = total_trees - n) ∧
  (n ≠ 8 ∨ signs_8 = total_trees - n) ∧
  (n ≠ 4 ∨ signs_4 = total_trees - n) ∧
  (n ≠ 3 ∨ signs_3 = total_trees - n)

/-- Theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasures_is_15 :
  ∃ (n : ℕ), n = 15 ∧ satisfies_conditions n ∧ ∀ (m : ℕ), m < n → ¬satisfies_conditions m :=
by sorry

end NUMINAMATH_CALUDE_min_treasures_is_15_l1102_110299


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1102_110292

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1102_110292


namespace NUMINAMATH_CALUDE_determinant_problem_l1102_110268

theorem determinant_problem (a b c d : ℝ) : 
  let M₁ : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let M₂ : Matrix (Fin 2) (Fin 2) ℝ := !![a+2*c, b+2*d; 3*c, 3*d]
  Matrix.det M₁ = -7 → Matrix.det M₂ = -21 := by
  sorry

end NUMINAMATH_CALUDE_determinant_problem_l1102_110268


namespace NUMINAMATH_CALUDE_optimal_ships_l1102_110262

/-- Revenue function -/
def R (x : ℕ) : ℚ := 3700 * x + 45 * x^2 - 10 * x^3

/-- Cost function -/
def C (x : ℕ) : ℚ := 460 * x + 5000

/-- Profit function -/
def P (x : ℕ) : ℚ := R x - C x

/-- The maximum number of ships that can be built annually -/
def max_capacity : ℕ := 20

/-- Theorem: The number of ships that maximizes annual profit is 12 -/
theorem optimal_ships :
  ∃ (x : ℕ), x ≤ max_capacity ∧ x > 0 ∧
  ∀ (y : ℕ), y ≤ max_capacity ∧ y > 0 → P x ≥ P y ∧
  x = 12 :=
sorry

end NUMINAMATH_CALUDE_optimal_ships_l1102_110262


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1102_110263

theorem complex_equation_solution : ∃ z : ℂ, (z + 2) * (1 + Complex.I ^ 3) = 2 * Complex.I ∧ z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1102_110263


namespace NUMINAMATH_CALUDE_function_minimum_implies_inequality_l1102_110220

open Real

/-- Given a function f(x) = ax^2 + bx - 2ln(x) where a > 0 and b is real,
    if f(x) ≥ f(2) for all x > 0, then ln(a) < -b - 1 -/
theorem function_minimum_implies_inequality (a b : ℝ) (ha : a > 0) :
  (∀ x > 0, a * x^2 + b * x - 2 * log x ≥ a * 2^2 + b * 2 - 2 * log 2) →
  log a < -b - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_implies_inequality_l1102_110220


namespace NUMINAMATH_CALUDE_average_star_rating_l1102_110258

def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

def total_star_points : ℕ := 5 * five_star_reviews + 4 * four_star_reviews + 3 * three_star_reviews + 2 * two_star_reviews

theorem average_star_rating :
  (total_star_points : ℚ) / total_reviews = 4 := by sorry

end NUMINAMATH_CALUDE_average_star_rating_l1102_110258


namespace NUMINAMATH_CALUDE_cost_of_mangos_rice_flour_l1102_110279

/-- The cost of mangos, rice, and flour given certain price relationships -/
theorem cost_of_mangos_rice_flour (mango_cost rice_cost flour_cost : ℝ) 
  (h1 : 10 * mango_cost = 24 * rice_cost)
  (h2 : flour_cost = 2 * rice_cost)
  (h3 : flour_cost = 21) : 
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 237.3 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_mangos_rice_flour_l1102_110279


namespace NUMINAMATH_CALUDE_x_24_equals_one_l1102_110283

theorem x_24_equals_one (x : ℂ) (h : x + 1/x = -Real.sqrt 3) : x^24 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_24_equals_one_l1102_110283


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l1102_110233

theorem sum_of_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ 
  (i + 2*i^2 + 3*i^3 + 4*i^4 + 5*i^5 + 6*i^6 + 7*i^7 + 8*i^8 + 9*i^9 = 4 + 5*i) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l1102_110233


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l1102_110249

def A : Set ℝ := {x | |x - 1| ≤ 1}
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -Real.sqrt 2 ≤ x ∧ x < 1}

theorem complement_intersection_A_B : 
  (A ∩ B)ᶜ = {x : ℝ | x ≠ 0} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l1102_110249


namespace NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1102_110238

/-- A function is an "H function" if for any two distinct real numbers x₁ and x₂,
    it satisfies x₁f(x₁) + x₂f(x₂) > x₁f(x₂) + x₂f(x₁) -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function is strictly increasing if for any two real numbers x₁ < x₂,
    we have f(x₁) < f(x₂) -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_strictly_increasing_l1102_110238


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1102_110294

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  a 1 = 2 ∧ 
  d ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (∃ r : ℝ, r ≠ 0 ∧ a 3 = r * a 1 ∧ a 11 = r * a 3) →
  (∃ r : ℝ, r = 4 ∧ a 3 = r * a 1 ∧ a 11 = r * a 3) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1102_110294


namespace NUMINAMATH_CALUDE_difference_is_integer_l1102_110241

/-- A linear function from ℝ to ℝ -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  map : ℝ → ℝ := fun x ↦ a * x + b
  increasing : 0 < a

/-- Two linear functions with the integer property -/
structure IntegerPropertyFunctions where
  f : LinearFunction
  g : LinearFunction
  integer_property : ∀ x : ℝ, Int.floor (f.map x) = f.map x ↔ Int.floor (g.map x) = g.map x

/-- The main theorem -/
theorem difference_is_integer (funcs : IntegerPropertyFunctions) :
  ∀ x : ℝ, ∃ n : ℤ, funcs.f.map x - funcs.g.map x = n :=
sorry

end NUMINAMATH_CALUDE_difference_is_integer_l1102_110241


namespace NUMINAMATH_CALUDE_distribute_9_4_l1102_110293

/-- The number of ways to distribute n identical items into k boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 220 ways to distribute 9 identical items into 4 boxes -/
theorem distribute_9_4 : distribute 9 4 = 220 := by
  sorry

end NUMINAMATH_CALUDE_distribute_9_4_l1102_110293


namespace NUMINAMATH_CALUDE_number_and_remainder_l1102_110282

theorem number_and_remainder : ∃ x : ℤ, 2 * x - 3 = 7 ∧ x % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_and_remainder_l1102_110282


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l1102_110276

theorem complex_square_one_plus_i : 
  (Complex.I + 1) ^ 2 = 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l1102_110276


namespace NUMINAMATH_CALUDE_divisibility_product_l1102_110221

theorem divisibility_product (a b c d : ℤ) : a ∣ b → c ∣ d → (a * c) ∣ (b * d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_product_l1102_110221


namespace NUMINAMATH_CALUDE_regular_dinosaur_weight_is_800_l1102_110288

/-- The weight of a regular dinosaur in pounds -/
def regular_dinosaur_weight : ℝ := sorry

/-- The weight of Barney the dinosaur in pounds -/
def barney_weight : ℝ := 5 * regular_dinosaur_weight + 1500

/-- The total weight of Barney and five regular dinosaurs in pounds -/
def total_weight : ℝ := 9500

/-- Theorem stating that each regular dinosaur weighs 800 pounds -/
theorem regular_dinosaur_weight_is_800 : regular_dinosaur_weight = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_dinosaur_weight_is_800_l1102_110288


namespace NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1102_110277

theorem arithmetic_mean_geq_geometric_mean {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_geq_geometric_mean_l1102_110277


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1102_110271

/-- Given a geometric sequence {aₙ}, prove that if a₁ + a₂ + a₃ = 2 and a₃ + a₄ + a₅ = 8, 
    then a₄ + a₅ + a₆ = ±16 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∃ (q : ℝ), ∀ n, a (n + 1) = q * a n) 
  (h_sum1 : a 1 + a 2 + a 3 = 2) (h_sum2 : a 3 + a 4 + a 5 = 8) : 
  a 4 + a 5 + a 6 = 16 ∨ a 4 + a 5 + a 6 = -16 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l1102_110271


namespace NUMINAMATH_CALUDE_highest_power_of_two_dividing_13_4_minus_11_4_l1102_110275

theorem highest_power_of_two_dividing_13_4_minus_11_4 :
  ∃ (n : ℕ), 2^n = (Nat.gcd (13^4 - 11^4) (2^32 : ℕ)) ∧
  ∀ (m : ℕ), 2^m ∣ (13^4 - 11^4) → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_highest_power_of_two_dividing_13_4_minus_11_4_l1102_110275


namespace NUMINAMATH_CALUDE_books_per_shelf_l1102_110278

theorem books_per_shelf (total_books : Nat) (total_shelves : Nat) 
  (h1 : total_books = 14240) (h2 : total_shelves = 1780) : 
  total_books / total_shelves = 8 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l1102_110278
