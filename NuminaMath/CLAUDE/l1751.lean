import Mathlib

namespace NUMINAMATH_CALUDE_no_roots_of_third_trinomial_l1751_175165

theorem no_roots_of_third_trinomial (a b : ℤ) : 
  (∃ x : ℤ, x^2 + a*x + b = 0) → 
  (∃ y : ℤ, y^2 + a*y + (b + 1) = 0) → 
  ∀ z : ℝ, z^2 + a*z + (b + 2) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_roots_of_third_trinomial_l1751_175165


namespace NUMINAMATH_CALUDE_sector_max_area_l1751_175104

/-- Given a sector of a circle with perimeter c (c > 0), 
    prove that the maximum area is c^2/16 and occurs when the arc length is c/2 -/
theorem sector_max_area (c : ℝ) (hc : c > 0) :
  let area (L : ℝ) := (c - L) * L / 4
  ∃ (L : ℝ), L = c / 2 ∧ 
    (∀ x, area x ≤ area L) ∧
    area L = c^2 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l1751_175104


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l1751_175140

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (t, -1 + Real.sqrt 3 * t)

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 2 * Real.sin θ + 2 * Real.cos θ

-- Define point P
def point_P : ℝ × ℝ := (0, -1)

-- Define the intersection points A and B
variable (A B : ℝ × ℝ)

-- Assume A and B are on both the line and the curve
axiom A_on_line : ∃ t : ℝ, line_l t = A
axiom B_on_line : ∃ t : ℝ, line_l t = B
axiom A_on_curve : ∃ θ : ℝ, (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = A
axiom B_on_curve : ∃ θ : ℝ, (curve_C θ * Real.cos θ, curve_C θ * Real.sin θ) = B

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem intersection_distance_sum :
  1 / distance point_P A + 1 / distance point_P B = (2 * Real.sqrt 3 + 1) / 3 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_sum_l1751_175140


namespace NUMINAMATH_CALUDE_sin_45_minus_sin_15_l1751_175117

theorem sin_45_minus_sin_15 : 
  Real.sin (45 * π / 180) - Real.sin (15 * π / 180) = (3 * Real.sqrt 2 - Real.sqrt 6) / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_45_minus_sin_15_l1751_175117


namespace NUMINAMATH_CALUDE_x_needs_18_days_l1751_175181

/-- The time needed for x to finish the remaining work after y leaves -/
def remaining_time_for_x (x_time y_time y_worked : ℚ) : ℚ :=
  (1 - y_worked / y_time) * x_time

/-- Proof that x needs 18 days to finish the remaining work -/
theorem x_needs_18_days (x_time y_time y_worked : ℚ) 
  (hx : x_time = 36)
  (hy : y_time = 24)
  (hw : y_worked = 12) :
  remaining_time_for_x x_time y_time y_worked = 18 := by
  sorry

#eval remaining_time_for_x 36 24 12

end NUMINAMATH_CALUDE_x_needs_18_days_l1751_175181


namespace NUMINAMATH_CALUDE_similarity_coefficients_are_valid_l1751_175125

/-- A triangle with sides 2, 3, and 3 -/
structure OriginalTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  h1 : side1 = 2
  h2 : side2 = 3
  h3 : side3 = 3

/-- Similarity coefficients for the four triangles -/
structure SimilarityCoefficients where
  k1 : ℝ
  k2 : ℝ
  k3 : ℝ
  k4 : ℝ

/-- Predicate to check if the similarity coefficients are valid -/
def valid_coefficients (sc : SimilarityCoefficients) : Prop :=
  (sc.k1 = 1/2 ∧ sc.k2 = 1/2 ∧ sc.k3 = 1/2 ∧ sc.k4 = 1/2) ∨
  (sc.k1 = 6/13 ∧ sc.k2 = 4/13 ∧ sc.k3 = 9/13 ∧ sc.k4 = 6/13)

/-- Theorem stating that the similarity coefficients for the divided triangles are valid -/
theorem similarity_coefficients_are_valid (t : OriginalTriangle) (sc : SimilarityCoefficients) :
  valid_coefficients sc := by sorry

end NUMINAMATH_CALUDE_similarity_coefficients_are_valid_l1751_175125


namespace NUMINAMATH_CALUDE_fraction_sum_l1751_175154

theorem fraction_sum : (3 : ℚ) / 5 + (2 : ℚ) / 15 = (11 : ℚ) / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1751_175154


namespace NUMINAMATH_CALUDE_complex_integer_calculation_l1751_175157

theorem complex_integer_calculation : (-7)^7 / 7^4 + 2^6 - 8^2 = -343 := by
  sorry

end NUMINAMATH_CALUDE_complex_integer_calculation_l1751_175157


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1751_175162

theorem sqrt_product_equality : Real.sqrt 128 * Real.sqrt 50 * Real.sqrt 18 = 240 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1751_175162


namespace NUMINAMATH_CALUDE_range_sum_bounds_l1751_175136

/-- The function f(x) = -2x^2 + 4x -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x

/-- The range of f is [m, n] -/
def m : ℝ := -6
def n : ℝ := 2

theorem range_sum_bounds :
  ∀ x, m ≤ f x ∧ f x ≤ n →
  0 ≤ m + n ∧ m + n ≤ 4 := by
  sorry

#check range_sum_bounds

end NUMINAMATH_CALUDE_range_sum_bounds_l1751_175136


namespace NUMINAMATH_CALUDE_det_A_eq_33_l1751_175147

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![5, 0, -2],
    ![1, 3,  4],
    ![0, -1, 1]]

theorem det_A_eq_33 : A.det = 33 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_33_l1751_175147


namespace NUMINAMATH_CALUDE_arrange_in_order_l1751_175146

def Ψ : ℤ := -(1006 : ℤ)

def Ω : ℤ := -(1007 : ℤ)

def Θ : ℤ := -(1008 : ℤ)

theorem arrange_in_order : Θ < Ω ∧ Ω < Ψ := by
  sorry

end NUMINAMATH_CALUDE_arrange_in_order_l1751_175146


namespace NUMINAMATH_CALUDE_both_pipes_open_time_l1751_175190

/-- The time it takes for pipe p to fill the cistern alone -/
def p_time : ℚ := 12

/-- The time it takes for pipe q to fill the cistern alone -/
def q_time : ℚ := 15

/-- The additional time it takes for pipe q to fill the cistern after pipe p is turned off -/
def additional_time : ℚ := 6

/-- The theorem stating that the time both pipes are open together is 4 minutes -/
theorem both_pipes_open_time : 
  ∃ (t : ℚ), 
    t * (1 / p_time + 1 / q_time) + additional_time * (1 / q_time) = 1 ∧ 
    t = 4 := by
  sorry

end NUMINAMATH_CALUDE_both_pipes_open_time_l1751_175190


namespace NUMINAMATH_CALUDE_algebraic_manipulation_l1751_175142

theorem algebraic_manipulation (a b : ℝ) :
  (-2 * a^2 * b)^2 * (3 * a * b^2 - 5 * a^2 * b) / (-a * b)^3 = -12 * a^2 * b + 20 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_manipulation_l1751_175142


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l1751_175184

theorem unique_solution_floor_equation :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 72 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l1751_175184


namespace NUMINAMATH_CALUDE_triangle_inequality_l1751_175107

open Real

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point M
def M : ℝ × ℝ := sorry

-- Define the semi-perimeter p
def semiPerimeter (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_inequality (t : Triangle) :
  let p := semiPerimeter t
  distance M t.A * cos (angle t.B t.A t.C / 2) +
  distance M t.B * cos (angle t.A t.B t.C / 2) +
  distance M t.C * cos (angle t.A t.C t.B / 2) ≥ p := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1751_175107


namespace NUMINAMATH_CALUDE_square_area_comparison_l1751_175183

theorem square_area_comparison (a b : ℝ) (h : b = 4 * a) :
  b ^ 2 = 16 * a ^ 2 := by sorry

end NUMINAMATH_CALUDE_square_area_comparison_l1751_175183


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_sum_sqrt_l1751_175112

theorem sqrt_sum_equals_sqrt_of_sum_sqrt (a b : ℚ) :
  Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3) ↔
  ({a, b} : Set ℚ) = {1/2, 3/2} :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_sum_sqrt_l1751_175112


namespace NUMINAMATH_CALUDE_factorization_3m_squared_minus_12_l1751_175173

theorem factorization_3m_squared_minus_12 (m : ℝ) : 3 * m^2 - 12 = 3 * (m + 2) * (m - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_3m_squared_minus_12_l1751_175173


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1751_175191

-- Define the inequality
def inequality (k x : ℝ) : Prop :=
  k * (x^2 + 6*x - k) * (x^2 + x - 12) > 0

-- Define the solution set
def solution_set (k : ℝ) : Set ℝ :=
  {x | inequality k x}

-- Theorem statement
theorem inequality_solution_set (k : ℝ) :
  solution_set k = Set.Ioo (-4 : ℝ) 3 ↔ k ∈ Set.Iic (-9 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1751_175191


namespace NUMINAMATH_CALUDE_m_range_l1751_175113

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → m * x^3 - x^2 + 4*x + 3 ≥ 0) → 
  m ∈ Set.Icc (-6) (-2) := by
sorry

end NUMINAMATH_CALUDE_m_range_l1751_175113


namespace NUMINAMATH_CALUDE_distance_between_specific_planes_l1751_175163

/-- Represents a plane in 3D space defined by ax + by + cz = d -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two planes -/
def distance_between_planes (p1 p2 : Plane) : ℝ :=
  sorry

/-- The first plane: 2x + 4y - 2z = 10 -/
def plane1 : Plane := ⟨2, 4, -2, 10⟩

/-- The second plane: x + 2y - z = -3 -/
def plane2 : Plane := ⟨1, 2, -1, -3⟩

theorem distance_between_specific_planes :
  distance_between_planes plane1 plane2 = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_specific_planes_l1751_175163


namespace NUMINAMATH_CALUDE_fraction_sum_and_multiply_l1751_175103

theorem fraction_sum_and_multiply :
  ((2 : ℚ) / 9 + 4 / 11) * 3 / 5 = 58 / 165 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_multiply_l1751_175103


namespace NUMINAMATH_CALUDE_digit_ratio_l1751_175151

/-- Given a 3-digit integer x with hundreds digit a, tens digit b, and units digit c,
    where a > 0 and the difference between the two greatest possible values of x is 241,
    prove that the ratio of b to a is 5:7. -/
theorem digit_ratio (x a b c : ℕ) : 
  (100 ≤ x) ∧ (x < 1000) ∧  -- x is a 3-digit integer
  (x = 100 * a + 10 * b + c) ∧  -- x is composed of digits a, b, c
  (a > 0) ∧  -- a is positive
  (999 - x = 241) →  -- difference between greatest possible value and x is 241
  (b : ℚ) / a = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_digit_ratio_l1751_175151


namespace NUMINAMATH_CALUDE_candy_ratio_problem_l1751_175187

/-- Proof of candy ratio problem -/
theorem candy_ratio_problem (chocolate_bars : ℕ) (mm_multiplier : ℕ) (candies_per_basket : ℕ) (num_baskets : ℕ)
  (h1 : chocolate_bars = 5)
  (h2 : mm_multiplier = 7)
  (h3 : candies_per_basket = 10)
  (h4 : num_baskets = 25) :
  (num_baskets * candies_per_basket - chocolate_bars - mm_multiplier * chocolate_bars) / (mm_multiplier * chocolate_bars) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_ratio_problem_l1751_175187


namespace NUMINAMATH_CALUDE_fraction_multiplication_division_main_proof_l1751_175196

theorem fraction_multiplication_division (a b c d e f : ℚ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → e ≠ 0 → f ≠ 0 →
  (a / b * c / d) / (e / f) = (a * c * f) / (b * d * e) :=
by sorry

theorem main_proof : (3 / 4 * 5 / 6) / (7 / 8) = 5 / 7 :=
by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_division_main_proof_l1751_175196


namespace NUMINAMATH_CALUDE_smallest_taco_packages_l1751_175159

/-- The number of tacos in each package -/
def tacos_per_package : ℕ := 4

/-- The number of taco shells in each package -/
def shells_per_package : ℕ := 6

/-- The minimum number of tacos and taco shells required -/
def min_required : ℕ := 60

/-- Proposition: The smallest number of taco packages to buy is 15 -/
theorem smallest_taco_packages : 
  (∃ (taco_packages shell_packages : ℕ),
    taco_packages * tacos_per_package = shell_packages * shells_per_package ∧
    taco_packages * tacos_per_package ≥ min_required ∧
    shell_packages * shells_per_package ≥ min_required ∧
    ∀ (t s : ℕ), 
      t * tacos_per_package = s * shells_per_package →
      t * tacos_per_package ≥ min_required →
      s * shells_per_package ≥ min_required →
      t ≥ taco_packages) →
  (∃ (shell_packages : ℕ),
    15 * tacos_per_package = shell_packages * shells_per_package ∧
    15 * tacos_per_package ≥ min_required ∧
    shell_packages * shells_per_package ≥ min_required) :=
by sorry

end NUMINAMATH_CALUDE_smallest_taco_packages_l1751_175159


namespace NUMINAMATH_CALUDE_smallest_number_problem_l1751_175158

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a ≤ b ∧ b ≤ c →
  b = 29 →
  c = b + 7 →
  (a + b + c) / 3 = 30 →
  a = 25 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l1751_175158


namespace NUMINAMATH_CALUDE_no_solution_for_coin_problem_l1751_175115

theorem no_solution_for_coin_problem : 
  ¬∃ (x y z : ℕ), x + y + z = 13 ∧ x + 3*y + 5*z = 200 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_coin_problem_l1751_175115


namespace NUMINAMATH_CALUDE_clothing_sale_profit_l1751_175133

def initial_cost : ℕ := 400
def num_sets : ℕ := 8
def sale_price : ℕ := 55
def adjustments : List ℤ := [2, -3, 2, 1, -2, -1, 0, -2]

theorem clothing_sale_profit :
  (num_sets * sale_price : ℤ) + (adjustments.sum) - initial_cost = 37 := by
  sorry

end NUMINAMATH_CALUDE_clothing_sale_profit_l1751_175133


namespace NUMINAMATH_CALUDE_alternating_work_completion_work_fully_completed_l1751_175100

/-- Represents the number of days it takes to complete the work when A and B work on alternate days, starting with B. -/
def alternating_work_days (a_days b_days : ℕ) : ℕ :=
  2 * (9 * b_days * a_days) / (b_days + 3 * a_days)

/-- Theorem stating that if A can complete the work in 12 days and B in 36 days,
    working on alternate days starting with B will complete the work in 18 days. -/
theorem alternating_work_completion :
  alternating_work_days 12 36 = 18 := by
  sorry

/-- Proof that the work is fully completed after 18 days. -/
theorem work_fully_completed (a_days b_days : ℕ) 
  (ha : a_days = 12) (hb : b_days = 36) :
  (9 : ℚ) * (1 / b_days + 1 / a_days) = 1 := by
  sorry

end NUMINAMATH_CALUDE_alternating_work_completion_work_fully_completed_l1751_175100


namespace NUMINAMATH_CALUDE_lara_flowers_in_vase_l1751_175124

/-- Calculates the number of flowers Lara put in the vase --/
def flowersInVase (totalFlowers : ℕ) (toMom : ℕ) (extraToGrandma : ℕ) : ℕ :=
  let toGrandma := toMom + extraToGrandma
  let remainingAfterMomAndGrandma := totalFlowers - toMom - toGrandma
  let toSister := remainingAfterMomAndGrandma / 3
  let toBestFriend := toSister + toSister / 4
  remainingAfterMomAndGrandma - toSister - toBestFriend

theorem lara_flowers_in_vase :
  flowersInVase 52 15 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_lara_flowers_in_vase_l1751_175124


namespace NUMINAMATH_CALUDE_pat_calculation_l1751_175108

theorem pat_calculation (x : ℝ) : (x / 6) - 14 = 16 → (x * 6) + 14 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_pat_calculation_l1751_175108


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l1751_175185

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y - 2)^2 = 4

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse x y ∧ hyperbola x y m ∧
  ∀ (x' y' : ℝ), ellipse x' y' ∧ hyperbola x' y' m → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_ellipse_hyperbola :
  ∀ m : ℝ, are_tangent m → m = 1/3 := by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l1751_175185


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l1751_175116

theorem least_three_digit_multiple : ∃ n : ℕ,
  (n ≥ 100 ∧ n < 1000) ∧
  2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n ∧
  ∀ m : ℕ, (m ≥ 100 ∧ m < 1000 ∧ 2 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 3 ∣ m) → n ≤ m :=
by
  use 210
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l1751_175116


namespace NUMINAMATH_CALUDE_expression_simplification_l1751_175128

theorem expression_simplification (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) (h3 : x ≠ 2) :
  ((x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2*x)) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1751_175128


namespace NUMINAMATH_CALUDE_f_at_3_l1751_175160

def f (x : ℝ) : ℝ := 9*x^3 - 5*x^2 - 3*x + 7

theorem f_at_3 : f 3 = 196 := by
  sorry

end NUMINAMATH_CALUDE_f_at_3_l1751_175160


namespace NUMINAMATH_CALUDE_machine_work_rate_l1751_175167

theorem machine_work_rate (x : ℝ) : 
  (1 / (x + 4) + 1 / (x + 3) + 1 / (x + 2) = 1 / x) → x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_machine_work_rate_l1751_175167


namespace NUMINAMATH_CALUDE_y_value_l1751_175118

theorem y_value (y : ℝ) (h : (9 : ℝ) / y^3 = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1751_175118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1751_175130

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : a 4 = 8) (h2 : a 5 = 12) (h3 : a 6 = 16) :
  a 1 + a 2 + a 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1751_175130


namespace NUMINAMATH_CALUDE_salary_distribution_l1751_175127

theorem salary_distribution (total : ℝ) :
  ∃ (a b c d : ℝ),
    a + b + c + d = total ∧
    2 * b = 3 * a ∧
    4 * b = 6 * a ∧
    3 * c = 4 * b ∧
    d = c + 700 ∧
    b = 1050 := by
  sorry

end NUMINAMATH_CALUDE_salary_distribution_l1751_175127


namespace NUMINAMATH_CALUDE_third_grade_students_l1751_175174

theorem third_grade_students (total_students : ℕ) (sample_size : ℕ) (first_grade_sample : ℕ) (second_grade_sample : ℕ) :
  total_students = 2000 →
  sample_size = 100 →
  first_grade_sample = 30 →
  second_grade_sample = 30 →
  (total_students * (sample_size - first_grade_sample - second_grade_sample)) / sample_size = 800 := by
  sorry

end NUMINAMATH_CALUDE_third_grade_students_l1751_175174


namespace NUMINAMATH_CALUDE_remainder_theorem_l1751_175197

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 60 * k - 1) :
  (n^2 + 2*n + 3) % 60 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1751_175197


namespace NUMINAMATH_CALUDE_work_completion_time_l1751_175192

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 10

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days A leaves before the work is completed -/
def a_leave_before : ℝ := 5

/-- The total number of days to complete the work -/
def total_days : ℝ := 10

/-- Theorem stating that given the conditions, the work is completed in 10 days -/
theorem work_completion_time :
  (1 / a_days + 1 / b_days) * (total_days - a_leave_before) + (1 / b_days) * a_leave_before = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1751_175192


namespace NUMINAMATH_CALUDE_mountaineer_arrangement_count_l1751_175110

/-- The number of ways to arrange mountaineers -/
def arrange_mountaineers (total : ℕ) (familiar : ℕ) (group_size : ℕ) : ℕ :=
  -- Number of ways to divide familiar mountaineers
  (familiar.choose (group_size / 2) * (familiar - group_size / 2).choose (group_size / 2) / 2) *
  -- Number of ways to divide unfamiliar mountaineers
  ((total - familiar).choose ((total - familiar) / 2) * ((total - familiar) / 2).choose ((total - familiar) / 2) / 2) *
  -- Number of ways to pair groups
  2 *
  -- Number of ways to order the groups
  2

/-- The theorem stating the number of arrangements for the given problem -/
theorem mountaineer_arrangement_count : 
  arrange_mountaineers 10 4 2 = 120 := by sorry

end NUMINAMATH_CALUDE_mountaineer_arrangement_count_l1751_175110


namespace NUMINAMATH_CALUDE_tim_morning_run_hours_l1751_175149

/-- Tim's running schedule -/
structure RunningSchedule where
  runs_per_week : ℕ
  total_hours_per_week : ℕ
  morning_equals_evening : Bool

/-- Calculate the number of hours Tim runs in the morning each day -/
def morning_run_hours (schedule : RunningSchedule) : ℚ :=
  if schedule.morning_equals_evening then
    (schedule.total_hours_per_week : ℚ) / (2 * schedule.runs_per_week)
  else
    0

/-- Theorem: Tim runs 1 hour in the morning each day -/
theorem tim_morning_run_hours :
  let tims_schedule : RunningSchedule := {
    runs_per_week := 5,
    total_hours_per_week := 10,
    morning_equals_evening := true
  }
  morning_run_hours tims_schedule = 1 := by sorry

end NUMINAMATH_CALUDE_tim_morning_run_hours_l1751_175149


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_once_l1751_175176

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define a point P on the ellipse
def point_on_ellipse (x₀ y₀ : ℝ) : Prop := ellipse x₀ y₀

-- Define the line l
def line (x₀ y₀ x y : ℝ) : Prop := 3 * x₀ * x + 4 * y₀ * y - 12 = 0

-- Theorem statement
theorem line_intersects_ellipse_once (x₀ y₀ : ℝ) 
  (h_point : point_on_ellipse x₀ y₀) :
  ∃! (x y : ℝ), ellipse x y ∧ line x₀ y₀ x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_once_l1751_175176


namespace NUMINAMATH_CALUDE_equation_solution_l1751_175109

theorem equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ 
  (∀ x : ℝ, (x + 1) * (x - 3) = 5 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1751_175109


namespace NUMINAMATH_CALUDE_min_value_abs_sum_min_value_abs_sum_achieved_l1751_175123

theorem min_value_abs_sum (x : ℝ) : |x - 1| + |x - 3| ≥ 2 := by sorry

theorem min_value_abs_sum_achieved : ∃ x : ℝ, |x - 1| + |x - 3| = 2 := by sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_min_value_abs_sum_achieved_l1751_175123


namespace NUMINAMATH_CALUDE_frank_floor_l1751_175120

/-- Given information about the floors where Dennis, Charlie, and Frank live,
    prove that Frank lives on the 16th floor. -/
theorem frank_floor (dennis_floor charlie_floor frank_floor : ℕ) 
  (h1 : dennis_floor = charlie_floor + 2)
  (h2 : charlie_floor = frank_floor / 4)
  (h3 : dennis_floor = 6) :
  frank_floor = 16 := by
  sorry

end NUMINAMATH_CALUDE_frank_floor_l1751_175120


namespace NUMINAMATH_CALUDE_simplify_quadratic_expression_l1751_175141

/-- Simplification of a quadratic expression -/
theorem simplify_quadratic_expression (y : ℝ) :
  4 * y + 8 * y^2 + 6 - (3 - 4 * y - 8 * y^2) = 16 * y^2 + 8 * y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_quadratic_expression_l1751_175141


namespace NUMINAMATH_CALUDE_t_range_theorem_l1751_175199

theorem t_range_theorem (t x y z : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 →
  3 * x^2 + 3 * z * x + z^2 = 1 →
  3 * y^2 + 3 * y * z + z^2 = 4 →
  x^2 - x * y + y^2 = t →
  (3 - Real.sqrt 5) / 2 ≤ t ∧ t ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_t_range_theorem_l1751_175199


namespace NUMINAMATH_CALUDE_max_value_inequality_l1751_175188

theorem max_value_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 6 + 8 * y * z ≤ Real.sqrt 22 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x^2 + y^2 + z^2 = 1 ∧
    2 * x * y * Real.sqrt 6 + 8 * y * z = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1751_175188


namespace NUMINAMATH_CALUDE_dogwood_tree_planting_l1751_175156

/-- The number of dogwood trees planted today -/
def trees_planted_today : ℕ := 41

/-- The initial number of trees in the park -/
def initial_trees : ℕ := 39

/-- The number of trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The final number of trees in the park -/
def final_trees : ℕ := 100

theorem dogwood_tree_planting :
  initial_trees + trees_planted_today + trees_planted_tomorrow = final_trees :=
by sorry

end NUMINAMATH_CALUDE_dogwood_tree_planting_l1751_175156


namespace NUMINAMATH_CALUDE_least_band_members_l1751_175153

/-- Represents the target ratio for each instrument -/
def target_ratio : Vector ℕ 5 := ⟨[5, 3, 6, 2, 4], by rfl⟩

/-- Represents the minimum number of successful candidates for each instrument -/
def min_candidates : Vector ℕ 5 := ⟨[16, 15, 20, 2, 12], by rfl⟩

/-- Checks if a given number of band members satisfies the target ratio and minimum requirements -/
def satisfies_requirements (total_members : ℕ) : Prop :=
  ∃ (x : ℕ), x > 0 ∧
    (∀ i : Fin 5, 
      (target_ratio.get i) * x ≥ min_candidates.get i) ∧
    (target_ratio.get 0) * x + 
    (target_ratio.get 1) * x + 
    (target_ratio.get 2) * x + 
    (target_ratio.get 3) * x + 
    (target_ratio.get 4) * x = total_members

/-- The main theorem stating that 100 is the least number of total band members satisfying the requirements -/
theorem least_band_members : 
  satisfies_requirements 100 ∧ 
  (∀ n : ℕ, n < 100 → ¬satisfies_requirements n) :=
sorry

end NUMINAMATH_CALUDE_least_band_members_l1751_175153


namespace NUMINAMATH_CALUDE_at_least_three_babies_speak_l1751_175132

def probability_baby_speaks : ℚ := 2/5

def number_of_babies : ℕ := 6

def probability_at_least_three_speak : ℚ := 7120/15625

theorem at_least_three_babies_speak :
  probability_at_least_three_speak =
    1 - (Nat.choose number_of_babies 0 * (1 - probability_baby_speaks)^number_of_babies +
         Nat.choose number_of_babies 1 * probability_baby_speaks * (1 - probability_baby_speaks)^(number_of_babies - 1) +
         Nat.choose number_of_babies 2 * probability_baby_speaks^2 * (1 - probability_baby_speaks)^(number_of_babies - 2)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_three_babies_speak_l1751_175132


namespace NUMINAMATH_CALUDE_problem1_problem2_problem3_problem4_l1751_175168

theorem problem1 : 5 / 7 + (-5 / 6) - (-2 / 7) + 1 + 1 / 6 = 4 / 3 := by sorry

theorem problem2 : (1 / 2 - (1 + 1 / 3) + 3 / 8) / (-1 / 24) = 11 := by sorry

theorem problem3 : (-3)^3 + (-5)^2 - |(-3)| * 4 = -14 := by sorry

theorem problem4 : -(1^101) - (-0.5 - (1 - 3 / 5 * 0.7) / (-1 / 2)^2) = 91 / 50 := by sorry

end NUMINAMATH_CALUDE_problem1_problem2_problem3_problem4_l1751_175168


namespace NUMINAMATH_CALUDE_fraction_simplification_l1751_175129

theorem fraction_simplification (x : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x + 1)) / ((2 / (x^2 - 1)) + (1 / (x + 1))) = (2*x - 2) / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1751_175129


namespace NUMINAMATH_CALUDE_total_cost_shirt_and_shoes_l1751_175111

/-- The total cost of a shirt and shoes, given the shirt cost and the relationship between shirt and shoe costs -/
theorem total_cost_shirt_and_shoes (shirt_cost : ℕ) (h1 : shirt_cost = 97) :
  let shoe_cost := 2 * shirt_cost + 9
  shirt_cost + shoe_cost = 300 := by
sorry


end NUMINAMATH_CALUDE_total_cost_shirt_and_shoes_l1751_175111


namespace NUMINAMATH_CALUDE_triangle_base_calculation_l1751_175155

theorem triangle_base_calculation (square_perimeter : ℝ) (triangle_area : ℝ) :
  square_perimeter = 60 →
  triangle_area = 150 →
  let square_side := square_perimeter / 4
  let triangle_height := square_side
  triangle_area = 1/2 * triangle_height * (triangle_base : ℝ) →
  triangle_base = 20 := by sorry

end NUMINAMATH_CALUDE_triangle_base_calculation_l1751_175155


namespace NUMINAMATH_CALUDE_cos_2x_plus_2y_l1751_175186

theorem cos_2x_plus_2y (x y : ℝ) (h : Real.cos x * Real.cos y - Real.sin x * Real.sin y = 1/4) :
  Real.cos (2*x + 2*y) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_plus_2y_l1751_175186


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l1751_175164

theorem simplify_nested_expression (x : ℝ) :
  2 * (1 - (2 * (1 - (1 + (2 * (1 - x)))))) = 8 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l1751_175164


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1751_175135

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 5 = 2 ∧ 
  (x : ℤ) % 7 = 3 ∧ 
  (x : ℤ) % 9 = 4 ∧
  ∀ y : ℕ+, ((y : ℤ) % 5 = 2 ∧ (y : ℤ) % 7 = 3 ∧ (y : ℤ) % 9 = 4) → x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1751_175135


namespace NUMINAMATH_CALUDE_basic_computer_printer_price_l1751_175138

/-- The total price of a basic computer and printer, given specific conditions -/
theorem basic_computer_printer_price : ∃ (printer_price : ℝ),
  let basic_computer_price : ℝ := 2000
  let enhanced_computer_price : ℝ := basic_computer_price + 500
  let total_price : ℝ := basic_computer_price + printer_price
  printer_price = (1 / 6) * (enhanced_computer_price + printer_price) →
  total_price = 2500 := by
sorry

end NUMINAMATH_CALUDE_basic_computer_printer_price_l1751_175138


namespace NUMINAMATH_CALUDE_arithmetic_sequence_square_root_l1751_175182

theorem arithmetic_sequence_square_root (x : ℝ) :
  x > 0 →
  (∃ d : ℝ, 2^2 + d = x^2 ∧ x^2 + d = 5^2) →
  x = Real.sqrt 14.5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_square_root_l1751_175182


namespace NUMINAMATH_CALUDE_sum_inequality_l1751_175198

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + b^2) + b / Real.sqrt (b^2 + c^2) + c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_sum_inequality_l1751_175198


namespace NUMINAMATH_CALUDE_system_solutions_l1751_175194

/-- The first equation of the system -/
def equation1 (x y z : ℝ) : Prop :=
  5 * x^2 + 3 * y^2 + 3 * x * y + 2 * x * z - y * z - 10 * y + 5 = 0

/-- The second equation of the system -/
def equation2 (x y z : ℝ) : Prop :=
  49 * x^2 + 65 * y^2 + 49 * z^2 - 14 * x * y - 98 * x * z + 14 * y * z - 182 * x - 102 * y + 182 * z + 233 = 0

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  equation1 x y z ∧ equation2 x y z

/-- The theorem stating that the given points are the only solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℝ, system x y z ↔ (x = 0 ∧ y = 1 ∧ z = -2) ∨ (x = 2/7 ∧ y = 1 ∧ z = -12/7) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l1751_175194


namespace NUMINAMATH_CALUDE_jackie_apples_l1751_175121

theorem jackie_apples (adam_apples : ℕ) (difference : ℕ) (jackie_apples : ℕ) : 
  adam_apples = 14 → 
  adam_apples = jackie_apples + difference → 
  difference = 5 →
  jackie_apples = 9 := by
sorry

end NUMINAMATH_CALUDE_jackie_apples_l1751_175121


namespace NUMINAMATH_CALUDE_polynomial_factors_imply_relation_l1751_175180

theorem polynomial_factors_imply_relation (h k : ℝ) : 
  (∃ a : ℝ, 2 * x^3 - h * x + k = (x + 2) * (x - 1) * a) → 
  2 * h - 3 * k = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factors_imply_relation_l1751_175180


namespace NUMINAMATH_CALUDE_power_of_power_of_three_l1751_175134

theorem power_of_power_of_three : (3^3)^(3^3) = 7625597484987 := by sorry

end NUMINAMATH_CALUDE_power_of_power_of_three_l1751_175134


namespace NUMINAMATH_CALUDE_smallest_m_no_real_roots_l1751_175126

theorem smallest_m_no_real_roots : ∃ (m : ℤ),
  (∀ (k : ℤ), k < m → ∃ (x : ℝ), 3 * x * (k * x - 6) - 2 * x^2 + 10 = 0) ∧
  (∀ (x : ℝ), 3 * x * (m * x - 6) - 2 * x^2 + 10 ≠ 0) ∧
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_m_no_real_roots_l1751_175126


namespace NUMINAMATH_CALUDE_georgia_yellow_buttons_l1751_175150

/-- The number of yellow buttons Georgia has -/
def yellow_buttons : ℕ := sorry

/-- The number of black buttons Georgia has -/
def black_buttons : ℕ := 2

/-- The number of green buttons Georgia has -/
def green_buttons : ℕ := 3

/-- The number of buttons Georgia gives to Mary -/
def buttons_given : ℕ := 4

/-- The number of buttons Georgia has left after giving buttons to Mary -/
def buttons_left : ℕ := 5

/-- Theorem stating that Georgia has 4 yellow buttons -/
theorem georgia_yellow_buttons : yellow_buttons = 4 := by
  sorry

end NUMINAMATH_CALUDE_georgia_yellow_buttons_l1751_175150


namespace NUMINAMATH_CALUDE_simplify_fraction_l1751_175169

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1751_175169


namespace NUMINAMATH_CALUDE_pablo_stack_difference_l1751_175139

/-- The height of Pablo's toy block stacks -/
def PabloStacks : ℕ → ℕ
| 0 => 5  -- First stack
| 1 => PabloStacks 0 + 2  -- Second stack
| 2 => PabloStacks 1 - 5  -- Third stack
| 3 => 21 - (PabloStacks 0 + PabloStacks 1 + PabloStacks 2)  -- Last stack
| _ => 0  -- Any other index

theorem pablo_stack_difference : PabloStacks 3 - PabloStacks 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_pablo_stack_difference_l1751_175139


namespace NUMINAMATH_CALUDE_part_one_part_two_l1751_175152

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := (k - 1) * x^2 - 4 * x + 3

-- Part 1
theorem part_one (k : ℝ) :
  (quadratic_equation k 1 = 0) → 
  (k = 2 ∧ ∃ x, x ≠ 1 ∧ quadratic_equation k x = 0 ∧ x = 3) :=
by sorry

-- Part 2
theorem part_two (k x₁ x₂ : ℝ) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) →
  (x₁^2 * x₂ + x₁ * x₂^2 = 3) →
  (k = -1) :=
by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1751_175152


namespace NUMINAMATH_CALUDE_carrots_grown_proof_l1751_175166

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 8

/-- The number of carrots Mary grew -/
def mary_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := sandy_carrots + mary_carrots

theorem carrots_grown_proof : total_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_carrots_grown_proof_l1751_175166


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l1751_175119

theorem largest_prime_factor_of_1001 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1001 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1001 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1001_l1751_175119


namespace NUMINAMATH_CALUDE_subtracted_value_l1751_175189

def original_number : ℝ := 54

theorem subtracted_value (x : ℝ) :
  ((original_number - x) / 7 = 7) ∧
  ((original_number - 34) / 10 = 2) →
  x = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l1751_175189


namespace NUMINAMATH_CALUDE_farmer_ploughing_problem_l1751_175143

/-- Represents the farmer's ploughing problem -/
def FarmerProblem (initial_productivity : ℝ) (productivity_increase : ℝ) 
  (total_area : ℝ) (days_ahead : ℕ) (initial_days : ℕ) : Prop :=
  let improved_productivity := initial_productivity * (1 + productivity_increase)
  let area_first_two_days := 2 * initial_productivity
  let remaining_area := total_area - area_first_two_days
  let remaining_days := remaining_area / improved_productivity
  initial_days = ⌈remaining_days⌉ + 2 + days_ahead

/-- The theorem statement for the farmer's ploughing problem -/
theorem farmer_ploughing_problem :
  FarmerProblem 120 0.25 1440 2 12 := by
  sorry

end NUMINAMATH_CALUDE_farmer_ploughing_problem_l1751_175143


namespace NUMINAMATH_CALUDE_f_properties_l1751_175137

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 + 1

theorem f_properties :
  let f := f
  ∃ (period : ℝ),
    (f (5 * Real.pi / 4) = Real.sqrt 3) ∧
    (period > 0 ∧ ∀ (x : ℝ), f (x + period) = f x) ∧
    (∀ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) → period ≤ p) ∧
    (f (-Real.pi / 5) < f (7 * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1751_175137


namespace NUMINAMATH_CALUDE_middle_term_coefficient_2x_plus_1_power_8_l1751_175102

theorem middle_term_coefficient_2x_plus_1_power_8 :
  let n : ℕ := 8
  let k : ℕ := n / 2
  let coeff : ℕ := Nat.choose n k * (2^k)
  coeff = 1120 :=
by sorry

end NUMINAMATH_CALUDE_middle_term_coefficient_2x_plus_1_power_8_l1751_175102


namespace NUMINAMATH_CALUDE_pants_price_problem_l1751_175193

theorem pants_price_problem (total_cost shirt_price pants_price shoes_price : ℚ) : 
  total_cost = 340 →
  shirt_price = (3/4) * pants_price →
  shoes_price = pants_price + 10 →
  total_cost = shirt_price + pants_price + shoes_price →
  pants_price = 120 := by
sorry

end NUMINAMATH_CALUDE_pants_price_problem_l1751_175193


namespace NUMINAMATH_CALUDE_triangle_circle_radii_relation_l1751_175177

/-- Given a triangle with sides of consecutive natural numbers, 
    the radius of its circumcircle (R) and the radius of its incircle (r) 
    satisfy the equation R = 2r + 1/(2r) -/
theorem triangle_circle_radii_relation (n : ℕ) (R r : ℝ) 
    (h1 : n > 1) 
    (h2 : R = (n^2 - 1) / (6 * r)) 
    (h3 : r^2 = (n^2 - 4) / 12) : 
  R = 2*r + 1/(2*r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_radii_relation_l1751_175177


namespace NUMINAMATH_CALUDE_tournament_prize_total_l1751_175145

def prize_money (first_place : ℕ) (interval : ℕ) : ℕ :=
  let second_place := first_place - interval
  let third_place := second_place - interval
  first_place + second_place + third_place

theorem tournament_prize_total :
  prize_money 2000 400 = 4800 :=
by sorry

end NUMINAMATH_CALUDE_tournament_prize_total_l1751_175145


namespace NUMINAMATH_CALUDE_right_triangle_angle_b_l1751_175195

/-- Given a right triangle ABC with ∠A = 70°, prove that ∠B = 20° -/
theorem right_triangle_angle_b (A B C : ℝ) : 
  A + B + C = 180 →  -- Sum of angles in a triangle is 180°
  C = 90 →           -- One angle is 90° (right angle)
  A = 70 →           -- Given ∠A = 70°
  B = 20 :=          -- To prove: ∠B = 20°
by sorry

end NUMINAMATH_CALUDE_right_triangle_angle_b_l1751_175195


namespace NUMINAMATH_CALUDE_sports_club_overlap_l1751_175122

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h1 : total = 35)
  (h2 : badminton = 15)
  (h3 : tennis = 18)
  (h4 : neither = 5) :
  badminton + tennis - (total - neither) = 3 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l1751_175122


namespace NUMINAMATH_CALUDE_f_zero_at_three_l1751_175172

def f (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 4 * x^2 - 5 * x + s

theorem f_zero_at_three (s : ℝ) : f 3 s = 0 ↔ s = -885 := by sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l1751_175172


namespace NUMINAMATH_CALUDE_parallel_lines_chord_distance_l1751_175144

theorem parallel_lines_chord_distance (r : ℝ) (d : ℝ) : 
  r > 0 → d > 0 →
  36 * r^2 = 36 * 324 + (1/4) * d^2 * 36 →
  40 * r^2 = 40 * 400 + 40 * d^2 →
  d = Real.sqrt (304/3) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_chord_distance_l1751_175144


namespace NUMINAMATH_CALUDE_no_single_digit_A_with_integer_solutions_l1751_175101

theorem no_single_digit_A_with_integer_solutions : 
  ∀ A : ℕ, 1 ≤ A ∧ A ≤ 9 → 
  ¬∃ x : ℕ, x > 0 ∧ x^2 - 2*A*x + A*10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_single_digit_A_with_integer_solutions_l1751_175101


namespace NUMINAMATH_CALUDE_equation_solutions_l1751_175178

theorem equation_solutions :
  (∃ x : ℝ, 8 * (x + 1)^3 = 64 ∧ x = 1) ∧
  (∃ x y : ℝ, (x + 1)^2 = 100 ∧ (y + 1)^2 = 100 ∧ x = 9 ∧ y = -11) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1751_175178


namespace NUMINAMATH_CALUDE_race_distance_proof_l1751_175148

/-- The total distance of a race where:
    - A covers the distance in 45 seconds
    - B covers the distance in 60 seconds
    - A beats B by 50 meters
-/
def race_distance : ℝ := 150

theorem race_distance_proof :
  ∀ (a_time b_time : ℝ) (lead : ℝ),
  a_time = 45 ∧ 
  b_time = 60 ∧ 
  lead = 50 →
  race_distance = (lead * b_time) / (b_time / a_time - 1) :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l1751_175148


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1751_175161

theorem line_intersects_circle (a : ℝ) (h : a ≠ 0) :
  let d := |2*a| / Real.sqrt (a^2 + 1)
  d < 3 := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1751_175161


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1751_175170

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 10) :
  a 3 + a 7 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1751_175170


namespace NUMINAMATH_CALUDE_arg_ratio_of_unit_complex_l1751_175114

theorem arg_ratio_of_unit_complex (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = 1)
  (h₃ : z₂ - z₁ = -1) :
  Complex.arg (z₁ / z₂) = π / 3 ∨ Complex.arg (z₁ / z₂) = 5 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arg_ratio_of_unit_complex_l1751_175114


namespace NUMINAMATH_CALUDE_abc_inequality_l1751_175106

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  a + b + c + 2 * a * b * c > a * b + b * c + c * a + 2 * Real.sqrt (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1751_175106


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1751_175105

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ := d.length * d.width * d.height

/-- Calculates the number of blocks that can fit in one layer of the larger box -/
def blocksPerLayer (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (largeBox.length / smallBox.length) * (largeBox.width / smallBox.width)

/-- The main theorem stating the maximum number of blocks that can fit -/
theorem max_blocks_fit (largeBox smallBox : BoxDimensions) :
  largeBox = BoxDimensions.mk 5 4 4 →
  smallBox = BoxDimensions.mk 3 2 1 →
  blocksPerLayer largeBox smallBox * (largeBox.height / smallBox.height) = 12 :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1751_175105


namespace NUMINAMATH_CALUDE_unique_assignment_l1751_175131

/-- Represents a valid assignment of digits to letters -/
structure Assignment where
  a : Fin 5
  m : Fin 5
  e : Fin 5
  h : Fin 5
  z : Fin 5
  different : a ≠ m ∧ a ≠ e ∧ a ≠ h ∧ a ≠ z ∧ m ≠ e ∧ m ≠ h ∧ m ≠ z ∧ e ≠ h ∧ e ≠ z ∧ h ≠ z

/-- The inequalities that must be satisfied -/
def satisfies_inequalities (assign : Assignment) : Prop :=
  3 > assign.a.val + 1 ∧
  assign.a.val + 1 > assign.m.val + 1 ∧
  assign.m.val + 1 < assign.e.val + 1 ∧
  assign.e.val + 1 < assign.h.val + 1 ∧
  assign.h.val + 1 < assign.a.val + 1

/-- The theorem stating that the only valid assignment results in ZAMENA = 541234 -/
theorem unique_assignment :
  ∀ (assign : Assignment),
    satisfies_inequalities assign →
    assign.z.val = 4 ∧
    assign.a.val = 3 ∧
    assign.m.val = 0 ∧
    assign.e.val = 1 ∧
    assign.h.val = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_assignment_l1751_175131


namespace NUMINAMATH_CALUDE_equality_addition_l1751_175175

theorem equality_addition (a b : ℝ) : a = b → a + 3 = 3 + b := by
  sorry

end NUMINAMATH_CALUDE_equality_addition_l1751_175175


namespace NUMINAMATH_CALUDE_arithmetic_equation_l1751_175171

theorem arithmetic_equation : 4 * (8 - 2 + 3) - 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l1751_175171


namespace NUMINAMATH_CALUDE_count_perfect_squares_l1751_175179

theorem count_perfect_squares (n : ℕ) : 
  (Finset.filter (fun k => 16 * k * k < 5000) (Finset.range n)).card = 17 :=
sorry

end NUMINAMATH_CALUDE_count_perfect_squares_l1751_175179
