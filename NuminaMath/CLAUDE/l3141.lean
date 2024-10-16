import Mathlib

namespace NUMINAMATH_CALUDE_average_of_last_part_calculation_l3141_314177

def average_of_last_part (total_count : ℕ) (total_average : ℚ) (first_part_count : ℕ) (first_part_average : ℚ) (middle_result : ℚ) : ℚ :=
  let last_part_count := total_count - first_part_count - 1
  let total_sum := total_count * total_average
  let first_part_sum := first_part_count * first_part_average
  (total_sum - first_part_sum - middle_result) / last_part_count

theorem average_of_last_part_calculation :
  average_of_last_part 25 50 12 14 878 = 204 / 13 := by
  sorry

end NUMINAMATH_CALUDE_average_of_last_part_calculation_l3141_314177


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3141_314114

theorem quadratic_function_property (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 1 = f 3 ∧ f 1 > f 4) → (a < 0 ∧ 4 * a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3141_314114


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l3141_314104

theorem inscribed_sphere_volume (r h l : ℝ) (V : ℝ) :
  r = 2 →
  2 * π * r * l = 8 * π →
  h^2 + r^2 = l^2 →
  (h - V^(1/3) * ((3 * r) / (4 * π))^(1/3)) / l = V^(1/3) * ((3 * r) / (4 * π))^(1/3) / r →
  V = (32 * Real.sqrt 3) / 27 * π :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l3141_314104


namespace NUMINAMATH_CALUDE_percentage_passed_both_l3141_314180

theorem percentage_passed_both (failed_hindi failed_english failed_both : ℚ) 
  (h1 : failed_hindi = 35 / 100)
  (h2 : failed_english = 45 / 100)
  (h3 : failed_both = 20 / 100) :
  1 - (failed_hindi + failed_english - failed_both) = 40 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_both_l3141_314180


namespace NUMINAMATH_CALUDE_supermarket_queue_clearing_time_l3141_314127

/-- The average number of people lining up to pay per hour -/
def average_customers_per_hour : ℝ := 60

/-- The number of people a single cashier can handle per hour -/
def cashier_capacity_per_hour : ℝ := 80

/-- The number of hours it takes for one cashier to clear the line -/
def hours_for_one_cashier : ℝ := 4

/-- The number of cashiers working in the second scenario -/
def num_cashiers : ℕ := 2

/-- The time it takes for two cashiers to clear the line -/
def time_for_two_cashiers : ℝ := 0.8

theorem supermarket_queue_clearing_time :
  2 * cashier_capacity_per_hour * time_for_two_cashiers = 
  average_customers_per_hour * time_for_two_cashiers + 
  (cashier_capacity_per_hour * hours_for_one_cashier - average_customers_per_hour * hours_for_one_cashier) :=
by sorry

end NUMINAMATH_CALUDE_supermarket_queue_clearing_time_l3141_314127


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3141_314162

theorem polynomial_evaluation :
  let a : ℚ := 7/3
  (4 * a^2 - 11 * a + 7) * (2 * a - 3) = 140/27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3141_314162


namespace NUMINAMATH_CALUDE_triangle_problem_l3141_314175

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : 1 + (Real.tan t.A / Real.tan t.B) = 2 * t.c / t.b)
  (h2 : t.a = Real.sqrt 3) :
  t.A = π/3 ∧ 
  (∀ (t' : Triangle), t'.a = Real.sqrt 3 → t'.b * t'.c ≤ t.b * t.c → 
    t.b = t.c ∧ t.b = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3141_314175


namespace NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l3141_314103

theorem tens_digit_of_8_pow_2023 : ∃ k : ℕ, 8^2023 = 100 * k + 12 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_8_pow_2023_l3141_314103


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3141_314106

theorem quadratic_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 6 * x + 1 ≠ 0) → a > 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3141_314106


namespace NUMINAMATH_CALUDE_becky_necklaces_l3141_314138

theorem becky_necklaces (initial : ℕ) : 
  initial - 3 + 5 - 15 = 37 → initial = 50 := by
  sorry

#check becky_necklaces

end NUMINAMATH_CALUDE_becky_necklaces_l3141_314138


namespace NUMINAMATH_CALUDE_sequence_bound_l3141_314140

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i : ℕ, i > 0 → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, i > 0 → j > 0 → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l3141_314140


namespace NUMINAMATH_CALUDE_convergence_and_bound_l3141_314193

def u : ℕ → ℚ
  | 0 => 1/3
  | k+1 => 3 * u k - 3 * (u k)^2

theorem convergence_and_bound :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n - 1/2| < ε) ∧
  (∀ k < 9, |u k - 1/2| > 1/2^500) ∧
  |u 9 - 1/2| ≤ 1/2^500 :=
sorry

end NUMINAMATH_CALUDE_convergence_and_bound_l3141_314193


namespace NUMINAMATH_CALUDE_propositions_relationship_l3141_314108

theorem propositions_relationship (x : ℝ) :
  (∀ x, x < 3 → x < 5) ↔ (∀ x, x ≥ 5 → x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_propositions_relationship_l3141_314108


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3141_314112

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, |x - 1| < 1 → x^2 - 5*x < 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x < 0 ∧ |x - 1| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3141_314112


namespace NUMINAMATH_CALUDE_equality_of_absolute_value_sums_l3141_314101

theorem equality_of_absolute_value_sums (a b c d : ℝ) 
  (h : ∀ x : ℝ, |2*x + 4| + |a*x + b| = |c*x + d|) : 
  d = 2*c := by sorry

end NUMINAMATH_CALUDE_equality_of_absolute_value_sums_l3141_314101


namespace NUMINAMATH_CALUDE_prime_factorization_property_l3141_314125

theorem prime_factorization_property (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃ y : ℕ, y ≤ p / 2 ∧ ¬∃ (a b : ℕ), a > y ∧ b > y ∧ p * y + 1 = a * b :=
by sorry

end NUMINAMATH_CALUDE_prime_factorization_property_l3141_314125


namespace NUMINAMATH_CALUDE_problem_solution_l3141_314116

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/5 = 5*y) : x = 625/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3141_314116


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3141_314168

theorem rationalize_denominator : 
  (50 + Real.sqrt 8) / (Real.sqrt 50 + Real.sqrt 8) = 
  (50 * (Real.sqrt 50 - Real.sqrt 8) + 12) / 42 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3141_314168


namespace NUMINAMATH_CALUDE_boxes_theorem_l3141_314139

def boxes_problem (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ) : Prop :=
  let neither := total - (markers + erasers - both)
  neither = 3

theorem boxes_theorem : boxes_problem 12 7 5 3 := by
  sorry

end NUMINAMATH_CALUDE_boxes_theorem_l3141_314139


namespace NUMINAMATH_CALUDE_height_is_four_l3141_314100

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- A right-angled triangle on a parabola -/
structure RightTriangleOnParabola where
  A : ParabolaPoint
  B : ParabolaPoint
  C : ParabolaPoint
  right_angle_at_C : (B.x - C.x) * (A.x - C.x) + (B.y - C.y) * (A.y - C.y) = 0
  hypotenuse_parallel_to_y : A.x = B.x

/-- The height from the hypotenuse of a right-angled triangle on a parabola -/
def height_from_hypotenuse (t : RightTriangleOnParabola) : ℝ :=
  |t.B.x - t.C.x|

/-- Theorem: The height from the hypotenuse is 4 -/
theorem height_is_four (t : RightTriangleOnParabola) : height_from_hypotenuse t = 4 := by
  sorry

end NUMINAMATH_CALUDE_height_is_four_l3141_314100


namespace NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l3141_314134

theorem min_product_of_reciprocal_sum (a b : ℕ+) : 
  (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 6 → 
  ∀ c d : ℕ+, (1 : ℚ) / c + (1 : ℚ) / (3 * d) = (1 : ℚ) / 6 → 
  a * b ≤ c * d ∧ a * b = 98 :=
by sorry

end NUMINAMATH_CALUDE_min_product_of_reciprocal_sum_l3141_314134


namespace NUMINAMATH_CALUDE_cos_plus_ax_monotonic_iff_l3141_314145

/-- A function f: ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def IsMonotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: f(x) = cos x + ax is monotonic iff a ∈ (-∞, -1] ∪ [1, +∞) -/
theorem cos_plus_ax_monotonic_iff (a : ℝ) :
  IsMonotonic (fun x => Real.cos x + a * x) ↔ a ≤ -1 ∨ a ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_cos_plus_ax_monotonic_iff_l3141_314145


namespace NUMINAMATH_CALUDE_parallel_condition_perpendicular_condition_l3141_314117

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := (2*a + 1)*x + (a + 2)*y + 3 = 0
def l2 (a x y : ℝ) : Prop := (a - 1)*x - 2*y + 2 = 0

-- Define parallel and perpendicular conditions
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ ∃ k, l2 a (x + k * (2*a + 1)) (y + k * (a + 2))

def perpendicular (a : ℝ) : Prop := ∀ x1 y1 x2 y2, 
  l1 a x1 y1 → l2 a x2 y2 → (x2 - x1) * (2*a + 1) + (y2 - y1) * (a + 2) = 0

-- State the theorems
theorem parallel_condition : ∀ a : ℝ, parallel a ↔ a = 0 := by sorry

theorem perpendicular_condition : ∀ a : ℝ, perpendicular a ↔ a = -1 ∨ a = 5/2 := by sorry

end NUMINAMATH_CALUDE_parallel_condition_perpendicular_condition_l3141_314117


namespace NUMINAMATH_CALUDE_sphere_division_l3141_314119

theorem sphere_division (π : ℝ) (h_π : π > 0) : 
  ∃ (R : ℝ), R > 0 ∧ 
  (4 / 3 * π * R^3 = 125 * (4 / 3 * π * 1^3)) ∧ 
  R = 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_division_l3141_314119


namespace NUMINAMATH_CALUDE_wendy_score_l3141_314129

/-- Wendy's video game scoring system -/
structure GameScore where
  points_per_treasure : ℕ
  treasures_level1 : ℕ
  treasures_level2 : ℕ

/-- Calculate the total score for Wendy's game -/
def total_score (game : GameScore) : ℕ :=
  (game.treasures_level1 + game.treasures_level2) * game.points_per_treasure

/-- Theorem: Wendy's total score is 35 points -/
theorem wendy_score : 
  ∀ (game : GameScore), 
  game.points_per_treasure = 5 → 
  game.treasures_level1 = 4 → 
  game.treasures_level2 = 3 → 
  total_score game = 35 := by
  sorry

end NUMINAMATH_CALUDE_wendy_score_l3141_314129


namespace NUMINAMATH_CALUDE_log_base_8_equals_3_l3141_314166

theorem log_base_8_equals_3 (y : ℝ) (h : Real.log y / Real.log 8 = 3) : y = 512 := by
  sorry

end NUMINAMATH_CALUDE_log_base_8_equals_3_l3141_314166


namespace NUMINAMATH_CALUDE_number_of_students_l3141_314174

theorem number_of_students (S : ℕ) (N : ℕ) : 
  (4 * S + 3 = N) → (5 * S = N + 6) → S = 9 := by
sorry

end NUMINAMATH_CALUDE_number_of_students_l3141_314174


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l3141_314152

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 36 * x + c = 0) →
  (a + c = 37) →
  (a < c) →
  (a = (37 - Real.sqrt 73) / 2 ∧ c = (37 + Real.sqrt 73) / 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l3141_314152


namespace NUMINAMATH_CALUDE_special_quadratic_relation_l3141_314179

theorem special_quadratic_relation (q a b : ℕ) (h : a^2 - q*a*b + b^2 = q) :
  ∃ (c : ℤ), c ≠ a ∧ c^2 - q*b*c + b^2 = q ∧ ∃ (k : ℕ), q = k^2 := by
  sorry

end NUMINAMATH_CALUDE_special_quadratic_relation_l3141_314179


namespace NUMINAMATH_CALUDE_remainder_x5_plus_3_div_x_minus_3_squared_l3141_314178

open Polynomial

theorem remainder_x5_plus_3_div_x_minus_3_squared (x : ℝ) :
  ∃ q : Polynomial ℝ, X^5 + C 3 = (X - C 3)^2 * q + (C 405 * X - C 969) := by
  sorry

end NUMINAMATH_CALUDE_remainder_x5_plus_3_div_x_minus_3_squared_l3141_314178


namespace NUMINAMATH_CALUDE_sequence_max_value_l3141_314165

def x (n : ℕ) : ℚ := (n - 1 : ℚ) / ((n : ℚ)^2 + 1)

theorem sequence_max_value :
  (∀ n : ℕ, x n ≤ (1 : ℚ) / 5) ∧
  x 2 = (1 : ℚ) / 5 ∧
  x 3 = (1 : ℚ) / 5 :=
sorry

end NUMINAMATH_CALUDE_sequence_max_value_l3141_314165


namespace NUMINAMATH_CALUDE_expensive_coat_savings_l3141_314121

/-- Represents a coat with its cost and lifespan. -/
structure Coat where
  cost : ℕ
  lifespan : ℕ

/-- Calculates the total cost of a coat over a given period. -/
def totalCost (coat : Coat) (period : ℕ) : ℕ :=
  (period + coat.lifespan - 1) / coat.lifespan * coat.cost

/-- Proves that buying the more expensive coat saves $120 over 30 years. -/
theorem expensive_coat_savings :
  let expensiveCoat : Coat := { cost := 300, lifespan := 15 }
  let cheapCoat : Coat := { cost := 120, lifespan := 5 }
  let period : ℕ := 30
  totalCost cheapCoat period - totalCost expensiveCoat period = 120 := by
  sorry


end NUMINAMATH_CALUDE_expensive_coat_savings_l3141_314121


namespace NUMINAMATH_CALUDE_min_fencing_length_l3141_314148

/-- Represents the dimensions of a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Calculates the perimeter of a rectangular garden, excluding one side (against the wall) -/
def Garden.fencingLength (g : Garden) : ℝ := g.length + 2 * g.width

/-- The minimum fencing length for a garden with area 50 m² is 20 meters -/
theorem min_fencing_length :
  ∀ g : Garden, g.area = 50 → g.fencingLength ≥ 20 ∧ 
  ∃ g' : Garden, g'.area = 50 ∧ g'.fencingLength = 20 := by
  sorry


end NUMINAMATH_CALUDE_min_fencing_length_l3141_314148


namespace NUMINAMATH_CALUDE_solution_satisfies_conditions_l3141_314198

/-- Represents the number of teeth each person has -/
structure TeethCount where
  dima : ℕ
  yulia : ℕ
  kolya : ℕ
  vanya : ℕ

/-- Checks if the given teeth count satisfies all conditions of the problem -/
def satisfiesConditions (tc : TeethCount) : Prop :=
  tc.dima = tc.yulia + 2 ∧
  tc.kolya = tc.dima + tc.yulia ∧
  tc.vanya = 2 * tc.kolya ∧
  tc.dima + tc.yulia + tc.kolya + tc.vanya = 64

/-- The theorem stating that the solution satisfies all conditions -/
theorem solution_satisfies_conditions : 
  satisfiesConditions ⟨9, 7, 16, 32⟩ := by sorry

end NUMINAMATH_CALUDE_solution_satisfies_conditions_l3141_314198


namespace NUMINAMATH_CALUDE_fraction_transformation_l3141_314122

theorem fraction_transformation (x y : ℝ) : 
  -(x - y) / (x + y) = (-x + y) / (x + y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3141_314122


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3141_314167

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | x > -2 ∧ x < 0}

theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | x > -2 ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3141_314167


namespace NUMINAMATH_CALUDE_tabithas_final_amount_l3141_314135

/-- Calculates Tabitha's remaining money after various transactions --/
def tabithas_remaining_money (initial_amount : ℚ) (given_to_mom : ℚ) (num_items : ℕ) (item_cost : ℚ) : ℚ :=
  let after_mom := initial_amount - given_to_mom
  let after_investment := after_mom / 2
  let spent_on_items := num_items * item_cost
  after_investment - spent_on_items

/-- Theorem stating that Tabitha's remaining money is 6 dollars --/
theorem tabithas_final_amount :
  tabithas_remaining_money 25 8 5 (1/2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_tabithas_final_amount_l3141_314135


namespace NUMINAMATH_CALUDE_surviving_cells_after_6_hours_l3141_314181

def cell_population (n : ℕ) : ℕ := 2^n + 1

theorem surviving_cells_after_6_hours :
  cell_population 6 = 65 :=
sorry

end NUMINAMATH_CALUDE_surviving_cells_after_6_hours_l3141_314181


namespace NUMINAMATH_CALUDE_calculation_proof_l3141_314187

theorem calculation_proof :
  let four_million : ℝ := 4 * 10^6
  let four_hundred_thousand : ℝ := 4 * 10^5
  let four_billion : ℝ := 4 * 10^9
  (four_million * four_hundred_thousand + four_billion) = 1.604 * 10^12 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3141_314187


namespace NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l3141_314123

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^3 - 8*x^2 + a*x - b = 0 ∧
    y^3 - 8*y^2 + a*y - b = 0 ∧
    z^3 - 8*z^2 + a*z - b = 0) →
  (∀ a' b' : ℝ, (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 8*x^2 + a'*x - b' = 0 ∧
    y^3 - 8*y^2 + a'*y - b' = 0 ∧
    z^3 - 8*z^2 + a'*z - b' = 0) →
  a + b ≤ a' + b') →
  a + b = 27 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l3141_314123


namespace NUMINAMATH_CALUDE_not_prime_4n_squared_minus_1_l3141_314110

theorem not_prime_4n_squared_minus_1 (n : ℤ) (h : n ≥ 2) : ¬ Prime (4 * n^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_4n_squared_minus_1_l3141_314110


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l3141_314115

open Real

theorem trigonometric_calculations :
  (sin (-60 * π / 180) = -Real.sqrt 3 / 2) ∧
  (cos (-45 * π / 180) = Real.sqrt 2 / 2) ∧
  (tan (-945 * π / 180) = -1) := by
  sorry

-- Definitions and properties used in the proof
axiom sine_odd (x : ℝ) : sin (-x) = -sin x
axiom cosine_even (x : ℝ) : cos (-x) = cos x
axiom tangent_odd (x : ℝ) : tan (-x) = -tan x
axiom sin_60 : sin (60 * π / 180) = Real.sqrt 3 / 2
axiom cos_45 : cos (45 * π / 180) = Real.sqrt 2 / 2
axiom tan_45 : tan (45 * π / 180) = 1
axiom tan_period (x : ℝ) (k : ℤ) : tan (x + k * π) = tan x

end NUMINAMATH_CALUDE_trigonometric_calculations_l3141_314115


namespace NUMINAMATH_CALUDE_popcorn_servings_needed_l3141_314173

/-- The number of pieces of popcorn in a serving -/
def serving_size : ℕ := 60

/-- The number of pieces Jared can eat -/
def jared_consumption : ℕ := 150

/-- The number of friends who can eat 80 pieces each -/
def friends_80 : ℕ := 3

/-- The number of friends who can eat 200 pieces each -/
def friends_200 : ℕ := 3

/-- The number of friends who can eat 100 pieces each -/
def friends_100 : ℕ := 4

/-- The number of pieces each friend in the first group can eat -/
def consumption_80 : ℕ := 80

/-- The number of pieces each friend in the second group can eat -/
def consumption_200 : ℕ := 200

/-- The number of pieces each friend in the third group can eat -/
def consumption_100 : ℕ := 100

/-- The theorem stating the number of servings needed -/
theorem popcorn_servings_needed : 
  (jared_consumption + 
   friends_80 * consumption_80 + 
   friends_200 * consumption_200 + 
   friends_100 * consumption_100 + 
   serving_size - 1) / serving_size = 24 :=
sorry

end NUMINAMATH_CALUDE_popcorn_servings_needed_l3141_314173


namespace NUMINAMATH_CALUDE_num_winning_scores_l3141_314183

/-- Represents a cross country meet with 3 teams of 4 runners each -/
structure CrossCountryMeet where
  numTeams : Nat
  runnersPerTeam : Nat
  totalRunners : Nat
  (team_count : numTeams = 3)
  (runner_count : runnersPerTeam = 4)
  (total_runners : totalRunners = numTeams * runnersPerTeam)

/-- Calculates the total score of all runners -/
def totalScore (meet : CrossCountryMeet) : Nat :=
  meet.totalRunners * (meet.totalRunners + 1) / 2

/-- Calculates the minimum possible winning score -/
def minWinningScore (meet : CrossCountryMeet) : Nat :=
  meet.runnersPerTeam * (meet.runnersPerTeam + 1) / 2

/-- Calculates the maximum possible winning score -/
def maxWinningScore (meet : CrossCountryMeet) : Nat :=
  totalScore meet / meet.numTeams

/-- Theorem stating the number of different winning scores possible -/
theorem num_winning_scores (meet : CrossCountryMeet) :
  (maxWinningScore meet - minWinningScore meet + 1) = 17 := by
  sorry


end NUMINAMATH_CALUDE_num_winning_scores_l3141_314183


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l3141_314128

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-5, 0)
def right_focus : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2

-- Define the right angle condition
def right_angle (P : ℝ × ℝ) : Prop :=
  let F₁ := left_focus
  let F₂ := right_focus
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0

-- Theorem statement
theorem hyperbola_triangle_area (P : ℝ × ℝ) :
  point_on_hyperbola P → right_angle P → 
  let F₁ := left_focus
  let F₂ := right_focus
  let area := (1/2) * ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt * 
              ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt
  area = 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l3141_314128


namespace NUMINAMATH_CALUDE_equal_digit_probability_l3141_314169

def num_dice : ℕ := 6
def sides_per_die : ℕ := 20
def one_digit_outcomes : ℕ := 9
def two_digit_outcomes : ℕ := 11

def prob_one_digit : ℚ := one_digit_outcomes / sides_per_die
def prob_two_digit : ℚ := two_digit_outcomes / sides_per_die

def equal_digit_prob : ℚ := (num_dice.choose (num_dice / 2)) *
  (prob_one_digit ^ (num_dice / 2)) * (prob_two_digit ^ (num_dice / 2))

theorem equal_digit_probability :
  equal_digit_prob = 4851495 / 16000000 := by sorry

end NUMINAMATH_CALUDE_equal_digit_probability_l3141_314169


namespace NUMINAMATH_CALUDE_range_of_a_l3141_314164

-- Define a decreasing function on (0, +∞)
variable (f : ℝ → ℝ)
variable (h_decreasing : ∀ x y, 0 < x → 0 < y → x < y → f y < f x)

-- Define the domain of f
variable (h_domain : ∀ x, 0 < x → f x ∈ Set.range f)

-- Define the variable a
variable (a : ℝ)

-- State the theorem
theorem range_of_a (h_ineq : f (2*a^2 + a + 1) < f (3*a^2 - 4*a + 1)) :
  (0 < a ∧ a < 1/3) ∨ (1 < a ∧ a < 5) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3141_314164


namespace NUMINAMATH_CALUDE_average_weight_a_and_b_l3141_314172

/-- Given three weights a, b, and c, prove that their average weight of a and b is 40 kg
    under certain conditions. -/
theorem average_weight_a_and_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- The average weight of a, b, and c is 45 kg
  (b + c) / 2 = 43 →       -- The average weight of b and c is 43 kg
  b = 31 →                 -- The weight of b is 31 kg
  (a + b) / 2 = 40 :=      -- The average weight of a and b is 40 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_a_and_b_l3141_314172


namespace NUMINAMATH_CALUDE_difference_of_squares_l3141_314192

theorem difference_of_squares (m : ℤ) : 
  (∃ x y : ℤ, m = x^2 - y^2) ↔ ¬(∃ k : ℤ, m = 4*k + 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3141_314192


namespace NUMINAMATH_CALUDE_michael_fish_count_l3141_314154

theorem michael_fish_count (initial_fish : Float) (given_fish : Float) : 
  initial_fish = 49.0 → given_fish = 18.0 → initial_fish + given_fish = 67.0 := by
  sorry

end NUMINAMATH_CALUDE_michael_fish_count_l3141_314154


namespace NUMINAMATH_CALUDE_running_time_difference_l3141_314142

/-- The time difference for running 5 miles between new and old shoes -/
theorem running_time_difference 
  (old_shoe_time : ℕ) -- Time to run one mile in old shoes
  (new_shoe_time : ℕ) -- Time to run one mile in new shoes
  (distance : ℕ) -- Distance to run in miles
  (h1 : old_shoe_time = 10)
  (h2 : new_shoe_time = 13)
  (h3 : distance = 5) :
  new_shoe_time * distance - old_shoe_time * distance = 15 :=
by sorry

end NUMINAMATH_CALUDE_running_time_difference_l3141_314142


namespace NUMINAMATH_CALUDE_conditional_extremum_l3141_314136

/-- The objective function to be optimized -/
def f (x₁ x₂ : ℝ) : ℝ := x₁^2 + x₂^2 - x₁*x₂ + x₁ + x₂ - 6

/-- The constraint function -/
def g (x₁ x₂ : ℝ) : ℝ := x₁ + x₂ + 3

/-- Theorem stating the conditional extremum of f subject to g -/
theorem conditional_extremum :
  ∃ (x₁ x₂ : ℝ), g x₁ x₂ = 0 ∧ 
    (∀ (y₁ y₂ : ℝ), g y₁ y₂ = 0 → f x₁ x₂ ≤ f y₁ y₂) ∧
    x₁ = -3/2 ∧ x₂ = -3/2 ∧ f x₁ x₂ = -9/2 :=
sorry

end NUMINAMATH_CALUDE_conditional_extremum_l3141_314136


namespace NUMINAMATH_CALUDE_solve_for_m_l3141_314197

theorem solve_for_m : ∀ m : ℝ, (∃ x : ℝ, x = 3 ∧ 3 * m - 2 * x = 6) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3141_314197


namespace NUMINAMATH_CALUDE_twenty_machines_four_minutes_l3141_314186

/-- The number of bottles produced by a given number of machines in a given time -/
def bottles_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  let bottles_per_minute_per_machine := 270 / 6
  machines * bottles_per_minute_per_machine * minutes

/-- Theorem stating that 20 machines produce 3600 bottles in 4 minutes -/
theorem twenty_machines_four_minutes :
  bottles_produced 20 4 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_twenty_machines_four_minutes_l3141_314186


namespace NUMINAMATH_CALUDE_part_one_part_two_l3141_314141

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h_not_necessary : ∃ x, ¬(p x a) ∧ q x) : 1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3141_314141


namespace NUMINAMATH_CALUDE_no_real_solutions_for_matrix_equation_l3141_314158

theorem no_real_solutions_for_matrix_equation : 
  ¬∃ (x : ℝ), (3 * x * x - 8 = 2 * x^2 - 3 * x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_matrix_equation_l3141_314158


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3141_314109

/-- Given an equilateral triangle with side length 2 and three right-angled isosceles triangles
    constructed on its sides, if the total area of the three right-angled isosceles triangles
    equals the area of the equilateral triangle, then the length of the congruent sides of
    one right-angled isosceles triangle is √(6√3)/3. -/
theorem isosceles_triangle_side_length :
  let equilateral_side : ℝ := 2
  let equilateral_area : ℝ := Real.sqrt 3 / 4 * equilateral_side^2
  let isosceles_area : ℝ := equilateral_area / 3
  let isosceles_side : ℝ := Real.sqrt (2 * isosceles_area)
  isosceles_side = Real.sqrt (6 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3141_314109


namespace NUMINAMATH_CALUDE_complex_point_on_line_l3141_314188

theorem complex_point_on_line (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  (z.re - z.im = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_point_on_line_l3141_314188


namespace NUMINAMATH_CALUDE_x_equals_six_l3141_314124

def floor (y : ℤ) : ℤ :=
  if y % 2 = 0 then y / 2 + 1 else 2 * y + 1

theorem x_equals_six :
  ∃ x : ℤ, floor x * floor 3 = 28 ∧ x = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_x_equals_six_l3141_314124


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3141_314170

theorem no_integer_solutions : 
  ¬∃ (y z : ℤ), (2*y^2 - 2*y*z - z^2 = 15) ∧ 
                (6*y*z + 2*z^2 = 60) ∧ 
                (y^2 + 8*z^2 = 90) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3141_314170


namespace NUMINAMATH_CALUDE_students_liking_both_desserts_l3141_314171

theorem students_liking_both_desserts 
  (total : Nat) 
  (like_apple : Nat) 
  (like_chocolate : Nat) 
  (like_neither : Nat) 
  (h1 : total = 40)
  (h2 : like_apple = 18)
  (h3 : like_chocolate = 15)
  (h4 : like_neither = 12) :
  like_apple + like_chocolate - (total - like_neither) = 5 := by
  sorry

end NUMINAMATH_CALUDE_students_liking_both_desserts_l3141_314171


namespace NUMINAMATH_CALUDE_minimum_candies_in_can_l3141_314143

theorem minimum_candies_in_can (red green : ℕ) : 
  (red > 0) →
  (green > 0) →
  ((3 * red) / 5 : ℚ) = (3 / 8) * ((3 * red) / 5 + (2 * green) / 5) →
  (∀ r g : ℕ, r > 0 ∧ g > 0 ∧ ((3 * r) / 5 : ℚ) = (3 / 8) * ((3 * r) / 5 + (2 * g) / 5) → r + g ≥ red + green) →
  red + green = 35 :=
by sorry

end NUMINAMATH_CALUDE_minimum_candies_in_can_l3141_314143


namespace NUMINAMATH_CALUDE_choir_arrangement_min_choir_members_l3141_314189

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
by sorry

theorem min_choir_members : 
  ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_min_choir_members_l3141_314189


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3141_314113

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 4 * x - 1
  ∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 6 / 2 ∧
              x₂ = -1 - Real.sqrt 6 / 2 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3141_314113


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3141_314156

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : is_geometric_sequence a q)
  (h_q_bounds : 0 < q ∧ q < 1/2)
  (h_property : ∀ k : ℕ, k > 0 → ∃ n : ℕ, a k - (a (k+1) + a (k+2)) = a n) :
  q = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3141_314156


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3141_314120

theorem simplify_trig_expression (x : ℝ) : 
  (Real.sqrt 3 / 2) * Real.sin x - (1 / 2) * Real.cos x = Real.sin (x - π / 6) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3141_314120


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l3141_314160

/-- Given that 34 cows eat 34 bags of husk in 34 days, 
    prove that one cow will eat one bag of husk in 34 days. -/
theorem one_cow_one_bag_days : 
  ∀ (cows bags days : ℕ), 
  cows = 34 → bags = 34 → days = 34 →
  (cows * bags = cows * days) →
  1 * days = 34 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l3141_314160


namespace NUMINAMATH_CALUDE_train_length_l3141_314102

/-- The length of a train given specific conditions. -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 200 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time - initial_distance = 200 := by
sorry

end NUMINAMATH_CALUDE_train_length_l3141_314102


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l3141_314184

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {y | ∃ x, y = 2^x}

def N : Set Int := {x | x^2 - x - 2 = 0}

theorem intersection_complement_M_and_N :
  (U \ M) ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l3141_314184


namespace NUMINAMATH_CALUDE_prob_reach_opposite_after_six_moves_l3141_314137

/-- Represents a cube with its vertices and edges. -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)

/-- Represents a bug's movement on the cube. -/
structure BugMovement (cube : Cube) where
  start_vertex : Fin 8
  num_moves : Nat
  prob_each_edge : ℝ

/-- The probability of the bug reaching the opposite vertex after a specific number of moves. -/
def prob_reach_opposite (cube : Cube) (movement : BugMovement cube) : ℝ :=
  sorry

/-- Theorem stating that the probability of reaching the opposite vertex after six moves is 1/8. -/
theorem prob_reach_opposite_after_six_moves (cube : Cube) (movement : BugMovement cube) :
  movement.num_moves = 6 →
  movement.prob_each_edge = 1/3 →
  prob_reach_opposite cube movement = 1/8 :=
sorry

end NUMINAMATH_CALUDE_prob_reach_opposite_after_six_moves_l3141_314137


namespace NUMINAMATH_CALUDE_age_difference_of_parents_l3141_314196

theorem age_difference_of_parents (albert_age brother_age father_age mother_age : ℕ) :
  father_age = albert_age + 48 →
  mother_age = brother_age + 46 →
  brother_age = albert_age - 2 →
  father_age - mother_age = 4 := by
sorry

end NUMINAMATH_CALUDE_age_difference_of_parents_l3141_314196


namespace NUMINAMATH_CALUDE_factor_expression_l3141_314146

theorem factor_expression (x : ℝ) : 4 * x * (x - 2) + 6 * (x - 2) = 2 * (x - 2) * (2 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3141_314146


namespace NUMINAMATH_CALUDE_max_value_fraction_l3141_314191

theorem max_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 1 ≤ y ∧ y ≤ 3) :
  (∀ a b : ℝ, -3 ≤ a ∧ a ≤ -1 → 1 ≤ b ∧ b ≤ 3 → (a + b) / (a - b) ≤ (x + y) / (x - y)) →
  (x + y) / (x - y) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3141_314191


namespace NUMINAMATH_CALUDE_min_value_a_squared_plus_4b_squared_l3141_314133

theorem min_value_a_squared_plus_4b_squared (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 2/a + 1/b = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2/x + 1/y = 1 → a^2 + 4*b^2 ≤ x^2 + 4*y^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_squared_plus_4b_squared_l3141_314133


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l3141_314126

theorem quadratic_two_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0) ↔ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l3141_314126


namespace NUMINAMATH_CALUDE_clara_cookie_sales_l3141_314182

/-- Represents the number of cookies in each type of box --/
structure CookieBox where
  type1 : Nat
  type2 : Nat
  type3 : Nat

/-- Represents the number of boxes sold for each type --/
structure BoxesSold where
  type1 : Nat
  type2 : Nat
  type3 : Nat

/-- Calculates the total number of cookies sold --/
def totalCookiesSold (c : CookieBox) (b : BoxesSold) : Nat :=
  c.type1 * b.type1 + c.type2 * b.type2 + c.type3 * b.type3

theorem clara_cookie_sales (c : CookieBox) (b : BoxesSold) 
    (h1 : c.type1 = 12)
    (h2 : c.type2 = 20)
    (h3 : c.type3 = 16)
    (h4 : b.type2 = 80)
    (h5 : b.type3 = 70)
    (h6 : totalCookiesSold c b = 3320) :
    b.type1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_clara_cookie_sales_l3141_314182


namespace NUMINAMATH_CALUDE_no_primes_in_range_l3141_314163

theorem no_primes_in_range (n : ℕ) (hn : n > 2) :
  ∀ k : ℕ, n! + 2 < k ∧ k < n! + n → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l3141_314163


namespace NUMINAMATH_CALUDE_gcd_5616_11609_l3141_314194

theorem gcd_5616_11609 : Nat.gcd 5616 11609 = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5616_11609_l3141_314194


namespace NUMINAMATH_CALUDE_tenth_term_is_18_l3141_314199

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ a 5 = 8 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The 10th term of the arithmetic sequence is 18 -/
theorem tenth_term_is_18 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 18 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_18_l3141_314199


namespace NUMINAMATH_CALUDE_square_sum_given_linear_equations_l3141_314153

theorem square_sum_given_linear_equations :
  ∀ x y : ℝ, 3 * x + 2 * y = 20 → 4 * x + 2 * y = 26 → x^2 + y^2 = 37 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_linear_equations_l3141_314153


namespace NUMINAMATH_CALUDE_same_color_probability_l3141_314176

/-- Represents the number of sides on each die -/
def totalSides : ℕ := 20

/-- Represents the number of orange sides on each die -/
def orangeSides : ℕ := 3

/-- Represents the number of purple sides on each die -/
def purpleSides : ℕ := 5

/-- Represents the number of green sides on each die -/
def greenSides : ℕ := 6

/-- Represents the number of blue sides on each die -/
def blueSides : ℕ := 5

/-- Represents the number of sparkly sides on each die -/
def sparklySides : ℕ := 1

/-- Theorem stating the probability of rolling the same color or shade on both dice -/
theorem same_color_probability : 
  (orangeSides^2 + purpleSides^2 + greenSides^2 + blueSides^2 + sparklySides^2) / totalSides^2 = 24 / 100 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3141_314176


namespace NUMINAMATH_CALUDE_smallest_difference_l3141_314147

theorem smallest_difference (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) :
  ∃ (m : ℤ), m = a - b ∧ (∀ (c d : ℤ), c + d < 11 → c > 6 → c - d ≥ m) := by
  sorry

end NUMINAMATH_CALUDE_smallest_difference_l3141_314147


namespace NUMINAMATH_CALUDE_equal_distances_exist_l3141_314190

/-- Represents a position on an 8x8 grid -/
structure Position where
  row : Fin 8
  col : Fin 8

/-- Calculates the squared Euclidean distance between two positions -/
def squaredDistance (p1 p2 : Position) : ℕ :=
  (p1.row - p2.row).val ^ 2 + (p1.col - p2.col).val ^ 2

/-- Represents a configuration of 8 rooks on a chessboard -/
structure RookConfiguration where
  positions : Fin 8 → Position
  no_attack : ∀ i j, i ≠ j → positions i ≠ positions j

theorem equal_distances_exist (config : RookConfiguration) :
  ∃ i j k l : Fin 8, i < j ∧ k < l ∧ (i, j) ≠ (k, l) ∧
    squaredDistance (config.positions i) (config.positions j) =
    squaredDistance (config.positions k) (config.positions l) :=
sorry

end NUMINAMATH_CALUDE_equal_distances_exist_l3141_314190


namespace NUMINAMATH_CALUDE_quartic_roots_l3141_314151

/-- Given a quadratic equation x² + px + q = 0 with roots x₁ and x₂,
    prove that x⁴ - (p² - 2q)x² + q² = 0 has roots x₁, x₂, -x₁, -x₂ -/
theorem quartic_roots (p q x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) → 
  (x₂^2 + p*x₂ + q = 0) → 
  (∀ x : ℝ, x^4 - (p^2 - 2*q)*x^2 + q^2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = -x₁ ∨ x = -x₂) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_l3141_314151


namespace NUMINAMATH_CALUDE_special_ellipse_properties_l3141_314161

/-- An ellipse with one vertex at (0,1) and focus on the x-axis -/
structure SpecialEllipse where
  /-- The right focus of the ellipse -/
  focus : ℝ × ℝ
  /-- The distance from the right focus to the line x-y+2√2=0 is 3 -/
  focus_distance : (|focus.1 + 2 * Real.sqrt 2| : ℝ) / Real.sqrt 2 = 3

/-- The equation of the ellipse -/
def ellipse_equation (e : SpecialEllipse) (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

/-- A line passing through (0,1) -/
structure LineThroughA where
  /-- The slope of the line -/
  k : ℝ

/-- The equation of a line passing through (0,1) -/
def line_equation (l : LineThroughA) (x y : ℝ) : Prop :=
  y = l.k * x + 1

/-- The theorem to be proved -/
theorem special_ellipse_properties (e : SpecialEllipse) :
  (∀ x y, ellipse_equation e x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ l : LineThroughA, line_equation l = line_equation ⟨1⟩ ∨ line_equation l = line_equation ⟨-1⟩ →
    ∀ l' : LineThroughA, ∃ x y, ellipse_equation e x y ∧ line_equation l x y ∧ line_equation l' x y →
      ∀ x' y', ellipse_equation e x' y' ∧ line_equation l' x' y' →
        (x - 0)^2 + (y - 1)^2 ≥ (x' - 0)^2 + (y' - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_special_ellipse_properties_l3141_314161


namespace NUMINAMATH_CALUDE_sin_squared_minus_cos_squared_l3141_314131

theorem sin_squared_minus_cos_squared (α : Real) (h : Real.sin α = Real.sqrt 5 / 5) :
  Real.sin α ^ 2 - Real.cos α ^ 2 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_cos_squared_l3141_314131


namespace NUMINAMATH_CALUDE_randy_piggy_bank_l3141_314130

/-- Calculates the remaining money in Randy's piggy bank after a year -/
theorem randy_piggy_bank (initial_amount : ℕ) (spend_per_trip : ℕ) (trips_per_month : ℕ) (months : ℕ) :
  initial_amount = 200 →
  spend_per_trip = 2 →
  trips_per_month = 4 →
  months = 12 →
  initial_amount - (spend_per_trip * trips_per_month * months) = 104 := by
  sorry

#check randy_piggy_bank

end NUMINAMATH_CALUDE_randy_piggy_bank_l3141_314130


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l3141_314118

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, is_two_digit n ∧ 17 ∣ n → 34 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l3141_314118


namespace NUMINAMATH_CALUDE_complex_on_negative_diagonal_l3141_314149

/-- A complex number z = a - ai corresponds to a point on the line y = -x in the complex plane. -/
theorem complex_on_negative_diagonal (a : ℝ) : 
  let z : ℂ := a - a * I
  (z.re, z.im) ∈ {p : ℝ × ℝ | p.2 = -p.1} :=
by
  sorry

end NUMINAMATH_CALUDE_complex_on_negative_diagonal_l3141_314149


namespace NUMINAMATH_CALUDE_production_days_calculation_l3141_314111

theorem production_days_calculation (n : ℕ) : 
  (n * 50 + 90) / (n + 1) = 52 → n = 19 := by sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3141_314111


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3141_314132

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

theorem geometric_arithmetic_sequence_problem
  (a b : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_arithmetic : is_arithmetic_sequence b)
  (h_a_prod : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
  (h_b_sum : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l3141_314132


namespace NUMINAMATH_CALUDE_distance_between_points_l3141_314155

/-- The distance between points (3, 5) and (-4, 1) is √65 -/
theorem distance_between_points : Real.sqrt 65 = Real.sqrt ((3 - (-4))^2 + (5 - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3141_314155


namespace NUMINAMATH_CALUDE_carpet_length_l3141_314185

/-- Given a rectangular carpet with width 4 feet covering an entire room floor of area 60 square feet, 
    prove that the length of the carpet is 15 feet. -/
theorem carpet_length (carpet_width : ℝ) (room_area : ℝ) (h1 : carpet_width = 4) (h2 : room_area = 60) :
  room_area / carpet_width = 15 := by
  sorry

end NUMINAMATH_CALUDE_carpet_length_l3141_314185


namespace NUMINAMATH_CALUDE_cricket_problem_solution_l3141_314144

def cricket_problem (team_scores : List Nat) (lost_matches : Nat) (triple_score_matches : Nat) (half_score_matches : Nat) : Prop :=
  let total_matches := team_scores.length
  let lost_scores := team_scores.take lost_matches
  let triple_scores := team_scores.drop lost_matches |>.take triple_score_matches
  let half_scores := team_scores.drop (lost_matches + triple_score_matches) |>.take half_score_matches
  
  total_matches = 12 ∧
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ∧
  lost_matches = 4 ∧
  triple_score_matches = 5 ∧
  half_score_matches = 3 ∧
  
  (lost_scores.map (λ x => x + 2)).sum +
  (triple_scores.map (λ x => x / 3)).sum +
  (half_scores.map (λ x => x * 2)).sum = 97

theorem cricket_problem_solution :
  cricket_problem [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 4 5 3 := by
  sorry

end NUMINAMATH_CALUDE_cricket_problem_solution_l3141_314144


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3141_314107

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ - 12 = 0 ∧ x₂^2 - 4*x₂ - 12 = 0 ∧ x₁ = 6 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 3 = 0 ∧ y₂^2 - 4*y₂ - 3 = 0 ∧ y₁ = 2 + Real.sqrt 7 ∧ y₂ = 2 - Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3141_314107


namespace NUMINAMATH_CALUDE_expression_evaluation_l3141_314159

theorem expression_evaluation :
  (∀ x : ℤ, x = -2 → (3*x + 1)*(2*x - 3) - (6*x - 5)*(x - 4) = -67) ∧
  (∀ x y : ℤ, x = 1 ∧ y = 2 → (2*x - y)*(x + y) - 2*x*(-2*x + 3*y) + 6*x*(-x - 5/2*y) = -44) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3141_314159


namespace NUMINAMATH_CALUDE_cost_of_balls_and_shuttlecocks_l3141_314105

/-- The cost of ping-pong balls and badminton shuttlecocks -/
theorem cost_of_balls_and_shuttlecocks 
  (ping_pong : ℝ) 
  (shuttlecock : ℝ) 
  (h1 : 3 * ping_pong + 2 * shuttlecock = 15.5)
  (h2 : 2 * ping_pong + 3 * shuttlecock = 17) :
  4 * ping_pong + 4 * shuttlecock = 26 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_balls_and_shuttlecocks_l3141_314105


namespace NUMINAMATH_CALUDE_dan_licks_l3141_314150

/-- The number of licks it takes for each person to get to the center of a lollipop -/
structure LollipopLicks where
  michael : ℕ
  sam : ℕ
  david : ℕ
  lance : ℕ
  dan : ℕ

/-- The average number of licks for all five people -/
def average (l : LollipopLicks) : ℚ :=
  (l.michael + l.sam + l.david + l.lance + l.dan) / 5

/-- Theorem stating that Dan takes 58 licks to get to the center of a lollipop -/
theorem dan_licks (l : LollipopLicks) : 
  l.michael = 63 → l.sam = 70 → l.david = 70 → l.lance = 39 → average l = 60 → l.dan = 58 := by
  sorry

end NUMINAMATH_CALUDE_dan_licks_l3141_314150


namespace NUMINAMATH_CALUDE_positive_numbers_equality_l3141_314157

theorem positive_numbers_equality (a b : ℝ) : 
  0 < a → 0 < b → a^b = b^a → b = 3*a → a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_equality_l3141_314157


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3141_314195

/-- The minimum distance between two points P and Q, where P lies on the curve y = x^2 - ln(x) 
    and Q lies on the line y = x - 2, and both P and Q have the same y-coordinate, is 2. -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → 
    let y₁ := x₁^2 - Real.log x₁
    let y₂ := x₂ - 2
    y₁ = y₂ → |x₂ - x₁| ≥ min_dist) ∧
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ 
    let y₁ := x₁^2 - Real.log x₁
    let y₂ := x₂ - 2
    y₁ = y₂ ∧ |x₂ - x₁| = min_dist) ∧
  min_dist = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3141_314195
