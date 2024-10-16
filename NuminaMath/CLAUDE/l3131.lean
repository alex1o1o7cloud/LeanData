import Mathlib

namespace NUMINAMATH_CALUDE_product_of_powers_l3131_313177

theorem product_of_powers (n : ℕ) : (500 ^ 50) * (2 ^ 100) = 10 ^ 75 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l3131_313177


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l3131_313120

/-- A function satisfying the given inequality is constant -/
theorem function_satisfying_inequality_is_constant
  (f : ℝ → ℝ)
  (h : ∀ (x y : ℝ), |f x - f y| ≤ (x - y)^2) :
  ∃ (c : ℝ), ∀ (x : ℝ), f x = c :=
sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l3131_313120


namespace NUMINAMATH_CALUDE_grasshopper_jump_l3131_313198

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump (frog_jump grasshopper_jump difference : ℕ) 
  (h1 : frog_jump = 39)
  (h2 : frog_jump = grasshopper_jump + difference)
  (h3 : difference = 22) :
  grasshopper_jump = 17 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_l3131_313198


namespace NUMINAMATH_CALUDE_calculation_proof_l3131_313113

theorem calculation_proof : 
  Real.sqrt 12 - abs (-1) + (1/2)⁻¹ + (2023 + Real.pi)^0 = 2 * Real.sqrt 3 + 2 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l3131_313113


namespace NUMINAMATH_CALUDE_sin_945_degrees_l3131_313157

theorem sin_945_degrees : Real.sin (945 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_945_degrees_l3131_313157


namespace NUMINAMATH_CALUDE_b_equals_484_l3131_313106

/-- Given two real numbers a and b satisfying certain conditions,
    prove that b equals 484. -/
theorem b_equals_484 (a b : ℝ) 
    (h1 : a + b = 1210)
    (h2 : (4/15) * a = (2/5) * b) : 
  b = 484 := by sorry

end NUMINAMATH_CALUDE_b_equals_484_l3131_313106


namespace NUMINAMATH_CALUDE_vote_alteration_l3131_313140

theorem vote_alteration (got twi tad : ℕ) (x : ℚ) : 
  got = 10 →
  twi = 12 →
  tad = 20 →
  2 * got = got + twi / 2 + tad * (1 - x / 100) →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_vote_alteration_l3131_313140


namespace NUMINAMATH_CALUDE_intersection_A_B_l3131_313187

def A : Set ℝ := {x : ℝ | |x - 2| < 1}
def B : Set ℝ := Set.range (Int.cast : ℤ → ℝ)

theorem intersection_A_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3131_313187


namespace NUMINAMATH_CALUDE_sum_consecutive_integers_n_plus_3_l3131_313196

theorem sum_consecutive_integers_n_plus_3 (n : ℕ) (h : n = 1) :
  (List.range (n + 3 + 1)).sum = ((n + 3) * (n + 4)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_consecutive_integers_n_plus_3_l3131_313196


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3131_313164

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | |x| < 2}
def N : Set ℝ := {y | ∃ x, y = 2^x - 1}

theorem intersection_complement_equality :
  M ∩ (U \ N) = Ioc (-2) (-1) :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3131_313164


namespace NUMINAMATH_CALUDE_joe_spending_l3131_313160

def entrance_fee_under_18 : ℝ := 5
def entrance_fee_over_18 : ℝ := entrance_fee_under_18 * 1.2
def group_discount_rate : ℝ := 0.15
def ride_cost : ℝ := 0.5
def joe_age : ℕ := 30
def twin_age : ℕ := 6
def joe_rides : ℕ := 4
def twin_a_rides : ℕ := 3
def twin_b_rides : ℕ := 5

def group_size : ℕ := 3

def total_entrance_fee : ℝ := 
  entrance_fee_over_18 + 2 * entrance_fee_under_18

def discounted_entrance_fee : ℝ := 
  total_entrance_fee * (1 - group_discount_rate)

def total_ride_cost : ℝ := 
  ride_cost * (joe_rides + twin_a_rides + twin_b_rides)

theorem joe_spending (joe_spending : ℝ) : 
  joe_spending = discounted_entrance_fee + total_ride_cost ∧ 
  joe_spending = 19.60 := by sorry

end NUMINAMATH_CALUDE_joe_spending_l3131_313160


namespace NUMINAMATH_CALUDE_quadratic_roots_cube_l3131_313184

theorem quadratic_roots_cube (A B C : ℝ) (r s : ℝ) (h1 : A ≠ 0) :
  A * r^2 + B * r + C = 0 →
  A * s^2 + B * s + C = 0 →
  r ≠ s →
  ∃ q, (r^3)^2 + ((B^3 - 3*A*B*C) / A^3) * r^3 + q = 0 ∧
       (s^3)^2 + ((B^3 - 3*A*B*C) / A^3) * s^3 + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_cube_l3131_313184


namespace NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l3131_313143

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sum_odd_integers (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1)

/-- The 15th odd positive integer starting from 5 -/
def last_term : ℕ := 5 + 2 * (15 - 1)

theorem sum_first_15_odd_from_5 :
  sum_odd_integers 5 15 = 285 := by sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l3131_313143


namespace NUMINAMATH_CALUDE_number_of_subsets_of_intersection_l3131_313144

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {0, 2, 4}

theorem number_of_subsets_of_intersection : Finset.card (Finset.powerset (M ∩ N)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_of_intersection_l3131_313144


namespace NUMINAMATH_CALUDE_min_zeros_in_special_set_l3131_313103

theorem min_zeros_in_special_set (n : ℕ) (a : Fin n → ℝ) 
  (h : n = 2011)
  (sum_property : ∀ i j k : Fin n, ∃ l : Fin n, a i + a j + a k = a l) :
  (Finset.filter (fun i => a i = 0) Finset.univ).card ≥ 2009 :=
sorry

end NUMINAMATH_CALUDE_min_zeros_in_special_set_l3131_313103


namespace NUMINAMATH_CALUDE_chord_circuit_l3131_313112

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the minimum number of such chords
    needed to form a complete circuit is 3. -/
theorem chord_circuit (angle : ℝ) (n : ℕ) : angle = 60 → n * angle = 360 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_circuit_l3131_313112


namespace NUMINAMATH_CALUDE_complex_expression_value_l3131_313111

theorem complex_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x^2 + x*y + y^2 = 0) : 
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l3131_313111


namespace NUMINAMATH_CALUDE_cubic_function_extreme_value_l3131_313117

/-- Given a cubic function f(x) = ax³ + bx + c that reaches an extreme value of c - 6 at x = 2,
    prove that a = 3/8 and b = -9/2 -/
theorem cubic_function_extreme_value (a b c : ℝ) :
  (∀ x, (fun x => a * x^3 + b * x + c) x = a * x^3 + b * x + c) →
  (∃ y, (fun x => a * x^3 + b * x + c) 2 = y ∧ 
        ∀ x, (fun x => a * x^3 + b * x + c) x ≤ y) →
  (fun x => a * x^3 + b * x + c) 2 = c - 6 →
  a = 3/8 ∧ b = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extreme_value_l3131_313117


namespace NUMINAMATH_CALUDE_expression_factorization_l3131_313168

theorem expression_factorization (x : ℝ) : 
  (4 * x^3 + 75 * x^2 - 12) - (-5 * x^3 + 3 * x^2 - 12) = 9 * x^2 * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3131_313168


namespace NUMINAMATH_CALUDE_specific_grid_square_count_l3131_313125

/-- Represents a square grid with some incomplete squares at the edges -/
structure SquareGrid :=
  (width : ℕ)
  (height : ℕ)
  (hasIncompleteEdges : Bool)

/-- Counts the number of squares of a given size in the grid -/
def countSquares (grid : SquareGrid) (size : ℕ) : ℕ :=
  sorry

/-- Counts the total number of squares in the grid -/
def totalSquares (grid : SquareGrid) : ℕ :=
  (countSquares grid 1) + (countSquares grid 2) + (countSquares grid 3)

/-- The main theorem stating that the total number of squares in the specific grid is 38 -/
theorem specific_grid_square_count :
  ∃ (grid : SquareGrid), grid.width = 5 ∧ grid.height = 5 ∧ grid.hasIncompleteEdges = true ∧ totalSquares grid = 38 :=
  sorry

end NUMINAMATH_CALUDE_specific_grid_square_count_l3131_313125


namespace NUMINAMATH_CALUDE_kaylin_age_is_33_l3131_313190

def freyja_age : ℕ := 10
def eli_age : ℕ := freyja_age + 9
def sarah_age : ℕ := 2 * eli_age
def kaylin_age : ℕ := sarah_age - 5

theorem kaylin_age_is_33 : kaylin_age = 33 := by
  sorry

end NUMINAMATH_CALUDE_kaylin_age_is_33_l3131_313190


namespace NUMINAMATH_CALUDE_circus_crowns_l3131_313102

theorem circus_crowns (feathers_per_crown : ℕ) (total_feathers : ℕ) (h1 : feathers_per_crown = 7) (h2 : total_feathers = 6538) :
  total_feathers / feathers_per_crown = 934 := by
  sorry

end NUMINAMATH_CALUDE_circus_crowns_l3131_313102


namespace NUMINAMATH_CALUDE_k_range_l3131_313176

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 4) + y^2 / (k - 6) = 1

-- Define the ellipse equation
def ellipse (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / 5 + y^2 / k = 1

-- Define the condition for the line through M(2,1) intersecting the ellipse
def line_intersects_ellipse (k : ℝ) : Prop :=
  ∀ m b : ℝ, ∃ x y : ℝ, y = m * x + b ∧ ellipse k x y ∧ (2 * m + b = 1)

-- Main theorem
theorem k_range :
  (∀ k : ℝ, is_hyperbola k → line_intersects_ellipse k → k > 5 ∧ k < 6) ∧
  (∀ k : ℝ, k > 5 ∧ k < 6 → is_hyperbola k ∧ line_intersects_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_k_range_l3131_313176


namespace NUMINAMATH_CALUDE_number_problem_l3131_313165

theorem number_problem (x : ℝ) : 2 * x - x / 2 = 45 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3131_313165


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3131_313154

theorem root_exists_in_interval : ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ 2^x = x^2 + 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l3131_313154


namespace NUMINAMATH_CALUDE_polynomial_sum_theorem_l3131_313188

theorem polynomial_sum_theorem (d : ℝ) :
  let f : ℝ → ℝ := λ x => 15 * x^3 + 17 * x + 18 + 19 * x^2
  let g : ℝ → ℝ := λ x => 3 * x^3 + 4 * x + 2
  ∃ (p q r s : ℤ),
    (∀ x, f x + g x = p * x^3 + q * x^2 + r * x + s) ∧
    p + q + r + s = 78 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_theorem_l3131_313188


namespace NUMINAMATH_CALUDE_intersection_y_intercept_sum_l3131_313199

/-- Given two lines that intersect at (2, 3), prove their y-intercepts sum to 10/3 -/
theorem intersection_y_intercept_sum (a b : ℚ) : 
  (2 = (1/3) * 3 + a) → 
  (3 = (1/3) * 2 + b) → 
  a + b = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_sum_l3131_313199


namespace NUMINAMATH_CALUDE_negative_max_inverse_is_max_of_negative_inverses_l3131_313142

/-- Given a non-empty set A of real numbers not containing zero,
    with a negative maximum value a, -a⁻¹ is the maximum value
    of the set {-x⁻¹ | x ∈ A}. -/
theorem negative_max_inverse_is_max_of_negative_inverses
  (A : Set ℝ)
  (hA_nonempty : A.Nonempty)
  (hA_no_zero : 0 ∉ A)
  (a : ℝ)
  (ha_max : ∀ x ∈ A, x ≤ a)
  (ha_neg : a < 0) :
  ∀ y ∈ {-x⁻¹ | x ∈ A}, y ≤ -a⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_negative_max_inverse_is_max_of_negative_inverses_l3131_313142


namespace NUMINAMATH_CALUDE_pool_filling_time_l3131_313128

theorem pool_filling_time (pool_capacity : ℝ) (num_hoses : ℕ) (flow_rate : ℝ) : 
  pool_capacity = 24000 ∧ 
  num_hoses = 4 ∧ 
  flow_rate = 2.5 → 
  pool_capacity / (num_hoses * flow_rate * 60) = 40 := by
sorry

end NUMINAMATH_CALUDE_pool_filling_time_l3131_313128


namespace NUMINAMATH_CALUDE_min_a_for_simplest_quadratic_root_l3131_313122

-- Define the property of being the simplest quadratic root
def is_simplest_quadratic_root (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ ∀ (m : ℕ), m < n → ¬(∃ (q : ℚ), q * q = m)

-- Define the main theorem
theorem min_a_for_simplest_quadratic_root :
  ∃ (a : ℤ), (∀ (b : ℤ), is_simplest_quadratic_root (Real.sqrt (3 * b + 1)) → a ≤ b) ∧
             is_simplest_quadratic_root (Real.sqrt (3 * a + 1)) ∧
             a = 2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_simplest_quadratic_root_l3131_313122


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l3131_313145

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → ⌈y⌉ * ⌊y⌋ = 110 → -11 < y ∧ y < -10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l3131_313145


namespace NUMINAMATH_CALUDE_dilation_image_l3131_313107

def dilation (center : ℂ) (scale : ℝ) (point : ℂ) : ℂ :=
  center + scale * (point - center)

theorem dilation_image : 
  let center : ℂ := -1 + 2*I
  let scale : ℝ := 2
  let point : ℂ := 3 + 4*I
  dilation center scale point = 7 + 6*I := by
  sorry

end NUMINAMATH_CALUDE_dilation_image_l3131_313107


namespace NUMINAMATH_CALUDE_section_B_students_l3131_313123

/-- The number of students in section A -/
def students_A : ℕ := 36

/-- The average weight of students in section A (in kg) -/
def avg_weight_A : ℚ := 40

/-- The average weight of students in section B (in kg) -/
def avg_weight_B : ℚ := 35

/-- The average weight of the whole class (in kg) -/
def avg_weight_total : ℚ := 37.25

/-- The number of students in section B -/
def students_B : ℕ := 44

theorem section_B_students : 
  (students_A : ℚ) * avg_weight_A + (students_B : ℚ) * avg_weight_B = 
  ((students_A : ℚ) + students_B) * avg_weight_total := by
  sorry

end NUMINAMATH_CALUDE_section_B_students_l3131_313123


namespace NUMINAMATH_CALUDE_inequality_solution_l3131_313131

theorem inequality_solution :
  ∀ x : ℝ, (x / 2 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-6) ∩ Set.Iio (-3/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3131_313131


namespace NUMINAMATH_CALUDE_gcd_126_105_l3131_313135

theorem gcd_126_105 : Nat.gcd 126 105 = 21 := by
  sorry

end NUMINAMATH_CALUDE_gcd_126_105_l3131_313135


namespace NUMINAMATH_CALUDE_range_of_f_l3131_313101

def f (x : ℝ) : ℝ := x^2 - 2*x

def domain : Set ℝ := {0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3131_313101


namespace NUMINAMATH_CALUDE_apples_per_child_l3131_313182

theorem apples_per_child (total_apples : ℕ) (num_children : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ)
  (h1 : total_apples = 450)
  (h2 : num_children = 33)
  (h3 : num_adults = 40)
  (h4 : apples_per_adult = 3) :
  (total_apples - num_adults * apples_per_adult) / num_children = 10 := by
sorry

end NUMINAMATH_CALUDE_apples_per_child_l3131_313182


namespace NUMINAMATH_CALUDE_distance_center_to_point_l3131_313171

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 6*x - 8*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, -4)

-- Define the given point
def given_point : ℝ × ℝ := (5, -3)

-- Statement to prove
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l3131_313171


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3131_313167

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n ≤ 999 ∧ 
  100 ≤ n ∧ 
  n % 17 = 0 ∧ 
  ∀ m : ℕ, m ≤ 999 ∧ 100 ≤ m ∧ m % 17 = 0 → m ≤ n :=
by
  use 986
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3131_313167


namespace NUMINAMATH_CALUDE_warren_guests_calculation_l3131_313133

/-- The number of tables Warren has -/
def num_tables : ℝ := 252.0

/-- The number of guests each table can hold -/
def guests_per_table : ℝ := 4.0

/-- The total number of guests Warren can accommodate -/
def total_guests : ℝ := num_tables * guests_per_table

theorem warren_guests_calculation : total_guests = 1008 := by
  sorry

end NUMINAMATH_CALUDE_warren_guests_calculation_l3131_313133


namespace NUMINAMATH_CALUDE_log_18_15_l3131_313137

-- Define the logarithm base 10 (lg) function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the variables a and b
variable (a b : ℝ)

-- State the theorem
theorem log_18_15 (h1 : lg 2 = a) (h2 : lg 3 = b) :
  (Real.log 15) / (Real.log 18) = (b - a + 1) / (a + 2 * b) := by
  sorry

end NUMINAMATH_CALUDE_log_18_15_l3131_313137


namespace NUMINAMATH_CALUDE_james_monthly_earnings_l3131_313149

/-- Calculates the monthly earnings of a Twitch streamer based on their subscribers and earnings per subscriber. -/
def monthly_earnings (initial_subscribers : ℕ) (gifted_subscribers : ℕ) (earnings_per_subscriber : ℕ) : ℕ :=
  (initial_subscribers + gifted_subscribers) * earnings_per_subscriber

/-- Theorem stating that James' monthly earnings from Twitch are $1800 -/
theorem james_monthly_earnings :
  monthly_earnings 150 50 9 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_james_monthly_earnings_l3131_313149


namespace NUMINAMATH_CALUDE_angle_symmetry_l3131_313169

/-- Given that the terminal side of angle α is symmetric to the terminal side of angle -690° about the y-axis, prove that α = k * 360° + 150° for some integer k. -/
theorem angle_symmetry (α : Real) : 
  (∃ k : ℤ, α = k * 360 + 150) ↔ 
  (∃ n : ℤ, α + (-690) = n * 360 + 180) :=
by sorry

end NUMINAMATH_CALUDE_angle_symmetry_l3131_313169


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l3131_313194

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : Real.cos A = 4/5)
  (h6 : Real.cos C = 5/13)
  (h7 : a = 13)
  (h8 : a / Real.sin A = b / Real.sin B)
  (h9 : b / Real.sin B = c / Real.sin C)
  : b = 21 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l3131_313194


namespace NUMINAMATH_CALUDE_inequality_proof_l3131_313147

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  3 + (a + b + c) + (1/a + 1/b + 1/c) + (a/b + b/c + c/a) ≥ (3*(a+1)*(b+1)*(c+1))/(a*b*c+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3131_313147


namespace NUMINAMATH_CALUDE_circle_C_radius_l3131_313139

-- Define the circle C
def Circle_C : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (-1, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 7 = 0

-- State that A is on the circle
axiom A_on_circle : A ∈ Circle_C

-- State that B is on the circle and the tangent line
axiom B_on_circle : B ∈ Circle_C
axiom B_on_tangent : tangent_line B.1 B.2

-- Define the radius of the circle
def radius (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_C_radius : radius Circle_C = 5 := by sorry

end NUMINAMATH_CALUDE_circle_C_radius_l3131_313139


namespace NUMINAMATH_CALUDE_watermelon_sales_theorem_l3131_313162

/-- Calculates the total income from selling watermelons -/
def watermelon_income (weight : ℕ) (price_per_pound : ℕ) (num_watermelons : ℕ) : ℕ :=
  weight * price_per_pound * num_watermelons

/-- Proves that selling 18 watermelons of 23 pounds each at $2 per pound yields $828 -/
theorem watermelon_sales_theorem :
  watermelon_income 23 2 18 = 828 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_sales_theorem_l3131_313162


namespace NUMINAMATH_CALUDE_f_composition_value_l3131_313175

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sin (Real.pi * x)
  else Real.cos (Real.pi * x / 2 + Real.pi / 3)

theorem f_composition_value : f (f (15/2)) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l3131_313175


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3131_313130

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ y, y = f 1 → y = (1 : ℝ) + 2) : 
  f 1 + deriv f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3131_313130


namespace NUMINAMATH_CALUDE_teapot_sale_cost_comparison_l3131_313108

/-- Represents the cost calculation for promotional methods in a teapot and teacup sale. -/
structure TeapotSale where
  teapot_price : ℝ
  teacup_price : ℝ
  discount_rate : ℝ
  teapots_bought : ℕ
  min_teacups : ℕ

/-- Calculates the cost under promotional method 1 (buy 1 teapot, get 1 teacup free) -/
def cost_method1 (sale : TeapotSale) (x : ℕ) : ℝ :=
  sale.teapot_price * sale.teapots_bought + sale.teacup_price * (x - sale.teapots_bought)

/-- Calculates the cost under promotional method 2 (9.2% discount on total price) -/
def cost_method2 (sale : TeapotSale) (x : ℕ) : ℝ :=
  (sale.teapot_price * sale.teapots_bought + sale.teacup_price * x) * (1 - sale.discount_rate)

/-- Theorem stating the relationship between costs of two promotional methods -/
theorem teapot_sale_cost_comparison (sale : TeapotSale)
    (h_teapot : sale.teapot_price = 20)
    (h_teacup : sale.teacup_price = 5)
    (h_discount : sale.discount_rate = 0.092)
    (h_teapots : sale.teapots_bought = 4)
    (h_min_teacups : sale.min_teacups = 4) :
    ∀ x : ℕ, x ≥ sale.min_teacups →
      (cost_method1 sale x < cost_method2 sale x ↔ x < 34) ∧
      (cost_method1 sale x = cost_method2 sale x ↔ x = 34) ∧
      (cost_method1 sale x > cost_method2 sale x ↔ x > 34) := by
  sorry


end NUMINAMATH_CALUDE_teapot_sale_cost_comparison_l3131_313108


namespace NUMINAMATH_CALUDE_number_problem_l3131_313178

theorem number_problem (x : ℚ) : x - (3/5) * x = 62 ↔ x = 155 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3131_313178


namespace NUMINAMATH_CALUDE_not_in_first_quadrant_l3131_313158

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def FirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Theorem: If mn ≤ 0, then point A(m,n) cannot be in the first quadrant -/
theorem not_in_first_quadrant (m n : ℝ) (h : m * n ≤ 0) :
  ¬FirstQuadrant ⟨m, n⟩ := by
  sorry


end NUMINAMATH_CALUDE_not_in_first_quadrant_l3131_313158


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3131_313124

-- Define the arithmetic sequence
def arithmetic_sequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

-- Define the increasing property
def increasing_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  arithmetic_sequence b 2 →
  increasing_sequence b →
  b 4 * b 5 = 15 →
  b 2 * b 7 = -9 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3131_313124


namespace NUMINAMATH_CALUDE_mirror_pieces_l3131_313141

theorem mirror_pieces (total : ℕ) (swept : ℕ) (stolen : ℕ) (picked_fraction : ℚ) : 
  total = 60 →
  swept = total / 2 →
  stolen = 3 →
  picked_fraction = 1 / 3 →
  (total - swept - stolen) * picked_fraction = 9 := by
sorry

end NUMINAMATH_CALUDE_mirror_pieces_l3131_313141


namespace NUMINAMATH_CALUDE_tim_singles_count_l3131_313116

-- Define the value of a single line
def single_value : ℕ := 1000

-- Define the value of a tetris
def tetris_value : ℕ := 8 * single_value

-- Define the number of tetrises Tim scored
def tim_tetrises : ℕ := 4

-- Define Tim's total score
def tim_total_score : ℕ := 38000

-- Theorem to prove
theorem tim_singles_count :
  ∃ (s : ℕ), s * single_value + tim_tetrises * tetris_value = tim_total_score ∧ s = 6 := by
  sorry

end NUMINAMATH_CALUDE_tim_singles_count_l3131_313116


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3131_313170

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  S : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  first_term : a 1 = 2
  third_sum : S 3 = 12

/-- The main theorem combining both parts of the problem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n) ∧
  (∃ k : ℕ, k > 0 ∧ (seq.a 3) * (seq.a (k + 1)) = (seq.S k)^2 ∧ k = 2) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3131_313170


namespace NUMINAMATH_CALUDE_percentage_of_students_owning_only_cats_l3131_313189

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ)
  (dog_owners : ℕ)
  (cat_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : dog_owners = 150)
  (h3 : cat_owners = 80)
  (h4 : both_owners = 25) :
  (cat_owners - both_owners) / total_students * 100 = 11 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_students_owning_only_cats_l3131_313189


namespace NUMINAMATH_CALUDE_zero_not_necessarily_in_2_5_l3131_313172

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of having a unique zero in an interval
def has_unique_zero_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

-- State the theorem
theorem zero_not_necessarily_in_2_5 :
  (has_unique_zero_in f 1 3) →
  (has_unique_zero_in f 1 4) →
  (has_unique_zero_in f 1 5) →
  ¬ (∀ g : ℝ → ℝ, (has_unique_zero_in g 1 3 ∧ has_unique_zero_in g 1 4 ∧ has_unique_zero_in g 1 5) → 
    (∃ x, 2 < x ∧ x < 5 ∧ g x = 0)) :=
by sorry

end NUMINAMATH_CALUDE_zero_not_necessarily_in_2_5_l3131_313172


namespace NUMINAMATH_CALUDE_one_third_to_fifth_power_l3131_313148

theorem one_third_to_fifth_power : (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end NUMINAMATH_CALUDE_one_third_to_fifth_power_l3131_313148


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3131_313186

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 2 = 0) ↔ k ≤ 2 ∧ k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3131_313186


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3131_313195

theorem complex_number_quadrant (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 2 * i / (1 + i)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3131_313195


namespace NUMINAMATH_CALUDE_intersection_AB_XOZ_plane_l3131_313121

/-- Given two points A and B in 3D space, this function returns the coordinates of the 
    intersection point of the line passing through A and B with the XOZ plane. -/
def intersectionWithXOZPlane (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem stating that the intersection of the line passing through A(1,-2,-3) and B(2,-1,-1) 
    with the XOZ plane is the point (3,0,1). -/
theorem intersection_AB_XOZ_plane :
  let A : ℝ × ℝ × ℝ := (1, -2, -3)
  let B : ℝ × ℝ × ℝ := (2, -1, -1)
  intersectionWithXOZPlane A B = (3, 0, 1) := by
  sorry

end NUMINAMATH_CALUDE_intersection_AB_XOZ_plane_l3131_313121


namespace NUMINAMATH_CALUDE_darnel_sprint_distance_l3131_313115

theorem darnel_sprint_distance (jogged_distance : Real) (additional_sprint : Real) :
  jogged_distance = 0.75 →
  additional_sprint = 0.13 →
  jogged_distance + additional_sprint = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_darnel_sprint_distance_l3131_313115


namespace NUMINAMATH_CALUDE_total_tax_collection_l3131_313156

/-- Represents the farm tax collection in a village -/
structure FarmTaxCollection where
  totalTax : ℝ
  farmerTax : ℝ
  farmerLandRatio : ℝ

/-- Theorem: Given a farmer's tax payment and land ratio, prove the total tax collected -/
theorem total_tax_collection (ftc : FarmTaxCollection) 
  (h1 : ftc.farmerTax = 480)
  (h2 : ftc.farmerLandRatio = 0.3125)
  : ftc.totalTax = 1536 := by
  sorry

#check total_tax_collection

end NUMINAMATH_CALUDE_total_tax_collection_l3131_313156


namespace NUMINAMATH_CALUDE_recurring_decimal_difference_l3131_313181

theorem recurring_decimal_difference : 
  let x : ℚ := 8/11  -- 0.overline{72}
  let y : ℚ := 18/25 -- 0.72
  x - y = 2/275 := by
sorry

end NUMINAMATH_CALUDE_recurring_decimal_difference_l3131_313181


namespace NUMINAMATH_CALUDE_binomial_equation_unique_solution_l3131_313173

theorem binomial_equation_unique_solution :
  ∃! m : ℕ, (Nat.choose 23 m) + (Nat.choose 23 12) = (Nat.choose 24 13) ∧ m = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_equation_unique_solution_l3131_313173


namespace NUMINAMATH_CALUDE_black_white_ratio_after_border_l3131_313110

/-- Represents a rectangular tile pattern -/
structure TilePattern where
  length : ℕ
  width : ℕ
  blackTiles : ℕ
  whiteTiles : ℕ

/-- Adds a border of black tiles to a tile pattern -/
def addBorder (pattern : TilePattern) (borderWidth : ℕ) : TilePattern :=
  { length := pattern.length + 2 * borderWidth,
    width := pattern.width + 2 * borderWidth,
    blackTiles := pattern.blackTiles + 
      (pattern.length + pattern.width + 2 * borderWidth) * 2 * borderWidth + 4 * borderWidth^2,
    whiteTiles := pattern.whiteTiles }

theorem black_white_ratio_after_border (initialPattern : TilePattern) :
  initialPattern.length = 4 →
  initialPattern.width = 8 →
  initialPattern.blackTiles = 10 →
  initialPattern.whiteTiles = 22 →
  let finalPattern := addBorder initialPattern 2
  (finalPattern.blackTiles : ℚ) / finalPattern.whiteTiles = 19 / 11 := by
  sorry

end NUMINAMATH_CALUDE_black_white_ratio_after_border_l3131_313110


namespace NUMINAMATH_CALUDE_bottle_production_theorem_l3131_313151

/-- Given a number of machines and their production rate, calculate the total bottles produced in a given time -/
def bottlesProduced (numMachines : ℕ) (ratePerMinute : ℕ) (minutes : ℕ) : ℕ :=
  numMachines * ratePerMinute * minutes

/-- The production rate of a single machine -/
def singleMachineRate (totalMachines : ℕ) (totalRate : ℕ) : ℕ :=
  totalRate / totalMachines

theorem bottle_production_theorem (initialMachines : ℕ) (initialRate : ℕ) (newMachines : ℕ) (time : ℕ) :
  initialMachines = 6 →
  initialRate = 270 →
  newMachines = 14 →
  time = 4 →
  bottlesProduced newMachines (singleMachineRate initialMachines initialRate) time = 2520 :=
by
  sorry

end NUMINAMATH_CALUDE_bottle_production_theorem_l3131_313151


namespace NUMINAMATH_CALUDE_total_distance_traveled_l3131_313183

/-- Calculates the total distance traveled by a man rowing upstream and downstream in a river. -/
theorem total_distance_traveled
  (man_speed : ℝ)
  (river_speed : ℝ)
  (total_time : ℝ)
  (h1 : man_speed = 6)
  (h2 : river_speed = 1.2)
  (h3 : total_time = 1)
  : ∃ (distance : ℝ), distance = 5.76 ∧ 
    (distance / (man_speed - river_speed) + distance / (man_speed + river_speed) = total_time) :=
by sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l3131_313183


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3131_313152

/-- Given a geometric sequence {a_n} where a_1 = 3 and 4a_1, 2a_2, a_3 form an arithmetic sequence,
    prove that a_3 + a_4 + a_5 = 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- a_1 = 3
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- arithmetic sequence condition
  a 3 + a 4 + a 5 = 84 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3131_313152


namespace NUMINAMATH_CALUDE_stream_rate_proof_l3131_313132

/-- Proves that the rate of a stream is 5 km/hr given the conditions of a boat's travel --/
theorem stream_rate_proof (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 16 →
  distance = 147 →
  time = 7 →
  (distance / time) - boat_speed = 5 := by
  sorry

#check stream_rate_proof

end NUMINAMATH_CALUDE_stream_rate_proof_l3131_313132


namespace NUMINAMATH_CALUDE_school_boys_count_l3131_313109

theorem school_boys_count :
  ∀ (x : ℕ),
  (x + x = 100) →
  (x = 50) :=
by
  sorry

#check school_boys_count

end NUMINAMATH_CALUDE_school_boys_count_l3131_313109


namespace NUMINAMATH_CALUDE_function_range_l3131_313100

theorem function_range (a : ℝ) : 
  (a > 0) →
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, x₂ ≥ -2 ∧ (x₁^2 - 2*x₁) > (a*x₂ + 2)) →
  a > 3/2 := by
sorry

end NUMINAMATH_CALUDE_function_range_l3131_313100


namespace NUMINAMATH_CALUDE_initial_number_count_l3131_313129

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  S / n = 20 →
  (S - 100) / (n - 2) = 18.75 →
  n = 110 := by
sorry

end NUMINAMATH_CALUDE_initial_number_count_l3131_313129


namespace NUMINAMATH_CALUDE_main_diagonal_squares_diagonal_5_composite_diagonal_21_composite_l3131_313180

def a (k : ℕ) : ℕ := (2*k + 1)^2

def b (k : ℕ) : ℕ := (4*k - 3) * (4*k + 1)

def c (k : ℕ) : ℕ := 4*((4*k + 3)*(4*k - 1)) + 1

theorem main_diagonal_squares (k : ℕ) :
  ∃ (n : ℕ), a k = 4*n + 1 :=
sorry

theorem diagonal_5_composite (k : ℕ) (h : k > 1) :
  ¬ Nat.Prime (b k) :=
sorry

theorem diagonal_21_composite (k : ℕ) :
  ¬ Nat.Prime (c k) :=
sorry

end NUMINAMATH_CALUDE_main_diagonal_squares_diagonal_5_composite_diagonal_21_composite_l3131_313180


namespace NUMINAMATH_CALUDE_franks_age_l3131_313150

theorem franks_age (frank_age : ℕ) (gabriel_age : ℕ) : 
  gabriel_age = frank_age - 3 →
  frank_age + gabriel_age = 17 →
  frank_age = 10 :=
by sorry

end NUMINAMATH_CALUDE_franks_age_l3131_313150


namespace NUMINAMATH_CALUDE_unique_a_value_l3131_313155

def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + 4 = 0}

theorem unique_a_value : ∃! a : ℝ, (B a).Nonempty ∧ B a ⊆ A ∧ a = 4 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3131_313155


namespace NUMINAMATH_CALUDE_novel_purchase_equation_l3131_313118

/-- Represents a bookstore's novel purchases --/
structure NovelPurchases where
  first_cost : ℝ
  second_cost : ℝ
  quantity_difference : ℕ
  first_quantity : ℝ

/-- The equation correctly represents the novel purchase situation --/
def correct_equation (p : NovelPurchases) : Prop :=
  p.first_cost / p.first_quantity = p.second_cost / (p.first_quantity + p.quantity_difference)

/-- Theorem stating that the given equation correctly represents the situation --/
theorem novel_purchase_equation (p : NovelPurchases) 
  (h1 : p.first_cost = 2000)
  (h2 : p.second_cost = 3000)
  (h3 : p.quantity_difference = 50)
  (h4 : p.first_quantity > 0) :
  correct_equation p := by
  sorry

end NUMINAMATH_CALUDE_novel_purchase_equation_l3131_313118


namespace NUMINAMATH_CALUDE_megan_country_albums_l3131_313134

/-- The number of country albums Megan bought -/
def num_country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def num_pop_albums : ℕ := 8

/-- The number of songs per album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := 70

/-- Proof that Megan bought 2 country albums -/
theorem megan_country_albums :
  num_country_albums * songs_per_album + num_pop_albums * songs_per_album = total_songs :=
by sorry

end NUMINAMATH_CALUDE_megan_country_albums_l3131_313134


namespace NUMINAMATH_CALUDE_product_of_primes_sum_31_l3131_313163

theorem product_of_primes_sum_31 (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_sum : p + q = 31) : p * q = 58 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_sum_31_l3131_313163


namespace NUMINAMATH_CALUDE_smallest_k_l3131_313114

theorem smallest_k (a b c k : ℤ) : 
  (a + 2 = b - 2) → 
  (a + 2 = (c : ℚ) / 2) → 
  (a + b + c = 2001 * k) → 
  (∀ m : ℤ, m > 0 → m < k → ¬(∃ a' b' c' : ℤ, 
    (a' + 2 = b' - 2) ∧ 
    (a' + 2 = (c' : ℚ) / 2) ∧ 
    (a' + b' + c' = 2001 * m))) → 
  k = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_l3131_313114


namespace NUMINAMATH_CALUDE_constant_sum_l3131_313138

theorem constant_sum (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 1 ↔ x = -1) ∧ (a + b / x = 5 ↔ x = -5)) →
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_constant_sum_l3131_313138


namespace NUMINAMATH_CALUDE_bound_step_difference_is_10_l3131_313185

/-- The number of steps Martha takes between consecutive lamp posts -/
def martha_steps : ℕ := 50

/-- The number of bounds Percy takes between consecutive lamp posts -/
def percy_bounds : ℕ := 15

/-- The total number of lamp posts -/
def total_posts : ℕ := 51

/-- The distance between the first and last lamp post in feet -/
def total_distance : ℕ := 10560

/-- The difference between Percy's bound length and Martha's step length in feet -/
def bound_step_difference : ℚ := 10

theorem bound_step_difference_is_10 :
  (total_distance : ℚ) / ((total_posts - 1) * percy_bounds) -
  (total_distance : ℚ) / ((total_posts - 1) * martha_steps) =
  bound_step_difference := by sorry

end NUMINAMATH_CALUDE_bound_step_difference_is_10_l3131_313185


namespace NUMINAMATH_CALUDE_basketball_shots_l3131_313119

theorem basketball_shots (total_points : ℕ) (three_point_shots : ℕ) : 
  total_points = 26 → 
  three_point_shots = 4 → 
  ∃ (two_point_shots : ℕ), 
    total_points = 3 * three_point_shots + 2 * two_point_shots ∧
    three_point_shots + two_point_shots = 11 :=
by sorry

end NUMINAMATH_CALUDE_basketball_shots_l3131_313119


namespace NUMINAMATH_CALUDE_prime_iff_no_equal_products_l3131_313153

theorem prime_iff_no_equal_products (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ 
  ∀ (a b c d : ℕ), 
    a > 0 → b > 0 → c > 0 → d > 0 → 
    a + b + c + d = p → 
    (a * b ≠ c * d ∧ a * c ≠ b * d ∧ a * d ≠ b * c) :=
by sorry

end NUMINAMATH_CALUDE_prime_iff_no_equal_products_l3131_313153


namespace NUMINAMATH_CALUDE_quadrilateral_angle_l3131_313166

/-- 
Given a quadrilateral with angles α₁, α₂, α₃, α₄, α₅ satisfying:
1) α₁ + α₂ = 180°
2) α₃ = α₄
3) α₂ + α₅ = 180°
Prove that α₄ = 90°
-/
theorem quadrilateral_angle (α₁ α₂ α₃ α₄ α₅ : ℝ) 
  (h1 : α₁ + α₂ = 180)
  (h2 : α₃ = α₄)
  (h3 : α₂ + α₅ = 180)
  (h4 : α₁ + α₂ + α₃ + α₄ = 360) :  -- sum of angles in a quadrilateral
  α₄ = 90 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_l3131_313166


namespace NUMINAMATH_CALUDE_company_signs_used_l3131_313104

/-- The number of signs in the special sign language --/
def total_signs : ℕ := 124

/-- The number of unused signs --/
def unused_signs : ℕ := 2

/-- The number of additional area codes if all signs were used --/
def additional_codes : ℕ := 488

/-- The number of signs in each area code --/
def signs_per_code : ℕ := 2

/-- The number of signs used fully by the company --/
def signs_used : ℕ := total_signs - unused_signs

theorem company_signs_used : signs_used = 120 := by
  sorry

end NUMINAMATH_CALUDE_company_signs_used_l3131_313104


namespace NUMINAMATH_CALUDE_square_root_reverses_squaring_l3131_313179

theorem square_root_reverses_squaring (x : ℝ) (hx : x = 25) : 
  Real.sqrt (x ^ 2) = x := by sorry

end NUMINAMATH_CALUDE_square_root_reverses_squaring_l3131_313179


namespace NUMINAMATH_CALUDE_cubic_expansion_sum_l3131_313105

theorem cubic_expansion_sum (a a₁ a₂ a₃ : ℝ) 
  (h : ∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
sorry

end NUMINAMATH_CALUDE_cubic_expansion_sum_l3131_313105


namespace NUMINAMATH_CALUDE_right_square_prism_properties_l3131_313192

/-- Right square prism -/
structure RightSquarePrism where
  base_edge : ℝ
  height : ℝ

/-- Calculates the lateral area of a right square prism -/
def lateral_area (p : RightSquarePrism) : ℝ :=
  4 * p.base_edge * p.height

/-- Calculates the volume of a right square prism -/
def volume (p : RightSquarePrism) : ℝ :=
  p.base_edge ^ 2 * p.height

theorem right_square_prism_properties :
  ∃ (p : RightSquarePrism), p.base_edge = 3 ∧ p.height = 2 ∧
    lateral_area p = 24 ∧ volume p = 18 := by
  sorry

end NUMINAMATH_CALUDE_right_square_prism_properties_l3131_313192


namespace NUMINAMATH_CALUDE_divisibleByTwo_infinite_lessThanBillion_finite_l3131_313197

-- Define the set of numbers divisible by 2
def divisibleByTwo : Set Int := {x | ∃ n : Int, x = 2 * n}

-- Define the set of positive integers less than 1 billion
def lessThanBillion : Set Nat := {x | x > 0 ∧ x < 1000000000}

-- Theorem 1: The set of numbers divisible by 2 is infinite
theorem divisibleByTwo_infinite : Set.Infinite divisibleByTwo := by
  sorry

-- Theorem 2: The set of positive integers less than 1 billion is finite
theorem lessThanBillion_finite : Set.Finite lessThanBillion := by
  sorry

end NUMINAMATH_CALUDE_divisibleByTwo_infinite_lessThanBillion_finite_l3131_313197


namespace NUMINAMATH_CALUDE_y_derivative_l3131_313161

open Real

noncomputable def y (x : ℝ) : ℝ := 
  (sin (tan (1/7)) * (cos (16*x))^2) / (32 * sin (32*x))

theorem y_derivative (x : ℝ) : 
  deriv y x = -sin (tan (1/7)) / (4 * (sin (16*x))^2) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l3131_313161


namespace NUMINAMATH_CALUDE_lizzy_money_calculation_l3131_313159

/-- Calculates Lizzy's final amount after lending money and receiving it back with interest -/
def lizzys_final_amount (initial_amount loan_amount interest_rate : ℚ) : ℚ :=
  initial_amount - loan_amount + loan_amount * (1 + interest_rate)

/-- Theorem stating that Lizzy will have $33 after lending $15 from her initial $30 and receiving it back with 20% interest -/
theorem lizzy_money_calculation :
  lizzys_final_amount 30 15 (1/5) = 33 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_money_calculation_l3131_313159


namespace NUMINAMATH_CALUDE_memorial_day_weather_probability_l3131_313127

/-- The probability of exactly k successes in n independent Bernoulli trials --/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the Memorial Day weekend --/
def num_days : ℕ := 5

/-- The probability of rain on each day --/
def rain_probability : ℝ := 0.8

/-- The number of desired sunny days --/
def desired_sunny_days : ℕ := 2

theorem memorial_day_weather_probability :
  binomial_probability num_days desired_sunny_days (1 - rain_probability) = 51 / 250 := by
  sorry

end NUMINAMATH_CALUDE_memorial_day_weather_probability_l3131_313127


namespace NUMINAMATH_CALUDE_perfect_square_from_divisibility_l3131_313193

theorem perfect_square_from_divisibility (n p : ℕ) : 
  n > 1 → 
  Nat.Prime p → 
  (p - 1) % n = 0 → 
  (n^3 - 1) % p = 0 → 
  ∃ (k : ℕ), 4*p - 3 = k^2 :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_from_divisibility_l3131_313193


namespace NUMINAMATH_CALUDE_exponent_division_l3131_313146

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3131_313146


namespace NUMINAMATH_CALUDE_third_player_games_l3131_313126

/-- Represents a chess tournament with three players. -/
structure ChessTournament where
  total_games : ℕ
  player1_games : ℕ
  player2_games : ℕ
  player3_games : ℕ

/-- The theorem stating the number of games played by the third player. -/
theorem third_player_games (t : ChessTournament) 
  (h1 : t.total_games = 27)
  (h2 : t.player1_games = 13)
  (h3 : t.player2_games = 27)
  (h4 : t.player1_games + t.player2_games + t.player3_games = 2 * t.total_games) :
  t.player3_games = 14 := by
  sorry


end NUMINAMATH_CALUDE_third_player_games_l3131_313126


namespace NUMINAMATH_CALUDE_safe_access_theorem_access_conditions_l3131_313191

/-- Represents the number of members in the commission -/
def commission_size : ℕ := 11

/-- Represents the minimum number of members needed for access -/
def min_access : ℕ := 6

/-- Calculates the number of locks needed -/
def num_locks : ℕ := Nat.choose commission_size (min_access - 1)

/-- Calculates the number of keys each member should have -/
def keys_per_member : ℕ := num_locks * min_access / commission_size

/-- Theorem stating the correct number of locks and keys per member -/
theorem safe_access_theorem :
  num_locks = 462 ∧ keys_per_member = 252 :=
sorry

/-- Theorem proving that the arrangement satisfies the access conditions -/
theorem access_conditions (members : Finset (Fin commission_size)) :
  (members.card ≥ min_access → ∃ (lock : Fin num_locks), ∀ k ∈ members, k.val < keys_per_member) ∧
  (members.card < min_access → ∃ (lock : Fin num_locks), ∀ k ∈ members, k.val ≥ keys_per_member) :=
sorry

end NUMINAMATH_CALUDE_safe_access_theorem_access_conditions_l3131_313191


namespace NUMINAMATH_CALUDE_bus_ride_cost_l3131_313136

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 6.85

/-- The theorem stating the cost of a bus ride from town P to town Q -/
theorem bus_ride_cost : bus_cost = 1.50 := by
  have h1 : train_cost = bus_cost + 6.85 := by rfl
  have h2 : bus_cost + train_cost = 9.85 := by sorry
  sorry

end NUMINAMATH_CALUDE_bus_ride_cost_l3131_313136


namespace NUMINAMATH_CALUDE_frog_reach_edge_prob_l3131_313174

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Defines the 4x4 grid with wraparound edges -/
def Grid := Set Position

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, considering wraparound -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Defines the probability of reaching an edge within n hops -/
def probReachEdge (start : Position) (n : Nat) : ℝ :=
  sorry

theorem frog_reach_edge_prob :
  probReachEdge ⟨3, 3⟩ 5 = 1 := by sorry

end NUMINAMATH_CALUDE_frog_reach_edge_prob_l3131_313174
