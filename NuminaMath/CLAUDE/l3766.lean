import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l3766_376677

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l3766_376677


namespace NUMINAMATH_CALUDE_relationship_2x_3sinx_l3766_376624

theorem relationship_2x_3sinx :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧
  (∀ x : ℝ, 0 < x → x < θ → 2 * x < 3 * Real.sin x) ∧
  (2 * θ = 3 * Real.sin θ) ∧
  (∀ x : ℝ, θ < x → x < π / 2 → 2 * x > 3 * Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_relationship_2x_3sinx_l3766_376624


namespace NUMINAMATH_CALUDE_intersection_M_N_l3766_376675

-- Define the sets M and N
def M : Set ℝ := {x | (x - 3) / (x + 1) > 0}
def N : Set ℝ := {x | 3 * x + 2 > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3766_376675


namespace NUMINAMATH_CALUDE_vector_simplification_l3766_376682

variable {V : Type*} [AddCommGroup V]

variable (A B C M O : V)

theorem vector_simplification :
  (A - B + M - B) + (B - O + B - C) + (O - M) = A - C :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3766_376682


namespace NUMINAMATH_CALUDE_bat_wings_area_is_three_and_half_l3766_376636

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  /-- The width of the rectangle -/
  width : ℝ
  /-- The height of the rectangle -/
  height : ℝ
  /-- The length of segments DC, CB, and BA -/
  segment_length : ℝ
  /-- Width is 3 -/
  width_is_three : width = 3
  /-- Height is 4 -/
  height_is_four : height = 4
  /-- Segment length is 1 -/
  segment_is_one : segment_length = 1

/-- The area of the "bat wings" in the special rectangle -/
def batWingsArea (r : SpecialRectangle) : ℝ := sorry

/-- Theorem stating that the area of the "bat wings" is 3 1/2 -/
theorem bat_wings_area_is_three_and_half (r : SpecialRectangle) :
  batWingsArea r = 3.5 := by sorry

end NUMINAMATH_CALUDE_bat_wings_area_is_three_and_half_l3766_376636


namespace NUMINAMATH_CALUDE_total_pencils_l3766_376686

/-- Given that each child has 2 pencils and there are 8 children, 
    prove that the total number of pencils is 16. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) 
  (h2 : num_children = 8) : 
  pencils_per_child * num_children = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3766_376686


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3766_376620

def vector_a (t : ℝ) : ℝ × ℝ := (t, 1)
def vector_b : ℝ × ℝ := (1, 2)

theorem perpendicular_vectors (t : ℝ) :
  (vector_a t).1 * vector_b.1 + (vector_a t).2 * vector_b.2 = 0 → t = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3766_376620


namespace NUMINAMATH_CALUDE_expression_simplification_l3766_376670

theorem expression_simplification (a b : ℝ) 
  (h : |a - 1| + b^2 - 6*b + 9 = 0) : 
  ((3*a + 2*b)*(3*a - 2*b) + (3*a - b)^2 - b*(2*a - 3*b)) / (2*a) = -3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l3766_376670


namespace NUMINAMATH_CALUDE_haley_cider_production_l3766_376695

/-- Represents the number of pints of cider Haley can make -/
def cider_pints (golden_per_pint pink_per_pint farmhands apples_per_hour work_hours golden_ratio pink_ratio : ℕ) : ℕ :=
  let total_apples := farmhands * apples_per_hour * work_hours
  let apples_per_pint := golden_per_pint + pink_per_pint
  total_apples / apples_per_pint

/-- Theorem stating that Haley can make 120 pints of cider given the conditions -/
theorem haley_cider_production :
  cider_pints 20 40 6 240 5 1 2 = 120 := by
  sorry

#eval cider_pints 20 40 6 240 5 1 2

end NUMINAMATH_CALUDE_haley_cider_production_l3766_376695


namespace NUMINAMATH_CALUDE_barry_average_proof_l3766_376637

def barry_yards : List ℕ := [98, 107, 85, 89, 91]
def next_game_target : ℕ := 130
def total_games : ℕ := 6

theorem barry_average_proof :
  (barry_yards.sum + next_game_target) / total_games = 100 := by
  sorry

end NUMINAMATH_CALUDE_barry_average_proof_l3766_376637


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l3766_376678

def problems_per_worksheet : ℕ := 4
def total_worksheets : ℕ := 9
def graded_worksheets : ℕ := 5

theorem problems_left_to_grade :
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 := by
  sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l3766_376678


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l3766_376643

theorem equal_roots_quadratic_equation :
  ∃! r : ℝ, ∀ x : ℝ, x^2 - r*x - r^2 = 0 → (∃! y : ℝ, y^2 - r*y - r^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_equation_l3766_376643


namespace NUMINAMATH_CALUDE_triangle_side_length_l3766_376629

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = 10 →
  a / Real.sin A = b / Real.sin B →
  b = 5 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3766_376629


namespace NUMINAMATH_CALUDE_divisors_of_36_l3766_376611

/-- The number of integer divisors of 36 -/
def num_divisors_36 : ℕ := 18

/-- A function that counts the number of integer divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ i => n % i = 0) (Finset.range (n + 1))).card * 2

theorem divisors_of_36 :
  count_divisors 36 = num_divisors_36 := by
  sorry

#eval count_divisors 36

end NUMINAMATH_CALUDE_divisors_of_36_l3766_376611


namespace NUMINAMATH_CALUDE_find_t_l3766_376640

/-- The number of hours I worked -/
def my_hours (t : ℝ) : ℝ := 2*t + 2

/-- My hourly rate in dollars -/
def my_rate (t : ℝ) : ℝ := 4*t - 4

/-- The number of hours Emily worked -/
def emily_hours (t : ℝ) : ℝ := 4*t - 2

/-- Emily's hourly rate in dollars -/
def emily_rate (t : ℝ) : ℝ := t + 3

/-- My total earnings -/
def my_earnings (t : ℝ) : ℝ := my_hours t * my_rate t

/-- Emily's total earnings -/
def emily_earnings (t : ℝ) : ℝ := emily_hours t * emily_rate t

theorem find_t : ∃ t : ℝ, t > 0 ∧ my_earnings t = emily_earnings t + 6 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l3766_376640


namespace NUMINAMATH_CALUDE_chord_intersection_probability_for_1996_points_chord_intersection_probability_general_l3766_376694

/-- The number of points on the circle -/
def n : ℕ := 1996

/-- The probability that two chords formed by four randomly selected points intersect -/
def chord_intersection_probability (n : ℕ) : ℚ :=
  if n ≥ 4 then 1 / 4 else 0

/-- Theorem stating that the probability of chord intersection is 1/4 for 1996 points -/
theorem chord_intersection_probability_for_1996_points :
  chord_intersection_probability n = 1 / 4 := by
  sorry

/-- Theorem stating that the probability of chord intersection is always 1/4 for n ≥ 4 -/
theorem chord_intersection_probability_general (n : ℕ) (h : n ≥ 4) :
  chord_intersection_probability n = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_for_1996_points_chord_intersection_probability_general_l3766_376694


namespace NUMINAMATH_CALUDE_submarine_age_conversion_l3766_376631

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ × ℕ × ℕ) : ℕ :=
  let (a, b, c) := octal
  a * 8^2 + b * 8^1 + c * 8^0

theorem submarine_age_conversion :
  octal_to_decimal (3, 6, 7) = 247 := by
  sorry

end NUMINAMATH_CALUDE_submarine_age_conversion_l3766_376631


namespace NUMINAMATH_CALUDE_rain_probability_both_days_l3766_376673

theorem rain_probability_both_days (prob_monday : ℝ) (prob_tuesday : ℝ) 
  (h1 : prob_monday = 0.4)
  (h2 : prob_tuesday = 0.3)
  (h3 : 0 ≤ prob_monday ∧ prob_monday ≤ 1)
  (h4 : 0 ≤ prob_tuesday ∧ prob_tuesday ≤ 1) :
  prob_monday * prob_tuesday = 0.12 :=
by
  sorry

end NUMINAMATH_CALUDE_rain_probability_both_days_l3766_376673


namespace NUMINAMATH_CALUDE_function_order_l3766_376691

/-- A quadratic function f(x) = x^2 + bx + c that satisfies f(x-1) = f(3-x) for all x ∈ ℝ -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The symmetry condition of the function -/
axiom symmetry (b c : ℝ) : ∀ x, f b c (x - 1) = f b c (3 - x)

/-- Theorem stating the order of f(0), f(-2), and f(5) -/
theorem function_order (b c : ℝ) : f b c 0 < f b c (-2) ∧ f b c (-2) < f b c 5 := by
  sorry

end NUMINAMATH_CALUDE_function_order_l3766_376691


namespace NUMINAMATH_CALUDE_quadratic_transformation_sum_l3766_376679

/-- Given a quadratic function y = x^2 - 4x - 12, when transformed into the form y = (x - m)^2 + p,
    the sum of m and p equals -14. -/
theorem quadratic_transformation_sum (x : ℝ) :
  ∃ (m p : ℝ), (∀ x, x^2 - 4*x - 12 = (x - m)^2 + p) ∧ (m + p = -14) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_sum_l3766_376679


namespace NUMINAMATH_CALUDE_interest_rate_proof_l3766_376600

theorem interest_rate_proof (P : ℝ) (n : ℕ) (diff : ℝ) (r : ℝ) : 
  P = 5399.999999999995 →
  n = 2 →
  P * ((1 + r)^n - 1) - P * r * n = diff →
  diff = 216 →
  r = 0.2 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l3766_376600


namespace NUMINAMATH_CALUDE_avery_egg_cartons_l3766_376699

theorem avery_egg_cartons (num_chickens : ℕ) (eggs_per_chicken : ℕ) (eggs_per_carton : ℕ) : 
  num_chickens = 20 →
  eggs_per_chicken = 6 →
  eggs_per_carton = 12 →
  (num_chickens * eggs_per_chicken) / eggs_per_carton = 10 := by
sorry

end NUMINAMATH_CALUDE_avery_egg_cartons_l3766_376699


namespace NUMINAMATH_CALUDE_sum_less_than_sqrt_three_sum_squares_l3766_376644

theorem sum_less_than_sqrt_three_sum_squares (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a + b + c < Real.sqrt (3 * (a^2 + b^2 + c^2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_less_than_sqrt_three_sum_squares_l3766_376644


namespace NUMINAMATH_CALUDE_cracker_problem_l3766_376680

/-- The number of crackers Darren and Calvin bought together -/
def total_crackers (darren_boxes calvin_boxes crackers_per_box : ℕ) : ℕ :=
  (darren_boxes + calvin_boxes) * crackers_per_box

theorem cracker_problem :
  ∀ (darren_boxes calvin_boxes crackers_per_box : ℕ),
    darren_boxes = 4 →
    crackers_per_box = 24 →
    calvin_boxes = 2 * darren_boxes - 1 →
    total_crackers darren_boxes calvin_boxes crackers_per_box = 264 := by
  sorry

end NUMINAMATH_CALUDE_cracker_problem_l3766_376680


namespace NUMINAMATH_CALUDE_dagger_example_l3766_376609

/-- The dagger operation on rational numbers -/
def dagger (a b : ℚ) : ℚ :=
  (a.num ^ 2 : ℚ) * b * (b.den : ℚ) / (a.den : ℚ)

/-- Theorem stating that 5/11 † 9/4 = 225/11 -/
theorem dagger_example : dagger (5 / 11) (9 / 4) = 225 / 11 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l3766_376609


namespace NUMINAMATH_CALUDE_triangle_side_length_l3766_376619

open Real

theorem triangle_side_length 
  (g : ℝ → ℝ)
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, g x = cos (2 * x + π / 6))
  (h2 : (1/2) * b * c * sin A = 2)
  (h3 : b = 2)
  (h4 : g A = -1/2)
  (h5 : a < c) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3766_376619


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3766_376628

theorem arithmetic_calculation : 2546 + 240 / 60 - 346 = 2204 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3766_376628


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3766_376647

theorem largest_angle_in_special_triangle (α β γ : Real) : 
  α + β + γ = π ∧ 
  0 < α ∧ 0 < β ∧ 0 < γ ∧
  Real.tan α + Real.tan β + Real.tan γ = 2016 →
  (max α (max β γ)) > π/2 - π/360 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3766_376647


namespace NUMINAMATH_CALUDE_total_peaches_l3766_376623

theorem total_peaches (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 13) 
  (h2 : green_peaches = 3) : 
  red_peaches + green_peaches = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_peaches_l3766_376623


namespace NUMINAMATH_CALUDE_correct_new_balance_l3766_376666

/-- Calculates the new credit card balance after transactions -/
def new_balance (initial_balance groceries_expense towels_return : ℚ) : ℚ :=
  initial_balance + groceries_expense + (groceries_expense / 2) - towels_return

/-- Proves that the new balance is correct given the specified transactions -/
theorem correct_new_balance :
  new_balance 126 60 45 = 171 := by
  sorry

end NUMINAMATH_CALUDE_correct_new_balance_l3766_376666


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3766_376685

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3766_376685


namespace NUMINAMATH_CALUDE_compare_n_squared_and_two_to_n_l3766_376683

theorem compare_n_squared_and_two_to_n (n : ℕ+) :
  (n = 1 → n.val^2 < 2^n.val) ∧
  (n = 2 → n.val^2 = 2^n.val) ∧
  (n = 3 → n.val^2 > 2^n.val) ∧
  (n = 4 → n.val^2 = 2^n.val) ∧
  (n ≥ 5 → n.val^2 < 2^n.val) := by
  sorry

end NUMINAMATH_CALUDE_compare_n_squared_and_two_to_n_l3766_376683


namespace NUMINAMATH_CALUDE_triangle_properties_l3766_376661

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  c * Real.cos B = (2 * a - b) * Real.cos C →
  c = 2 →
  a + b + c = 2 * Real.sqrt 3 + 2 →
  -- Triangle validity conditions
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  -- Theorem statements
  C = Real.pi / 3 ∧
  (1/2) * a * b * Real.sin C = (2 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3766_376661


namespace NUMINAMATH_CALUDE_gcd_product_l3766_376655

theorem gcd_product (a b n : ℕ) (ha : Nat.gcd a n = 1) (hb : Nat.gcd b n = 1) : 
  Nat.gcd (a * b) n = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_product_l3766_376655


namespace NUMINAMATH_CALUDE_train_length_problem_l3766_376684

/-- The length of two trains passing each other --/
theorem train_length_problem (speed1 speed2 : ℝ) (passing_time : ℝ) (h1 : speed1 = 65) (h2 : speed2 = 50) (h3 : passing_time = 11.895652173913044) :
  let relative_speed := (speed1 + speed2) * (1000 / 3600)
  let total_distance := relative_speed * passing_time
  let train_length := total_distance / 2
  train_length = 190 := by sorry

end NUMINAMATH_CALUDE_train_length_problem_l3766_376684


namespace NUMINAMATH_CALUDE_vector_b_proof_l3766_376645

def vector_a : Fin 2 → ℝ := ![2, -1]

theorem vector_b_proof (b : Fin 2 → ℝ) 
  (collinear : ∃ k : ℝ, k > 0 ∧ b = k • vector_a)
  (magnitude : Real.sqrt ((b 0) ^ 2 + (b 1) ^ 2) = 2 * Real.sqrt 5) :
  b = ![4, -2] := by
  sorry

end NUMINAMATH_CALUDE_vector_b_proof_l3766_376645


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3766_376635

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, k > 0 ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l3766_376635


namespace NUMINAMATH_CALUDE_perfect_number_examples_sum_xy_is_one_k_equals_36_when_S_is_perfect_number_l3766_376663

/-- Definition of a perfect number -/
def is_perfect_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

/-- Theorem 1: 11 is not a perfect number and 53 is a perfect number -/
theorem perfect_number_examples :
  (¬ is_perfect_number 11) ∧ (is_perfect_number 53) := by sorry

/-- Theorem 2: Given x^2 + y^2 - 4x + 2y + 5 = 0, prove x + y = 1 -/
theorem sum_xy_is_one (x y : ℝ) (h : x^2 + y^2 - 4*x + 2*y + 5 = 0) :
  x + y = 1 := by sorry

/-- Definition of S -/
def S (x y k : ℝ) : ℝ := 2*x^2 + y^2 + 2*x*y + 12*x + k

/-- Theorem 3: Given S = 2x^2 + y^2 + 2xy + 12x + k, 
    prove that k = 36 when S is a perfect number -/
theorem k_equals_36_when_S_is_perfect_number (x y : ℝ) :
  (∃ a b : ℝ, S x y 36 = a^2 + b^2) → 
  (∀ k : ℝ, (∃ a b : ℝ, S x y k = a^2 + b^2) → k = 36) := by sorry

end NUMINAMATH_CALUDE_perfect_number_examples_sum_xy_is_one_k_equals_36_when_S_is_perfect_number_l3766_376663


namespace NUMINAMATH_CALUDE_a_99_value_l3766_376656

def is_increasing (s : ℕ → ℝ) := ∀ n, s n ≤ s (n + 1)
def is_decreasing (s : ℕ → ℝ) := ∀ n, s n ≥ s (n + 1)

theorem a_99_value (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n ≥ 2, |a n - a (n-1)| = (n : ℝ)^2)
  (h3 : is_increasing (λ n => a (2*n - 1)))
  (h4 : is_decreasing (λ n => a (2*n)))
  (h5 : a 1 > a 2) :
  a 99 = 4950 := by sorry

end NUMINAMATH_CALUDE_a_99_value_l3766_376656


namespace NUMINAMATH_CALUDE_spade_calculation_l3766_376648

/-- The spade operation for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- The main theorem -/
theorem spade_calculation : spade 5 (spade 7 8) = -200 := by sorry

end NUMINAMATH_CALUDE_spade_calculation_l3766_376648


namespace NUMINAMATH_CALUDE_quadratic_completion_sum_l3766_376668

theorem quadratic_completion_sum (x : ℝ) : ∃ (m n : ℝ), 
  (x^2 - 8*x + 3 = 0 ↔ (x - m)^2 = n) ∧ m + n = 17 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_sum_l3766_376668


namespace NUMINAMATH_CALUDE_min_value_theorem_l3766_376653

theorem min_value_theorem (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀ + y₀ + z₀ = 1 ∧ 
    1/x₀ + 4/y₀ + 9/z₀ = 36 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3766_376653


namespace NUMINAMATH_CALUDE_sum_inequality_l3766_376662

theorem sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 3) : 
  1 / (x^5 - x^2 + 3) + 1 / (y^5 - y^2 + 3) + 1 / (z^5 - z^2 + 3) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l3766_376662


namespace NUMINAMATH_CALUDE_independence_test_type_I_error_l3766_376607

/-- Represents the observed value of the χ² statistic -/
def k : ℝ := sorry

/-- Represents the probability of making a Type I error -/
def type_I_error_prob : ℝ → ℝ := sorry

/-- States that as k decreases, the probability of Type I error increases -/
theorem independence_test_type_I_error (h : k₁ < k₂) :
  type_I_error_prob k₁ > type_I_error_prob k₂ := by sorry

end NUMINAMATH_CALUDE_independence_test_type_I_error_l3766_376607


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l3766_376658

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_third : a 3 = 1)
  (h_product : a 5 * a 6 * a 7 = 8) :
  a 9 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l3766_376658


namespace NUMINAMATH_CALUDE_factor_expression_l3766_376613

theorem factor_expression (x : ℝ) : 4*x*(x+2) + 9*(x+2) = (x+2)*(4*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3766_376613


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3766_376610

/-- Tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- The volume of a tetrahedron given its edge lengths -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: The volume of the specific tetrahedron is 10.25 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 4,
    PR := 5,
    PS := 6,
    QR := 3,
    QS := Real.sqrt 37,
    RS := 7
  }
  volume t = 10.25 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3766_376610


namespace NUMINAMATH_CALUDE_remainder_of_S_mod_512_l3766_376617

def R : Finset ℕ := Finset.image (λ n => (3^n) % 512) (Finset.range 12)

def S : ℕ := Finset.sum R id

theorem remainder_of_S_mod_512 : S % 512 = 72 := by sorry

end NUMINAMATH_CALUDE_remainder_of_S_mod_512_l3766_376617


namespace NUMINAMATH_CALUDE_count_special_integers_l3766_376649

def is_even_digit (d : Nat) : Bool :=
  d % 2 = 0 ∧ d ≤ 9

def has_only_even_digits (n : Nat) : Bool :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def is_five_digit (n : Nat) : Bool :=
  10000 ≤ n ∧ n ≤ 99999

theorem count_special_integers :
  (Finset.filter (λ n : Nat => is_five_digit n ∧ has_only_even_digits n ∧ n % 5 = 0)
    (Finset.range 100000)).card = 500 := by
  sorry

end NUMINAMATH_CALUDE_count_special_integers_l3766_376649


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l3766_376697

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 8 →
  initial_avg = 27 →
  leaving_age = 21 →
  remaining_people = 7 →
  (initial_people : ℚ) * initial_avg - leaving_age ≥ 0 →
  (((initial_people : ℚ) * initial_avg - leaving_age) / remaining_people : ℚ) = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l3766_376697


namespace NUMINAMATH_CALUDE_smallest_gcd_yz_l3766_376660

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x.val y.val = 360) (h2 : Nat.gcd x.val z.val = 1176) :
  ∃ (k : ℕ+), (∀ (w : ℕ+), Nat.gcd y.val z.val ≥ k.val) ∧ Nat.gcd y.val z.val = k.val :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_yz_l3766_376660


namespace NUMINAMATH_CALUDE_bus_line_count_l3766_376601

theorem bus_line_count (people_in_front people_behind : ℕ) 
  (h1 : people_in_front = 6) 
  (h2 : people_behind = 5) : 
  people_in_front + 1 + people_behind = 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_line_count_l3766_376601


namespace NUMINAMATH_CALUDE_num_motorcycles_in_parking_lot_l3766_376604

-- Define the number of wheels for each vehicle type
def car_wheels : ℕ := 5
def motorcycle_wheels : ℕ := 2
def tricycle_wheels : ℕ := 3

-- Define the number of cars and tricycles
def num_cars : ℕ := 19
def num_tricycles : ℕ := 11

-- Define the total number of wheels
def total_wheels : ℕ := 184

-- Theorem to prove
theorem num_motorcycles_in_parking_lot :
  ∃ (num_motorcycles : ℕ),
    num_motorcycles = 28 ∧
    num_motorcycles * motorcycle_wheels +
    num_cars * car_wheels +
    num_tricycles * tricycle_wheels = total_wheels :=
by sorry

end NUMINAMATH_CALUDE_num_motorcycles_in_parking_lot_l3766_376604


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l3766_376612

theorem cubic_equation_integer_solutions (a b : ℤ) :
  a^3 + b^3 + 3*a*b = 1 ↔ (b = 1 - a) ∨ (a = -1 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l3766_376612


namespace NUMINAMATH_CALUDE_f_sum_derivative_equals_two_l3766_376622

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem f_sum_derivative_equals_two :
  let f' := deriv f
  f 2017 + f' 2017 + f (-2017) - f' (-2017) = 2 := by sorry

end NUMINAMATH_CALUDE_f_sum_derivative_equals_two_l3766_376622


namespace NUMINAMATH_CALUDE_domino_tiling_theorem_l3766_376633

/-- Represents a rectangular grid -/
structure Rectangle where
  m : ℕ
  n : ℕ

/-- Represents a domino tile -/
structure Domino where
  length : ℕ
  width : ℕ

/-- Represents a tiling of a rectangle with dominoes -/
def Tiling (r : Rectangle) (d : Domino) := Unit

/-- Predicate to check if a tiling has no straight cuts -/
def has_no_straight_cuts (t : Tiling r d) : Prop := sorry

theorem domino_tiling_theorem :
  /- Part a -/
  (∀ (r : Rectangle) (d : Domino), r.m = 6 ∧ r.n = 6 ∧ d.length = 1 ∧ d.width = 2 →
    ¬ ∃ (t : Tiling r d), has_no_straight_cuts t) ∧
  /- Part b -/
  (∀ (r : Rectangle) (d : Domino), r.m > 6 ∧ r.n > 6 ∧ (r.m * r.n) % 2 = 0 ∧ d.length = 1 ∧ d.width = 2 →
    ∃ (t : Tiling r d), has_no_straight_cuts t) ∧
  /- Part c -/
  (∃ (r : Rectangle) (d : Domino) (t : Tiling r d), r.m = 6 ∧ r.n = 8 ∧ d.length = 1 ∧ d.width = 2 ∧
    has_no_straight_cuts t) :=
by sorry

end NUMINAMATH_CALUDE_domino_tiling_theorem_l3766_376633


namespace NUMINAMATH_CALUDE_shrimp_earnings_l3766_376615

theorem shrimp_earnings (victor_shrimp : ℕ) (austin_less : ℕ) (price : ℚ) (tails_per_set : ℕ) : 
  victor_shrimp = 26 →
  austin_less = 8 →
  price = 7 →
  tails_per_set = 11 →
  let austin_shrimp := victor_shrimp - austin_less
  let total_victor_austin := victor_shrimp + austin_shrimp
  let brian_shrimp := total_victor_austin / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let sets_sold := total_shrimp / tails_per_set
  let total_earnings := price * (sets_sold : ℚ)
  let each_boy_earnings := total_earnings / 3
  each_boy_earnings = 14 := by
sorry

end NUMINAMATH_CALUDE_shrimp_earnings_l3766_376615


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_value_l3766_376625

theorem square_plus_inverse_square_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_value_l3766_376625


namespace NUMINAMATH_CALUDE_equation_solution_l3766_376638

theorem equation_solution : ∃ x : ℝ, 11 + Real.sqrt (x + 6 * 4 / 3) = 13 ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3766_376638


namespace NUMINAMATH_CALUDE_odd_product_plus_one_is_odd_l3766_376692

theorem odd_product_plus_one_is_odd (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : 0 < p) (hq_pos : 0 < q) : 
  Odd (4 * p * q + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_product_plus_one_is_odd_l3766_376692


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3766_376630

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {3, 4, 5, 6}

theorem intersection_of_M_and_N : M ∩ N = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3766_376630


namespace NUMINAMATH_CALUDE_probability_one_instrument_l3766_376674

theorem probability_one_instrument (total : ℕ) (at_least_one : ℚ) (two_or_more : ℕ) : 
  total = 800 →
  at_least_one = 1/5 →
  two_or_more = 128 →
  (((at_least_one * total) - two_or_more) / total : ℚ) = 1/25 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_instrument_l3766_376674


namespace NUMINAMATH_CALUDE_work_completion_time_l3766_376667

/-- Worker rates and work completion time -/
theorem work_completion_time
  (rate_a rate_b rate_c rate_d : ℝ)
  (total_work : ℝ)
  (h1 : rate_a = 1.5 * rate_b)
  (h2 : rate_a * 30 = total_work)
  (h3 : rate_c = 2 * rate_b)
  (h4 : rate_d = 0.5 * rate_a)
  : ∃ (days : ℕ), days = 12 ∧ 
    (1.25 * rate_b + 2.75 * rate_b) * (days : ℝ) ≥ total_work ∧
    (1.25 * rate_b + 2.75 * rate_b) * ((days - 1) : ℝ) < total_work :=
by sorry


end NUMINAMATH_CALUDE_work_completion_time_l3766_376667


namespace NUMINAMATH_CALUDE_work_completion_time_l3766_376641

/-- The number of days y needs to finish the work alone -/
def y_days : ℝ := 15

/-- The number of days y worked before leaving -/
def y_worked : ℝ := 5

/-- The number of days x needed to finish the remaining work after y left -/
def x_remaining : ℝ := 12

/-- The number of days x needs to finish the work alone -/
def x_days : ℝ := 18

theorem work_completion_time :
  (y_worked / y_days) + (x_remaining / x_days) = 1 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3766_376641


namespace NUMINAMATH_CALUDE_playground_area_l3766_376681

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ

/-- The landscape satisfies the given conditions -/
def validLandscape (l : Landscape) : Prop :=
  l.length = 8 * l.breadth ∧
  l.length = 240 ∧
  l.playgroundArea = (1 / 6) * (l.length * l.breadth)

/-- Theorem: The playground area is 1200 square meters -/
theorem playground_area (l : Landscape) (h : validLandscape l) : 
  l.playgroundArea = 1200 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_l3766_376681


namespace NUMINAMATH_CALUDE_otimes_inequality_range_l3766_376618

-- Define the ⊗ operation
def otimes (x y : ℝ) := x * (2 - y)

-- Theorem statement
theorem otimes_inequality_range (m : ℝ) :
  (∀ x : ℝ, otimes (x + m) x < 1) ↔ -4 < m ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_otimes_inequality_range_l3766_376618


namespace NUMINAMATH_CALUDE_mary_circus_change_l3766_376634

/-- Calculates the change Mary receives after buying circus tickets for herself and her children -/
theorem mary_circus_change (num_children : ℕ) (adult_price child_price payment : ℚ) : 
  num_children = 3 ∧ 
  adult_price = 2 ∧ 
  child_price = 1 ∧ 
  payment = 20 → 
  payment - (adult_price + num_children * child_price) = 15 := by
  sorry

end NUMINAMATH_CALUDE_mary_circus_change_l3766_376634


namespace NUMINAMATH_CALUDE_age_difference_l3766_376693

/-- Given that the sum of X and Y is 15 years greater than the sum of Y and Z,
    prove that Z is 1.5 decades younger than X. -/
theorem age_difference (X Y Z : ℕ) (h : X + Y = Y + Z + 15) :
  (X - Z : ℚ) / 10 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3766_376693


namespace NUMINAMATH_CALUDE_line_slope_l3766_376605

/-- The slope of the line given by the equation x/4 + y/5 = 1 is -5/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_l3766_376605


namespace NUMINAMATH_CALUDE_cube_surface_area_l3766_376672

theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1331 →
  volume = side ^ 3 →
  surface_area = 6 * side ^ 2 →
  surface_area = 726 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3766_376672


namespace NUMINAMATH_CALUDE_square_of_product_l3766_376671

theorem square_of_product (p q : ℝ) : (-3 * p * q)^2 = 9 * p^2 * q^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_l3766_376671


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3766_376664

theorem trigonometric_identities (x : Real) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.tan x = -2) : 
  (Real.sin x - Real.cos x = -3 * Real.sqrt 5 / 5) ∧ 
  ((Real.sin (2 * π - x) * Real.cos (π - x) - Real.sin x ^ 2) / 
   (Real.cos (π + x) * Real.cos (π/2 - x) + Real.cos x ^ 2) = -2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3766_376664


namespace NUMINAMATH_CALUDE_backpack_price_l3766_376608

theorem backpack_price (t_shirt_price cap_price discount total_after_discount : ℕ) 
  (ht : t_shirt_price = 30)
  (hc : cap_price = 5)
  (hd : discount = 2)
  (ht : total_after_discount = 43) :
  ∃ backpack_price : ℕ, 
    t_shirt_price + backpack_price + cap_price - discount = total_after_discount ∧ 
    backpack_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_backpack_price_l3766_376608


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l3766_376659

theorem difference_of_squares_division : (121^2 - 112^2) / 9 = 233 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l3766_376659


namespace NUMINAMATH_CALUDE_division_by_fraction_twelve_divided_by_three_fifths_l3766_376652

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b := by sorry

theorem twelve_divided_by_three_fifths :
  12 / (3 / 5) = 20 := by sorry

end NUMINAMATH_CALUDE_division_by_fraction_twelve_divided_by_three_fifths_l3766_376652


namespace NUMINAMATH_CALUDE_M_invertible_iff_square_free_l3766_376627

def M (n : ℕ+) : Matrix (Fin n) (Fin n) ℤ :=
  Matrix.of (fun i j => if (i.val + 1) % j.val = 0 then 1 else 0)

def square_free (m : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → m % (k * k) ≠ 0

theorem M_invertible_iff_square_free (n : ℕ+) :
  IsUnit (M n).det ↔ square_free (n + 1) :=
sorry

end NUMINAMATH_CALUDE_M_invertible_iff_square_free_l3766_376627


namespace NUMINAMATH_CALUDE_pyramid_intersection_volume_l3766_376676

/-- The length of each edge of the pyramids -/
def edge_length : ℝ := 12

/-- The volume of the solid of intersection of two regular square pyramids -/
def intersection_volume : ℝ := 72

/-- Theorem stating the volume of the solid of intersection of two regular square pyramids -/
theorem pyramid_intersection_volume :
  let pyramids : ℕ := 2
  let base_parallel : Prop := True  -- Represents that bases are parallel
  let edges_parallel : Prop := True  -- Represents that edges are parallel
  let apex_at_center : Prop := True  -- Represents that each apex is at the center of the other base
  intersection_volume = 72 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_intersection_volume_l3766_376676


namespace NUMINAMATH_CALUDE_library_books_l3766_376690

theorem library_books (initial_books : ℕ) : 
  (initial_books : ℚ) * (2 / 6) = 3300 → initial_books = 9900 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l3766_376690


namespace NUMINAMATH_CALUDE_det_equals_polynomial_l3766_376665

/-- The determinant of a 3x3 matrix with polynomial entries -/
def matrix_det (y : ℝ) : ℝ :=
  let a11 := 2*y + 3
  let a12 := y - 1
  let a13 := y + 2
  let a21 := y + 1
  let a22 := 2*y
  let a23 := y
  let a31 := y
  let a32 := y
  let a33 := 2*y - 1
  a11 * (a22 * a33 - a23 * a32) - 
  a12 * (a21 * a33 - a23 * a31) + 
  a13 * (a21 * a32 - a22 * a31)

theorem det_equals_polynomial (y : ℝ) : 
  matrix_det y = 4*y^3 + 8*y^2 - 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_det_equals_polynomial_l3766_376665


namespace NUMINAMATH_CALUDE_unique_solution_3644_l3766_376642

def repeating_decimal_ab (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

def repeating_decimal_abcd (a b c d : ℕ) : ℚ := (1000 * a + 100 * b + 10 * c + d : ℚ) / 9999

theorem unique_solution_3644 (a b c d : ℕ) :
  a ∈ Finset.range 10 →
  b ∈ Finset.range 10 →
  c ∈ Finset.range 10 →
  d ∈ Finset.range 10 →
  repeating_decimal_ab a b + repeating_decimal_abcd a b c d = 27 / 37 →
  a = 3 ∧ b = 6 ∧ c = 4 ∧ d = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_3644_l3766_376642


namespace NUMINAMATH_CALUDE_lower_right_is_one_l3766_376626

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Check if a number appears exactly once in each row --/
def valid_rows (g : Grid) : Prop :=
  ∀ i : Fin 5, ∀ n : Fin 5, (∃! j : Fin 5, g i j = n)

/-- Check if a number appears exactly once in each column --/
def valid_columns (g : Grid) : Prop :=
  ∀ j : Fin 5, ∀ n : Fin 5, (∃! i : Fin 5, g i j = n)

/-- Check if no number repeats on the main diagonal --/
def valid_diagonal (g : Grid) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i i ≠ g j j

/-- Check if the grid satisfies the initial placements --/
def valid_initial (g : Grid) : Prop :=
  g 0 0 = 0 ∧ g 1 1 = 1 ∧ g 2 2 = 2 ∧ g 0 3 = 3 ∧ g 3 3 = 4 ∧ g 1 4 = 0

theorem lower_right_is_one (g : Grid) 
  (h_rows : valid_rows g) 
  (h_cols : valid_columns g) 
  (h_diag : valid_diagonal g) 
  (h_init : valid_initial g) : 
  g 4 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_lower_right_is_one_l3766_376626


namespace NUMINAMATH_CALUDE_double_wardrobe_with_socks_l3766_376639

/-- Represents the number of pairs of an item in a wardrobe -/
structure WardobeItem where
  pairs : Nat

/-- Represents a wardrobe with various clothing items -/
structure Wardrobe where
  socks : WardobeItem
  shoes : WardobeItem
  pants : WardobeItem
  tshirts : WardobeItem

/-- Calculates the total number of individual items in a wardrobe -/
def totalItems (w : Wardrobe) : Nat :=
  w.socks.pairs * 2 + w.shoes.pairs * 2 + w.pants.pairs + w.tshirts.pairs

/-- Theorem: Buying 35 pairs of socks doubles the number of items in Jonas' wardrobe -/
theorem double_wardrobe_with_socks (jonas : Wardrobe)
    (h1 : jonas.socks.pairs = 20)
    (h2 : jonas.shoes.pairs = 5)
    (h3 : jonas.pants.pairs = 10)
    (h4 : jonas.tshirts.pairs = 10) :
    totalItems { socks := ⟨jonas.socks.pairs + 35⟩,
                 shoes := jonas.shoes,
                 pants := jonas.pants,
                 tshirts := jonas.tshirts } = 2 * totalItems jonas := by
  sorry


end NUMINAMATH_CALUDE_double_wardrobe_with_socks_l3766_376639


namespace NUMINAMATH_CALUDE_bryans_deposit_l3766_376614

theorem bryans_deposit (mark_deposit : ℕ) (bryan_deposit : ℕ) : 
  mark_deposit = 88 →
  bryan_deposit = 5 * mark_deposit - 40 →
  bryan_deposit = 400 := by
sorry

end NUMINAMATH_CALUDE_bryans_deposit_l3766_376614


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3766_376603

theorem contrapositive_equivalence (p q : Prop) :
  (p → q) ↔ (¬q → ¬p) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3766_376603


namespace NUMINAMATH_CALUDE_max_triangle_area_l3766_376602

theorem max_triangle_area (a b c : Real) (h1 : 0 < a) (h2 : a ≤ 1) (h3 : 1 ≤ b) 
  (h4 : b ≤ 2) (h5 : 2 ≤ c) (h6 : c ≤ 3) :
  ∃ (area : Real), area ≤ 1 ∧ 
    ∀ (A : Real), (∃ (α : Real), A = 1/2 * a * b * Real.sin α ∧ 
      a + b > c ∧ b + c > a ∧ c + a > b) → A ≤ area :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l3766_376602


namespace NUMINAMATH_CALUDE_original_number_exists_and_unique_l3766_376616

theorem original_number_exists_and_unique : 
  ∃! x : ℝ, 3 * (2 * x + 9) = 63 := by
  sorry

end NUMINAMATH_CALUDE_original_number_exists_and_unique_l3766_376616


namespace NUMINAMATH_CALUDE_speed_limit_inequality_l3766_376698

/-- Given a speed limit of 40 km/h, prove that it can be expressed as v ≤ 40, where v is the speed of a vehicle. -/
theorem speed_limit_inequality (v : ℝ) (speed_limit : ℝ) (h : speed_limit = 40) :
  v ≤ speed_limit ↔ v ≤ 40 := by sorry

end NUMINAMATH_CALUDE_speed_limit_inequality_l3766_376698


namespace NUMINAMATH_CALUDE_phi_value_l3766_376651

theorem phi_value (φ : Real) (a : Real) :
  φ ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ x₁ x₂ x₃ : Real,
    x₁ ∈ Set.Icc 0 Real.pi ∧
    x₂ ∈ Set.Icc 0 Real.pi ∧
    x₃ ∈ Set.Icc 0 Real.pi ∧
    Real.sin (2 * x₁ + φ) = a ∧
    Real.sin (2 * x₂ + φ) = a ∧
    Real.sin (2 * x₃ + φ) = a ∧
    x₁ + x₂ + x₃ = 7 * Real.pi / 6) →
  φ = Real.pi / 3 ∨ φ = 4 * Real.pi / 3 :=
by sorry

end NUMINAMATH_CALUDE_phi_value_l3766_376651


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l3766_376646

theorem ellipse_parabola_intersection (p : ℝ) (h_p : p > 0) : 
  (∃ A B : ℝ × ℝ, 
    (A.1^2 / 8 + A.2^2 / 2 = 1) ∧ 
    (B.1^2 / 8 + B.2^2 / 2 = 1) ∧
    (A.2^2 = 2 * p * A.1) ∧ 
    (B.2^2 = 2 * p * B.1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)) → 
  p = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l3766_376646


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l3766_376687

theorem modular_arithmetic_problem :
  (3 * (7⁻¹ : ZMod 97) + 5 * (13⁻¹ : ZMod 97)) = (73 : ZMod 97) := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l3766_376687


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3766_376650

theorem x_squared_plus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  x^4 + 1/x^4 = 23 → x^2 + 1/x^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l3766_376650


namespace NUMINAMATH_CALUDE_cos_four_thirds_pi_plus_alpha_l3766_376696

theorem cos_four_thirds_pi_plus_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos ((4 / 3) * π + α) = -(1 / 3) := by
  sorry

end NUMINAMATH_CALUDE_cos_four_thirds_pi_plus_alpha_l3766_376696


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l3766_376632

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 156 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l3766_376632


namespace NUMINAMATH_CALUDE_pauls_toys_l3766_376654

theorem pauls_toys (toys_per_box : ℕ) (number_of_boxes : ℕ) (h1 : toys_per_box = 8) (h2 : number_of_boxes = 4) :
  toys_per_box * number_of_boxes = 32 := by
  sorry

end NUMINAMATH_CALUDE_pauls_toys_l3766_376654


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3766_376621

theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (a 2 / a 1)) 
  (h_sum : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) 
  (h_arith : 2 * (S 1 + 2 * a 2) = (S 3 + a 3) + (S 2 + a 2)) :
  a 2 / a 1 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3766_376621


namespace NUMINAMATH_CALUDE_circle_motion_speeds_l3766_376606

/-- Represents the state of two circles moving towards the vertex of a right angle -/
structure CircleMotion where
  r1 : ℝ  -- radius of first circle
  r2 : ℝ  -- radius of second circle
  d1 : ℝ  -- initial distance of first circle from vertex
  d2 : ℝ  -- initial distance of second circle from vertex
  t_external : ℝ  -- time when circles touch externally
  t_internal : ℝ  -- time when circles touch internally

/-- Represents a pair of speeds for the two circles -/
structure SpeedPair where
  s1 : ℝ  -- speed of first circle
  s2 : ℝ  -- speed of second circle

/-- Checks if a given speed pair satisfies the conditions for the circle motion -/
def satisfiesConditions (cm : CircleMotion) (sp : SpeedPair) : Prop :=
  let d1_external := cm.d1 - sp.s1 * cm.t_external
  let d2_external := cm.d2 - sp.s2 * cm.t_external
  let d1_internal := cm.d1 - sp.s1 * cm.t_internal
  let d2_internal := cm.d2 - sp.s2 * cm.t_internal
  d1_external^2 + d2_external^2 = (cm.r1 + cm.r2)^2 ∧
  d1_internal^2 + d2_internal^2 = (cm.r1 - cm.r2)^2

/-- The main theorem stating that given the conditions, only two speed pairs satisfy the motion -/
theorem circle_motion_speeds (cm : CircleMotion)
  (h_r1 : cm.r1 = 9)
  (h_r2 : cm.r2 = 4)
  (h_d1 : cm.d1 = 48)
  (h_d2 : cm.d2 = 14)
  (h_t_external : cm.t_external = 9)
  (h_t_internal : cm.t_internal = 11) :
  ∃ (sp1 sp2 : SpeedPair),
    satisfiesConditions cm sp1 ∧
    satisfiesConditions cm sp2 ∧
    ((sp1.s1 = 4 ∧ sp1.s2 = 1) ∨ (sp1.s1 = 3.9104 ∧ sp1.s2 = 1.3072)) ∧
    ((sp2.s1 = 4 ∧ sp2.s2 = 1) ∨ (sp2.s1 = 3.9104 ∧ sp2.s2 = 1.3072)) ∧
    sp1 ≠ sp2 ∧
    ∀ (sp : SpeedPair), satisfiesConditions cm sp → (sp = sp1 ∨ sp = sp2) := by
  sorry

end NUMINAMATH_CALUDE_circle_motion_speeds_l3766_376606


namespace NUMINAMATH_CALUDE_divisibility_property_l3766_376657

theorem divisibility_property (n : ℕ) (h1 : n > 2) (h2 : Even n) :
  (n + 1) ∣ (n + 1)^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3766_376657


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3766_376689

/-- A rhombus with given diagonal lengths has a specific perimeter -/
theorem rhombus_perimeter (AC BD : ℝ) (h1 : AC = 8) (h2 : BD = 6) :
  let side_length := Real.sqrt ((AC / 2) ^ 2 + (BD / 2) ^ 2)
  4 * side_length = 20 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3766_376689


namespace NUMINAMATH_CALUDE_tan_cos_expression_equals_negative_one_l3766_376669

theorem tan_cos_expression_equals_negative_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_cos_expression_equals_negative_one_l3766_376669


namespace NUMINAMATH_CALUDE_number_puzzle_l3766_376688

theorem number_puzzle : ∃ x : ℝ, ((2 * x - 37 + 25) / 8 = 5) ∧ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3766_376688
