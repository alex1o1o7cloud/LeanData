import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3813_381364

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3813_381364


namespace NUMINAMATH_CALUDE_frog_safety_probability_l3813_381334

/-- Represents the probability of the frog reaching stone 14 safely when starting from stone n -/
def safe_probability (n : ℕ) : ℚ := sorry

/-- The total number of stones -/
def total_stones : ℕ := 15

/-- The probability of jumping backwards from stone n -/
def back_prob (n : ℕ) : ℚ := (n + 1) / total_stones

/-- The probability of jumping forwards from stone n -/
def forward_prob (n : ℕ) : ℚ := 1 - back_prob n

theorem frog_safety_probability :
  0 < 2 ∧ 2 < 14 →
  (∀ n : ℕ, 0 < n ∧ n < 14 →
    safe_probability n = back_prob n * safe_probability (n - 1) +
                         forward_prob n * safe_probability (n + 1)) →
  safe_probability 0 = 0 →
  safe_probability 14 = 1 →
  safe_probability 2 = 85 / 256 :=
sorry

end NUMINAMATH_CALUDE_frog_safety_probability_l3813_381334


namespace NUMINAMATH_CALUDE_chorus_arrangement_l3813_381359

/-- The maximum number of chorus members that satisfies both arrangements -/
def max_chorus_members : ℕ := 300

/-- The number of columns in the rectangular formation -/
def n : ℕ := 15

/-- The side length of the square formation -/
def k : ℕ := 17

theorem chorus_arrangement :
  (∃ m : ℕ, m = max_chorus_members) ∧
  (∃ k : ℕ, max_chorus_members = k^2 + 11) ∧
  (max_chorus_members = n * (n + 5)) ∧
  (∀ m : ℕ, m > max_chorus_members →
    (¬∃ k : ℕ, m = k^2 + 11) ∨ (¬∃ n : ℕ, m = n * (n + 5))) :=
by sorry

#eval max_chorus_members
#eval n
#eval k

end NUMINAMATH_CALUDE_chorus_arrangement_l3813_381359


namespace NUMINAMATH_CALUDE_log_function_fixed_point_l3813_381358

theorem log_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x => Real.log x / Real.log a + 1
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_log_function_fixed_point_l3813_381358


namespace NUMINAMATH_CALUDE_paige_pencils_l3813_381392

theorem paige_pencils (initial_pencils : ℕ) : 
  (initial_pencils - 3 = 91) → initial_pencils = 94 := by
  sorry

end NUMINAMATH_CALUDE_paige_pencils_l3813_381392


namespace NUMINAMATH_CALUDE_log_equation_solution_l3813_381384

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 →
  x = 3^(10/3) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3813_381384


namespace NUMINAMATH_CALUDE_cornelia_age_proof_l3813_381352

/-- Cornelia's current age -/
def cornelia_age : ℕ := 80

/-- Kilee's current age -/
def kilee_age : ℕ := 20

/-- In 10 years, Cornelia will be three times as old as Kilee -/
theorem cornelia_age_proof :
  cornelia_age + 10 = 3 * (kilee_age + 10) :=
by sorry

end NUMINAMATH_CALUDE_cornelia_age_proof_l3813_381352


namespace NUMINAMATH_CALUDE_transport_theorem_l3813_381380

-- Define the capacity of a worker per hour
def worker_capacity : ℝ := 30

-- Define the capacity of a robot per hour
def robot_capacity : ℝ := 450

-- Define the number of robots
def num_robots : ℕ := 3

-- Define the total amount to be transported
def total_amount : ℝ := 3600

-- Define the time limit
def time_limit : ℝ := 2

-- Define the function to calculate the minimum number of additional workers
def min_additional_workers : ℕ := 15

theorem transport_theorem :
  -- Condition 1: Robot carries 420kg more than a worker
  (robot_capacity = worker_capacity + 420) →
  -- Condition 2: Time for robot to carry 900kg equals time for 10 workers to carry 600kg
  (900 / robot_capacity = 600 / (10 * worker_capacity)) →
  -- Condition 3 & 4 are implicitly used in the conclusion
  -- Conclusion 1: Robot capacity is 450kg per hour
  (robot_capacity = 450) ∧
  -- Conclusion 2: Worker capacity is 30kg per hour
  (worker_capacity = 30) ∧
  -- Conclusion 3: Minimum additional workers needed is 15
  (min_additional_workers = 15 ∧
   robot_capacity * num_robots * time_limit + worker_capacity * min_additional_workers * time_limit ≥ total_amount ∧
   ∀ n : ℕ, n < 15 → robot_capacity * num_robots * time_limit + worker_capacity * n * time_limit < total_amount) :=
by sorry

end NUMINAMATH_CALUDE_transport_theorem_l3813_381380


namespace NUMINAMATH_CALUDE_friend_bicycles_count_friend_owns_ten_bicycles_l3813_381372

theorem friend_bicycles_count (ignatius_bicycles : ℕ) (tires_per_bicycle : ℕ) 
  (friend_unicycles : ℕ) (friend_tricycles : ℕ) : ℕ :=
  let ignatius_total_tires := ignatius_bicycles * tires_per_bicycle
  let friend_total_tires := 3 * ignatius_total_tires
  let friend_other_tires := friend_unicycles * 1 + friend_tricycles * 3
  let friend_bicycle_tires := friend_total_tires - friend_other_tires
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_ten_bicycles :
  friend_bicycles_count 4 2 1 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_friend_bicycles_count_friend_owns_ten_bicycles_l3813_381372


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l3813_381306

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l3813_381306


namespace NUMINAMATH_CALUDE_systematic_sampling_first_number_l3813_381303

/-- Systematic sampling function -/
def systematicSample (firstSelected : ℕ) (groupSize : ℕ) (groupNumber : ℕ) : ℕ :=
  firstSelected + groupSize * (groupNumber - 1)

theorem systematic_sampling_first_number 
  (totalStudents : ℕ) 
  (sampleSize : ℕ) 
  (selectedNumber : ℕ) 
  (selectedGroup : ℕ) 
  (h1 : totalStudents = 800) 
  (h2 : sampleSize = 50) 
  (h3 : selectedNumber = 503) 
  (h4 : selectedGroup = 32) :
  ∃ (firstSelected : ℕ), 
    firstSelected = 7 ∧ 
    systematicSample firstSelected (totalStudents / sampleSize) selectedGroup = selectedNumber :=
by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_number_l3813_381303


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3813_381315

theorem simplify_trig_expression :
  (Real.sqrt (1 - 2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180))) /
  (Real.cos (40 * π / 180) - Real.sqrt (1 - Real.sin (50 * π / 180) ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3813_381315


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3813_381347

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l3813_381347


namespace NUMINAMATH_CALUDE_find_B_l3813_381396

theorem find_B : ∃ B : ℕ, 
  (632 - 591 = 41) ∧ 
  (∃ (AB1 : ℕ), AB1 = 500 + 90 + B ∧ AB1 < 1000) → 
  B = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_B_l3813_381396


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l3813_381313

/-- Represents a fruit drink composed of grapefruit, lemon, and orange juice -/
structure FruitDrink where
  total : ℝ
  grapefruit : ℝ
  lemon : ℝ
  orange : ℝ

/-- Theorem stating the total volume of the fruit drink -/
theorem fruit_drink_volume (drink : FruitDrink)
  (h1 : drink.grapefruit = 0.25 * drink.total)
  (h2 : drink.lemon = 0.35 * drink.total)
  (h3 : drink.orange = 20)
  (h4 : drink.total = drink.grapefruit + drink.lemon + drink.orange) :
  drink.total = 50 := by
  sorry


end NUMINAMATH_CALUDE_fruit_drink_volume_l3813_381313


namespace NUMINAMATH_CALUDE_selection_theorem_l3813_381350

def male_teachers : ℕ := 5
def female_teachers : ℕ := 3
def total_selection : ℕ := 3

def select_with_both_genders (m f s : ℕ) : ℕ :=
  Nat.choose m 2 * Nat.choose f 1 + Nat.choose m 1 * Nat.choose f 2

theorem selection_theorem :
  select_with_both_genders male_teachers female_teachers total_selection = 45 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l3813_381350


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3813_381321

/-- Parabola defined by y² = 2x -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

/-- Line defined by y = -1/2x + b -/
def line (x y b : ℝ) : Prop := y = -1/2*x + b

/-- Point on both parabola and line -/
def intersection_point (x y b : ℝ) : Prop :=
  parabola x y ∧ line x y b

/-- Circle with diameter AB is tangent to x-axis -/
def circle_tangent_to_x_axis (xA yA xB yB : ℝ) : Prop :=
  (yA + yB) / 2 = (xB - xA) / 4

theorem parabola_line_intersection (b : ℝ) :
  (∃ xA yA xB yB : ℝ,
    intersection_point xA yA b ∧
    intersection_point xB yB b ∧
    xA ≠ xB ∧
    circle_tangent_to_x_axis xA yA xB yB) →
  b = -4/5 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3813_381321


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3813_381307

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi

-- Define the conditions of the problem
def problem_conditions (t : Triangle) : Prop :=
  5 * Real.sin t.A + 3 * Real.cos t.B = 7 ∧
  3 * Real.sin t.B + 5 * Real.cos t.A = 3

-- Theorem statement
theorem triangle_angle_C (t : Triangle) :
  problem_conditions t → Real.sin t.C = 4/5 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3813_381307


namespace NUMINAMATH_CALUDE_jar_flipping_problem_l3813_381339

theorem jar_flipping_problem (total_jars : Nat) (max_flip_per_move : Nat) (n_upper_bound : Nat) : 
  total_jars = 343 →
  max_flip_per_move = 27 →
  n_upper_bound = 2021 →
  (∃ (n : Nat), n ≥ (total_jars + max_flip_per_move - 1) / max_flip_per_move ∧ 
                n ≤ n_upper_bound ∧
                n % 2 = 1) →
  (Finset.filter (fun x => x % 2 = 1) (Finset.range (n_upper_bound + 1))).card = 1005 := by
sorry

end NUMINAMATH_CALUDE_jar_flipping_problem_l3813_381339


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3813_381360

theorem complex_number_quadrant : ∃ (a b : ℝ), (a > 0 ∧ b < 0) ∧ (Complex.mk a b = 5 / (Complex.mk 2 1)) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3813_381360


namespace NUMINAMATH_CALUDE_minimum_cost_for_all_entries_l3813_381336

/-- The cost of a single entry in yuan -/
def entry_cost : ℕ := 2

/-- The number of ways to choose 3 consecutive numbers from 01 to 17 -/
def ways_first_segment : ℕ := 15

/-- The number of ways to choose 2 consecutive numbers from 19 to 29 -/
def ways_second_segment : ℕ := 10

/-- The number of ways to choose 1 number from 30 to 36 -/
def ways_third_segment : ℕ := 7

/-- The total number of possible entries -/
def total_entries : ℕ := ways_first_segment * ways_second_segment * ways_third_segment

/-- The theorem stating the minimum amount of money needed -/
theorem minimum_cost_for_all_entries : 
  entry_cost * total_entries = 2100 := by sorry

end NUMINAMATH_CALUDE_minimum_cost_for_all_entries_l3813_381336


namespace NUMINAMATH_CALUDE_article_cost_price_l3813_381304

theorem article_cost_price (loss_percentage : Real) (gain_percentage : Real) (price_increase : Real) 
  (cost_price : Real) :
  loss_percentage = 0.15 →
  gain_percentage = 0.125 →
  price_increase = 72.50 →
  (1 - loss_percentage) * cost_price + price_increase = (1 + gain_percentage) * cost_price →
  cost_price = 263.64 := by
sorry

end NUMINAMATH_CALUDE_article_cost_price_l3813_381304


namespace NUMINAMATH_CALUDE_distance_not_equal_sum_l3813_381386

theorem distance_not_equal_sum : ∀ (a b : ℤ), 
  a = -2 ∧ b = 10 → |b - a| ≠ -2 + 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_not_equal_sum_l3813_381386


namespace NUMINAMATH_CALUDE_production_increase_l3813_381328

theorem production_increase (original_hours original_output : ℝ) 
  (h_positive_hours : original_hours > 0)
  (h_positive_output : original_output > 0) :
  let new_hours := 0.9 * original_hours
  let new_rate := 2 * (original_output / original_hours)
  let new_output := new_hours * new_rate
  (new_output - original_output) / original_output = 0.8 := by
sorry

end NUMINAMATH_CALUDE_production_increase_l3813_381328


namespace NUMINAMATH_CALUDE_factorable_quadratic_b_eq_42_l3813_381379

/-- A quadratic expression that can be factored into two linear binomials with integer coefficients -/
structure FactorableQuadratic where
  b : ℤ
  factored : ∃ (d e f g : ℤ), 28 * x^2 + b * x + 14 = (d * x + e) * (f * x + g)

/-- Theorem stating that for a FactorableQuadratic, b must equal 42 -/
theorem factorable_quadratic_b_eq_42 (q : FactorableQuadratic) : q.b = 42 := by
  sorry

end NUMINAMATH_CALUDE_factorable_quadratic_b_eq_42_l3813_381379


namespace NUMINAMATH_CALUDE_f_decreasing_interval_f_min_value_f_max_value_l3813_381369

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Theorem for monotonically decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x > f y :=
sorry

-- Theorem for minimum value on [-3, 3]
theorem f_min_value :
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f y ∧ f x = -49 :=
sorry

-- Theorem for maximum value on [-3, 3]
theorem f_max_value :
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f x ≥ f y ∧ f x = 59 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_f_min_value_f_max_value_l3813_381369


namespace NUMINAMATH_CALUDE_condition_implication_l3813_381378

theorem condition_implication (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
sorry

end NUMINAMATH_CALUDE_condition_implication_l3813_381378


namespace NUMINAMATH_CALUDE_hunter_frog_count_l3813_381341

/-- The total number of frogs Hunter saw in the pond -/
def total_frogs (initial : ℕ) (on_logs : ℕ) (babies : ℕ) : ℕ :=
  initial + on_logs + babies

/-- Theorem stating the total number of frogs Hunter saw -/
theorem hunter_frog_count :
  total_frogs 5 3 24 = 32 := by
  sorry

end NUMINAMATH_CALUDE_hunter_frog_count_l3813_381341


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3813_381322

-- Define the quadratic equation
def quadratic_equation (x a : ℝ) : Prop := x^2 + 3*x - a = 0

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation x a ∧ quadratic_equation y a

-- Theorem statement
theorem quadratic_roots_condition (a : ℝ) :
  has_two_distinct_real_roots a ↔ a > -9/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3813_381322


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l3813_381391

def angle_set (k : ℤ) : ℝ := k * 360 - 1560

theorem same_terminal_side_angles :
  (∃ k₁ : ℤ, angle_set k₁ = 240) ∧
  (∃ k₂ : ℤ, angle_set k₂ = -120) ∧
  (∀ α : ℝ, (∃ k : ℤ, angle_set k = α) →
    (α > 0 → α ≥ 240) ∧
    (α < 0 → α ≤ -120)) :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_angles_l3813_381391


namespace NUMINAMATH_CALUDE_roots_sum_ln_abs_l3813_381355

theorem roots_sum_ln_abs (m : ℝ) (x₁ x₂ : ℝ) :
  (Real.log (|x₁ - 2|) = m) ∧ (Real.log (|x₂ - 2|) = m) →
  x₁ + x₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_ln_abs_l3813_381355


namespace NUMINAMATH_CALUDE_john_money_left_l3813_381399

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 4 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem: John has 50 - 10q dollars left after his purchases -/
theorem john_money_left (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end NUMINAMATH_CALUDE_john_money_left_l3813_381399


namespace NUMINAMATH_CALUDE_min_value_2x_l3813_381327

theorem min_value_2x (x y z : ℕ+) (h1 : 2 * x = 6 * z) (h2 : x + y + z = 26) : 2 * x = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_2x_l3813_381327


namespace NUMINAMATH_CALUDE_units_digit_of_57_to_57_l3813_381319

theorem units_digit_of_57_to_57 : (57^57) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_57_to_57_l3813_381319


namespace NUMINAMATH_CALUDE_pizza_theorem_l3813_381312

def pizza_problem (total_pepperoni : ℕ) (fallen_pepperoni : ℕ) : Prop :=
  let half_pizza_pepperoni := total_pepperoni / 2
  let quarter_pizza_pepperoni := half_pizza_pepperoni / 2
  quarter_pizza_pepperoni - fallen_pepperoni = 9

theorem pizza_theorem : pizza_problem 40 1 := by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l3813_381312


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3813_381387

theorem rectangle_area_change (L W : ℝ) (h1 : L > 0) (h2 : W > 0) :
  let new_length := 1.4 * L
  let new_width := 0.5 * W
  let original_area := L * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = -0.3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3813_381387


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l3813_381329

theorem chess_game_draw_probability 
  (p_jian_win : ℝ) 
  (p_gu_not_win : ℝ) 
  (h1 : p_jian_win = 0.4) 
  (h2 : p_gu_not_win = 0.6) : 
  p_gu_not_win - p_jian_win = 0.2 := by
sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l3813_381329


namespace NUMINAMATH_CALUDE_largest_n_proof_l3813_381362

/-- Binary operation @ defined as n @ n = n - (n * 5) -/
def binary_op (n : ℤ) : ℤ := n - (n * 5)

/-- The largest positive integer n such that n @ n < 21 -/
def largest_n : ℕ := 1

theorem largest_n_proof :
  (∀ (m : ℕ), m > largest_n → binary_op m ≥ 21) ∧
  binary_op largest_n < 21 :=
sorry

end NUMINAMATH_CALUDE_largest_n_proof_l3813_381362


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3813_381397

/-- Prove that the given equation represents a circle with center (2, -1) passing through (-1, 3) -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -1)
  let point : ℝ × ℝ := (-1, 3)
  ((x - center.1)^2 + (y - center.2)^2 = 
   (point.1 - center.1)^2 + (point.2 - center.2)^2) ↔
  ((x - 2)^2 + (y + 1)^2 = 25) :=
by sorry


end NUMINAMATH_CALUDE_circle_equation_proof_l3813_381397


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l3813_381375

/-- If the graph of the quadratic function y = mx^2 + x + m(m-3) passes through the origin, then m = 3 -/
theorem quadratic_through_origin (m : ℝ) : 
  (∀ x y : ℝ, y = m * x^2 + x + m * (m - 3)) → 
  (0 = m * 0^2 + 0 + m * (m - 3)) → 
  m = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l3813_381375


namespace NUMINAMATH_CALUDE_sin_pi_half_equals_one_l3813_381308

theorem sin_pi_half_equals_one : 
  let f : ℝ → ℝ := fun x ↦ Real.sin (x / 2 + π / 4)
  f (π / 2) = 1 := by
sorry

end NUMINAMATH_CALUDE_sin_pi_half_equals_one_l3813_381308


namespace NUMINAMATH_CALUDE_parallel_line_construction_l3813_381377

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Predicate to check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Predicate to check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- Theorem: Given a line and a point not on the line, 
    it's possible to construct a parallel line through the point 
    using only compass and straightedge -/
theorem parallel_line_construction 
  (l : Line) (A : Point) (h : ¬A.onLine l) :
  ∃ (l' : Line), A.onLine l' ∧ l.parallel l' :=
sorry

end NUMINAMATH_CALUDE_parallel_line_construction_l3813_381377


namespace NUMINAMATH_CALUDE_solution_equivalence_l3813_381342

-- Define the set of real numbers greater than 1
def greaterThanOne : Set ℝ := {x | x > 1}

-- Define the solution set of ax - b > 0
def solutionSet (a b : ℝ) : Set ℝ := {x | a * x - b > 0}

-- Define the set (-∞,-1)∪(2,+∞)
def targetSet : Set ℝ := {x | x < -1 ∨ x > 2}

-- Theorem statement
theorem solution_equivalence (a b : ℝ) :
  solutionSet a b = greaterThanOne →
  {x : ℝ | (a * x + b) / (x - 2) > 0} = targetSet := by
  sorry

end NUMINAMATH_CALUDE_solution_equivalence_l3813_381342


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l3813_381320

theorem solve_cubic_equation (t p s : ℝ) : 
  t = 3 * s^3 + 2 * p → t = 29 → p = 3 → s = (23/3)^(1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l3813_381320


namespace NUMINAMATH_CALUDE_patricia_candy_count_l3813_381376

theorem patricia_candy_count (initial_candy : ℕ) (taken_candy : ℕ) : 
  initial_candy = 76 → taken_candy = 5 → initial_candy - taken_candy = 71 := by
  sorry

end NUMINAMATH_CALUDE_patricia_candy_count_l3813_381376


namespace NUMINAMATH_CALUDE_triangle_side_length_l3813_381300

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 2 →
  b = Real.sqrt 6 →
  A = π / 6 →
  c^2 - 2 * Real.sqrt 6 * c * Real.cos A + 2 = 6 →
  c = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3813_381300


namespace NUMINAMATH_CALUDE_paddington_washington_goats_difference_l3813_381316

theorem paddington_washington_goats_difference 
  (washington_goats : ℕ) 
  (total_goats : ℕ) 
  (h1 : washington_goats = 140)
  (h2 : total_goats = 320)
  (h3 : washington_goats < total_goats - washington_goats) : 
  total_goats - washington_goats - washington_goats = 40 := by
  sorry

end NUMINAMATH_CALUDE_paddington_washington_goats_difference_l3813_381316


namespace NUMINAMATH_CALUDE_infinite_binomial_congruence_pairs_l3813_381338

theorem infinite_binomial_congruence_pairs :
  ∀ p : ℕ, Prime p → p ≠ 2 →
  ∃ a b : ℕ,
    a > b ∧
    a + b = 2 * p ∧
    (Nat.choose (2 * p) a) % (2 * p) = (Nat.choose (2 * p) b) % (2 * p) ∧
    (Nat.choose (2 * p) a) % (2 * p) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_infinite_binomial_congruence_pairs_l3813_381338


namespace NUMINAMATH_CALUDE_michael_small_birdhouses_sold_l3813_381305

/-- Proves that Michael sold 3 small birdhouses given the prices and sales information. -/
theorem michael_small_birdhouses_sold :
  let large_price : ℕ := 22
  let medium_price : ℕ := 16
  let small_price : ℕ := 7
  let large_sold : ℕ := 2
  let medium_sold : ℕ := 2
  let total_amount : ℕ := 97
  let small_sold : ℕ := (total_amount - (large_price * large_sold + medium_price * medium_sold)) / small_price
  small_sold = 3 := by sorry

end NUMINAMATH_CALUDE_michael_small_birdhouses_sold_l3813_381305


namespace NUMINAMATH_CALUDE_kaleb_initial_savings_l3813_381317

/-- The amount of money Kaleb had initially saved up. -/
def initial_savings : ℕ := sorry

/-- The cost of each toy. -/
def toy_cost : ℕ := 6

/-- The number of toys Kaleb can buy after receiving his allowance. -/
def num_toys : ℕ := 6

/-- The amount of allowance Kaleb received. -/
def allowance : ℕ := 15

/-- Theorem stating that Kaleb's initial savings were $21. -/
theorem kaleb_initial_savings :
  initial_savings = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_kaleb_initial_savings_l3813_381317


namespace NUMINAMATH_CALUDE_square_lake_area_l3813_381309

/-- Represents a square lake with a given boat speed and crossing times -/
structure SquareLake where
  boat_speed : ℝ  -- Speed of the boat in miles per hour
  length_time : ℝ  -- Time to cross the length in hours
  width_time : ℝ  -- Time to cross the width in hours

/-- Calculates the area of a square lake based on boat speed and crossing times -/
def lake_area (lake : SquareLake) : ℝ :=
  (lake.boat_speed * lake.length_time) * (lake.boat_speed * lake.width_time)

/-- Theorem: The area of the specified square lake is 100 square miles -/
theorem square_lake_area :
  let lake := SquareLake.mk 10 2 (1/2)
  lake_area lake = 100 := by
  sorry


end NUMINAMATH_CALUDE_square_lake_area_l3813_381309


namespace NUMINAMATH_CALUDE_max_binder_price_l3813_381368

/-- Proves that the maximum whole-dollar price of a binder is $7 given the conditions --/
theorem max_binder_price (total_money : ℕ) (num_binders : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : 
  total_money = 160 →
  num_binders = 18 →
  entrance_fee = 5 →
  tax_rate = 8 / 100 →
  ∃ (price : ℕ), price = 7 ∧ 
    price = ⌊(total_money - entrance_fee) / ((1 + tax_rate) * num_binders)⌋ ∧
    ∀ (p : ℕ), p > price → 
      p * num_binders * (1 + tax_rate) + entrance_fee > total_money :=
by
  sorry

#check max_binder_price

end NUMINAMATH_CALUDE_max_binder_price_l3813_381368


namespace NUMINAMATH_CALUDE_kho_kho_problem_l3813_381340

/-- Represents the number of students who left to play kho-kho -/
def students_who_left (initial_boys initial_girls remaining_girls : ℕ) : ℕ :=
  initial_girls - remaining_girls

/-- Proves that 8 girls left to play kho-kho given the problem conditions -/
theorem kho_kho_problem (initial_boys initial_girls remaining_girls : ℕ) :
  initial_boys = initial_girls →
  initial_boys + initial_girls = 32 →
  initial_boys = 2 * remaining_girls →
  students_who_left initial_boys initial_girls remaining_girls = 8 :=
by
  sorry

#check kho_kho_problem

end NUMINAMATH_CALUDE_kho_kho_problem_l3813_381340


namespace NUMINAMATH_CALUDE_adam_purchases_cost_l3813_381363

/-- Represents the cost of Adam's purchases -/
def total_cost (nuts_quantity : ℝ) (dried_fruits_quantity : ℝ) (nuts_price : ℝ) (dried_fruits_price : ℝ) : ℝ :=
  nuts_quantity * nuts_price + dried_fruits_quantity * dried_fruits_price

/-- Theorem stating that Adam's purchases cost $56 -/
theorem adam_purchases_cost :
  total_cost 3 2.5 12 8 = 56 := by
  sorry

end NUMINAMATH_CALUDE_adam_purchases_cost_l3813_381363


namespace NUMINAMATH_CALUDE_last_digit_of_4139_power_467_l3813_381349

theorem last_digit_of_4139_power_467 : (4139^467) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_4139_power_467_l3813_381349


namespace NUMINAMATH_CALUDE_prime_greater_than_five_form_l3813_381388

theorem prime_greater_than_five_form (p : ℕ) (h_prime : Nat.Prime p) (h_gt_five : p > 5) :
  ∃ k : ℕ, p = 6 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_prime_greater_than_five_form_l3813_381388


namespace NUMINAMATH_CALUDE_stating_time_is_seven_thirty_two_l3813_381382

/-- Represents the number of minutes in an hour -/
def minutesInHour : ℕ := 60

/-- Represents the time in minutes after 7:00 a.m. -/
def minutesAfterSeven (x : ℚ) : ℚ := 8 * x

/-- Represents the time in minutes before 8:00 a.m. -/
def minutesBeforeEight (x : ℚ) : ℚ := 7 * x

/-- 
Theorem stating that if a time is 8x minutes after 7:00 a.m. and 7x minutes before 8:00 a.m.,
then the time is 32 minutes after 7:00 a.m. (which is 7:32 a.m.)
-/
theorem time_is_seven_thirty_two (x : ℚ) :
  minutesAfterSeven x + minutesBeforeEight x = minutesInHour →
  minutesAfterSeven x = 32 :=
by sorry

end NUMINAMATH_CALUDE_stating_time_is_seven_thirty_two_l3813_381382


namespace NUMINAMATH_CALUDE_integral_calculation_l3813_381345

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 2) * x + a^2

-- Define the property of f being an even function
def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- State the theorem
theorem integral_calculation (a : ℝ) (h : is_even_function (f a)) :
  ∫ x in -a..a, (x^2 + x + Real.sqrt (4 - x^2)) = 16/3 + 2 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_integral_calculation_l3813_381345


namespace NUMINAMATH_CALUDE_smallest_valid_N_sum_of_digits_l3813_381370

def P (N : ℕ) : ℚ := (N + 1 - Int.ceil (N / 3 : ℚ)) / (N + 1 : ℚ)

def is_valid (N : ℕ) : Prop :=
  N > 0 ∧ N % 5 = 0 ∧ N % 6 = 0 ∧ P N < 2/3

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_valid_N_sum_of_digits :
  ∃ N, is_valid N ∧
    (∀ M, is_valid M → N ≤ M) ∧
    sum_of_digits N = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_N_sum_of_digits_l3813_381370


namespace NUMINAMATH_CALUDE_cosine_period_l3813_381389

/-- The period of the cosine function with a modified argument -/
theorem cosine_period (f : ℝ → ℝ) (h : f = λ x => Real.cos ((3 * x) / 4 + π / 6)) :
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ p = 8 * π / 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_period_l3813_381389


namespace NUMINAMATH_CALUDE_candles_remaining_l3813_381330

/-- Calculates the number of candles remaining after three people use them according to specific rules. -/
theorem candles_remaining (total : ℕ) (alyssa_fraction : ℚ) (chelsea_fraction : ℚ) (bianca_fraction : ℚ) : 
  total = 60 ∧ 
  alyssa_fraction = 1/2 ∧ 
  chelsea_fraction = 7/10 ∧ 
  bianca_fraction = 4/5 →
  ↑total - (alyssa_fraction * ↑total + 
    chelsea_fraction * (↑total - alyssa_fraction * ↑total) + 
    ⌊bianca_fraction * (↑total - alyssa_fraction * ↑total - chelsea_fraction * (↑total - alyssa_fraction * ↑total))⌋) = 2 := by
  sorry

#check candles_remaining

end NUMINAMATH_CALUDE_candles_remaining_l3813_381330


namespace NUMINAMATH_CALUDE_line_moved_down_three_units_l3813_381332

/-- Represents a linear function of the form y = mx + b -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Moves a linear function vertically by a given amount -/
def moveVertically (f : LinearFunction) (amount : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.intercept - amount }

theorem line_moved_down_three_units :
  let original := LinearFunction.mk 2 5
  let moved := moveVertically original 3
  moved = LinearFunction.mk 2 2 := by
  sorry

end NUMINAMATH_CALUDE_line_moved_down_three_units_l3813_381332


namespace NUMINAMATH_CALUDE_reflection_in_first_quadrant_l3813_381337

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the reflection across y-axis
def reflect_y (p : Point) : Point :=
  (-p.1, p.2)

-- Define the first quadrant
def in_first_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 > 0

-- Theorem statement
theorem reflection_in_first_quadrant :
  let P : Point := (-3, 1)
  in_first_quadrant (reflect_y P) := by sorry

end NUMINAMATH_CALUDE_reflection_in_first_quadrant_l3813_381337


namespace NUMINAMATH_CALUDE_initial_shells_l3813_381381

theorem initial_shells (initial_amount added_amount total_amount : ℕ) 
  (h1 : added_amount = 12)
  (h2 : total_amount = 17)
  (h3 : initial_amount + added_amount = total_amount) :
  initial_amount = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_shells_l3813_381381


namespace NUMINAMATH_CALUDE_brownie_solution_l3813_381383

/-- Represents the brownie distribution problem --/
def brownie_problem (total_brownies : ℕ) (total_cost : ℚ) (faculty_fraction : ℚ) 
  (faculty_price_increase : ℚ) (carl_fraction : ℚ) (simon_brownies : ℕ) 
  (friends_fraction : ℚ) (num_friends : ℕ) : Prop :=
  let original_price := total_cost / total_brownies
  let faculty_brownies := (faculty_fraction * total_brownies).floor
  let faculty_price := original_price + faculty_price_increase
  let remaining_after_faculty := total_brownies - faculty_brownies
  let carl_brownies := (carl_fraction * remaining_after_faculty).floor
  let remaining_after_carl := remaining_after_faculty - carl_brownies - simon_brownies
  let friends_brownies := (friends_fraction * remaining_after_carl).floor
  let annie_brownies := remaining_after_carl - friends_brownies
  let annie_cost := annie_brownies * original_price
  let faculty_cost := faculty_brownies * faculty_price
  annie_cost = 5.1 ∧ faculty_cost = 45

/-- Theorem stating the solution to the brownie problem --/
theorem brownie_solution : 
  brownie_problem 150 45 (3/5) 0.2 (1/4) 3 (2/3) 5 := by
  sorry

end NUMINAMATH_CALUDE_brownie_solution_l3813_381383


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_p_and_q_l3813_381390

theorem not_p_sufficient_not_necessary_for_not_p_and_q
  (p q : Prop) :
  (∀ (h : ¬p), ¬(p ∧ q)) ∧
  ¬(∀ (h : ¬(p ∧ q)), ¬p) :=
by sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_p_and_q_l3813_381390


namespace NUMINAMATH_CALUDE_max_divisor_of_f_l3813_381348

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_of_f :
  ∃ (m : ℕ), (∀ (n : ℕ), m ∣ f n) ∧ 
  (∀ (k : ℕ), (∀ (n : ℕ), k ∣ f n) → k ≤ 36) ∧
  (∀ (n : ℕ), 36 ∣ f n) :=
sorry

end NUMINAMATH_CALUDE_max_divisor_of_f_l3813_381348


namespace NUMINAMATH_CALUDE_vector_length_on_number_line_l3813_381394

theorem vector_length_on_number_line : 
  ∀ (A B : ℝ), A = -1 → B = 2 → abs (B - A) = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_vector_length_on_number_line_l3813_381394


namespace NUMINAMATH_CALUDE_work_day_meetings_percentage_l3813_381356

/-- Proves that given a 10-hour work day and two meetings, where the first meeting is 60 minutes long
    and the second is three times as long, the percentage of the work day spent in meetings is 40%. -/
theorem work_day_meetings_percentage (work_day_hours : ℕ) (first_meeting_minutes : ℕ) :
  work_day_hours = 10 →
  first_meeting_minutes = 60 →
  let work_day_minutes : ℕ := work_day_hours * 60
  let second_meeting_minutes : ℕ := 3 * first_meeting_minutes
  let total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes
  let meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100
  meeting_percentage = 40 := by
  sorry


end NUMINAMATH_CALUDE_work_day_meetings_percentage_l3813_381356


namespace NUMINAMATH_CALUDE_symmetry_center_of_sine_function_l3813_381311

/-- Given a function f(x) = 1/2 * sin(ω * x + π/6) where ω > 0,
    and its graph is tangent to the line y = m with adjacent tangent points
    separated by a distance of π, prove that the symmetry center x₀
    in the interval [0, π/2] is equal to 5π/12. -/
theorem symmetry_center_of_sine_function (ω : ℝ) (m : ℝ) (x₀ : ℝ) :
  ω > 0 →
  (∀ x, ∃ k : ℤ, (x + k * π) = (π / 2 - π / (6 * ω)) + n * π / ω → 
    1/2 * Real.sin (ω * x + π/6) = m) →
  x₀ ∈ Set.Icc 0 (π/2) →
  (∀ x, 1/2 * Real.sin (ω * x + π/6) = 
        1/2 * Real.sin (ω * (2 * x₀ - x) + π/6)) →
  x₀ = 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_center_of_sine_function_l3813_381311


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l3813_381335

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def isPointOnPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_not_on_transformed_plane :
  let A : Point3D := { x := -3, y := -2, z := 4 }
  let a : Plane3D := { a := 2, b := -3, c := 1, d := -5 }
  let k : ℝ := -4/5
  let a' : Plane3D := transformPlane a k
  ¬ isPointOnPlane A a' := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l3813_381335


namespace NUMINAMATH_CALUDE_sail_pressure_velocity_l3813_381333

/-- The pressure-area-velocity relationship for a boat sail -/
theorem sail_pressure_velocity 
  (k : ℝ) 
  (A₁ A₂ V₁ V₂ P₁ P₂ : ℝ) 
  (h1 : P₁ = k * A₁ * V₁^2) 
  (h2 : P₂ = k * A₂ * V₂^2) 
  (h3 : A₁ = 2) 
  (h4 : V₁ = 20) 
  (h5 : P₁ = 5) 
  (h6 : A₂ = 4) 
  (h7 : P₂ = 20) : 
  V₂ = 20 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sail_pressure_velocity_l3813_381333


namespace NUMINAMATH_CALUDE_least_sum_p_q_l3813_381393

theorem least_sum_p_q (p q : ℕ) (hp : p > 1) (hq : q > 1) 
  (h_eq : 17 * (p + 1) = 25 * (q + 1)) : 
  (∀ p' q' : ℕ, p' > 1 → q' > 1 → 17 * (p' + 1) = 25 * (q' + 1) → p' + q' ≥ p + q) → 
  p + q = 168 := by
sorry

end NUMINAMATH_CALUDE_least_sum_p_q_l3813_381393


namespace NUMINAMATH_CALUDE_migration_distance_l3813_381310

/-- The distance between lake Jim and lake Disney -/
def distance_jim_disney : ℝ := 50

/-- The number of migrating birds -/
def num_birds : ℕ := 20

/-- The distance between lake Disney and lake London -/
def distance_disney_london : ℝ := 60

/-- The total distance traveled by all birds in two seasons -/
def total_distance : ℝ := 2200

theorem migration_distance :
  distance_jim_disney * num_birds + distance_disney_london * num_birds = total_distance :=
by sorry

end NUMINAMATH_CALUDE_migration_distance_l3813_381310


namespace NUMINAMATH_CALUDE_remainder_problem_l3813_381354

theorem remainder_problem (d : ℕ) (a b : ℕ) (h1 : d > 0) (h2 : d ≤ a ∧ d ≤ b) 
  (h3 : ∀ k > d, k ∣ a ∨ k ∣ b) (h4 : b % d = 5) : a % d = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3813_381354


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3813_381365

/-- An isosceles triangle with altitude 8 and perimeter 32 has area 48 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 → -- b and s are positive real numbers
  2 * s + 2 * b = 32 → -- perimeter condition
  b ^ 2 + 8 ^ 2 = s ^ 2 → -- Pythagorean theorem for half the triangle
  (2 * b) * 8 / 2 = 48 := by 
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_area_l3813_381365


namespace NUMINAMATH_CALUDE_sales_solution_l3813_381344

def sales_problem (m1 m2 m3 m5 m6 average : ℕ) : Prop :=
  let total := 6 * average
  let known_sum := m1 + m2 + m3 + m5 + m6
  let m4 := total - known_sum
  m4 = 8200

theorem sales_solution :
  sales_problem 5400 9000 6300 4500 1200 5600 := by
  sorry

end NUMINAMATH_CALUDE_sales_solution_l3813_381344


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l3813_381373

theorem right_triangle_third_side_product (a b c : ℝ) : 
  (a = 6 ∧ b = 8 ∧ a^2 + b^2 = c^2) ∨ (a = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) → 
  c * b = 20 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l3813_381373


namespace NUMINAMATH_CALUDE_special_triangle_secant_sum_range_l3813_381314

-- Define a structure for a triangle with the given condition
structure SpecialTriangle where
  A : Real
  B : Real
  C : Real
  angle_sum : A + B + C = π
  special_condition : A + C = 2 * B

-- Define the secant function
noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

-- State the theorem
theorem special_triangle_secant_sum_range (t : SpecialTriangle) :
  ∃ (f : Real → Real), 
    (∀ x, f x = sec t.A + sec t.C) ∧ 
    (Set.range f = {y | y < -1 ∨ y ≥ 4}) := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_secant_sum_range_l3813_381314


namespace NUMINAMATH_CALUDE_arithmetic_sequence_separable_special_sequence_a_value_complex_sequence_separable_values_l3813_381351

/-- A sequence is m-th degree separable if there exists an n such that a_{m+n} = a_m + a_n -/
def IsNthDegreeSeparable (a : ℕ → ℝ) (m : ℕ) : Prop :=
  ∃ n : ℕ, a (m + n) = a m + a n

/-- An arithmetic sequence with first term 2 and common difference 2 -/
def ArithmeticSequence (n : ℕ) : ℝ := 2 * n

/-- A sequence with sum of first n terms S_n = 2^n - a where a > 0 -/
def SpecialSequence (a : ℝ) (n : ℕ) : ℝ := 2^n - a

/-- A sequence defined by a_n = 2^n + n^2 + 12 -/
def ComplexSequence (n : ℕ) : ℝ := 2^n + n^2 + 12

theorem arithmetic_sequence_separable :
  IsNthDegreeSeparable ArithmeticSequence 3 :=
sorry

theorem special_sequence_a_value (a : ℝ) (h : a > 0) :
  IsNthDegreeSeparable (SpecialSequence a) 1 → a = 1 :=
sorry

theorem complex_sequence_separable_values :
  (∃ m : ℕ, IsNthDegreeSeparable ComplexSequence m) ∧
  (∀ m : ℕ, IsNthDegreeSeparable ComplexSequence m → (m = 1 ∨ m = 3)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_separable_special_sequence_a_value_complex_sequence_separable_values_l3813_381351


namespace NUMINAMATH_CALUDE_cubic_root_sum_product_l3813_381366

theorem cubic_root_sum_product (a b : ℝ) : 
  (a^3 - 4*a^2 - a + 4 = 0) → 
  (b^3 - 4*b^2 - b + 4 = 0) → 
  (a ≠ b) →
  (a + b + a*b = -1) := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_product_l3813_381366


namespace NUMINAMATH_CALUDE_slope_equals_one_implies_m_equals_one_l3813_381323

/-- Given two points M(-2, m) and N(m, 4), if the slope of the line passing through M and N
    is equal to 1, then m = 1. -/
theorem slope_equals_one_implies_m_equals_one (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_equals_one_implies_m_equals_one_l3813_381323


namespace NUMINAMATH_CALUDE_rebecca_work_hours_l3813_381302

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Rebecca worked 56 hours. -/
theorem rebecca_work_hours :
  ∀ x : ℕ,
  let thomas_hours := x
  let toby_hours := 2 * x - 10
  let rebecca_hours := toby_hours - 8
  (thomas_hours + toby_hours + rebecca_hours = 157) →
  rebecca_hours = 56 :=
by
  sorry

end NUMINAMATH_CALUDE_rebecca_work_hours_l3813_381302


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l3813_381371

def f (x : ℝ) : ℝ := x * abs x

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l3813_381371


namespace NUMINAMATH_CALUDE_xyz_sum_root_l3813_381331

theorem xyz_sum_root (x y z : ℝ) 
  (eq1 : y + z = 14)
  (eq2 : z + x = 15)
  (eq3 : x + y = 16) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 134.24375 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_root_l3813_381331


namespace NUMINAMATH_CALUDE_diamond_calculation_l3813_381318

def diamond (A B : ℚ) : ℚ := (A - B) / 5

theorem diamond_calculation : (diamond (diamond 7 15) 2) = -18/25 := by
  sorry

end NUMINAMATH_CALUDE_diamond_calculation_l3813_381318


namespace NUMINAMATH_CALUDE_smallest_three_square_representations_l3813_381343

/-- A function that represents the number of ways a positive integer can be expressed as the sum of three squares -/
def numThreeSquareRepresentations (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is expressible as the sum of three squares in three different ways -/
def hasThreeRepresentations (n : ℕ) : Prop :=
  numThreeSquareRepresentations n = 3

/-- Theorem stating that 30 is the smallest positive integer with three different representations as the sum of three squares -/
theorem smallest_three_square_representations :
  (∀ m : ℕ, m > 0 → m < 30 → ¬(hasThreeRepresentations m)) ∧
  hasThreeRepresentations 30 := by sorry

end NUMINAMATH_CALUDE_smallest_three_square_representations_l3813_381343


namespace NUMINAMATH_CALUDE_cube_space_division_theorem_l3813_381301

/-- The number of parts that space is divided into by the planes containing the faces of a cube -/
def cube_space_division : ℕ := 33

/-- The number of faces a cube has -/
def cube_faces : ℕ := 6

/-- Theorem stating that the planes containing the faces of a cube divide space into 33 parts -/
theorem cube_space_division_theorem :
  cube_space_division = 33 ∧ cube_faces = 6 :=
sorry

end NUMINAMATH_CALUDE_cube_space_division_theorem_l3813_381301


namespace NUMINAMATH_CALUDE_mango_price_reduction_mango_price_reduction_result_l3813_381326

/-- Calculates the percentage reduction in mango prices --/
theorem mango_price_reduction (original_cost : ℝ) (original_quantity : ℕ) 
  (reduced_cost : ℝ) (original_purchase : ℕ) (additional_mangoes : ℕ) : ℝ :=
  let original_price_per_mango := original_cost / original_quantity
  let original_purchase_quantity := reduced_cost / original_price_per_mango
  let new_purchase_quantity := original_purchase_quantity + additional_mangoes
  let new_price_per_mango := reduced_cost / new_purchase_quantity
  let price_reduction_percentage := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100
  price_reduction_percentage

/-- The percentage reduction in mango prices is approximately 9.91% --/
theorem mango_price_reduction_result : 
  abs (mango_price_reduction 450 135 360 108 12 - 9.91) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_mango_price_reduction_mango_price_reduction_result_l3813_381326


namespace NUMINAMATH_CALUDE_perimeter_difference_is_zero_l3813_381398

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- The perimeter of Shape 1 -/
def shape1_perimeter : ℕ :=
  rectangle_perimeter 4 3

/-- The perimeter of Shape 2 -/
def shape2_perimeter : ℕ :=
  rectangle_perimeter 6 1

/-- The positive difference between the perimeters of Shape 1 and Shape 2 -/
def perimeter_difference : ℕ :=
  Int.natAbs (shape1_perimeter - shape2_perimeter)

theorem perimeter_difference_is_zero : perimeter_difference = 0 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_difference_is_zero_l3813_381398


namespace NUMINAMATH_CALUDE_becky_lunch_days_proof_l3813_381361

/-- The number of school days in an academic year -/
def school_days : ℕ := 180

/-- The fraction of time Aliyah packs her lunch -/
def aliyah_lunch_fraction : ℚ := 1/2

/-- The fraction of Aliyah's lunch-packing frequency that Becky packs her lunch -/
def becky_lunch_fraction : ℚ := 1/2

/-- The number of days Becky packs her lunch in a school year -/
def becky_lunch_days : ℕ := 45

theorem becky_lunch_days_proof :
  (school_days : ℚ) * aliyah_lunch_fraction * becky_lunch_fraction = becky_lunch_days := by
  sorry

end NUMINAMATH_CALUDE_becky_lunch_days_proof_l3813_381361


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3813_381374

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  let norm_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product := a.1 * b.1 + a.2 * b.2
  angle = Real.arccos (dot_product / (norm_a * norm_b)) →
  norm_a = 1 →
  norm_b = 1 / 2 →
  Real.sqrt (((a.1 - 2 * b.1) ^ 2) + ((a.2 - 2 * b.2) ^ 2)) = 1 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3813_381374


namespace NUMINAMATH_CALUDE_sin_equality_l3813_381346

theorem sin_equality (x : ℝ) (h : Real.sin (x + π/4) = 1/3) :
  Real.sin (4*x) - 2 * Real.cos (3*x) * Real.sin x = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_l3813_381346


namespace NUMINAMATH_CALUDE_leading_coefficient_of_f_l3813_381324

/-- Given a polynomial f satisfying f(x + 1) - f(x) = 6x + 4 for all x,
    prove that the leading coefficient of f is 3. -/
theorem leading_coefficient_of_f (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 6 * x + 4) →
  ∃ c, ∀ x, f x = 3 * x^2 + x + c :=
sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_f_l3813_381324


namespace NUMINAMATH_CALUDE_trajectory_equation_l3813_381353

/-- 
Given a point P in the plane, if its distance to the line y=-3 is equal to 
its distance to the point (0,3), then the equation of its trajectory is x^2 = 12y.
-/
theorem trajectory_equation (P : ℝ × ℝ) : 
  (∀ (x y : ℝ), P = (x, y) → |y + 3| = ((x - 0)^2 + (y - 3)^2).sqrt) →
  (∃ (x y : ℝ), P = (x, y) ∧ x^2 = 12*y) :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l3813_381353


namespace NUMINAMATH_CALUDE_largest_x_floor_ratio_l3813_381395

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(⌊x⌋) / x = 8 / 9) → x ≤ 63 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_ratio_l3813_381395


namespace NUMINAMATH_CALUDE_eugene_purchase_cost_l3813_381325

def tshirt_price : ℚ := 20
def pants_price : ℚ := 80
def shoes_price : ℚ := 150
def hat_price : ℚ := 25
def jacket_price : ℚ := 120

def tshirt_discount : ℚ := 0.1
def pants_discount : ℚ := 0.1
def shoes_discount : ℚ := 0.15
def hat_discount : ℚ := 0.05
def jacket_discount : ℚ := 0.2

def sales_tax : ℚ := 0.06

def tshirt_quantity : ℕ := 4
def pants_quantity : ℕ := 3
def shoes_quantity : ℕ := 2
def hat_quantity : ℕ := 3
def jacket_quantity : ℕ := 1

theorem eugene_purchase_cost :
  let discounted_tshirt := tshirt_price * (1 - tshirt_discount)
  let discounted_pants := pants_price * (1 - pants_discount)
  let discounted_shoes := shoes_price * (1 - shoes_discount)
  let discounted_hat := hat_price * (1 - hat_discount)
  let discounted_jacket := jacket_price * (1 - jacket_discount)
  
  let total_before_tax := 
    discounted_tshirt * tshirt_quantity +
    discounted_pants * pants_quantity +
    discounted_shoes * shoes_quantity +
    discounted_hat * hat_quantity +
    discounted_jacket * jacket_quantity
  
  let total_with_tax := total_before_tax * (1 + sales_tax)
  
  total_with_tax = 752.87 := by sorry

end NUMINAMATH_CALUDE_eugene_purchase_cost_l3813_381325


namespace NUMINAMATH_CALUDE_range_of_b_l3813_381367

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.sqrt (9 - p.1^2)}
def N (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + b}

-- State the theorem
theorem range_of_b (b : ℝ) : M ∩ N b = ∅ ↔ b > 3 * Real.sqrt 2 ∨ b < -3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l3813_381367


namespace NUMINAMATH_CALUDE_number_division_l3813_381357

theorem number_division (x : ℝ) : x - 17 = 55 → x / 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l3813_381357


namespace NUMINAMATH_CALUDE_jerry_stickers_l3813_381385

theorem jerry_stickers (fred_stickers : ℕ) (george_stickers : ℕ) (jerry_stickers : ℕ) : 
  fred_stickers = 18 →
  george_stickers = fred_stickers - 6 →
  jerry_stickers = 3 * george_stickers →
  jerry_stickers = 36 := by
sorry

end NUMINAMATH_CALUDE_jerry_stickers_l3813_381385
