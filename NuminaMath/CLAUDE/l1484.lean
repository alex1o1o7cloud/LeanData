import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_new_length_l1484_148443

/-- Given a rectangle with original length 18 cm and breadth 10 cm,
    if the breadth is changed to 7.2 cm while maintaining the same area,
    the new length will be 25 cm. -/
theorem rectangle_new_length (original_length original_breadth new_breadth new_length : ℝ) :
  original_length = 18 ∧
  original_breadth = 10 ∧
  new_breadth = 7.2 ∧
  original_length * original_breadth = new_length * new_breadth →
  new_length = 25 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_new_length_l1484_148443


namespace NUMINAMATH_CALUDE_inequality_order_l1484_148446

theorem inequality_order (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a + 3*b) / 4 < (a^2 * b)^(1/3) ∧ (a^2 * b)^(1/3) < (a + 3*b)^2 / (4*(a + b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_order_l1484_148446


namespace NUMINAMATH_CALUDE_fence_cost_l1484_148473

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 55) :
  4 * price_per_foot * Real.sqrt area = 3740 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_l1484_148473


namespace NUMINAMATH_CALUDE_smallest_square_side_length_l1484_148494

theorem smallest_square_side_length :
  ∀ (n : ℕ),
  (∃ (a b c d : ℕ),
    n * n = a + 4 * b + 9 * c ∧
    14 = a + b + c ∧
    a ≥ 10 ∧ b ≥ 3 ∧ c ≥ 1) →
  n ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_side_length_l1484_148494


namespace NUMINAMATH_CALUDE_workers_calculation_l1484_148492

/-- The initial number of workers on a job -/
def initial_workers : ℕ := 20

/-- The number of days to complete the job with the initial number of workers -/
def initial_days : ℕ := 30

/-- The number of days worked before some workers leave -/
def days_before_leaving : ℕ := 15

/-- The number of workers that leave the job -/
def workers_leaving : ℕ := 5

/-- The total number of days to complete the job after some workers leave -/
def total_days : ℕ := 35

theorem workers_calculation :
  (initial_workers * days_before_leaving = (initial_workers - workers_leaving) * (total_days - days_before_leaving)) ∧
  (initial_workers * initial_days = initial_workers * days_before_leaving + (initial_workers - workers_leaving) * (total_days - days_before_leaving)) :=
sorry

end NUMINAMATH_CALUDE_workers_calculation_l1484_148492


namespace NUMINAMATH_CALUDE_simplify_expression_l1484_148448

theorem simplify_expression (α : ℝ) (h : π < α ∧ α < (3*π)/2) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2*α))) = Real.sin (α/2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1484_148448


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1484_148437

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 2}
def B : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {y | -1 < y ∧ y ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1484_148437


namespace NUMINAMATH_CALUDE_clock_synchronization_l1484_148414

/-- The number of minutes in 12 hours -/
def minutes_in_12_hours : ℕ := 12 * 60

/-- The number of minutes Arthur's clock gains per day -/
def arthur_gain : ℕ := 15

/-- The number of minutes Oleg's clock gains per day -/
def oleg_gain : ℕ := 12

/-- The number of days it takes for Arthur's clock to gain 12 hours -/
def arthur_days : ℕ := minutes_in_12_hours / arthur_gain

/-- The number of days it takes for Oleg's clock to gain 12 hours -/
def oleg_days : ℕ := minutes_in_12_hours / oleg_gain

theorem clock_synchronization :
  Nat.lcm arthur_days oleg_days = 240 := by sorry

end NUMINAMATH_CALUDE_clock_synchronization_l1484_148414


namespace NUMINAMATH_CALUDE_bara_numbers_l1484_148478

theorem bara_numbers (a b : ℤ) (h1 : a ≠ b) 
  (h2 : (a + b) + (a - b) + a * b + a / b = -100)
  (h3 : (a - b) + a * b + a / b = -100) :
  (a = -9 ∧ b = 9) ∨ (a = 11 ∧ b = -11) := by
sorry

end NUMINAMATH_CALUDE_bara_numbers_l1484_148478


namespace NUMINAMATH_CALUDE_g_equals_2x_minus_1_l1484_148418

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the property of g in relation to f
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_equals_2x_minus_1 (g : ℝ → ℝ) (h : g_property g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_equals_2x_minus_1_l1484_148418


namespace NUMINAMATH_CALUDE_f_comparison_l1484_148475

def f (a b x : ℝ) := a * x^2 - 2 * b * x + 1

theorem f_comparison (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_increasing : ∀ x y, x ≤ y → y ≤ 0 → f a b x ≤ f a b y) :
  f a b (a - 2) < f a b (b + 1) :=
by sorry

end NUMINAMATH_CALUDE_f_comparison_l1484_148475


namespace NUMINAMATH_CALUDE_power_function_increasing_m_l1484_148453

/-- A function f is a power function if it can be written as f(x) = ax^n for some constants a and n, where a ≠ 0 -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a n : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^n

/-- A function f is increasing on (0, +∞) if for any x1 < x2 in (0, +∞), f(x1) < f(x2) -/
def isIncreasingOnPositiveReals (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2

/-- The main theorem -/
theorem power_function_increasing_m (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (m^2 - m - 5) * x^m
  isPowerFunction f ∧ isIncreasingOnPositiveReals f → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_increasing_m_l1484_148453


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l1484_148461

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_duration : ℕ
  change_interval : ℕ

/-- Calculates the probability of observing a color change -/
def probability_of_change (cycle : TrafficLightCycle) : ℚ :=
  cycle.change_interval / cycle.total_duration

theorem traffic_light_change_probability :
  let cycle : TrafficLightCycle := ⟨93, 15⟩
  probability_of_change cycle = 5 / 31 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l1484_148461


namespace NUMINAMATH_CALUDE_xibing_purchase_problem_l1484_148423

/-- Xibing purchase problem -/
theorem xibing_purchase_problem 
  (initial_price : ℚ) 
  (person_a_spent : ℚ) 
  (person_b_spent : ℚ) 
  (box_difference : ℕ) 
  (price_reduction : ℚ) :
  person_a_spent = 2400 →
  person_b_spent = 3000 →
  box_difference = 10 →
  price_reduction = 20 →
  ∃ (person_a_boxes : ℕ),
    person_a_boxes = 40 ∧
    initial_price = person_a_spent / person_a_boxes ∧
    initial_price = person_b_spent / (person_a_boxes + box_difference) ∧
    (2 * person_a_spent) / (person_a_boxes + person_a_spent / (initial_price - price_reduction)) = 48 ∧
    (person_b_spent + (initial_price - price_reduction) * (person_a_boxes + box_difference)) / 
      (2 * (person_a_boxes + box_difference)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_xibing_purchase_problem_l1484_148423


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l1484_148481

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h_initial : initial_price = 10)
  (h_new : new_price = 13) :
  let reduction_percentage := (new_price - initial_price) / initial_price * 100
  reduction_percentage = 30 := by
  sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l1484_148481


namespace NUMINAMATH_CALUDE_john_needs_four_planks_l1484_148462

/-- The number of planks John needs for the house wall -/
def num_planks (total_nails : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  (total_nails - additional_nails) / nails_per_plank

/-- Theorem stating that John needs 4 planks for the house wall -/
theorem john_needs_four_planks :
  num_planks 43 7 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_four_planks_l1484_148462


namespace NUMINAMATH_CALUDE_range_of_m_l1484_148486

/-- Given that p: m - 1 < x < m + 1, q: (x - 2)(x - 6) < 0, and q is a necessary but not sufficient
condition for p, prove that the range of values for m is [3, 5]. -/
theorem range_of_m (m x : ℝ) 
  (hp : m - 1 < x ∧ x < m + 1)
  (hq : (x - 2) * (x - 6) < 0)
  (h_nec_not_suff : ∀ y, (m - 1 < y ∧ y < m + 1) → (y - 2) * (y - 6) < 0)
  (h_not_suff : ∃ z, (z - 2) * (z - 6) < 0 ∧ ¬(m - 1 < z ∧ z < m + 1)) :
  3 ≤ m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1484_148486


namespace NUMINAMATH_CALUDE_alberts_age_l1484_148447

theorem alberts_age (dad_age : ℕ) (h1 : dad_age = 48) : ∃ (albert_age : ℕ),
  (albert_age = 15) ∧ 
  (dad_age - 4 = 4 * (albert_age - 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_alberts_age_l1484_148447


namespace NUMINAMATH_CALUDE_line_inclination_theorem_l1484_148434

/-- Given a line ax + by + c = 0 with inclination angle α, and sin α + cos α = 0, then a - b = 0 -/
theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ x y, a * x + b * y + c = 0) →  -- line exists
  (Real.tan α = -a / b) →           -- definition of inclination angle
  (Real.sin α + Real.cos α = 0) →   -- given condition
  a - b = 0 := by
sorry

end NUMINAMATH_CALUDE_line_inclination_theorem_l1484_148434


namespace NUMINAMATH_CALUDE_program_output_l1484_148490

def S : ℕ → ℕ
  | 0 => 1
  | (n + 1) => S n + (2 * (n + 1) - 1)

theorem program_output :
  (S 1 = 2) ∧ (S 2 = 5) ∧ (S 3 = 10) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l1484_148490


namespace NUMINAMATH_CALUDE_map_scale_conversion_l1484_148469

/-- Given a map scale where 8 cm represents 40 km, prove that 20 cm represents 100 km -/
theorem map_scale_conversion (map_scale : ℝ → ℝ) 
  (h1 : map_scale 8 = 40) -- 8 cm represents 40 km
  (h2 : ∀ x y : ℝ, map_scale (x + y) = map_scale x + map_scale y) -- Linear scaling
  (h3 : ∀ x : ℝ, map_scale x ≥ 0) -- Non-negative scaling
  : map_scale 20 = 100 := by sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l1484_148469


namespace NUMINAMATH_CALUDE_expand_and_simplify_product_l1484_148429

theorem expand_and_simplify_product (x : ℝ) : 
  (5 * x + 3) * (2 * x^2 - x + 4) = 10 * x^3 + x^2 + 17 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_product_l1484_148429


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1484_148470

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1484_148470


namespace NUMINAMATH_CALUDE_max_value_log_expression_l1484_148400

open Real

theorem max_value_log_expression (x : ℝ) (h : x > -1) :
  ∃ M, M = -2 ∧ 
  (log (x + 1 / (x + 1) + 3) / log (1/2) ≤ M) ∧
  ∃ x₀, x₀ > -1 ∧ log (x₀ + 1 / (x₀ + 1) + 3) / log (1/2) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_log_expression_l1484_148400


namespace NUMINAMATH_CALUDE_triangle_shape_l1484_148452

theorem triangle_shape (A : ℝ) (hA : 0 < A ∧ A < π) 
  (h : Real.sin A + Real.cos A = 12/25) : A > π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l1484_148452


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l1484_148424

-- Equation 1
theorem equation_one_solutions (x : ℝ) : 
  (x + 2)^2 = 2*x + 4 ↔ x = 0 ∨ x = -2 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) : 
  x^2 - 2*x - 5 = 0 ↔ x = 1 + Real.sqrt 6 ∨ x = 1 - Real.sqrt 6 := by sorry

-- Equation 3
theorem equation_three_solutions (x : ℝ) : 
  x^2 - 5*x - 6 = 0 ↔ x = -1 ∨ x = 6 := by sorry

-- Equation 4
theorem equation_four_solutions (x : ℝ) : 
  (x + 3)^2 = (1 - 2*x)^2 ↔ x = -2/3 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solutions_equation_four_solutions_l1484_148424


namespace NUMINAMATH_CALUDE_total_pencils_is_52_l1484_148458

/-- The number of pencils in a pack -/
def pencils_per_pack : ℕ := 12

/-- The number of packs Jimin has -/
def jimin_packs : ℕ := 2

/-- The number of individual pencils Jimin has -/
def jimin_individual : ℕ := 7

/-- The number of packs Yuna has -/
def yuna_packs : ℕ := 1

/-- The number of individual pencils Yuna has -/
def yuna_individual : ℕ := 9

/-- The total number of pencils Jimin and Yuna have -/
def total_pencils : ℕ := 
  jimin_packs * pencils_per_pack + jimin_individual +
  yuna_packs * pencils_per_pack + yuna_individual

theorem total_pencils_is_52 : total_pencils = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_52_l1484_148458


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1484_148474

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1484_148474


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1484_148431

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (2 - Complex.I) * (a + 2 * Complex.I) = b * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1484_148431


namespace NUMINAMATH_CALUDE_octal_difference_multiple_of_seven_fifty_six_possible_difference_l1484_148419

-- Define a two-digit number in base 8
def octal_number (tens units : Nat) : Nat :=
  8 * tens + units

-- Define the reversed number
def reversed_octal_number (tens units : Nat) : Nat :=
  8 * units + tens

-- Define the difference between the original and reversed number
def octal_difference (tens units : Nat) : Int :=
  (octal_number tens units : Int) - (reversed_octal_number tens units : Int)

-- Theorem stating that the difference is always a multiple of 7
theorem octal_difference_multiple_of_seven (tens units : Nat) :
  ∃ k : Int, octal_difference tens units = 7 * k :=
sorry

-- Theorem stating that 56 is a possible difference
theorem fifty_six_possible_difference :
  ∃ tens units : Nat, octal_difference tens units = 56 :=
sorry

end NUMINAMATH_CALUDE_octal_difference_multiple_of_seven_fifty_six_possible_difference_l1484_148419


namespace NUMINAMATH_CALUDE_gcd_75_225_l1484_148477

theorem gcd_75_225 : Nat.gcd 75 225 = 75 := by
  sorry

end NUMINAMATH_CALUDE_gcd_75_225_l1484_148477


namespace NUMINAMATH_CALUDE_tiffany_sunscreen_cost_l1484_148451

/-- Calculates the cost of sunscreen for a beach visit given the specified parameters. -/
def sunscreenCost (reapplyInterval : ℕ) (amountPerApplication : ℕ) (bottleSize : ℕ) (bottleCost : ℚ) (visitDuration : ℕ) : ℚ :=
  let applications := visitDuration / reapplyInterval
  let totalAmount := applications * amountPerApplication
  let bottlesNeeded := (totalAmount + bottleSize - 1) / bottleSize  -- Ceiling division
  bottlesNeeded * bottleCost

/-- Theorem stating that the sunscreen cost for Tiffany's beach visit is $7. -/
theorem tiffany_sunscreen_cost :
  sunscreenCost 2 3 12 (7/2) 16 = 7 := by
  sorry

#eval sunscreenCost 2 3 12 (7/2) 16

end NUMINAMATH_CALUDE_tiffany_sunscreen_cost_l1484_148451


namespace NUMINAMATH_CALUDE_percent_relation_l1484_148421

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (2/5) * (x + y)) :
  y = (1/9) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1484_148421


namespace NUMINAMATH_CALUDE_quadratic_inequality_integer_solution_l1484_148468

theorem quadratic_inequality_integer_solution (a : ℤ) : 
  (∀ x : ℝ, x^2 + 2*↑a*x + 1 > 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_integer_solution_l1484_148468


namespace NUMINAMATH_CALUDE_inequality_problem_l1484_148467

theorem inequality_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 1) :
  (∃ (max_val : ℝ), a = 2 → (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/2 + 1/x + 1/y = 1 →
    1/(x + y) ≤ max_val) ∧ max_val = 1/8) ∧
  1/(a + b) + 1/(b + c) + 1/(a + c) ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l1484_148467


namespace NUMINAMATH_CALUDE_min_operations_for_two_pints_l1484_148465

/-- Represents the state of the two vessels -/
structure VesselState :=
  (v7 : ℕ)
  (v11 : ℕ)

/-- Represents an operation on the vessels -/
inductive Operation
  | Fill7
  | Fill11
  | Empty7
  | Empty11
  | Pour7To11
  | Pour11To7

/-- Applies an operation to a vessel state -/
def applyOperation (state : VesselState) (op : Operation) : VesselState :=
  match op with
  | Operation.Fill7 => ⟨7, state.v11⟩
  | Operation.Fill11 => ⟨state.v7, 11⟩
  | Operation.Empty7 => ⟨0, state.v11⟩
  | Operation.Empty11 => ⟨state.v7, 0⟩
  | Operation.Pour7To11 => 
      let amount := min state.v7 (11 - state.v11)
      ⟨state.v7 - amount, state.v11 + amount⟩
  | Operation.Pour11To7 => 
      let amount := min state.v11 (7 - state.v7)
      ⟨state.v7 + amount, state.v11 - amount⟩

/-- Checks if a sequence of operations results in 2 pints in either vessel -/
def isValidSolution (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.v7 = 2 ∨ finalState.v11 = 2

/-- The main theorem stating that 14 is the minimum number of operations -/
theorem min_operations_for_two_pints :
  (∃ (ops : List Operation), ops.length = 14 ∧ isValidSolution ops) ∧
  (∀ (ops : List Operation), ops.length < 14 → ¬isValidSolution ops) :=
sorry

end NUMINAMATH_CALUDE_min_operations_for_two_pints_l1484_148465


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l1484_148411

def isComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ a + b = n

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬isSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l1484_148411


namespace NUMINAMATH_CALUDE_max_tangent_segment_length_l1484_148487

/-- Given a triangle ABC with perimeter 2p, the maximum length of a segment
    parallel to BC and tangent to the inscribed circle is p/4, and this
    maximum is achieved when BC = p/2. -/
theorem max_tangent_segment_length (p : ℝ) (h : p > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 2 * p ∧
    (∀ (x y z : ℝ),
      x > 0 → y > 0 → z > 0 → x + y + z = 2 * p →
      x * (p - x) / p ≤ p / 4) ∧
    a * (p - a) / p = p / 4 ∧
    a = p / 2 := by
  sorry


end NUMINAMATH_CALUDE_max_tangent_segment_length_l1484_148487


namespace NUMINAMATH_CALUDE_taxi_fare_problem_l1484_148407

/-- The fare structure for a taxi ride -/
structure TaxiFare where
  fixedCharge : ℝ
  ratePerMile : ℝ

/-- Calculate the total fare for a given distance -/
def totalFare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  fare.fixedCharge + fare.ratePerMile * distance

/-- The problem statement -/
theorem taxi_fare_problem (fare : TaxiFare) 
  (h1 : totalFare fare 80 = 200)
  (h2 : fare.fixedCharge = 20) :
  totalFare fare 100 = 245 := by
  sorry


end NUMINAMATH_CALUDE_taxi_fare_problem_l1484_148407


namespace NUMINAMATH_CALUDE_system_solution_iff_m_neq_one_l1484_148404

/-- The system of equations has at least one solution if and only if m ≠ 1 -/
theorem system_solution_iff_m_neq_one (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_iff_m_neq_one_l1484_148404


namespace NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l1484_148401

/-- The maximum number of parts that three planes can divide 3D space into -/
def max_parts_three_planes : ℕ := 8

/-- Theorem stating that the maximum number of parts that three planes can divide 3D space into is 8 -/
theorem max_parts_three_planes_is_eight :
  max_parts_three_planes = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_parts_three_planes_is_eight_l1484_148401


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l1484_148471

/-- A regular dodecahedron -/
structure RegularDodecahedron :=
  (vertices : ℕ)
  (edges_per_vertex : ℕ)
  (h_vertices : vertices = 20)
  (h_edges_per_vertex : edges_per_vertex = 3)

/-- The probability of two randomly chosen vertices being connected by an edge -/
def edge_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_probability d = 3 / 19 :=
by sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l1484_148471


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1484_148403

theorem adult_ticket_cost 
  (total_spent : ℕ) 
  (family_size : ℕ) 
  (child_ticket_cost : ℕ) 
  (adult_tickets : ℕ) 
  (h1 : total_spent = 119)
  (h2 : family_size = 7)
  (h3 : child_ticket_cost = 14)
  (h4 : adult_tickets = 4) :
  ∃ (adult_ticket_cost : ℕ), 
    adult_ticket_cost * adult_tickets + 
    child_ticket_cost * (family_size - adult_tickets) = total_spent ∧ 
    adult_ticket_cost = 14 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1484_148403


namespace NUMINAMATH_CALUDE_integral_sin_cos_identity_l1484_148420

theorem integral_sin_cos_identity : 
  ∫ x in (0)..(π / 2), (Real.sin (Real.sin x))^2 + (Real.cos (Real.cos x))^2 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_sin_cos_identity_l1484_148420


namespace NUMINAMATH_CALUDE_four_digit_number_proof_l1484_148438

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the largest number that can be formed by rearranging the digits of a given number -/
def largest_rearrangement (n : FourDigitNumber) : ℕ :=
  sorry

/-- Returns the smallest number that can be formed by rearranging the digits of a given number -/
def smallest_rearrangement (n : FourDigitNumber) : ℕ :=
  sorry

/-- Checks if a number has any digit equal to 0 -/
def has_zero_digit (n : ℕ) : Bool :=
  sorry

theorem four_digit_number_proof :
  ∃ (A : FourDigitNumber),
    largest_rearrangement A = A.value + 7668 ∧
    smallest_rearrangement A = A.value - 594 ∧
    ¬ has_zero_digit A.value ∧
    A.value = 1963 :=
  sorry

end NUMINAMATH_CALUDE_four_digit_number_proof_l1484_148438


namespace NUMINAMATH_CALUDE_sum_of_powers_l1484_148409

theorem sum_of_powers (x : ℝ) (h : x + 1/x = 4) : x^6 + 1/x^6 = 2702 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1484_148409


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_one_l1484_148456

theorem points_three_units_from_negative_one : 
  ∀ x : ℝ, abs (x - (-1)) = 3 ↔ x = 2 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_one_l1484_148456


namespace NUMINAMATH_CALUDE_josh_bought_six_cds_l1484_148460

/-- Represents the shopping problem where Josh buys films, books, and CDs. -/
def ShoppingProblem (num_films num_books total_spent : ℕ) (film_cost book_cost cd_cost : ℚ) : Prop :=
  ∃ (num_cds : ℕ),
    (num_films : ℚ) * film_cost + (num_books : ℚ) * book_cost + (num_cds : ℚ) * cd_cost = total_spent

/-- Proves that Josh bought 6 CDs given the problem conditions. -/
theorem josh_bought_six_cds :
  ShoppingProblem 9 4 79 5 4 3 → (∃ (num_cds : ℕ), num_cds = 6) :=
by
  sorry

#check josh_bought_six_cds

end NUMINAMATH_CALUDE_josh_bought_six_cds_l1484_148460


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l1484_148441

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the asymptote condition
def asymptote_condition (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3

-- Define the focus condition
def focus_condition (c : ℝ) : Prop :=
  Real.sqrt ((1 + c)^2 + 3) = 2

-- Define the point condition
def point_condition (x y c : ℝ) : Prop :=
  Real.sqrt ((x + c)^2 + y^2) = 5/2

-- Main theorem
theorem hyperbola_theorem (a b c : ℝ) (x y : ℝ) :
  hyperbola a b x y →
  asymptote_condition a b →
  focus_condition c →
  point_condition x y c →
  Real.sqrt ((x - c)^2 + y^2) = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l1484_148441


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l1484_148415

theorem unique_two_digit_integer (t : ℕ) : 
  (t ≥ 10 ∧ t ≤ 99) ∧ (13 * t) % 100 = 47 ↔ t = 19 :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l1484_148415


namespace NUMINAMATH_CALUDE_pythagorean_triple_has_even_number_l1484_148472

theorem pythagorean_triple_has_even_number (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  Even a ∨ Even b ∨ Even c := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_has_even_number_l1484_148472


namespace NUMINAMATH_CALUDE_line_segment_polar_equation_l1484_148480

/-- The polar equation of the line segment y = 1 - x where 0 ≤ x ≤ 1 -/
theorem line_segment_polar_equation (θ : Real) (ρ : Real) :
  (0 ≤ θ) ∧ (θ ≤ Real.pi / 2) →
  (ρ * Real.cos θ + ρ * Real.sin θ = 1) ↔
  (ρ * Real.sin θ = 1 - ρ * Real.cos θ) ∧
  (0 ≤ ρ * Real.cos θ) ∧ (ρ * Real.cos θ ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_polar_equation_l1484_148480


namespace NUMINAMATH_CALUDE_pistachio_price_per_can_l1484_148496

/-- The price of a can of pistachios given James' consumption habits and weekly spending -/
theorem pistachio_price_per_can 
  (can_size : ℝ) 
  (consumption_per_5_days : ℝ) 
  (weekly_spending : ℝ) 
  (h1 : can_size = 5) 
  (h2 : consumption_per_5_days = 30) 
  (h3 : weekly_spending = 84) : 
  weekly_spending / ((7 / 5) * consumption_per_5_days / can_size) = 10 := by
sorry

end NUMINAMATH_CALUDE_pistachio_price_per_can_l1484_148496


namespace NUMINAMATH_CALUDE_p_or_q_can_be_either_l1484_148440

theorem p_or_q_can_be_either (p q : Prop) (h : ¬(p ∧ q)) : 
  (∃ b : Bool, (p ∨ q) = b) ∧ (∃ b : Bool, (p ∨ q) ≠ b) := by
sorry

end NUMINAMATH_CALUDE_p_or_q_can_be_either_l1484_148440


namespace NUMINAMATH_CALUDE_average_rainfall_virginia_l1484_148488

theorem average_rainfall_virginia (march april may june july : ℝ) 
  (h_march : march = 3.79)
  (h_april : april = 4.5)
  (h_may : may = 3.95)
  (h_june : june = 3.09)
  (h_july : july = 4.67) :
  (march + april + may + june + july) / 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_rainfall_virginia_l1484_148488


namespace NUMINAMATH_CALUDE_fresh_fruits_ratio_l1484_148449

/-- Represents the quantity of fruits in the store -/
structure FruitQuantity where
  pineapples : ℕ
  apples : ℕ
  oranges : ℕ

/-- Represents the spoilage rates of fruits -/
structure SpoilageRate where
  pineapples : ℚ
  apples : ℚ
  oranges : ℚ

def initialQuantity : FruitQuantity :=
  { pineapples := 200, apples := 300, oranges := 100 }

def soldQuantity : FruitQuantity :=
  { pineapples := 56, apples := 128, oranges := 22 }

def spoilageRate : SpoilageRate :=
  { pineapples := 1/10, apples := 15/100, oranges := 1/20 }

def remainingFruits (initial : FruitQuantity) (sold : FruitQuantity) : FruitQuantity :=
  { pineapples := initial.pineapples - sold.pineapples,
    apples := initial.apples - sold.apples,
    oranges := initial.oranges - sold.oranges }

def spoiledFruits (remaining : FruitQuantity) (rate : SpoilageRate) : FruitQuantity :=
  { pineapples := (remaining.pineapples : ℚ) * rate.pineapples |> round |> Int.toNat,
    apples := (remaining.apples : ℚ) * rate.apples |> round |> Int.toNat,
    oranges := (remaining.oranges : ℚ) * rate.oranges |> round |> Int.toNat }

def freshFruits (remaining : FruitQuantity) (spoiled : FruitQuantity) : FruitQuantity :=
  { pineapples := remaining.pineapples - spoiled.pineapples,
    apples := remaining.apples - spoiled.apples,
    oranges := remaining.oranges - spoiled.oranges }

theorem fresh_fruits_ratio :
  let remaining := remainingFruits initialQuantity soldQuantity
  let spoiled := spoiledFruits remaining spoilageRate
  let fresh := freshFruits remaining spoiled
  fresh.pineapples = 130 ∧ fresh.apples = 146 ∧ fresh.oranges = 74 := by sorry

end NUMINAMATH_CALUDE_fresh_fruits_ratio_l1484_148449


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1484_148405

theorem inscribed_circle_radius (PQ QR : Real) (h1 : PQ = 15) (h2 : QR = 8) : 
  let PR := Real.sqrt (PQ^2 + QR^2)
  let s := (PQ + QR + PR) / 2
  let area := PQ * QR / 2
  area / s = 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1484_148405


namespace NUMINAMATH_CALUDE_sum_of_digits_of_X_squared_sum_of_digits_of_111111111_squared_l1484_148416

/-- 
Given a natural number n, we define X as the number consisting of n ones.
For example, if n = 3, then X = 111.
-/
def X (n : ℕ) : ℕ := (10^n - 1) / 9

/-- 
The sum of digits function for a natural number.
-/
def sumOfDigits (m : ℕ) : ℕ := sorry

/-- 
Theorem: For a number X consisting of n ones, the sum of the digits of X^2 is equal to n^2.
-/
theorem sum_of_digits_of_X_squared (n : ℕ) : 
  sumOfDigits ((X n)^2) = n^2 := by sorry

/-- 
Corollary: For the specific case where n = 9 (corresponding to 111111111), 
the sum of the digits of X^2 is 81.
-/
theorem sum_of_digits_of_111111111_squared : 
  sumOfDigits ((X 9)^2) = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_X_squared_sum_of_digits_of_111111111_squared_l1484_148416


namespace NUMINAMATH_CALUDE_part1_solution_set_part2_solution_set_l1484_148406

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem part1_solution_set :
  {x : ℝ | f 1 x < |2*x - 1| - 1} = {x : ℝ | x < -1 ∨ x > 1} := by sorry

-- Part 2
theorem part2_solution_set :
  ∀ x ∈ Set.Ioo (-2) 1, {a : ℝ | |x - 1| > |2*x - a - 1| - f a x} = Set.Iic (-2) := by sorry

end NUMINAMATH_CALUDE_part1_solution_set_part2_solution_set_l1484_148406


namespace NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1484_148432

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (2*x - 1)^6 = a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 729 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_coefficients_l1484_148432


namespace NUMINAMATH_CALUDE_ap_80th_term_l1484_148445

/-- An arithmetic progression (AP) with given properties -/
structure AP where
  /-- Sum of the first 20 terms -/
  sum20 : ℚ
  /-- Sum of the first 60 terms -/
  sum60 : ℚ
  /-- The property that sum20 = 200 -/
  sum20_eq : sum20 = 200
  /-- The property that sum60 = 180 -/
  sum60_eq : sum60 = 180

/-- The 80th term of the AP -/
def term80 (ap : AP) : ℚ := -573/40

/-- Theorem stating that the 80th term of the AP with given properties is -573/40 -/
theorem ap_80th_term (ap : AP) : term80 ap = -573/40 := by
  sorry

end NUMINAMATH_CALUDE_ap_80th_term_l1484_148445


namespace NUMINAMATH_CALUDE_system_solution_quadratic_expression_l1484_148482

theorem system_solution_quadratic_expression :
  ∀ x y z : ℚ,
  (2 * x + 3 * y + z = 20) →
  (x + 2 * y + 3 * z = 26) →
  (3 * x + y + 2 * z = 29) →
  ∃ k : ℚ, 12 * x^2 + 22 * x * y + 12 * y^2 + 12 * x * z + 12 * y * z + 12 * z^2 = k :=
by
  sorry


end NUMINAMATH_CALUDE_system_solution_quadratic_expression_l1484_148482


namespace NUMINAMATH_CALUDE_smallest_rectangle_cover_l1484_148427

/-- The width of the rectangle -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the smallest square that can be covered by whole rectangles -/
def square_side : ℕ := lcm rectangle_width rectangle_height

/-- The area of the square -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_rectangle_cover :
  num_rectangles = 12 ∧
  ∀ n : ℕ, n < num_rectangles → 
    ¬ (∃ s : ℕ, s * s = n * rectangle_area) :=
sorry

end NUMINAMATH_CALUDE_smallest_rectangle_cover_l1484_148427


namespace NUMINAMATH_CALUDE_quadrilateral_second_offset_l1484_148499

/-- Given a quadrilateral with one diagonal of 50 cm, one offset of 10 cm, and an area of 450 cm^2,
    prove that the length of the second offset is 8 cm. -/
theorem quadrilateral_second_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) (offset2 : ℝ) :
  diagonal = 50 → offset1 = 10 → area = 450 →
  area = 1/2 * diagonal * (offset1 + offset2) →
  offset2 = 8 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_second_offset_l1484_148499


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l1484_148413

theorem recurring_decimal_sum : 
  (2 : ℚ) / 3 + 7 / 9 = 13 / 9 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l1484_148413


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_odd_l1484_148428

def isOdd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

def inRange (n : ℕ) : Prop := 5 ≤ n ∧ n ≤ 12

theorem sum_of_largest_and_smallest_odd : 
  ∃ (a b : ℕ), 
    isOdd a ∧ isOdd b ∧ 
    inRange a ∧ inRange b ∧
    (∀ x, isOdd x ∧ inRange x → a ≤ x ∧ x ≤ b) ∧
    a + b = 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_odd_l1484_148428


namespace NUMINAMATH_CALUDE_opponent_total_score_l1484_148466

/-- Represents the score of a basketball game -/
structure GameScore where
  team : ℕ
  opponent : ℕ

/-- Calculates the total opponent score given a list of game scores -/
def totalOpponentScore (games : List GameScore) : ℕ :=
  games.foldr (fun game acc => game.opponent + acc) 0

theorem opponent_total_score : 
  ∃ (games : List GameScore),
    games.length = 12 ∧ 
    (∀ g ∈ games, 1 ≤ g.team ∧ g.team ≤ 12) ∧
    (games.filter (fun g => g.opponent = g.team + 2)).length = 6 ∧
    (∀ g ∈ games.filter (fun g => g.opponent ≠ g.team + 2), g.team = 3 * g.opponent) ∧
    totalOpponentScore games = 50 := by
  sorry


end NUMINAMATH_CALUDE_opponent_total_score_l1484_148466


namespace NUMINAMATH_CALUDE_repeating_decimal_multiplication_l1484_148495

theorem repeating_decimal_multiplication (x : ℝ) : 
  (∀ n : ℕ, (x * 10^(4 + 2*n)) % 1 = 0.3131) → 
  (10^5 - 10^3) * x = 309.969 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_multiplication_l1484_148495


namespace NUMINAMATH_CALUDE_b_third_place_four_times_l1484_148426

-- Define the structure for a contestant
structure Contestant where
  name : String
  firstPlace : Nat
  secondPlace : Nat
  thirdPlace : Nat

-- Define the competition parameters
def numCompetitions : Nat := 6
def firstPlaceScore : Nat := 5
def secondPlaceScore : Nat := 2
def thirdPlaceScore : Nat := 1

-- Define the contestants
def contestantA : Contestant := ⟨"A", 4, 1, 1⟩
def contestantB : Contestant := ⟨"B", 1, 0, 4⟩
def contestantC : Contestant := ⟨"C", 0, 3, 2⟩

-- Define the score calculation function
def calculateScore (c : Contestant) : Nat :=
  c.firstPlace * firstPlaceScore + c.secondPlace * secondPlaceScore + c.thirdPlace * thirdPlaceScore

-- Theorem to prove
theorem b_third_place_four_times :
  (calculateScore contestantA = 26) ∧
  (calculateScore contestantB = 11) ∧
  (calculateScore contestantC = 11) ∧
  (contestantB.firstPlace = 1) ∧
  (contestantA.firstPlace + contestantB.firstPlace + contestantC.firstPlace +
   contestantA.secondPlace + contestantB.secondPlace + contestantC.secondPlace +
   contestantA.thirdPlace + contestantB.thirdPlace + contestantC.thirdPlace = numCompetitions) →
  contestantB.thirdPlace = 4 := by
  sorry


end NUMINAMATH_CALUDE_b_third_place_four_times_l1484_148426


namespace NUMINAMATH_CALUDE_factorization_sum_l1484_148402

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∃ (x y : ℝ), 27 * x^6 - 512 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 152 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1484_148402


namespace NUMINAMATH_CALUDE_park_oaks_l1484_148498

/-- The number of huge ancient oaks in a park -/
def huge_ancient_oaks (total_trees medium_firs saplings : ℕ) : ℕ :=
  total_trees - medium_firs - saplings

/-- Theorem: There are 15 huge ancient oaks in the park -/
theorem park_oaks : huge_ancient_oaks 96 23 58 = 15 := by
  sorry

end NUMINAMATH_CALUDE_park_oaks_l1484_148498


namespace NUMINAMATH_CALUDE_price_reduction_for_1200_profit_no_solution_for_1600_profit_l1484_148410

-- Define the initial conditions
def initial_sales : ℕ := 30
def initial_profit : ℕ := 40
def sales_increase_rate : ℕ := 2

-- Define the profit function
def daily_profit (price_reduction : ℝ) : ℝ :=
  (initial_profit - price_reduction) * (initial_sales + sales_increase_rate * price_reduction)

-- Theorem for part 1
theorem price_reduction_for_1200_profit :
  ∃ (x : ℝ), x > 0 ∧ daily_profit x = 1200 ∧ 
  (∀ (y : ℝ), y > 0 ∧ y ≠ x → daily_profit y ≠ 1200) :=
sorry

-- Theorem for part 2
theorem no_solution_for_1600_profit :
  ¬∃ (x : ℝ), daily_profit x = 1600 :=
sorry

end NUMINAMATH_CALUDE_price_reduction_for_1200_profit_no_solution_for_1600_profit_l1484_148410


namespace NUMINAMATH_CALUDE_arrangements_eight_athletes_three_consecutive_l1484_148430

/-- The number of tracks and athletes -/
def n : ℕ := 8

/-- The number of specified athletes that must be in consecutive tracks -/
def k : ℕ := 3

/-- The number of ways to arrange n athletes on n tracks, 
    where k specified athletes must be in consecutive tracks -/
def arrangements (n k : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem arrangements_eight_athletes_three_consecutive : 
  arrangements n k = 4320 := by sorry

end NUMINAMATH_CALUDE_arrangements_eight_athletes_three_consecutive_l1484_148430


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1484_148425

theorem fraction_sum_equality : (18 : ℚ) / 42 - 2 / 9 + 1 / 14 = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1484_148425


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1484_148491

theorem vector_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : a • b = 0) 
  (h2 : ‖a‖ = 2) 
  (h3 : ‖b‖ = 1) : 
  ‖a + 2 • b‖ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1484_148491


namespace NUMINAMATH_CALUDE_original_price_calculation_shirt_price_proof_l1484_148497

/-- 
Given two successive discounts and a final sale price, 
calculate the original price of an item.
-/
theorem original_price_calculation 
  (discount1 : ℝ) 
  (discount2 : ℝ) 
  (final_price : ℝ) : ℝ :=
  let remaining_factor1 := 1 - discount1
  let remaining_factor2 := 1 - discount2
  let original_price := final_price / (remaining_factor1 * remaining_factor2)
  original_price

/-- 
Prove that given discounts of 15% and 2%, 
if the final sale price is 830, 
then the original price is approximately 996.40.
-/
theorem shirt_price_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (original_price_calculation 0.15 0.02 830 - 996.40) < ε :=
sorry

end NUMINAMATH_CALUDE_original_price_calculation_shirt_price_proof_l1484_148497


namespace NUMINAMATH_CALUDE_product_of_four_numbers_l1484_148450

theorem product_of_four_numbers (a b c d : ℚ) : 
  a + b + c + d = 36 →
  a = 3 * (b + c + d) →
  b = 5 * c →
  d = (1/2) * c →
  a * b * c * d = 178.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_four_numbers_l1484_148450


namespace NUMINAMATH_CALUDE_bee_paths_count_l1484_148464

/-- Represents the number of beehives in the row -/
def n : ℕ := 6

/-- Represents the possible moves of the bee -/
inductive BeeMove
  | Right
  | UpperRight
  | LowerRight

/-- Represents a path of the bee as a list of moves -/
def BeePath := List BeeMove

/-- Checks if a path is valid (ends at hive number 6) -/
def isValidPath (path : BeePath) : Bool :=
  sorry

/-- Counts the number of valid paths to hive number 6 -/
def countValidPaths : ℕ :=
  sorry

/-- Theorem: The number of valid paths to hive number 6 is 21 -/
theorem bee_paths_count : countValidPaths = 21 := by
  sorry

end NUMINAMATH_CALUDE_bee_paths_count_l1484_148464


namespace NUMINAMATH_CALUDE_some_number_value_l1484_148454

theorem some_number_value (x : ℝ) : 
  7^8 - 6/x + 9^3 + 3 + 12 = 95 → x = 1 / 960908.333 :=
by sorry

end NUMINAMATH_CALUDE_some_number_value_l1484_148454


namespace NUMINAMATH_CALUDE_total_money_proof_l1484_148412

/-- Represents the ratio of money shares for Jonah, Kira, and Liam respectively -/
def money_ratio : Fin 3 → ℕ
| 0 => 2  -- Jonah's ratio
| 1 => 3  -- Kira's ratio
| 2 => 8  -- Liam's ratio

/-- Kira's share of the money -/
def kiras_share : ℕ := 45

/-- The total amount of money shared -/
def total_money : ℕ := 195

/-- Theorem stating that given the conditions, the total amount of money shared is $195 -/
theorem total_money_proof :
  (∃ (multiplier : ℚ), 
    (multiplier * money_ratio 1 = kiras_share) ∧ 
    (multiplier * (money_ratio 0 + money_ratio 1 + money_ratio 2) = total_money)) :=
by sorry

end NUMINAMATH_CALUDE_total_money_proof_l1484_148412


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1484_148455

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def A : Set ℕ := {2,5,8}
def B : Set ℕ := {1,3,5,7}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {1,3,7} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1484_148455


namespace NUMINAMATH_CALUDE_sushi_lollipops_l1484_148439

theorem sushi_lollipops (x y : ℕ) : x + y = 27 :=
  by
    have h1 : x + y = 5 + (3 * 5) + 7 := by sorry
    have h2 : 5 + (3 * 5) + 7 = 27 := by sorry
    rw [h1, h2]

end NUMINAMATH_CALUDE_sushi_lollipops_l1484_148439


namespace NUMINAMATH_CALUDE_locus_of_points_l1484_148408

/-- Two lines in a plane --/
structure TwoLines where
  l₁ : Set (ℝ × ℝ)
  l₃ : Set (ℝ × ℝ)

/-- Distance from a point to a line --/
def distanceToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- Translate a line by a distance --/
def translateLine (l : Set (ℝ × ℝ)) (d : ℝ) : Set (ℝ × ℝ) := sorry

/-- Angle bisector of two lines --/
def angleBisector (l1 l2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The theorem statement --/
theorem locus_of_points (lines : TwoLines) (a : ℝ) :
  ∀ (M : ℝ × ℝ), 
    (distanceToLine M lines.l₁ + distanceToLine M lines.l₃ = a) →
    ∃ (d : ℝ), M ∈ angleBisector lines.l₁ (translateLine lines.l₃ d) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_points_l1484_148408


namespace NUMINAMATH_CALUDE_sam_has_148_balls_l1484_148442

-- Define the number of tennis balls for each person
def lily_balls : ℕ := 84

-- Define Frodo's tennis balls in terms of Lily's
def frodo_balls : ℕ := (lily_balls * 135 + 50) / 100

-- Define Brian's tennis balls in terms of Frodo's
def brian_balls : ℕ := (frodo_balls * 35 + 5) / 10

-- Define Sam's tennis balls
def sam_balls : ℕ := ((frodo_balls + lily_balls) * 3 + 2) / 4

-- Theorem statement
theorem sam_has_148_balls : sam_balls = 148 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_148_balls_l1484_148442


namespace NUMINAMATH_CALUDE_tesla_ownership_l1484_148489

/-- The number of Teslas owned by different individuals and their relationships. -/
theorem tesla_ownership (chris sam elon : ℕ) : 
  chris = 6 → 
  sam = chris / 2 → 
  elon = 13 → 
  elon - sam = 10 := by
sorry

end NUMINAMATH_CALUDE_tesla_ownership_l1484_148489


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1484_148463

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 2) * x^(m^2 - 2) + 2 * x + 1 = a * x^2 + b * x + c) ∧ 
  (m + 2 ≠ 0) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1484_148463


namespace NUMINAMATH_CALUDE_dig_time_proof_l1484_148493

/-- Represents the time (in days) it takes for a person to dig a well alone -/
structure DigTime :=
  (days : ℝ)
  (pos : days > 0)

/-- Given the dig times for three people and their combined dig time,
    proves that if two people's dig times are 24 and 48 days,
    the third person's dig time is 16 days -/
theorem dig_time_proof
  (combined_time : ℝ)
  (combined_time_pos : combined_time > 0)
  (combined_time_eq : combined_time = 8)
  (time1 time2 time3 : DigTime)
  (time2_eq : time2.days = 24)
  (time3_eq : time3.days = 48)
  (combined_rate_eq : 1 / combined_time = 1 / time1.days + 1 / time2.days + 1 / time3.days) :
  time1.days = 16 := by
sorry


end NUMINAMATH_CALUDE_dig_time_proof_l1484_148493


namespace NUMINAMATH_CALUDE_prob_same_color_left_right_is_31_138_l1484_148417

def total_pairs : ℕ := 12
def blue_pairs : ℕ := 7
def red_pairs : ℕ := 3
def green_pairs : ℕ := 2

def total_shoes : ℕ := total_pairs * 2

def prob_same_color_left_right : ℚ :=
  (blue_pairs * total_pairs + red_pairs * total_pairs + green_pairs * total_pairs) / 
  (total_shoes * (total_shoes - 1))

theorem prob_same_color_left_right_is_31_138 : 
  prob_same_color_left_right = 31 / 138 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_left_right_is_31_138_l1484_148417


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l1484_148422

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 := by
  sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l1484_148422


namespace NUMINAMATH_CALUDE_complement_of_beta_l1484_148436

-- Define angles α and β
variable (α β : Real)

-- Define the conditions
def complementary : Prop := α + β = 180
def alpha_greater : Prop := α > β

-- Define the complement of an angle
def complement (θ : Real) : Real := 90 - θ

-- State the theorem
theorem complement_of_beta (h1 : complementary α β) (h2 : alpha_greater α β) :
  complement β = (α - β) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_beta_l1484_148436


namespace NUMINAMATH_CALUDE_sum_of_factors_40_l1484_148459

theorem sum_of_factors_40 : (Finset.filter (· ∣ 40) (Finset.range 41)).sum id = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_40_l1484_148459


namespace NUMINAMATH_CALUDE_cab_base_price_l1484_148479

/-- Represents the base price of a cab ride -/
def base_price : ℝ := sorry

/-- Represents the per-mile charge of a cab ride -/
def per_mile_charge : ℝ := 4

/-- Represents the total distance traveled in miles -/
def distance : ℝ := 5

/-- Represents the total cost of the cab ride -/
def total_cost : ℝ := 23

/-- Theorem stating that the base price of the cab ride is $3 -/
theorem cab_base_price : base_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_cab_base_price_l1484_148479


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1484_148433

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 = 3 →
  a 6 = 1 / 9 →
  a 4 + a 5 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1484_148433


namespace NUMINAMATH_CALUDE_alice_has_winning_strategy_l1484_148457

/-- A game played on a complete graph -/
structure Graph :=
  (n : ℕ)  -- number of vertices
  (is_complete : n > 0)

/-- A player in the game -/
inductive Player
| Alice
| Bob

/-- A move in the game -/
structure Move :=
  (player : Player)
  (edges_oriented : ℕ)

/-- The game state -/
structure GameState :=
  (graph : Graph)
  (moves : List Move)
  (remaining_edges : ℕ)

/-- Alice's strategy -/
def alice_strategy (state : GameState) : Move :=
  { player := Player.Alice, edges_oriented := 1 }

/-- Bob's strategy -/
def bob_strategy (state : GameState) (m : ℕ) : Move :=
  { player := Player.Bob, edges_oriented := m }

/-- The winning condition for Alice -/
def alice_wins (final_state : GameState) : Prop :=
  ∃ (cycle : List ℕ), cycle.length > 0 ∧ cycle.Nodup

/-- The main theorem -/
theorem alice_has_winning_strategy :
  ∀ (g : Graph),
    g.n = 2014 →
    ∀ (bob_moves : GameState → ℕ),
      (∀ (state : GameState), 1 ≤ bob_moves state ∧ bob_moves state ≤ 1000) →
      ∃ (final_state : GameState),
        final_state.graph = g ∧
        final_state.remaining_edges = 0 ∧
        alice_wins final_state :=
  sorry

end NUMINAMATH_CALUDE_alice_has_winning_strategy_l1484_148457


namespace NUMINAMATH_CALUDE_prob_at_least_two_tails_in_three_flips_prob_at_least_two_tails_in_three_flips_is_half_l1484_148476

/-- The probability of getting at least two tails in three independent flips of a fair coin -/
theorem prob_at_least_two_tails_in_three_flips : ℝ :=
  let p_head : ℝ := 1/2  -- probability of getting heads on a single flip
  let p_tail : ℝ := 1 - p_head  -- probability of getting tails on a single flip
  let p_all_heads : ℝ := p_head ^ 3  -- probability of getting all heads
  let p_one_tail : ℝ := 3 * p_head ^ 2 * p_tail  -- probability of getting exactly one tail
  1 - (p_all_heads + p_one_tail)

/-- The probability of getting at least two tails in three independent flips of a fair coin is 1/2 -/
theorem prob_at_least_two_tails_in_three_flips_is_half :
  prob_at_least_two_tails_in_three_flips = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_tails_in_three_flips_prob_at_least_two_tails_in_three_flips_is_half_l1484_148476


namespace NUMINAMATH_CALUDE_share_ratio_proof_l1484_148435

theorem share_ratio_proof (total : ℝ) (c_share : ℝ) (f : ℝ) :
  total = 700 →
  c_share = 400 →
  0 < f →
  f ≤ 1 →
  total = f^2 * c_share + f * c_share + c_share →
  (f^2 * c_share) / (f * c_share) = 1 / 2 ∧
  (f * c_share) / c_share = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_proof_l1484_148435


namespace NUMINAMATH_CALUDE_quadrilaterals_from_circle_points_l1484_148485

/-- The number of distinct points on the circumference of a circle -/
def num_points : ℕ := 10

/-- The number of vertices required to form a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

/-- The number of different quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose num_points vertices_per_quadrilateral

theorem quadrilaterals_from_circle_points : num_quadrilaterals = 300 := by
  sorry

end NUMINAMATH_CALUDE_quadrilaterals_from_circle_points_l1484_148485


namespace NUMINAMATH_CALUDE_simple_interest_double_rate_l1484_148484

/-- The rate of interest for simple interest when a sum doubles in 10 years -/
theorem simple_interest_double_rate : 
  ∀ (principal : ℝ) (rate : ℝ),
  principal > 0 →
  principal * (1 + rate * 10) = 2 * principal →
  rate = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_double_rate_l1484_148484


namespace NUMINAMATH_CALUDE_amy_and_noah_total_books_l1484_148444

/-- The number of books owned by different people -/
structure BookCounts where
  maddie : ℕ
  luisa : ℕ
  amy : ℕ
  noah : ℕ

/-- The conditions of the book counting problem -/
def BookProblemConditions (bc : BookCounts) : Prop :=
  bc.maddie = 15 ∧
  bc.luisa = 18 ∧
  bc.amy + bc.luisa = bc.maddie + 9 ∧
  bc.noah = bc.amy / 3

/-- The theorem stating that under the given conditions, Amy and Noah have 8 books in total -/
theorem amy_and_noah_total_books (bc : BookCounts) 
  (h : BookProblemConditions bc) : bc.amy + bc.noah = 8 := by
  sorry

end NUMINAMATH_CALUDE_amy_and_noah_total_books_l1484_148444


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_neg_one_subset_iff_m_range_l1484_148483

-- Define sets A and B
def A : Set ℝ := {x | x > 1}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3}

-- Theorem for part (1)
theorem intersection_and_union_when_m_neg_one :
  (A ∩ B (-1) = {x | 1 < x ∧ x ≤ 2}) ∧
  (A ∪ B (-1) = {x | x ≥ -1}) := by sorry

-- Theorem for part (2)
theorem subset_iff_m_range :
  ∀ m : ℝ, B m ⊆ A ↔ m > 1 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_neg_one_subset_iff_m_range_l1484_148483
