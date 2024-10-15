import Mathlib

namespace NUMINAMATH_CALUDE_solve_for_y_l3323_332358

theorem solve_for_y (x y : ℚ) (h1 : x = 102) (h2 : x^3*y - 3*x^2*y + 3*x*y = 106200) : 
  y = 10/97 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l3323_332358


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3323_332337

theorem min_value_expression (x : ℝ) :
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition :
  ∃ x : ℝ, x = 2/3 ∧ 
    Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3323_332337


namespace NUMINAMATH_CALUDE_carl_weight_l3323_332386

theorem carl_weight (billy brad carl : ℕ) 
  (h1 : billy = brad + 9)
  (h2 : brad = carl + 5)
  (h3 : billy = 159) : 
  carl = 145 := by sorry

end NUMINAMATH_CALUDE_carl_weight_l3323_332386


namespace NUMINAMATH_CALUDE_players_who_quit_correct_players_who_quit_l3323_332321

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let remaining_players := total_lives / lives_per_player
  initial_players - remaining_players

theorem correct_players_who_quit :
  players_who_quit 10 8 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_players_who_quit_correct_players_who_quit_l3323_332321


namespace NUMINAMATH_CALUDE_inequality_solution_equivalence_l3323_332306

def satisfies_inequality (x : ℝ) : Prop :=
  1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 4

def solution_set : Set ℝ :=
  {x | x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x}

theorem inequality_solution_equivalence :
  ∀ x : ℝ, satisfies_inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_equivalence_l3323_332306


namespace NUMINAMATH_CALUDE_any_nonzero_to_zero_power_is_one_l3323_332379

theorem any_nonzero_to_zero_power_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_any_nonzero_to_zero_power_is_one_l3323_332379


namespace NUMINAMATH_CALUDE_area_of_region_l3323_332394

theorem area_of_region (x y : ℝ) : 
  (∃ A : ℝ, A = Real.pi * 23 ∧ 
   A = Real.pi * (Real.sqrt ((x + 3)^2 + (y - 2)^2))^2 ∧
   x^2 + y^2 + 6*x - 4*y - 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l3323_332394


namespace NUMINAMATH_CALUDE_machine_parts_processed_l3323_332345

/-- Given two machines processing parts for 'a' hours, where the second machine
    processed 'n' fewer parts and takes 'b' minutes longer per part than the first,
    prove the number of parts processed by each machine. -/
theorem machine_parts_processed
  (a b n : ℝ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  let x := (b * n + Real.sqrt (b^2 * n^2 + 240 * a * b * n)) / (2 * b)
  let y := (-b * n + Real.sqrt (b^2 * n^2 + 240 * a * b * n)) / (2 * b)
  (∀ t, 0 < t ∧ t < a → (t / x = t / (x - n) - b / 60)) ∧
  x > 0 ∧ y > 0 ∧ x - y = n :=
sorry


end NUMINAMATH_CALUDE_machine_parts_processed_l3323_332345


namespace NUMINAMATH_CALUDE_stating_correct_deposit_equation_l3323_332366

/-- Represents the annual interest rate as a decimal -/
def annual_rate : ℝ := 0.0369

/-- Represents the number of years for the fixed deposit -/
def years : ℕ := 3

/-- Represents the tax rate on interest as a decimal -/
def tax_rate : ℝ := 0.2

/-- Represents the final withdrawal amount in yuan -/
def final_amount : ℝ := 5442.8

/-- 
Theorem stating the correct equation for calculating the initial deposit amount,
given the annual interest rate, number of years, tax rate, and final withdrawal amount.
-/
theorem correct_deposit_equation (x : ℝ) :
  x + x * annual_rate * (years : ℝ) * (1 - tax_rate) = final_amount :=
sorry

end NUMINAMATH_CALUDE_stating_correct_deposit_equation_l3323_332366


namespace NUMINAMATH_CALUDE_sarah_homework_problem_l3323_332342

/-- The number of homework problems Sarah had initially -/
def total_problems (finished_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished_problems + remaining_pages * problems_per_page

/-- Theorem stating that Sarah had 60 homework problems initially -/
theorem sarah_homework_problem :
  total_problems 20 5 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_sarah_homework_problem_l3323_332342


namespace NUMINAMATH_CALUDE_infinite_indices_inequality_l3323_332380

def FastGrowingSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ C : ℝ, ∃ N : ℕ, ∀ k > N, (a k : ℝ) > C * k)

theorem infinite_indices_inequality
  (a : ℕ → ℕ)
  (h : FastGrowingSequence a) :
  ∀ M : ℕ, ∃ k > M, 2 * (a k) < (a (k - 1)) + (a (k + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinite_indices_inequality_l3323_332380


namespace NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l3323_332349

/-- The number of ordered quadruplets (a, b, c, d) with given gcd and lcm -/
def count_quadruplets (gcd lcm : ℕ) : ℕ := sorry

/-- The theorem stating the smallest n satisfying the conditions -/
theorem smallest_n_for_quadruplets :
  ∃ n : ℕ, n > 0 ∧ 
  count_quadruplets 72 n = 72000 ∧
  (∀ m : ℕ, m > 0 → m < n → count_quadruplets 72 m ≠ 72000) ∧
  n = 36288 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l3323_332349


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3323_332312

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3323_332312


namespace NUMINAMATH_CALUDE_solve_for_q_l3323_332395

theorem solve_for_q (n m q : ℚ) 
  (h1 : 5/6 = n/72)
  (h2 : 5/6 = (m+n)/90)
  (h3 : 5/6 = (q-m)/150) : q = 140 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l3323_332395


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3323_332327

theorem solve_exponential_equation :
  ∃ x : ℝ, 5^(3*x) = Real.sqrt 625 ∧ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3323_332327


namespace NUMINAMATH_CALUDE_rectangle_triangles_l3323_332371

/-- Represents a rectangle divided into triangles -/
structure DividedRectangle where
  horizontal_divisions : Nat
  vertical_divisions : Nat

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (rect : DividedRectangle) : Nat :=
  sorry

/-- Theorem: A rectangle divided into 6 horizontal and 3 vertical parts contains 48 triangles -/
theorem rectangle_triangles :
  let rect : DividedRectangle := { horizontal_divisions := 6, vertical_divisions := 3 }
  count_triangles rect = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangles_l3323_332371


namespace NUMINAMATH_CALUDE_salad_ratio_l3323_332326

theorem salad_ratio (mushrooms cherry_tomatoes pickles bacon_bits red_bacon_bits : ℕ) : 
  mushrooms = 3 →
  cherry_tomatoes = 2 * mushrooms →
  pickles = 4 * cherry_tomatoes →
  bacon_bits = 4 * pickles →
  red_bacon_bits = 32 →
  (red_bacon_bits : ℚ) / bacon_bits = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_salad_ratio_l3323_332326


namespace NUMINAMATH_CALUDE_q_round_time_l3323_332355

/-- The time it takes for two runners to meet at the starting point again -/
def meeting_time : ℕ := 2772

/-- The time it takes for runner P to complete one round -/
def p_round_time : ℕ := 252

/-- Theorem stating that under given conditions, runner Q takes 2772 seconds to complete a round -/
theorem q_round_time : ∀ (q_round_time : ℕ), 
  (meeting_time % p_round_time = 0) →
  (meeting_time % q_round_time = 0) →
  (meeting_time / p_round_time ≠ meeting_time / q_round_time) →
  q_round_time = meeting_time :=
by sorry

end NUMINAMATH_CALUDE_q_round_time_l3323_332355


namespace NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3323_332382

/-- A 4x4 grid of points separated by unit distances -/
def Grid := Fin 5 × Fin 5

/-- A rectangle on the grid is defined by two vertical lines and two horizontal lines -/
def Rectangle := (Fin 5 × Fin 5) × (Fin 5 × Fin 5)

/-- The number of rectangles on a 4x4 grid -/
def num_rectangles : ℕ := sorry

theorem rectangles_on_4x4_grid : num_rectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangles_on_4x4_grid_l3323_332382


namespace NUMINAMATH_CALUDE_bicycle_license_combinations_l3323_332378

def license_letter : Nat := 2  -- B or C
def license_digits : Nat := 6
def free_digit_positions : Nat := license_digits - 1  -- All but the last digit
def digits_per_position : Nat := 10  -- 0 to 9
def last_digit : Nat := 1  -- Only 5 is allowed

theorem bicycle_license_combinations :
  license_letter * digits_per_position ^ free_digit_positions * last_digit = 200000 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_license_combinations_l3323_332378


namespace NUMINAMATH_CALUDE_blue_beads_count_l3323_332313

theorem blue_beads_count (total : ℕ) (blue_neighbors : ℕ) (green_neighbors : ℕ) :
  total = 30 →
  blue_neighbors = 26 →
  green_neighbors = 20 →
  ∃ blue_count : ℕ,
    blue_count = 18 ∧
    blue_count ≤ total ∧
    blue_count * 2 ≥ blue_neighbors ∧
    (total - blue_count) * 2 ≥ green_neighbors :=
by
  sorry


end NUMINAMATH_CALUDE_blue_beads_count_l3323_332313


namespace NUMINAMATH_CALUDE_inequality_solution_l3323_332354

theorem inequality_solution (x : ℝ) : 
  (2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 6) ↔ (7 / 3 < x ∧ x ≤ 14 / 5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3323_332354


namespace NUMINAMATH_CALUDE_egg_order_problem_l3323_332335

theorem egg_order_problem (total : ℚ) : 
  (total > 0) →
  (total * (1 - 1/4) * (1 - 2/3) = 9) →
  total = 18 := by
sorry

end NUMINAMATH_CALUDE_egg_order_problem_l3323_332335


namespace NUMINAMATH_CALUDE_rectangle_division_exists_l3323_332367

/-- A rectangle in a 2D plane --/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- Predicate to check if a set of points forms a rectangle --/
def IsRectangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- A division of a rectangle into smaller rectangles --/
def RectangleDivision (r : Rectangle) (divisions : List Rectangle) : Prop := sorry

/-- Check if the union of two rectangles forms a rectangle --/
def UnionIsRectangle (r1 r2 : Rectangle) : Prop := sorry

/-- Main theorem: There exists a division of a rectangle into 5 smaller rectangles
    such that the union of any two of them is not a rectangle --/
theorem rectangle_division_exists :
  ∃ (r : Rectangle) (divisions : List Rectangle),
    RectangleDivision r divisions ∧
    divisions.length = 5 ∧
    ∀ (r1 r2 : Rectangle), r1 ∈ divisions → r2 ∈ divisions → r1 ≠ r2 →
      ¬UnionIsRectangle r1 r2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_exists_l3323_332367


namespace NUMINAMATH_CALUDE_garden_area_increase_l3323_332381

theorem garden_area_increase : 
  let original_length : ℝ := 40
  let original_width : ℝ := 10
  let original_perimeter : ℝ := 2 * (original_length + original_width)
  let new_side_length : ℝ := original_perimeter / 4
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_side_length ^ 2
  new_area - original_area = 225 := by sorry

end NUMINAMATH_CALUDE_garden_area_increase_l3323_332381


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l3323_332331

theorem simplify_fraction_product : 
  (360 : ℚ) / 24 * (10 : ℚ) / 240 * (6 : ℚ) / 3 * (9 : ℚ) / 18 = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l3323_332331


namespace NUMINAMATH_CALUDE_ladonnas_cans_correct_l3323_332340

/-- The number of cans collected by LaDonna, given that:
    - The total number of cans collected is 85
    - Prikya collected twice as many cans as LaDonna
    - Yoki collected 10 cans
-/
def ladonnas_cans : ℕ := 25

/-- The total number of cans collected -/
def total_cans : ℕ := 85

/-- The number of cans collected by Yoki -/
def yokis_cans : ℕ := 10

theorem ladonnas_cans_correct :
  ladonnas_cans + 2 * ladonnas_cans + yokis_cans = total_cans :=
by sorry

end NUMINAMATH_CALUDE_ladonnas_cans_correct_l3323_332340


namespace NUMINAMATH_CALUDE_m_range_theorem_l3323_332332

/-- Proposition p: The solution set of the inequality |x-1| > m-1 is ℝ -/
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 1| > m - 1

/-- Proposition q: f(x) = -(5-2m)x is a decreasing function -/
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → -(5 - 2*m)*x > -(5 - 2*m)*y

/-- Either p or q is true -/
def either_p_or_q (m : ℝ) : Prop := p m ∨ q m

/-- Both p and q are false propositions -/
def both_p_and_q_false (m : ℝ) : Prop := ¬(p m) ∧ ¬(q m)

/-- The range of m satisfying the given conditions -/
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m < 2

theorem m_range_theorem :
  ∀ m : ℝ, (either_p_or_q m ∧ ¬(both_p_and_q_false m)) ↔ m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3323_332332


namespace NUMINAMATH_CALUDE_x_value_on_line_k_l3323_332384

/-- A line passing through the origin with slope 1/2 -/
def line_k (x y : ℝ) : Prop := y = (1/2) * x

theorem x_value_on_line_k (x y : ℝ) :
  line_k x 6 → 
  line_k 10 y → 
  x * y = 60 →
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_value_on_line_k_l3323_332384


namespace NUMINAMATH_CALUDE_min_value_sum_fractions_l3323_332303

theorem min_value_sum_fractions (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a ≥ 9 ∧
  (∃ (x : ℝ), 0 < x → (x + x + k) / x + (x + x + k) / x + (x + x + k) / x = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_fractions_l3323_332303


namespace NUMINAMATH_CALUDE_floor_sqrt_23_squared_l3323_332372

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_23_squared_l3323_332372


namespace NUMINAMATH_CALUDE_sum_of_digits_l3323_332391

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem sum_of_digits (x y : ℕ) : 
  (x < 10) → 
  (y < 10) → 
  is_divisible_by (653 * 100 + x * 10 + y) 80 → 
  x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l3323_332391


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l3323_332347

theorem x_range_for_inequality (x : ℝ) :
  (∀ m : ℝ, -2 ≤ m ∧ m ≤ 2 → 2*x - 1 > m*(x^2 - 1)) ↔ 
  ((Real.sqrt 7 - 1) / 2 < x ∧ x < (Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l3323_332347


namespace NUMINAMATH_CALUDE_cubic_root_function_l3323_332376

/-- Given a function y = kx^(1/3) where y = 4 when x = 8, 
    prove that y = 6 when x = 27 -/
theorem cubic_root_function (k : ℝ) :
  (∃ y : ℝ, y = k * (8 : ℝ)^(1/3) ∧ y = 4) →
  k * (27 : ℝ)^(1/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_function_l3323_332376


namespace NUMINAMATH_CALUDE_sam_has_sixteen_dimes_l3323_332305

/-- The number of dimes Sam has after receiving some from his dad -/
def total_dimes (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Sam has 16 dimes after receiving some from his dad -/
theorem sam_has_sixteen_dimes : total_dimes 9 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_sixteen_dimes_l3323_332305


namespace NUMINAMATH_CALUDE_cos_n_equals_sin_712_l3323_332399

theorem cos_n_equals_sin_712 (n : ℤ) :
  -90 ≤ n ∧ n ≤ 90 ∧ Real.cos (n * π / 180) = Real.sin (712 * π / 180) → n = -82 := by
  sorry

end NUMINAMATH_CALUDE_cos_n_equals_sin_712_l3323_332399


namespace NUMINAMATH_CALUDE_oldest_babysat_age_jane_l3323_332390

/-- Represents a person with their current age and baby-sitting history. -/
structure Person where
  currentAge : ℕ
  babySittingStartAge : ℕ
  babySittingEndAge : ℕ

/-- Calculates the maximum age of a child that a person could have babysat. -/
def maxBabysatChildAge (p : Person) : ℕ :=
  p.babySittingEndAge / 2

/-- Calculates the current age of the oldest person that could have been babysat. -/
def oldestBabysatPersonCurrentAge (p : Person) : ℕ :=
  maxBabysatChildAge p + (p.currentAge - p.babySittingEndAge)

/-- Theorem stating the age of the oldest person Jane could have babysat. -/
theorem oldest_babysat_age_jane :
  let jane : Person := {
    currentAge := 32,
    babySittingStartAge := 18,
    babySittingEndAge := 20
  }
  oldestBabysatPersonCurrentAge jane = 22 := by
  sorry


end NUMINAMATH_CALUDE_oldest_babysat_age_jane_l3323_332390


namespace NUMINAMATH_CALUDE_faye_remaining_money_l3323_332300

/-- Calculates the remaining money for Faye after her purchases -/
def remaining_money (initial_money : ℚ) (cupcake_price : ℚ) (cupcake_quantity : ℕ) 
  (cookie_box_price : ℚ) (cookie_box_quantity : ℕ) : ℚ :=
  let mother_gift := 2 * initial_money
  let total_money := initial_money + mother_gift
  let cupcake_cost := cupcake_price * cupcake_quantity
  let cookie_cost := cookie_box_price * cookie_box_quantity
  let total_spent := cupcake_cost + cookie_cost
  total_money - total_spent

/-- Theorem stating that Faye's remaining money is $30 -/
theorem faye_remaining_money :
  remaining_money 20 1.5 10 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_faye_remaining_money_l3323_332300


namespace NUMINAMATH_CALUDE_height_difference_l3323_332316

theorem height_difference (height_a height_b : ℝ) :
  height_b = height_a * (1 + 66.67 / 100) →
  (height_b - height_a) / height_b * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l3323_332316


namespace NUMINAMATH_CALUDE_negation_of_existence_l3323_332320

theorem negation_of_existence (T S : Type → Prop) : 
  (¬ ∃ x, T x ∧ S x) ↔ (∀ x, T x → ¬ S x) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3323_332320


namespace NUMINAMATH_CALUDE_swap_digits_result_l3323_332398

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : tens < 10 ∧ units < 10

/-- Swaps the digits of a two-digit number -/
def swap_digits (n : TwoDigitNumber) : TwoDigitNumber where
  tens := n.units
  units := n.tens
  is_valid := by
    simp [n.is_valid]

/-- Theorem stating the result of swapping digits in the given conditions -/
theorem swap_digits_result (n : TwoDigitNumber) (h_sum : n.tens + n.units = 13) :
  ∃ a : Nat, n.units = a ∧ (swap_digits n).tens * 10 + (swap_digits n).units = 9 * a + 13 := by
  sorry

end NUMINAMATH_CALUDE_swap_digits_result_l3323_332398


namespace NUMINAMATH_CALUDE_percentage_relation_l3323_332304

theorem percentage_relation (x a b : ℝ) (ha : a = 0.06 * x) (hb : b = 0.3 * x) :
  a = 0.2 * b := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3323_332304


namespace NUMINAMATH_CALUDE_divisors_of_300_l3323_332356

/-- Given that 300 = 2 × 2 × 3 × 5 × 5, prove that 300 has 18 divisors -/
theorem divisors_of_300 : ∃ (d : Finset Nat), Finset.card d = 18 ∧ 
  (∀ x : Nat, x ∈ d ↔ (x ∣ 300)) := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_300_l3323_332356


namespace NUMINAMATH_CALUDE_field_division_fraction_l3323_332343

theorem field_division_fraction (total_area smaller_area larger_area : ℝ) : 
  total_area = 500 →
  smaller_area = 225 →
  larger_area = total_area - smaller_area →
  (larger_area - smaller_area) / ((smaller_area + larger_area) / 2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_field_division_fraction_l3323_332343


namespace NUMINAMATH_CALUDE_function_has_one_zero_l3323_332336

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 - x

theorem function_has_one_zero (a : ℝ) (h1 : |a| ≥ 1 / (2 * Real.exp 1)) 
  (h2 : ∃ x₀ : ℝ, ∀ x : ℝ, f a x ≥ f a x₀) :
  ∃! x : ℝ, f a x = 0 :=
sorry

end NUMINAMATH_CALUDE_function_has_one_zero_l3323_332336


namespace NUMINAMATH_CALUDE_six_and_neg_six_are_opposite_l3323_332341

/-- Two real numbers are opposite if one is the negative of the other -/
def are_opposite (a b : ℝ) : Prop := b = -a

/-- 6 and -6 are opposite numbers -/
theorem six_and_neg_six_are_opposite : are_opposite 6 (-6) := by
  sorry

end NUMINAMATH_CALUDE_six_and_neg_six_are_opposite_l3323_332341


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l3323_332353

/-- A triangle with special median properties -/
structure SpecialTriangle where
  /-- The length of side EF -/
  EF : ℝ
  /-- The length of side DF -/
  DF : ℝ
  /-- The median from D is perpendicular to the median from E -/
  medians_perpendicular : Bool

/-- Theorem: In a special triangle with EF = 10, DF = 8, and perpendicular medians, DE = 18 -/
theorem special_triangle_side_length (t : SpecialTriangle) 
  (h1 : t.EF = 10) 
  (h2 : t.DF = 8) 
  (h3 : t.medians_perpendicular = true) : 
  ∃ DE : ℝ, DE = 18 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_side_length_l3323_332353


namespace NUMINAMATH_CALUDE_student_increase_proof_l3323_332374

/-- Represents the increase in the number of students in a hostel -/
def student_increase : ℕ := sorry

/-- The initial number of students in the hostel -/
def initial_students : ℕ := 35

/-- The original daily expenditure of the mess in rupees -/
def original_expenditure : ℕ := 420

/-- The increase in daily mess expenses in rupees when the number of students increases -/
def expense_increase : ℕ := 42

/-- The decrease in average expenditure per student in rupees when the number of students increases -/
def average_expense_decrease : ℕ := 1

/-- Calculates the new total expenditure after the increase in students -/
def new_total_expenditure : ℕ := (initial_students + student_increase) * 
  (original_expenditure / initial_students - average_expense_decrease)

theorem student_increase_proof : 
  new_total_expenditure = original_expenditure + expense_increase ∧ 
  student_increase = 7 := by sorry

end NUMINAMATH_CALUDE_student_increase_proof_l3323_332374


namespace NUMINAMATH_CALUDE_polygon_sides_l3323_332392

theorem polygon_sides (n : ℕ) : 
  (n ≥ 3) →  -- Ensure it's a valid polygon
  (180 * (n - 2) : ℝ) / 360 = 5 / 2 → 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3323_332392


namespace NUMINAMATH_CALUDE_customers_left_l3323_332348

/-- Given a waiter with an initial number of customers and a number of remaining customers,
    prove that the number of customers who left is the difference between the initial and remaining customers. -/
theorem customers_left (initial remaining : ℕ) (h1 : initial = 21) (h2 : remaining = 12) :
  initial - remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_customers_left_l3323_332348


namespace NUMINAMATH_CALUDE_math_and_lang_not_science_l3323_332388

def students : ℕ := 120
def math_students : ℕ := 80
def lang_students : ℕ := 70
def science_students : ℕ := 50
def all_three_students : ℕ := 20

theorem math_and_lang_not_science :
  ∃ (math_and_lang math_and_science lang_and_science : ℕ),
    math_and_lang + math_and_science + lang_and_science = 
      math_students + lang_students + science_students - students + all_three_students ∧
    math_and_lang - all_three_students = 30 := by
  sorry

end NUMINAMATH_CALUDE_math_and_lang_not_science_l3323_332388


namespace NUMINAMATH_CALUDE_simplify_power_expression_l3323_332389

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l3323_332389


namespace NUMINAMATH_CALUDE_jewelry_pattern_purple_beads_jewelry_pattern_purple_beads_proof_l3323_332364

theorem jewelry_pattern_purple_beads : ℕ → Prop :=
  fun purple_beads =>
    let green_beads : ℕ := 3
    let red_beads : ℕ := 2 * green_beads
    let pattern_total : ℕ := green_beads + purple_beads + red_beads
    let bracelet_repeats : ℕ := 3
    let necklace_repeats : ℕ := 5
    let bracelet_beads : ℕ := bracelet_repeats * pattern_total
    let necklace_beads : ℕ := necklace_repeats * pattern_total
    let total_beads : ℕ := 742
    let num_bracelets : ℕ := 1
    let num_necklaces : ℕ := 10
    num_bracelets * bracelet_beads + num_necklaces * necklace_beads = total_beads →
    purple_beads = 5

-- Proof
theorem jewelry_pattern_purple_beads_proof : jewelry_pattern_purple_beads 5 := by
  sorry

end NUMINAMATH_CALUDE_jewelry_pattern_purple_beads_jewelry_pattern_purple_beads_proof_l3323_332364


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3323_332333

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ = -n² + 4n,
    prove that the common difference d is -2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : ∀ n, S n = -n^2 + 4*n)  -- The given sum formula
  (h2 : ∀ n, S (n+1) - S n = a (n+1))  -- Definition of sum function
  (h3 : ∀ n, a (n+1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3323_332333


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l3323_332373

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x < 3 ∧ x ≠ 2 ∧ (1 - x) / (x - 2) = m / (2 - x) - 2) → 
  m < 6 ∧ m ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l3323_332373


namespace NUMINAMATH_CALUDE_range_of_a_l3323_332330

theorem range_of_a (a : ℝ) : Real.sqrt (a^2) = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3323_332330


namespace NUMINAMATH_CALUDE_triangle_properties_l3323_332377

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - t.a^2 = t.b * t.c) 
  (h2 : t.a = Real.sqrt 3)
  (h3 : Real.cos t.C = Real.sqrt 3 / 3) :
  (t.A = π / 3) ∧ (t.c = 2 * Real.sqrt 6 / 3) := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3323_332377


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3323_332317

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_formula 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_q : ∃ q : ℝ, q ∈ Set.Ioo 0 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n)
  (h_sum : a 1 * a 5 + 2 * a 3 * a 5 + a 2 * a 8 = 25)
  (h_mean : Real.sqrt (a 3 * a 5) = 2) :
  ∀ n : ℕ, a n = 2^(5 - n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3323_332317


namespace NUMINAMATH_CALUDE_money_distribution_l3323_332301

/-- Given three people A, B, and C with money, prove that A and C together have 200 Rs. -/
theorem money_distribution (A B C : ℕ) : 
  A + B + C = 450 →
  B + C = 350 →
  C = 100 →
  A + C = 200 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l3323_332301


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_9_with_digit_sum_27_l3323_332346

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem smallest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ digit_sum n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ digit_sum m = 27 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_9_with_digit_sum_27_l3323_332346


namespace NUMINAMATH_CALUDE_batsman_average_excluding_extremes_l3323_332338

def batting_average : ℝ := 60
def num_innings : ℕ := 46
def highest_score : ℕ := 194
def score_difference : ℕ := 180

theorem batsman_average_excluding_extremes :
  let total_runs : ℝ := batting_average * num_innings
  let lowest_score : ℕ := highest_score - score_difference
  let runs_excluding_extremes : ℝ := total_runs - highest_score - lowest_score
  let innings_excluding_extremes : ℕ := num_innings - 2
  (runs_excluding_extremes / innings_excluding_extremes : ℝ) = 58 := by sorry

end NUMINAMATH_CALUDE_batsman_average_excluding_extremes_l3323_332338


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l3323_332369

theorem modular_inverse_of_5_mod_23 :
  ∃ x : ℕ, x < 23 ∧ (5 * x) % 23 = 1 ∧ x = 14 := by
sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l3323_332369


namespace NUMINAMATH_CALUDE_price_change_l3323_332351

/-- Theorem: Price change after 50% decrease and 60% increase --/
theorem price_change (P : ℝ) (P_pos : P > 0) :
  P * (1 - 0.5) * (1 + 0.6) = P * 0.8 := by
  sorry

#check price_change

end NUMINAMATH_CALUDE_price_change_l3323_332351


namespace NUMINAMATH_CALUDE_blackboard_divisibility_l3323_332362

/-- Represents the transformation process on the blackboard -/
def transform (n : ℕ) : ℕ := sorry

/-- The number on the blackboard after n minutes -/
def blackboard_number (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | n+1 => transform (blackboard_number n)

/-- The final number N on the blackboard -/
def N : ℕ := blackboard_number (sorry : ℕ)

theorem blackboard_divisibility :
  (9 ∣ N) → (99 ∣ N) := by sorry

end NUMINAMATH_CALUDE_blackboard_divisibility_l3323_332362


namespace NUMINAMATH_CALUDE_seeds_solution_l3323_332361

def seeds_problem (wednesday thursday friday : ℕ) : Prop :=
  wednesday = 5 * thursday ∧
  wednesday + thursday = 156 ∧
  friday = 4

theorem seeds_solution :
  ∃ (wednesday thursday friday : ℕ),
    seeds_problem wednesday thursday friday ∧
    wednesday = 130 ∧
    thursday = 26 ∧
    friday = 4 ∧
    wednesday + thursday + friday = 160 := by
  sorry

end NUMINAMATH_CALUDE_seeds_solution_l3323_332361


namespace NUMINAMATH_CALUDE_real_root_range_l3323_332323

theorem real_root_range (a : ℝ) : 
  (∃ x : ℝ, (2 : ℝ)^(2*x) + (2 : ℝ)^x * a + a + 1 = 0) → 
  a ≤ 2 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_real_root_range_l3323_332323


namespace NUMINAMATH_CALUDE_lavender_bouquet_cost_l3323_332334

/-- The cost of a bouquet is directly proportional to the number of lavenders it contains. -/
def is_proportional (cost : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, n ≠ 0 → m ≠ 0 → cost n / n = cost m / m

/-- Given that a bouquet of 15 lavenders costs $25 and the price is directly proportional
    to the number of lavenders, prove that a bouquet of 50 lavenders costs $250/3. -/
theorem lavender_bouquet_cost (cost : ℕ → ℚ)
    (h_prop : is_proportional cost)
    (h_15 : cost 15 = 25) :
    cost 50 = 250 / 3 := by
  sorry

end NUMINAMATH_CALUDE_lavender_bouquet_cost_l3323_332334


namespace NUMINAMATH_CALUDE_student_task_assignment_l3323_332396

/-- The number of ways to assign students to tasks under specific conditions -/
def assignment_count (n : ℕ) (m : ℕ) (k : ℕ) : ℕ :=
  Nat.choose k 1 * Nat.choose m 2 * (n - 1)^(n - 1) + Nat.choose k 2 * (n - 1)^(n - 1)

/-- Theorem stating the number of ways to assign 5 students to 4 tasks under given conditions -/
theorem student_task_assignment :
  assignment_count 4 4 3 = Nat.choose 3 1 * Nat.choose 4 2 * 3^3 + Nat.choose 3 2 * 3^3 :=
by sorry

end NUMINAMATH_CALUDE_student_task_assignment_l3323_332396


namespace NUMINAMATH_CALUDE_common_factor_is_gcf_l3323_332309

-- Define the polynomial terms
def term1 (x y : ℤ) : ℤ := 7 * x^2 * y
def term2 (x y : ℤ) : ℤ := 21 * x * y^2

-- Define the common factor
def common_factor (x y : ℤ) : ℤ := 7 * x * y

-- Theorem statement
theorem common_factor_is_gcf :
  ∀ (x y : ℤ), 
    (∃ (a b : ℤ), term1 x y = common_factor x y * a ∧ term2 x y = common_factor x y * b) ∧
    (∀ (z : ℤ), (∃ (c d : ℤ), term1 x y = z * c ∧ term2 x y = z * d) → z ∣ common_factor x y) :=
sorry

end NUMINAMATH_CALUDE_common_factor_is_gcf_l3323_332309


namespace NUMINAMATH_CALUDE_apple_pear_basket_weights_l3323_332350

/-- Given the conditions of the apple and pear basket problem, prove the weights of individual baskets. -/
theorem apple_pear_basket_weights :
  ∀ (apple_weight pear_weight : ℕ),
  -- Total weight of all baskets is 692 kg
  12 * apple_weight + 14 * pear_weight = 692 →
  -- Weight of pear basket is 10 kg less than apple basket
  pear_weight = apple_weight - 10 →
  -- Prove that apple_weight is 32 kg and pear_weight is 22 kg
  apple_weight = 32 ∧ pear_weight = 22 := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_basket_weights_l3323_332350


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l3323_332370

theorem cubic_roots_sum_squares (p q r : ℝ) : 
  (p + q + r = 15) → (p * q + q * r + r * p = 25) → 
  (p^3 - 15*p^2 + 25*p - 10 = 0) → 
  (q^3 - 15*q^2 + 25*q - 10 = 0) → 
  (r^3 - 15*r^2 + 25*r - 10 = 0) → 
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l3323_332370


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_with_negation_greater_than_neg_200_l3323_332315

theorem largest_multiple_of_8_with_negation_greater_than_neg_200 :
  ∃ (n : ℤ), n = 192 ∧ 
  (∀ (m : ℤ), m % 8 = 0 ∧ -m > -200 → m ≤ n) ∧
  192 % 8 = 0 ∧
  -192 > -200 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_with_negation_greater_than_neg_200_l3323_332315


namespace NUMINAMATH_CALUDE_megan_museum_pictures_l3323_332318

/-- Represents the number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- Represents the number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- Represents the number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- Represents the number of pictures Megan had left after deleting -/
def remaining_pictures : ℕ := 2

theorem megan_museum_pictures :
  zoo_pictures + museum_pictures = remaining_pictures + deleted_pictures :=
by sorry

end NUMINAMATH_CALUDE_megan_museum_pictures_l3323_332318


namespace NUMINAMATH_CALUDE_equal_segments_iff_proportion_l3323_332325

/-- A triangle with side lengths a, b, and c where a ≤ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_le_c : a ≤ c

/-- The internal bisector of the incenter divides the median from point B into three equal segments -/
def has_equal_segments (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ 
    let m := (t.a^2 + t.c^2 - t.b^2/2) / 2
    (3*x)^2 = m ∧
    ((t.a + t.c - t.b)/2)^2 = 2*x^2 ∧
    ((t.c - t.a)/2)^2 = 2*x^2

/-- The side lengths satisfy the proportion a/5 = b/10 = c/13 -/
def satisfies_proportion (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 5*k ∧ t.b = 10*k ∧ t.c = 13*k

theorem equal_segments_iff_proportion (t : Triangle) :
  has_equal_segments t ↔ satisfies_proportion t := by
  sorry

end NUMINAMATH_CALUDE_equal_segments_iff_proportion_l3323_332325


namespace NUMINAMATH_CALUDE_special_number_divisibility_l3323_332329

/-- Represents a 4-digit number with the given properties -/
structure SpecialNumber where
  value : Nat
  is_four_digit : value ≥ 1000 ∧ value < 10000
  has_three_unique_digits : ∃ (a b c : Nat), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ((value / 1000 = a ∧ (value / 100) % 10 = a) ∨
     (value / 1000 = a ∧ (value / 10) % 10 = a) ∨
     (value / 1000 = a ∧ value % 10 = a) ∨
     ((value / 100) % 10 = a ∧ (value / 10) % 10 = a) ∨
     ((value / 100) % 10 = a ∧ value % 10 = a) ∨
     ((value / 10) % 10 = a ∧ value % 10 = a)) ∧
    value = a * 1000 + b * 100 + c * 10 + (if value / 1000 = a then b else a)

/-- Mrs. Smith's age is the last two digits of the special number -/
def mrs_smith_age (n : SpecialNumber) : Nat := n.value % 100

/-- The ages of Mrs. Smith's children -/
def children_ages : Finset Nat := Finset.range 12 \ {0}

theorem special_number_divisibility (n : SpecialNumber) :
  ∃ (x : Nat), x ∈ children_ages ∧ ¬(n.value % x = 0) ∧
  ∀ (y : Nat), y ∈ children_ages ∧ y ≠ x → n.value % y = 0 →
  x = 3 := by sorry

#check special_number_divisibility

end NUMINAMATH_CALUDE_special_number_divisibility_l3323_332329


namespace NUMINAMATH_CALUDE_total_cards_is_690_l3323_332368

/-- The number of get well cards Mariela received in the hospital -/
def cards_in_hospital : ℕ := 403

/-- The number of get well cards Mariela received at home -/
def cards_at_home : ℕ := 287

/-- The total number of get well cards Mariela received -/
def total_cards : ℕ := cards_in_hospital + cards_at_home

/-- Theorem stating that the total number of get well cards Mariela received is 690 -/
theorem total_cards_is_690 : total_cards = 690 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_is_690_l3323_332368


namespace NUMINAMATH_CALUDE_power_inequality_l3323_332385

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ a^b * b^a :=
by sorry

end NUMINAMATH_CALUDE_power_inequality_l3323_332385


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l3323_332344

theorem inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l3323_332344


namespace NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l3323_332314

/-- S(n) is defined as n minus the largest perfect square less than or equal to n -/
def S (n : ℕ) : ℕ := n - (Nat.sqrt n) ^ 2

/-- The sequence a_n is defined recursively -/
def a (A : ℕ) : ℕ → ℕ
  | 0 => A
  | n + 1 => a A n + S (a A n)

/-- A non-negative integer is a perfect square if it's equal to some integer squared -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 2

/-- The main theorem: the sequence becomes constant iff A is a perfect square -/
theorem sequence_constant_iff_perfect_square (A : ℕ) :
  (∃ N : ℕ, ∀ n ≥ N, a A n = a A N) ↔ is_perfect_square A := by
  sorry

end NUMINAMATH_CALUDE_sequence_constant_iff_perfect_square_l3323_332314


namespace NUMINAMATH_CALUDE_circle_radius_from_polar_l3323_332324

/-- The radius of a circle defined by the polar equation ρ = 6cosθ is 3 -/
theorem circle_radius_from_polar (θ : ℝ) :
  let ρ := 6 * Real.cos θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    radius = 3 ∧
    ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔
      x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_polar_l3323_332324


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3323_332397

theorem cubic_equation_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_solution : ∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = a ∨ x = -b ∨ x = c) :
  a = 1 ∧ b = -1 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3323_332397


namespace NUMINAMATH_CALUDE_area_is_nine_halves_l3323_332365

/-- The line in the Cartesian coordinate system -/
def line (x y : ℝ) : Prop := x - y = 0

/-- The curve in the Cartesian coordinate system -/
def curve (x y : ℝ) : Prop := y = x^2 - 2*x

/-- The area enclosed by the line and the curve -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is equal to 9/2 -/
theorem area_is_nine_halves : enclosed_area = 9/2 := by sorry

end NUMINAMATH_CALUDE_area_is_nine_halves_l3323_332365


namespace NUMINAMATH_CALUDE_concatenated_square_exists_l3323_332352

theorem concatenated_square_exists : ∃ (A : ℕ), ∃ (n : ℕ), ∃ (B : ℕ), 
  (10^n + 1) * A = B^2 ∧ A > 0 ∧ A < 10^n := by
  sorry

end NUMINAMATH_CALUDE_concatenated_square_exists_l3323_332352


namespace NUMINAMATH_CALUDE_square_value_l3323_332311

theorem square_value (square q : ℤ) 
  (eq1 : square + q = 74)
  (eq2 : square + 2 * q ^ 2 = 180) : 
  square = 66 := by sorry

end NUMINAMATH_CALUDE_square_value_l3323_332311


namespace NUMINAMATH_CALUDE_radius_C₁_is_sqrt_30_l3323_332308

/-- Two circles C₁ and C₂ with the following properties:
    1. The center O of C₁ lies on C₂
    2. C₁ and C₂ intersect at points X and Y
    3. There exists a point Z on C₂ exterior to C₁
    4. XZ = 13, OZ = 11, YZ = 7 -/
structure TwoCircles where
  O : ℝ × ℝ  -- Center of C₁
  X : ℝ × ℝ  -- Intersection point
  Y : ℝ × ℝ  -- Intersection point
  Z : ℝ × ℝ  -- Point on C₂ exterior to C₁
  C₁ : Set (ℝ × ℝ)  -- Circle C₁
  C₂ : Set (ℝ × ℝ)  -- Circle C₂
  O_on_C₂ : O ∈ C₂
  X_on_both : X ∈ C₁ ∧ X ∈ C₂
  Y_on_both : Y ∈ C₁ ∧ Y ∈ C₂
  Z_on_C₂ : Z ∈ C₂
  Z_exterior_C₁ : Z ∉ C₁
  XZ_length : dist X Z = 13
  OZ_length : dist O Z = 11
  YZ_length : dist Y Z = 7

/-- The radius of C₁ is √30 -/
theorem radius_C₁_is_sqrt_30 (tc : TwoCircles) : 
  ∃ (center : ℝ × ℝ) (r : ℝ), tc.C₁ = {p : ℝ × ℝ | dist p center = r} ∧ r = Real.sqrt 30 :=
sorry

end NUMINAMATH_CALUDE_radius_C₁_is_sqrt_30_l3323_332308


namespace NUMINAMATH_CALUDE_park_bushes_count_l3323_332363

def park_bushes (initial_orchids initial_roses initial_tulips added_orchids removed_roses : ℕ) : ℕ × ℕ × ℕ :=
  let final_orchids := initial_orchids + added_orchids
  let final_roses := initial_roses - removed_roses
  let final_tulips := initial_tulips * 2
  (final_orchids, final_roses, final_tulips)

theorem park_bushes_count : park_bushes 2 5 3 4 1 = (6, 4, 6) := by sorry

end NUMINAMATH_CALUDE_park_bushes_count_l3323_332363


namespace NUMINAMATH_CALUDE_athlete_distance_difference_l3323_332328

theorem athlete_distance_difference : 
  let field_length : ℚ := 24
  let mary_fraction : ℚ := 3/8
  let edna_fraction : ℚ := 2/3
  let lucy_fraction : ℚ := 5/6
  let mary_distance : ℚ := field_length * mary_fraction
  let edna_distance : ℚ := mary_distance * edna_fraction
  let lucy_distance : ℚ := edna_distance * lucy_fraction
  mary_distance - lucy_distance = 4 := by
sorry

end NUMINAMATH_CALUDE_athlete_distance_difference_l3323_332328


namespace NUMINAMATH_CALUDE_water_tank_evaporation_l3323_332302

/-- Calculates the remaining water in a tank after evaporation --/
def remaining_water (initial_amount : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_amount - evaporation_rate * days

/-- Proves that 450 gallons remain after 50 days of evaporation --/
theorem water_tank_evaporation :
  remaining_water 500 1 50 = 450 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_evaporation_l3323_332302


namespace NUMINAMATH_CALUDE_lily_pad_coverage_time_l3323_332387

def days_to_half_coverage : ℕ := 57

theorem lily_pad_coverage_time :
  ∀ (total_coverage : ℝ) (daily_growth_factor : ℝ),
    total_coverage > 0 →
    daily_growth_factor = 2 →
    (daily_growth_factor ^ days_to_half_coverage : ℝ) * (1 / 2) = 1 →
    (daily_growth_factor ^ (days_to_half_coverage + 1) : ℝ) = total_coverage :=
by sorry

end NUMINAMATH_CALUDE_lily_pad_coverage_time_l3323_332387


namespace NUMINAMATH_CALUDE_rachel_book_count_l3323_332319

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves with mystery books -/
def mystery_shelves : ℕ := 6

/-- The number of shelves with picture books -/
def picture_shelves : ℕ := 2

/-- The total number of books Rachel has -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem rachel_book_count : total_books = 72 := by
  sorry

end NUMINAMATH_CALUDE_rachel_book_count_l3323_332319


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l3323_332339

-- Define the repeating decimal 0.080808...
def repeating_08 : ℚ := 8 / 99

-- Define the repeating decimal 0.333333...
def repeating_3 : ℚ := 1 / 3

-- Theorem statement
theorem product_of_repeating_decimals : 
  repeating_08 * repeating_3 = 8 / 297 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l3323_332339


namespace NUMINAMATH_CALUDE_rod_lengths_at_zero_celsius_l3323_332359

/-- Theorem: Rod Lengths at 0°C
Given:
- Total length at 0°C is 1 m
- Total length at 100°C is 1.0024 m
- Coefficient of linear expansion for steel is 0.000011
- Coefficient of linear expansion for zinc is 0.000031

Prove:
- Length of steel rod at 0°C is 0.35 m
- Length of zinc rod at 0°C is 0.65 m
-/
theorem rod_lengths_at_zero_celsius 
  (total_length_zero : Real) 
  (total_length_hundred : Real)
  (steel_expansion : Real)
  (zinc_expansion : Real)
  (h1 : total_length_zero = 1)
  (h2 : total_length_hundred = 1.0024)
  (h3 : steel_expansion = 0.000011)
  (h4 : zinc_expansion = 0.000031) :
  ∃ (steel_length zinc_length : Real),
    steel_length = 0.35 ∧ 
    zinc_length = 0.65 ∧
    steel_length + zinc_length = total_length_zero ∧
    steel_length * (1 + 100 * steel_expansion) + 
    zinc_length * (1 + 100 * zinc_expansion) = total_length_hundred :=
by sorry

end NUMINAMATH_CALUDE_rod_lengths_at_zero_celsius_l3323_332359


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3323_332383

/-- Given an arithmetic sequence {a_n} where a_3 + a_4 + a_5 = 12, 
    the sum of the first seven terms is 28. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 3 + a 4 + a 5 = 12 →                    -- given condition
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3323_332383


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3323_332375

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 3) ∧ (x = 47/9) ∧ (y = 17/3) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3323_332375


namespace NUMINAMATH_CALUDE_inequality_solution_l3323_332307

theorem inequality_solution (x : ℝ) :
  x > 2 →
  (((x - 2) ^ (x^2 - 6*x + 8)) > 1) ↔ (x > 2 ∧ x < 3) ∨ x > 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3323_332307


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l3323_332310

-- Define the properties of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : is_decreasing_on_nonneg f) : 
  f 1 > f (-2) ∧ f (-2) > f 3 :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l3323_332310


namespace NUMINAMATH_CALUDE_figure_area_is_79_l3323_332360

/-- Calculates the area of a rectangle -/
def rectangleArea (width : ℕ) (height : ℕ) : ℕ := width * height

/-- Represents the dimensions of the figure -/
structure FigureDimensions where
  leftWidth : ℕ
  leftHeight : ℕ
  middleWidth : ℕ
  middleHeight : ℕ
  rightWidth : ℕ
  rightHeight : ℕ

/-- Calculates the total area of the figure -/
def totalArea (d : FigureDimensions) : ℕ :=
  rectangleArea d.leftWidth d.leftHeight +
  rectangleArea d.middleWidth d.middleHeight +
  rectangleArea d.rightWidth d.rightHeight

/-- Theorem: The total area of the figure is 79 square units -/
theorem figure_area_is_79 (d : FigureDimensions) 
  (h1 : d.leftWidth = 6 ∧ d.leftHeight = 7)
  (h2 : d.middleWidth = 4 ∧ d.middleHeight = 3)
  (h3 : d.rightWidth = 5 ∧ d.rightHeight = 5) :
  totalArea d = 79 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_is_79_l3323_332360


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3323_332393

theorem absolute_value_equation_solution :
  ∃! n : ℝ, |n + 6| = 2 - n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3323_332393


namespace NUMINAMATH_CALUDE_emily_earnings_theorem_l3323_332322

/-- The amount of money Emily makes by selling chocolate bars -/
def emily_earnings (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Theorem: Emily makes $77 by selling all but 4 bars from a box of 15 bars costing $7 each -/
theorem emily_earnings_theorem : emily_earnings 15 7 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_emily_earnings_theorem_l3323_332322


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l3323_332357

theorem smallest_four_digit_mod_8 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 8 = 3 → n ≥ 1003 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l3323_332357
