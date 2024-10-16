import Mathlib

namespace NUMINAMATH_CALUDE_count_even_perfect_square_factors_l394_39478

/-- The number of even perfect square factors of 2^6 * 7^9 * 3^2 -/
def evenPerfectSquareFactors : ℕ :=
  30

/-- The exponent of 2 in the given number -/
def exponent2 : ℕ := 6

/-- The exponent of 7 in the given number -/
def exponent7 : ℕ := 9

/-- The exponent of 3 in the given number -/
def exponent3 : ℕ := 2

/-- Theorem stating that the number of even perfect square factors
    of 2^6 * 7^9 * 3^2 is equal to 30 -/
theorem count_even_perfect_square_factors :
  (∀ a b c : ℕ,
    0 ≤ a ∧ a ≤ exponent2 ∧
    0 ≤ b ∧ b ≤ exponent7 ∧
    0 ≤ c ∧ c ≤ exponent3 ∧
    Even a ∧ Even b ∧ Even c ∧
    a ≥ 1) →
  evenPerfectSquareFactors = 30 := by
  sorry

end NUMINAMATH_CALUDE_count_even_perfect_square_factors_l394_39478


namespace NUMINAMATH_CALUDE_cylinder_radius_with_prisms_l394_39420

theorem cylinder_radius_with_prisms (h₁ h₂ d : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (d_pos : d > 0) 
  (h₁_eq : h₁ = 9) (h₂_eq : h₂ = 2) (d_eq : d = 23) : ∃ R : ℝ, 
  R > 0 ∧ 
  R^2 = (R - h₁)^2 + (d - x)^2 ∧ 
  R^2 = (R - h₂)^2 + x^2 ∧ 
  R = 17 :=
sorry

end NUMINAMATH_CALUDE_cylinder_radius_with_prisms_l394_39420


namespace NUMINAMATH_CALUDE_isosceles_triangle_parallel_lines_l394_39460

theorem isosceles_triangle_parallel_lines (base : ℝ) (line1 line2 : ℝ) : 
  base = 20 →
  line2 > line1 →
  line1 * line1 = (1/3) * base * base →
  line2 * line2 = (2/3) * base * base →
  line2 - line1 = (20 * (Real.sqrt 6 - Real.sqrt 3)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_parallel_lines_l394_39460


namespace NUMINAMATH_CALUDE_hyperbola_focus_coordinates_l394_39468

/-- Given a hyperbola with equation (x-5)^2/7^2 - (y-10)^2/15^2 = 1, 
    the focus with the larger x-coordinate has coordinates (5 + √274, 10) -/
theorem hyperbola_focus_coordinates (x y : ℝ) : 
  ((x - 5)^2 / 7^2) - ((y - 10)^2 / 15^2) = 1 →
  ∃ (f_x f_y : ℝ), f_x > 5 ∧ f_y = 10 ∧ 
  f_x = 5 + Real.sqrt 274 ∧
  ((f_x - 5)^2 / 7^2) - ((f_y - 10)^2 / 15^2) = 1 ∧
  ∀ (x' y' : ℝ), x' > 5 ∧ 
    ((x' - 5)^2 / 7^2) - ((y' - 10)^2 / 15^2) = 1 →
    x' ≤ f_x :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_coordinates_l394_39468


namespace NUMINAMATH_CALUDE_stating_largest_valid_n_l394_39431

/-- 
Given a positive integer n, this function checks if n! can be expressed as 
the product of n - 4 consecutive positive integers.
-/
def is_valid (n : ℕ) : Prop :=
  ∃ (b : ℕ), b ≥ 4 ∧ n.factorial = ((n - 4 + b).factorial / b.factorial)

/-- 
Theorem stating that 119 is the largest positive integer n for which n! 
can be expressed as the product of n - 4 consecutive positive integers.
-/
theorem largest_valid_n : 
  (is_valid 119) ∧ (∀ m : ℕ, m > 119 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_stating_largest_valid_n_l394_39431


namespace NUMINAMATH_CALUDE_whitewash_fence_l394_39496

theorem whitewash_fence (k : ℕ) : 
  ∀ (x y : Fin (2^(k+1))), 
    (∃ (z : Fin (2^(k+1))), z ≠ x ∧ z ≠ y ∧ 
      (2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (y.val^2 + 3*y.val - 2)) ↔ 
       2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (z.val^2 + 3*z.val - 2)))) ∧
    (∀ (w : Fin (2^(k+1))), 
      2^(k+1) ∣ ((x.val^2 + 3*x.val - 2) - (w.val^2 + 3*w.val - 2)) → 
      w = x ∨ w = y) :=
by sorry

#check whitewash_fence

end NUMINAMATH_CALUDE_whitewash_fence_l394_39496


namespace NUMINAMATH_CALUDE_sum_of_roots_for_f_l394_39439

theorem sum_of_roots_for_f (f : ℝ → ℝ) : 
  (∀ x, f (x / 4) = x^2 + 3*x + 2) →
  (∃ z₁ z₂, f (4*z₁) = 8 ∧ f (4*z₂) = 8 ∧ z₁ ≠ z₂ ∧ 
    (∀ z, f (4*z) = 8 → z = z₁ ∨ z = z₂) ∧
    z₁ + z₂ = -3/16) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_for_f_l394_39439


namespace NUMINAMATH_CALUDE_solve_equation_l394_39415

theorem solve_equation (a b c : ℤ) (x : ℝ) (h : 5 / (a^2 + b * Real.log x) = c) :
  x = 10 ^ ((5 / c - a^2) / b) :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_l394_39415


namespace NUMINAMATH_CALUDE_least_common_multiple_18_35_l394_39450

theorem least_common_multiple_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_18_35_l394_39450


namespace NUMINAMATH_CALUDE_percentage_problem_l394_39495

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 700 → 
  0.3 * N = (P / 100) * 150 + 120 → 
  P = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l394_39495


namespace NUMINAMATH_CALUDE_movie_ticket_final_price_l394_39497

def movie_ticket_price (initial_price : ℝ) (year1_increase year2_decrease year3_increase year4_decrease year5_increase tax discount : ℝ) : ℝ :=
  let price1 := initial_price * (1 + year1_increase)
  let price2 := price1 * (1 - year2_decrease)
  let price3 := price2 * (1 + year3_increase)
  let price4 := price3 * (1 - year4_decrease)
  let price5 := price4 * (1 + year5_increase)
  let price_with_tax := price5 * (1 + tax)
  price_with_tax * (1 - discount)

theorem movie_ticket_final_price :
  let initial_price : ℝ := 100
  let year1_increase : ℝ := 0.12
  let year2_decrease : ℝ := 0.05
  let year3_increase : ℝ := 0.08
  let year4_decrease : ℝ := 0.04
  let year5_increase : ℝ := 0.06
  let tax : ℝ := 0.07
  let discount : ℝ := 0.10
  ∃ ε > 0, |movie_ticket_price initial_price year1_increase year2_decrease year3_increase year4_decrease year5_increase tax discount - 112.61| < ε :=
sorry

end NUMINAMATH_CALUDE_movie_ticket_final_price_l394_39497


namespace NUMINAMATH_CALUDE_solve_system_l394_39480

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l394_39480


namespace NUMINAMATH_CALUDE_quadratic_polynomial_coefficients_l394_39406

theorem quadratic_polynomial_coefficients (m n : ℚ) : 
  (∀ x, (x^2 + m*x + n) % (x - m) = m ∧ (x^2 + m*x + n) % (x - n) = n) → 
  ((m = 0 ∧ n = 0) ∨ (m = 1/2 ∧ n = 0) ∨ (m = 1 ∧ n = -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_coefficients_l394_39406


namespace NUMINAMATH_CALUDE_correct_problem_percentage_l394_39452

/-- Given a total number of problems and the number of missed problems,
    calculate the percentage of correctly solved problems. -/
theorem correct_problem_percentage
  (x : ℕ) -- x represents the number of missed problems
  (h : x > 0) -- ensure x is positive to avoid division by zero
  : (((7 : ℚ) * x - x) / (7 * x)) * 100 = (6 : ℚ) / 7 * 100 := by
  sorry

#eval (6 : ℚ) / 7 * 100 -- To show the approximate result

end NUMINAMATH_CALUDE_correct_problem_percentage_l394_39452


namespace NUMINAMATH_CALUDE_henry_age_is_27_l394_39434

/-- Henry's present age -/
def henry_age : ℕ := sorry

/-- Jill's present age -/
def jill_age : ℕ := sorry

/-- The sum of Henry and Jill's present ages is 43 -/
axiom sum_of_ages : henry_age + jill_age = 43

/-- 5 years ago, Henry was twice the age of Jill -/
axiom age_relation : henry_age - 5 = 2 * (jill_age - 5)

/-- Theorem: Henry's present age is 27 years -/
theorem henry_age_is_27 : henry_age = 27 := by sorry

end NUMINAMATH_CALUDE_henry_age_is_27_l394_39434


namespace NUMINAMATH_CALUDE_solve_steak_problem_l394_39453

def steak_problem (cost_per_pound change_received : ℕ) : Prop :=
  let amount_paid : ℕ := 20
  let amount_spent : ℕ := amount_paid - change_received
  let pounds_bought : ℕ := amount_spent / cost_per_pound
  (cost_per_pound = 7 ∧ change_received = 6) → pounds_bought = 2

theorem solve_steak_problem :
  ∀ (cost_per_pound change_received : ℕ),
    steak_problem cost_per_pound change_received :=
by
  sorry

end NUMINAMATH_CALUDE_solve_steak_problem_l394_39453


namespace NUMINAMATH_CALUDE_ends_in_zero_l394_39482

theorem ends_in_zero (a : ℤ) (n : ℕ) (h : n ≥ 2) :
  ∃ k : ℤ, a^(2^n + 1) - a = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_ends_in_zero_l394_39482


namespace NUMINAMATH_CALUDE_inequality_proof_l394_39442

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / Real.sqrt (y + z)) + (y / Real.sqrt (z + x)) + (z / Real.sqrt (x + y)) ≥ Real.sqrt ((3 / 2) * (x + y + z)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l394_39442


namespace NUMINAMATH_CALUDE_probability_adjacent_ascending_five_cds_l394_39462

/-- The probability of two specific CDs being adjacent in ascending order when n CDs are randomly arranged -/
def probability_adjacent_ascending (n : ℕ) : ℚ :=
  if n ≥ 2 then (4 * (n - 2).factorial) / n.factorial else 0

/-- Theorem: The probability of CDs 1 and 2 being next to each other in ascending order 
    when 5 CDs are randomly placed in a cassette holder is 1/5 -/
theorem probability_adjacent_ascending_five_cds : 
  probability_adjacent_ascending 5 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_adjacent_ascending_five_cds_l394_39462


namespace NUMINAMATH_CALUDE_angle_c_is_right_angle_l394_39419

theorem angle_c_is_right_angle 
  (A B C : ℝ) 
  (triangle_condition : A + B + C = Real.pi)
  (condition1 : Real.sin A + Real.cos B = Real.sqrt 2)
  (condition2 : Real.cos A + Real.sin B = Real.sqrt 2) : 
  C = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_is_right_angle_l394_39419


namespace NUMINAMATH_CALUDE_x_value_l394_39464

theorem x_value : ∃ x : ℚ, (3 * x + 4) / 5 = 15 ∧ x = 71 / 3 := by sorry

end NUMINAMATH_CALUDE_x_value_l394_39464


namespace NUMINAMATH_CALUDE_vector_addition_l394_39465

theorem vector_addition (a b : ℝ × ℝ) : 
  a = (-1, 6) → b = (3, -2) → a + b = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l394_39465


namespace NUMINAMATH_CALUDE_num_divisors_2310_l394_39448

/-- The number of positive divisors of 2310 is 32 -/
theorem num_divisors_2310 : Nat.card (Nat.divisors 2310) = 32 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_2310_l394_39448


namespace NUMINAMATH_CALUDE_weight_order_l394_39438

-- Define the weights as real numbers
variable (B S C K : ℝ)

-- State the given conditions
axiom suitcase_heavier : S > B
axiom satchel_backpack_heavier : C + B > S + K
axiom basket_satchel_equal_suitcase_backpack : K + C = S + B

-- Theorem to prove
theorem weight_order : C > S ∧ S > B ∧ B > K := by sorry

end NUMINAMATH_CALUDE_weight_order_l394_39438


namespace NUMINAMATH_CALUDE_unique_solution_floor_ceiling_l394_39422

theorem unique_solution_floor_ceiling (a : ℝ) :
  (⌊a⌋ = 3 * a + 6) ∧ (⌈a⌉ = 4 * a + 9) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_ceiling_l394_39422


namespace NUMINAMATH_CALUDE_x_minus_y_equals_eight_l394_39485

theorem x_minus_y_equals_eight (x y : ℝ) 
  (h1 : |x| + x - y = 14) 
  (h2 : x + |y| + y = 6) : 
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_eight_l394_39485


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l394_39454

theorem negation_of_existence_inequality : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l394_39454


namespace NUMINAMATH_CALUDE_largest_square_area_proof_l394_39405

/-- The side length of the original square in inches -/
def original_side : ℝ := 5

/-- The side length of the squares cut from each corner in inches -/
def cut_side : ℝ := 1

/-- The area of the largest square that can fit in the remaining space -/
def largest_square_area : ℝ := 12.5

/-- Theorem stating that the area of the largest square that can fit in the remaining space
    after cutting 1-inch squares from each corner of a 5-inch square is 12.5 square inches -/
theorem largest_square_area_proof :
  let remaining_side := original_side - 2 * cut_side
  let diagonal_space := remaining_side + 2 * cut_side
  let largest_side := diagonal_space / Real.sqrt 2
  largest_side ^ 2 = largest_square_area :=
by sorry

end NUMINAMATH_CALUDE_largest_square_area_proof_l394_39405


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l394_39455

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real

-- Define the theorem
theorem triangle_angle_difference (t : Triangle) (h1 : t.Y = 2 * t.X) (h2 : t.X = 30) 
  (Z₁ Z₂ : Real) (h3 : Z₁ + Z₂ = t.Z) : Z₁ - Z₂ = 30 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_difference_l394_39455


namespace NUMINAMATH_CALUDE_f_difference_l394_39423

/-- The function f(x) = 2x^3 - 3x^2 + 4x - 5 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 4 * x - 5

/-- Theorem stating that f(x + h) - f(x) equals the given expression -/
theorem f_difference (x h : ℝ) : 
  f (x + h) - f x = 6 * x^2 - 6 * x + 6 * x * h + 2 * h^2 - 3 * h + 4 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l394_39423


namespace NUMINAMATH_CALUDE_normal_trip_time_l394_39471

theorem normal_trip_time 
  (normal_distance : ℝ) 
  (additional_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : normal_distance = 150) 
  (h2 : additional_distance = 100) 
  (h3 : total_time = 5) :
  (normal_distance / ((normal_distance + additional_distance) / total_time)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_normal_trip_time_l394_39471


namespace NUMINAMATH_CALUDE_binomial_expression_approx_l394_39458

/-- Calculates the binomial coefficient for real x and nonnegative integer k -/
def binomial (x : ℝ) (k : ℕ) : ℝ := sorry

/-- The main theorem stating that the given expression is approximately equal to -1.243 -/
theorem binomial_expression_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  |((binomial (3/2 : ℝ) 10) * 3^10) / (binomial 20 10) + 1.243| < ε :=
sorry

end NUMINAMATH_CALUDE_binomial_expression_approx_l394_39458


namespace NUMINAMATH_CALUDE_farm_food_calculation_l394_39440

/-- Given a farm with sheep and horses, calculate the daily food requirement per horse -/
theorem farm_food_calculation (sheep_count horse_count total_food : ℕ) 
  (h1 : sheep_count = 56)
  (h2 : sheep_count = horse_count)
  (h3 : total_food = 12880) :
  total_food / horse_count = 230 := by
sorry

end NUMINAMATH_CALUDE_farm_food_calculation_l394_39440


namespace NUMINAMATH_CALUDE_function_domain_implies_m_range_l394_39461

theorem function_domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sqrt (m * x^2 + m * x + 1)) ↔ 0 ≤ m ∧ m ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_function_domain_implies_m_range_l394_39461


namespace NUMINAMATH_CALUDE_one_and_half_of_number_l394_39494

theorem one_and_half_of_number (x : ℚ) : (3 / 2) * x = 30 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_one_and_half_of_number_l394_39494


namespace NUMINAMATH_CALUDE_max_expensive_price_is_11000_l394_39489

/-- Represents a company's product line -/
structure ProductLine where
  num_products : ℕ
  average_price : ℝ
  min_price : ℝ
  num_below_threshold : ℕ
  price_threshold : ℝ

/-- The maximum possible price for the most expensive product -/
def max_expensive_price (pl : ProductLine) : ℝ :=
  let total_price := pl.num_products * pl.average_price
  let min_price_sum := pl.num_below_threshold * pl.min_price
  let remaining_price := total_price - min_price_sum
  let remaining_products := pl.num_products - pl.num_below_threshold
  remaining_price - (remaining_products - 1) * pl.price_threshold

/-- Theorem stating the maximum price of the most expensive product -/
theorem max_expensive_price_is_11000 (c : ProductLine) 
  (h1 : c.num_products = 20)
  (h2 : c.average_price = 1200)
  (h3 : c.min_price = 400)
  (h4 : c.num_below_threshold = 10)
  (h5 : c.price_threshold = 1000) :
  max_expensive_price c = 11000 := by
  sorry


end NUMINAMATH_CALUDE_max_expensive_price_is_11000_l394_39489


namespace NUMINAMATH_CALUDE_tournament_probability_l394_39487

/-- The probability of two specific participants playing each other in a tournament --/
theorem tournament_probability (n : ℕ) (h : n = 26) :
  let total_matches := n - 1
  let total_pairs := n * (n - 1) / 2
  (total_matches : ℚ) / total_pairs = 1 / 13 :=
by sorry

end NUMINAMATH_CALUDE_tournament_probability_l394_39487


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l394_39498

/-- Given a rectangle with length to width ratio of 5:2 and diagonal d, 
    its area A can be expressed as A = (10/29)d^2 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l394_39498


namespace NUMINAMATH_CALUDE_brian_chris_fishing_l394_39435

theorem brian_chris_fishing (brian_trips chris_trips : ℕ) 
  (brian_fish_per_trip : ℕ) (total_fish : ℕ) :
  brian_trips = 2 * chris_trips →
  brian_fish_per_trip = 400 →
  chris_trips = 10 →
  total_fish = 13600 →
  (1 - (brian_fish_per_trip : ℚ) / ((total_fish - brian_trips * brian_fish_per_trip) / chris_trips : ℚ)) = 2/7 := by
sorry

end NUMINAMATH_CALUDE_brian_chris_fishing_l394_39435


namespace NUMINAMATH_CALUDE_toby_work_hours_l394_39481

/-- Given the working hours of Thomas, Toby, and Rebecca, prove that Toby worked 10 hours less than twice what Thomas worked. -/
theorem toby_work_hours (x : ℕ) : 
  -- Total hours worked
  x + (2 * x - 10) + 56 = 157 →
  -- Rebecca worked 56 hours
  56 = 56 →
  -- Rebecca worked 8 hours less than Toby
  56 = (2 * x - 10) - 8 →
  -- Toby worked 10 hours less than twice what Thomas worked
  (2 * x - (2 * x - 10)) = 10 := by
sorry

end NUMINAMATH_CALUDE_toby_work_hours_l394_39481


namespace NUMINAMATH_CALUDE_min_value_complex_l394_39473

theorem min_value_complex (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min_val : ℝ), min_val = Real.sqrt (13 + 6 * Real.sqrt 7) ∧
    ∀ w : ℂ, Complex.abs w = 2 → Complex.abs (w + 3 - 4 * Complex.I) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_l394_39473


namespace NUMINAMATH_CALUDE_max_notebook_price_l394_39447

def entrance_fee : ℕ := 3
def total_budget : ℕ := 160
def num_notebooks : ℕ := 15
def tax_rate : ℚ := 8 / 100

theorem max_notebook_price :
  ∃ (price : ℕ),
    price ≤ 9 ∧
    (price : ℚ) * (1 + tax_rate) * num_notebooks + entrance_fee ≤ total_budget ∧
    ∀ (p : ℕ), p > price →
      (p : ℚ) * (1 + tax_rate) * num_notebooks + entrance_fee > total_budget :=
by sorry

end NUMINAMATH_CALUDE_max_notebook_price_l394_39447


namespace NUMINAMATH_CALUDE_cube_root_negative_equals_negative_cube_root_l394_39490

theorem cube_root_negative_equals_negative_cube_root (x : ℝ) (h : x > 0) :
  ((-x : ℝ) ^ (1/3 : ℝ)) = -(x ^ (1/3 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_cube_root_negative_equals_negative_cube_root_l394_39490


namespace NUMINAMATH_CALUDE_officer_selection_count_l394_39410

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  girls * boys * (girls - 1)

/-- Theorem stating the number of ways to choose officers under given conditions --/
theorem officer_selection_count :
  let total_members : ℕ := 24
  let boys : ℕ := 12
  let girls : ℕ := 12
  choose_officers total_members boys girls = 1584 := by
  sorry

#eval choose_officers 24 12 12

end NUMINAMATH_CALUDE_officer_selection_count_l394_39410


namespace NUMINAMATH_CALUDE_odd_multiples_of_three_count_l394_39402

theorem odd_multiples_of_three_count : 
  (Finset.filter (fun n => n % 2 = 1 ∧ n % 3 = 0) (Finset.range 1001)).card = 167 := by
  sorry

end NUMINAMATH_CALUDE_odd_multiples_of_three_count_l394_39402


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l394_39443

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 6 →
  b = Real.sqrt 3 →
  b + a * (Real.sin C - Real.cos C) = 0 →
  A + B + C = π →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l394_39443


namespace NUMINAMATH_CALUDE_smallest_n_with_common_factor_l394_39403

def has_common_factor_greater_than_one (a b : ℤ) : Prop :=
  ∃ (k : ℤ), k > 1 ∧ k ∣ a ∧ k ∣ b

theorem smallest_n_with_common_factor : 
  (∀ n : ℕ, n > 0 ∧ n < 19 → ¬(has_common_factor_greater_than_one (11*n - 3) (8*n + 2))) ∧ 
  (has_common_factor_greater_than_one (11*19 - 3) (8*19 + 2)) := by
  sorry

#check smallest_n_with_common_factor

end NUMINAMATH_CALUDE_smallest_n_with_common_factor_l394_39403


namespace NUMINAMATH_CALUDE_integer_part_of_sqrt18_minus_2_l394_39418

theorem integer_part_of_sqrt18_minus_2 :
  ⌊Real.sqrt 18 - 2⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_part_of_sqrt18_minus_2_l394_39418


namespace NUMINAMATH_CALUDE_percentage_of_democrat_voters_prove_percentage_of_democrat_voters_l394_39441

theorem percentage_of_democrat_voters : ℝ → ℝ → Prop :=
  fun d r =>
    d + r = 100 →
    0.7 * d + 0.2 * r = 50 →
    d = 60

-- Proof
theorem prove_percentage_of_democrat_voters :
  ∃ d r : ℝ, percentage_of_democrat_voters d r :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_democrat_voters_prove_percentage_of_democrat_voters_l394_39441


namespace NUMINAMATH_CALUDE_money_distribution_l394_39444

/-- Given three people with a total of $4000, where one person has two-thirds of the amount
    the other two have combined, prove that this person has $1600. -/
theorem money_distribution (total : ℚ) (r_share : ℚ) : 
  total = 4000 →
  r_share = (2/3) * (total - r_share) →
  r_share = 1600 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l394_39444


namespace NUMINAMATH_CALUDE_quadratic_equation_single_solution_sum_l394_39432

theorem quadratic_equation_single_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + b₁ * x + 12 * x + 11 = 0 → (∀ y : ℝ, 3 * y^2 + b₁ * y + 12 * y + 11 = 0 → x = y)) ∧
  (∀ x : ℝ, 3 * x^2 + b₂ * x + 12 * x + 11 = 0 → (∀ y : ℝ, 3 * y^2 + b₂ * y + 12 * y + 11 = 0 → x = y)) ∧
  (∃ x : ℝ, 3 * x^2 + b₁ * x + 12 * x + 11 = 0) ∧
  (∃ x : ℝ, 3 * x^2 + b₂ * x + 12 * x + 11 = 0) ∧
  (b₁ ≠ b₂) →
  b₁ + b₂ = -24 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_single_solution_sum_l394_39432


namespace NUMINAMATH_CALUDE_sum_of_squared_medians_l394_39429

/-- The sum of squares of medians in a triangle with sides 13, 14, and 15 --/
theorem sum_of_squared_medians (a b c : ℝ) (ha : a = 13) (hb : b = 14) (hc : c = 15) :
  let m_a := (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2)
  let m_b := (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2)
  let m_c := (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2)
  m_a^2 + m_b^2 + m_c^2 = 442.5 := by
  sorry

#check sum_of_squared_medians

end NUMINAMATH_CALUDE_sum_of_squared_medians_l394_39429


namespace NUMINAMATH_CALUDE_nancy_folders_l394_39467

-- Define the problem parameters
def initial_files : ℕ := 80
def deleted_files : ℕ := 31
def files_per_folder : ℕ := 7

-- Define the function to calculate the number of folders
def calculate_folders (initial : ℕ) (deleted : ℕ) (per_folder : ℕ) : ℕ :=
  (initial - deleted) / per_folder

-- State the theorem
theorem nancy_folders :
  calculate_folders initial_files deleted_files files_per_folder = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_folders_l394_39467


namespace NUMINAMATH_CALUDE_percentage_relation_l394_39408

theorem percentage_relation (A B T : ℝ) 
  (h1 : A = 0.2 * B) 
  (h2 : B = 0.3 * T) : 
  A = 0.06 * T := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l394_39408


namespace NUMINAMATH_CALUDE_car_cost_l394_39463

/-- The cost of a car given an initial payment and monthly installments -/
theorem car_cost (initial_payment : ℕ) (num_installments : ℕ) (installment_amount : ℕ) : 
  initial_payment + num_installments * installment_amount = 18000 :=
by
  sorry

#check car_cost 3000 6 2500

end NUMINAMATH_CALUDE_car_cost_l394_39463


namespace NUMINAMATH_CALUDE_abs_equation_solution_l394_39412

theorem abs_equation_solution : ∃! x : ℝ, |x - 3| = 5 - x := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l394_39412


namespace NUMINAMATH_CALUDE_password_guess_probabilities_l394_39425

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts -/
def guess_probability (total_digits : ℕ) (max_attempts : ℕ) : ℚ :=
  1 / total_digits + (1 - 1 / total_digits) * (1 / (total_digits - 1))

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts, given that the last digit is even -/
def guess_probability_even (total_even_digits : ℕ) (max_attempts : ℕ) : ℚ :=
  1 / total_even_digits + (1 - 1 / total_even_digits) * (1 / (total_even_digits - 1))

theorem password_guess_probabilities :
  (guess_probability 10 2 = 1/5) ∧ (guess_probability_even 5 2 = 2/5) :=
sorry

end NUMINAMATH_CALUDE_password_guess_probabilities_l394_39425


namespace NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_l394_39486

theorem not_p_or_q_false_implies_p_or_q (p q : Prop) :
  ¬(¬(p ∨ q)) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_p_or_q_false_implies_p_or_q_l394_39486


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l394_39469

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 - 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l394_39469


namespace NUMINAMATH_CALUDE_senate_committee_seating_arrangements_l394_39401

def circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem senate_committee_seating_arrangements :
  circular_arrangements 10 = 362880 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_seating_arrangements_l394_39401


namespace NUMINAMATH_CALUDE_red_highest_probability_l394_39457

/-- A color of a ball -/
inductive Color
  | Red
  | Yellow
  | White

/-- The number of balls of each color in the bag -/
def ballCount (c : Color) : ℕ :=
  match c with
  | Color.Red => 6
  | Color.Yellow => 4
  | Color.White => 1

/-- The total number of balls in the bag -/
def totalBalls : ℕ := ballCount Color.Red + ballCount Color.Yellow + ballCount Color.White

/-- The probability of drawing a ball of a given color -/
def probability (c : Color) : ℚ :=
  ballCount c / totalBalls

/-- Theorem: The probability of drawing a red ball is the highest -/
theorem red_highest_probability :
  probability Color.Red > probability Color.Yellow ∧
  probability Color.Red > probability Color.White :=
sorry

end NUMINAMATH_CALUDE_red_highest_probability_l394_39457


namespace NUMINAMATH_CALUDE_base4_even_digits_145_l394_39456

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

theorem base4_even_digits_145 :
  countEvenDigits (toBase4 145) = 2 := by
  sorry

end NUMINAMATH_CALUDE_base4_even_digits_145_l394_39456


namespace NUMINAMATH_CALUDE_area_of_parallelogram_EFGH_l394_39474

/-- The area of parallelogram EFGH --/
def area_EFGH : ℝ := 15

/-- The base of parallelogram EFGH --/
def base_FG : ℝ := 3

/-- The height from point E to line FG --/
def height_E_to_FG : ℝ := 5

/-- The theorem stating that the area of parallelogram EFGH is 15 square units --/
theorem area_of_parallelogram_EFGH :
  area_EFGH = base_FG * height_E_to_FG :=
by sorry

end NUMINAMATH_CALUDE_area_of_parallelogram_EFGH_l394_39474


namespace NUMINAMATH_CALUDE_binomial_13_8_l394_39483

theorem binomial_13_8 (h1 : Nat.choose 14 7 = 3432) 
                      (h2 : Nat.choose 14 8 = 3003) 
                      (h3 : Nat.choose 12 7 = 792) : 
  Nat.choose 13 8 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_binomial_13_8_l394_39483


namespace NUMINAMATH_CALUDE_pyramid_division_theorem_l394_39404

/-- A structure representing a pyramid divided by planes parallel to its base -/
structure DividedPyramid (n : ℕ) where
  volumePlanes : Fin (n + 1) → ℝ
  surfacePlanes : Fin (n + 1) → ℝ

/-- The condition for a common plane between volume and surface divisions -/
def hasCommonPlane (n : ℕ) : Prop :=
  ∃ (i k : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ k ∧ k ≤ n ∧ (n + 1) * i^2 = k^3

/-- The list of n values up to 100 that satisfy the common plane condition -/
def validNValues : List ℕ :=
  [7, 15, 23, 26, 31, 39, 47, 53, 55, 63, 71, 79, 80, 87, 95]

/-- The condition for multiple common planes -/
def hasMultipleCommonPlanes (n : ℕ) : Prop :=
  ∃ (i₁ k₁ i₂ k₂ : ℕ),
    1 ≤ i₁ ∧ i₁ ≤ n ∧ 1 ≤ k₁ ∧ k₁ ≤ n ∧
    1 ≤ i₂ ∧ i₂ ≤ n ∧ 1 ≤ k₂ ∧ k₂ ≤ n ∧
    (n + 1) * i₁^2 = k₁^3 ∧ (n + 1) * i₂^2 = k₂^3 ∧
    (i₁ ≠ i₂ ∨ k₁ ≠ k₂)

theorem pyramid_division_theorem :
  (∀ n ∈ validNValues, hasCommonPlane n) ∧
  (∀ n ∈ validNValues, n ≠ 63 → ¬hasMultipleCommonPlanes n) ∧
  hasMultipleCommonPlanes 63 :=
sorry

end NUMINAMATH_CALUDE_pyramid_division_theorem_l394_39404


namespace NUMINAMATH_CALUDE_horse_speed_calculation_l394_39491

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in speed between firing in the same direction as the horse
    and the opposite direction, in feet per second -/
def speed_difference : ℝ := 40

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- Theorem stating that given the bullet speed and speed difference,
    the horse's speed is 20 feet per second -/
theorem horse_speed_calculation :
  (bullet_speed + horse_speed) - (bullet_speed - horse_speed) = speed_difference :=
by sorry

end NUMINAMATH_CALUDE_horse_speed_calculation_l394_39491


namespace NUMINAMATH_CALUDE_smallest_area_squared_l394_39484

/-- A regular hexagon ABCDEF with side length 10 inscribed in a circle ω -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 10)

/-- Points X, Y, Z on minor arcs AB, CD, EF respectively -/
structure TriangleXYZ (h : RegularHexagon) :=
  (X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (Z : ℝ × ℝ)
  (X_on_AB : True)  -- Placeholder for the condition that X is on minor arc AB
  (Y_on_CD : True)  -- Placeholder for the condition that Y is on minor arc CD
  (Z_on_EF : True)  -- Placeholder for the condition that Z is on minor arc EF

/-- The area of triangle XYZ -/
def triangle_area (h : RegularHexagon) (t : TriangleXYZ h) : ℝ :=
  sorry  -- Definition of triangle area

/-- The theorem stating the smallest possible area squared -/
theorem smallest_area_squared (h : RegularHexagon) :
  ∃ (t : TriangleXYZ h), ∀ (t' : TriangleXYZ h), (triangle_area h t)^2 ≤ (triangle_area h t')^2 ∧ (triangle_area h t)^2 = 7500 :=
sorry

end NUMINAMATH_CALUDE_smallest_area_squared_l394_39484


namespace NUMINAMATH_CALUDE_bottles_poured_is_four_l394_39451

def cylinder_capacity : ℚ := 80

def initial_fullness : ℚ := 3/4

def final_fullness : ℚ := 4/5

def bottles_poured (capacity : ℚ) (initial : ℚ) (final : ℚ) : ℚ :=
  capacity * final - capacity * initial

theorem bottles_poured_is_four :
  bottles_poured cylinder_capacity initial_fullness final_fullness = 4 := by
  sorry

end NUMINAMATH_CALUDE_bottles_poured_is_four_l394_39451


namespace NUMINAMATH_CALUDE_parabola_shift_parabola_transformation_l394_39433

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := (x - 1)^2 + 2

-- Theorem stating the equivalence of the original parabola after transformation and the shifted parabola
theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by
  sorry

-- Theorem stating that the shifted parabola is the result of the described transformations
theorem parabola_transformation :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - 1) + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_parabola_transformation_l394_39433


namespace NUMINAMATH_CALUDE_shirt_price_satisfies_conditions_l394_39492

/-- The original price of a shirt, given the following conditions:
  1. Three items: shirt, pants, jacket
  2. Shirt: 25% discount, then additional 25% discount
  3. Pants: 30% discount, original price $50
  4. Jacket: two successive 20% discounts, original price $75
  5. 10% loyalty discount on total after individual discounts
  6. 15% sales tax on final price
  7. Total price paid: $150
-/
def shirt_price : ℝ :=
  let pants_price : ℝ := 50
  let jacket_price : ℝ := 75
  let pants_discount : ℝ := 0.30
  let jacket_discount : ℝ := 0.20
  let loyalty_discount : ℝ := 0.10
  let sales_tax : ℝ := 0.15
  let total_paid : ℝ := 150
  sorry

/-- Theorem stating that the calculated shirt price satisfies the given conditions -/
theorem shirt_price_satisfies_conditions :
  let S := shirt_price
  let pants_discounted := 50 * (1 - 0.30)
  let jacket_discounted := 75 * (1 - 0.20) * (1 - 0.20)
  (S * 0.75 * 0.75 + pants_discounted + jacket_discounted) * (1 - 0.10) * (1 + 0.15) = 150 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_satisfies_conditions_l394_39492


namespace NUMINAMATH_CALUDE_x_squared_less_than_x_l394_39499

theorem x_squared_less_than_x (x : ℝ) : x^2 < x ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_less_than_x_l394_39499


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l394_39416

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution
    that is 30% alcohol results in a 50% alcohol solution -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.3
  let final_concentration : ℝ := 0.5
  let added_alcohol : ℝ := 2.4

  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol : ℝ := initial_alcohol + added_alcohol

  final_alcohol / final_volume = final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l394_39416


namespace NUMINAMATH_CALUDE_special_array_determination_l394_39446

/-- Represents an m×n array of positive integers -/
def SpecialArray (m n : ℕ) := Fin m → Fin n → ℕ+

/-- The condition that must hold for any four numbers in the array -/
def SpecialCondition (A : SpecialArray m n) : Prop :=
  ∀ (i₁ i₂ : Fin m) (j₁ j₂ : Fin n),
    A i₁ j₁ + A i₂ j₂ = A i₁ j₂ + A i₂ j₁

/-- The theorem stating that m+n-1 elements are sufficient to determine the entire array -/
theorem special_array_determination (m n : ℕ) (A : SpecialArray m n) 
  (hA : SpecialCondition A) :
  ∃ (S : Finset ((Fin m) × (Fin n))),
    S.card = m + n - 1 ∧ 
    (∀ (B : SpecialArray m n), 
      SpecialCondition B → 
      (∀ (p : (Fin m) × (Fin n)), p ∈ S → A p.1 p.2 = B p.1 p.2) → 
      A = B) :=
sorry

end NUMINAMATH_CALUDE_special_array_determination_l394_39446


namespace NUMINAMATH_CALUDE_length_of_AB_prime_l394_39426

/-- Given points A, B, and C in the plane, with A' and B' on the line y = x,
    and lines AA' and BB' intersecting at C, prove that the length of A'B' is 120√2/11 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 5) →
  B = (0, 15) →
  C = (3, 7) →
  A'.1 = A'.2 →
  B'.1 = B'.2 →
  (∃ t : ℝ, A' = (1 - t) • A + t • C) →
  (∃ s : ℝ, B' = (1 - s) • B + s • C) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 120 * Real.sqrt 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_prime_l394_39426


namespace NUMINAMATH_CALUDE_a_minus_b_equals_two_l394_39459

theorem a_minus_b_equals_two (a b : ℝ) 
  (h1 : |a| = 1) 
  (h2 : |b - 1| = 2) 
  (h3 : a > b) : 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_two_l394_39459


namespace NUMINAMATH_CALUDE_coconut_cost_is_fifty_cents_l394_39479

/-- Represents the cost per coconut on Rohan's farm -/
def coconut_cost (farm_size : ℕ) (trees_per_sqm : ℕ) (coconuts_per_tree : ℕ) 
  (harvest_interval : ℕ) (months : ℕ) (total_earnings : ℚ) : ℚ :=
  let total_trees := farm_size * trees_per_sqm
  let total_coconuts := total_trees * coconuts_per_tree
  let harvests := months / harvest_interval
  let total_harvested := total_coconuts * harvests
  total_earnings / total_harvested

/-- Proves that the cost per coconut on Rohan's farm is $0.50 -/
theorem coconut_cost_is_fifty_cents :
  coconut_cost 20 2 6 3 6 240 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_coconut_cost_is_fifty_cents_l394_39479


namespace NUMINAMATH_CALUDE_number_problem_l394_39472

theorem number_problem (x : ℝ) : (x - 14) / 10 = 4 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l394_39472


namespace NUMINAMATH_CALUDE_floor_paving_cost_l394_39476

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 5) 
  (h2 : width = 4.75) 
  (h3 : rate = 900) : 
  length * width * rate = 21375 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l394_39476


namespace NUMINAMATH_CALUDE_product_of_divisors_60_has_three_prime_factors_l394_39436

def divisors (n : ℕ) : Finset ℕ :=
  sorry

def product_of_divisors (n : ℕ) : ℕ :=
  (divisors n).prod id

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  sorry

theorem product_of_divisors_60_has_three_prime_factors :
  num_distinct_prime_factors (product_of_divisors 60) = 3 :=
sorry

end NUMINAMATH_CALUDE_product_of_divisors_60_has_three_prime_factors_l394_39436


namespace NUMINAMATH_CALUDE_carbon_processing_optimization_l394_39421

-- Define the processing volume range
def ProcessingRange : Set ℝ := {x : ℝ | 300 ≤ x ∧ x ≤ 600}

-- Define the cost function
def CostFunction (x : ℝ) : ℝ := 0.5 * x^2 - 200 * x + 45000

-- Define the revenue function
def RevenueFunction (x : ℝ) : ℝ := 200 * x

-- Define the profit function
def ProfitFunction (x : ℝ) : ℝ := RevenueFunction x - CostFunction x

-- Theorem statement
theorem carbon_processing_optimization :
  ∃ (x_min : ℝ) (max_profit : ℝ),
    x_min ∈ ProcessingRange ∧
    (∀ x ∈ ProcessingRange, CostFunction x_min / x_min ≤ CostFunction x / x) ∧
    x_min = 300 ∧
    (∀ x ∈ ProcessingRange, ProfitFunction x > 0) ∧
    (∀ x ∈ ProcessingRange, ProfitFunction x ≤ max_profit) ∧
    max_profit = 35000 := by
  sorry

end NUMINAMATH_CALUDE_carbon_processing_optimization_l394_39421


namespace NUMINAMATH_CALUDE_tan_sum_minus_product_62_73_l394_39407

theorem tan_sum_minus_product_62_73 :
  Real.tan (62 * π / 180) + Real.tan (73 * π / 180) - 
  Real.tan (62 * π / 180) * Real.tan (73 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_minus_product_62_73_l394_39407


namespace NUMINAMATH_CALUDE_number_plus_two_equals_six_l394_39430

theorem number_plus_two_equals_six :
  ∃ x : ℝ, (2 + x = 6) ∧ (x = 4) := by
  sorry

end NUMINAMATH_CALUDE_number_plus_two_equals_six_l394_39430


namespace NUMINAMATH_CALUDE_four_thirds_of_product_l394_39466

theorem four_thirds_of_product (a b : ℚ) (ha : a = 15/4) (hb : b = 5/2) : 
  (4/3 : ℚ) * (a * b) = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_product_l394_39466


namespace NUMINAMATH_CALUDE_expression_simplification_l394_39417

theorem expression_simplification (p q r : ℝ) 
  (hp : p ≠ 2) (hq : q ≠ 3) (hr : r ≠ 4) : 
  (p - 2) / (4 - r) * (q - 3) / (2 - p) * (r - 4) / (3 - q) * (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l394_39417


namespace NUMINAMATH_CALUDE_purple_position_correct_l394_39445

/-- The position of "PURPLE" in the alphabetized list of all its distinguishable rearrangements -/
def purple_position : ℕ := 226

/-- The word to be rearranged -/
def word : String := "PURPLE"

/-- The theorem stating that the position of "PURPLE" in the alphabetized list of all its distinguishable rearrangements is 226 -/
theorem purple_position_correct : 
  purple_position = 226 ∧ 
  word = "PURPLE" ∧
  purple_position = (List.filter (· ≤ word) (List.map String.mk (List.permutations word.data))).length :=
by sorry

end NUMINAMATH_CALUDE_purple_position_correct_l394_39445


namespace NUMINAMATH_CALUDE_problem_solution_l394_39409

def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m

theorem problem_solution (m : ℝ) (h_m : m > 0) 
  (h_solution_set : {x : ℝ | f m (x - 3) ≥ 0} = Set.Iic (-2) ∪ Set.Ici 2) :
  m = 2 ∧ 
  ∀ (x t : ℝ), f 2 x ≥ |2 * x - 1| - t^2 + (3/2) * t + 1 → 
    t ∈ Set.Iic (1/2) ∪ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l394_39409


namespace NUMINAMATH_CALUDE_jack_multiple_is_ten_l394_39428

/-- The multiple of Michael's current trophies that Jack will have in three years -/
def jack_multiple (michael_current : ℕ) (michael_increase : ℕ) (total_after : ℕ) : ℕ :=
  (total_after - (michael_current + michael_increase)) / michael_current

theorem jack_multiple_is_ten :
  jack_multiple 30 100 430 = 10 := by sorry

end NUMINAMATH_CALUDE_jack_multiple_is_ten_l394_39428


namespace NUMINAMATH_CALUDE_triangle_inequality_l394_39475

theorem triangle_inequality (a b c S : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : S > 0) (h₅ : a + b > c) (h₆ : b + c > a) (h₇ : c + a > b)
  (h₈ : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  c^2 - a^2 - b^2 + 4*a*b ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l394_39475


namespace NUMINAMATH_CALUDE_ap_to_gp_ratio_is_positive_integer_l394_39413

/-- An arithmetic progression starting with 1 -/
def AP (x : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => AP x n + (x - 1)

/-- A geometric progression starting with 1 -/
def GP (a : ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => GP a n * a

/-- The property that a GP is formed by deleting some terms from an AP -/
def isSubsequence (x a : ℝ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, GP a n = AP x m

theorem ap_to_gp_ratio_is_positive_integer (x : ℝ) (hx : x ≥ 1) (a : ℝ) (ha : a > 0)
    (h : isSubsequence x a) : ∃ k : ℕ+, a = k :=
  sorry

end NUMINAMATH_CALUDE_ap_to_gp_ratio_is_positive_integer_l394_39413


namespace NUMINAMATH_CALUDE_no_1989_digit_number_sum_equals_product_l394_39424

theorem no_1989_digit_number_sum_equals_product : ¬ ∃ (n : ℕ), 
  (n ≥ 10^1988 ∧ n < 10^1989) ∧  -- n has 1989 digits
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < 1989 ∧ d₂ < 1989 ∧ d₃ < 1989 ∧ 
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧
    (n / 10^d₁ % 10 = 5) ∧ (n / 10^d₂ % 10 = 5) ∧ (n / 10^d₃ % 10 = 5)) ∧  -- at least three digits are 5
  (List.sum (List.map (λ i => n / 10^i % 10) (List.range 1989)) = 
   List.prod (List.map (λ i => n / 10^i % 10) (List.range 1989))) :=  -- sum of digits equals product of digits
by sorry

end NUMINAMATH_CALUDE_no_1989_digit_number_sum_equals_product_l394_39424


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l394_39414

theorem smallest_prime_dividing_sum : ∃ (p : Nat), 
  Prime p ∧ 
  p ∣ (2^14 + 7^9) ∧ 
  ∀ (q : Nat), Prime q → q ∣ (2^14 + 7^9) → p ≤ q :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l394_39414


namespace NUMINAMATH_CALUDE_fraction_simplification_l394_39427

theorem fraction_simplification : (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l394_39427


namespace NUMINAMATH_CALUDE_max_missed_problems_l394_39437

theorem max_missed_problems (total_problems : ℕ) (pass_percentage : ℚ) 
  (h1 : total_problems = 50)
  (h2 : pass_percentage = 85 / 100) : 
  ∃ (max_missed : ℕ), 
    (max_missed ≤ total_problems) ∧ 
    ((total_problems - max_missed : ℚ) / total_problems ≥ pass_percentage) ∧
    ∀ (n : ℕ), n > max_missed → 
      ((total_problems - n : ℚ) / total_problems < pass_percentage) :=
by
  sorry

end NUMINAMATH_CALUDE_max_missed_problems_l394_39437


namespace NUMINAMATH_CALUDE_ellipse_iff_k_in_range_l394_39449

/-- The equation of an ellipse in the form (x^2 / (3+k)) + (y^2 / (2-k)) = 1 -/
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
def k_range : Set ℝ :=
  {k | k ∈ (Set.Ioo (-3) (-1/2) ∪ Set.Ioo (-1/2) 2)}

/-- Theorem stating that the equation represents an ellipse if and only if k is in the specified range -/
theorem ellipse_iff_k_in_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ k_range :=
sorry

end NUMINAMATH_CALUDE_ellipse_iff_k_in_range_l394_39449


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l394_39470

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15) ∧ 
  (|x₂ - 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧ 
  (|x₁ - x₂| = 30) := by
sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l394_39470


namespace NUMINAMATH_CALUDE_tan_equality_345_degrees_l394_39400

theorem tan_equality_345_degrees (n : ℤ) :
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (345 * π / 180) → n = -15 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_345_degrees_l394_39400


namespace NUMINAMATH_CALUDE_max_cells_visitable_l394_39493

/-- Represents a rectangular board -/
structure Board where
  rows : Nat
  cols : Nat

/-- Represents a cube with one painted face -/
structure Cube where
  side : Nat
  painted_face : Nat

/-- Defines the maximum number of cells a cube can visit on a board without the painted face touching -/
def max_visitable_cells (b : Board) (c : Cube) : Nat :=
  b.rows * b.cols

/-- Theorem stating that the maximum number of visitable cells equals the total number of cells on the board -/
theorem max_cells_visitable (b : Board) (c : Cube) 
  (h1 : b.rows = 7) 
  (h2 : b.cols = 12) 
  (h3 : c.side = 1) 
  (h4 : c.painted_face ≤ 6) :
  max_visitable_cells b c = b.rows * b.cols := by
  sorry

end NUMINAMATH_CALUDE_max_cells_visitable_l394_39493


namespace NUMINAMATH_CALUDE_fraction_simplification_l394_39477

theorem fraction_simplification (x : ℝ) : (3*x - 2)/4 + (5 - 2*x)/3 = (x + 14)/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l394_39477


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l394_39411

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 6 * x^2 + 6 * x - 12

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-2 : ℝ) 1, f' x < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l394_39411


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l394_39488

theorem simple_interest_calculation (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal = 10040.625 →
  rate = 8 →
  time = 5 →
  principal * rate * time / 100 = 40162.5 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l394_39488
