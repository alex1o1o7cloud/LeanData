import Mathlib

namespace NUMINAMATH_CALUDE_min_first_row_sum_l608_60874

/-- Represents a grid with 9 rows and 2004 columns -/
def Grid := Fin 9 → Fin 2004 → ℕ

/-- The condition that each integer from 1 to 2004 appears exactly 9 times in the grid -/
def validDistribution (g : Grid) : Prop :=
  ∀ n : Fin 2004, (Finset.univ.sum fun i => (Finset.univ.filter (fun j => g i j = n.val + 1)).card) = 9

/-- The condition that no integer appears more than 3 times in any column -/
def validColumn (g : Grid) : Prop :=
  ∀ j : Fin 2004, ∀ n : Fin 2004, (Finset.univ.filter (fun i => g i j = n.val + 1)).card ≤ 3

/-- The sum of the numbers in the first row -/
def firstRowSum (g : Grid) : ℕ :=
  Finset.univ.sum (fun j => g 0 j)

theorem min_first_row_sum :
  ∀ g : Grid, validDistribution g → validColumn g →
  firstRowSum g ≥ 2005004 :=
sorry

end NUMINAMATH_CALUDE_min_first_row_sum_l608_60874


namespace NUMINAMATH_CALUDE_project_cost_increase_l608_60843

def initial_lumber_cost : ℝ := 450
def initial_nails_cost : ℝ := 30
def initial_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

theorem project_cost_increase :
  let initial_total_cost := initial_lumber_cost + initial_nails_cost + initial_fabric_cost
  let new_lumber_cost := initial_lumber_cost * (1 + lumber_inflation_rate)
  let new_nails_cost := initial_nails_cost * (1 + nails_inflation_rate)
  let new_fabric_cost := initial_fabric_cost * (1 + fabric_inflation_rate)
  let new_total_cost := new_lumber_cost + new_nails_cost + new_fabric_cost
  new_total_cost - initial_total_cost = 97 := by
sorry

end NUMINAMATH_CALUDE_project_cost_increase_l608_60843


namespace NUMINAMATH_CALUDE_vector_problem_solution_l608_60858

def vector_problem (a b : ℝ × ℝ) (m : ℝ) : Prop :=
  let norm_a : ℝ := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b : ℝ := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
  norm_a = 3 ∧
  norm_b = 2 ∧
  dot_product a b = norm_a * norm_b * (-1/2) ∧
  dot_product (a.1 + m * b.1, a.2 + m * b.2) a = 0

theorem vector_problem_solution (a b : ℝ × ℝ) (m : ℝ) 
  (h : vector_problem a b m) : m = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_solution_l608_60858


namespace NUMINAMATH_CALUDE_definite_integral_sqrt_minus_x_l608_60888

open Set
open MeasureTheory
open Interval

theorem definite_integral_sqrt_minus_x :
  ∫ (x : ℝ) in (Icc 0 1), (Real.sqrt (1 - (x - 1)^2) - x) = π/4 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_sqrt_minus_x_l608_60888


namespace NUMINAMATH_CALUDE_circle_point_x_value_l608_60801

/-- Given a circle with diameter endpoints (-8, 0) and (32, 0), 
    if the point (x, 20) lies on this circle, then x = 12. -/
theorem circle_point_x_value 
  (x : ℝ) 
  (h : (x - 12)^2 + 20^2 = ((32 - (-8)) / 2)^2) : 
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_circle_point_x_value_l608_60801


namespace NUMINAMATH_CALUDE_fraction_equivalence_l608_60853

theorem fraction_equivalence : ∃ n : ℚ, (4 + n) / (7 + n) = 2 / 3 :=
by
  use 2
  sorry

#check fraction_equivalence

end NUMINAMATH_CALUDE_fraction_equivalence_l608_60853


namespace NUMINAMATH_CALUDE_expression_is_perfect_square_l608_60813

/-- The expression is a perfect square when x = 0.04 -/
theorem expression_is_perfect_square : 
  ∃ y : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + 0.02 * 0.02) = y * y := by
  sorry

end NUMINAMATH_CALUDE_expression_is_perfect_square_l608_60813


namespace NUMINAMATH_CALUDE_min_four_dollar_frisbees_l608_60809

/-- Given the conditions of frisbee sales, proves the minimum number of $4 frisbees sold -/
theorem min_four_dollar_frisbees 
  (total_frisbees : ℕ) 
  (price_low price_high : ℕ) 
  (total_receipts : ℕ) 
  (h_total : total_frisbees = 60)
  (h_price_low : price_low = 3)
  (h_price_high : price_high = 4)
  (h_receipts : total_receipts = 200) :
  ∃ (x y : ℕ), 
    x + y = total_frisbees ∧ 
    price_low * x + price_high * y = total_receipts ∧
    y ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_four_dollar_frisbees_l608_60809


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l608_60898

theorem complex_number_quadrant : ∃ (z : ℂ), z = (I : ℂ) / (Real.sqrt 3 - 3 * I) ∧ z.re < 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l608_60898


namespace NUMINAMATH_CALUDE_circle_equation_l608_60812

/-- Given a circle with radius 5 and a line l: x + 2y - 3 = 0 tangent to the circle at point P(1,1),
    prove that the equations of the circle are:
    (x-1-√5)² + (y-1-2√5)² = 25 and (x-1+√5)² + (y-1+2√5)² = 25 -/
theorem circle_equation (x y : ℝ) :
  let r : ℝ := 5
  let l : ℝ → ℝ → ℝ := fun x y ↦ x + 2*y - 3
  let P : ℝ × ℝ := (1, 1)
  (∃ (center : ℝ × ℝ), (center.1 - P.1)^2 + (center.2 - P.2)^2 = r^2 ∧
    l P.1 P.2 = 0 ∧
    (∀ (t : ℝ), t ≠ 0 → l (P.1 + t) (P.2 + t * ((center.2 - P.2) / (center.1 - P.1))) ≠ 0)) →
  ((x - (1 - Real.sqrt 5))^2 + (y - (1 - 2 * Real.sqrt 5))^2 = 25) ∨
  ((x - (1 + Real.sqrt 5))^2 + (y - (1 + 2 * Real.sqrt 5))^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l608_60812


namespace NUMINAMATH_CALUDE_first_prime_in_special_product_l608_60804

theorem first_prime_in_special_product (x y z : Nat) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧  -- x, y, z are prime
  x ≠ y ∧ x ≠ z ∧ y ≠ z ∧  -- x, y, z are different
  (∃ (divisors : Finset Nat), divisors.card = 12 ∧ 
    ∀ d ∈ divisors, (x^2 * y * z) % d = 0) →  -- x^2 * y * z has 12 divisors
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_first_prime_in_special_product_l608_60804


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l608_60893

/-- The trajectory of point P given a moving point M on a circle and a fixed point B -/
theorem trajectory_of_midpoint (x y : ℝ) : 
  (∃ m n : ℝ, m^2 + n^2 = 1 ∧ 
              x = (m + 3) / 2 ∧ 
              y = n / 2) → 
  (2*x - 3)^2 + 4*y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l608_60893


namespace NUMINAMATH_CALUDE_solution_is_correct_l608_60819

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the equation
def equation (y : ℝ) : Prop :=
  log 3 ((4*y + 16) / (6*y - 9)) + log 3 ((6*y - 9) / (2*y - 5)) = 3

-- Theorem statement
theorem solution_is_correct :
  equation (151/50) := by sorry

end NUMINAMATH_CALUDE_solution_is_correct_l608_60819


namespace NUMINAMATH_CALUDE_special_function_monotonicity_l608_60803

/-- A function f: ℝ → ℝ satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (1 - x) = f x) ∧
  (∀ x, (x - 1/2) * (deriv^[2] f x) > 0)

/-- Theorem stating the monotonicity property of the special function -/
theorem special_function_monotonicity 
  (f : ℝ → ℝ) (hf : SpecialFunction f) (x₁ x₂ : ℝ) 
  (h_order : x₁ < x₂) (h_sum : x₁ + x₂ > 1) : 
  f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_special_function_monotonicity_l608_60803


namespace NUMINAMATH_CALUDE_frank_lawn_money_l608_60894

/-- The amount of money Frank made mowing lawns -/
def lawn_money : ℕ := 19

/-- The cost of mower blades -/
def blade_cost : ℕ := 11

/-- The number of games Frank could buy -/
def num_games : ℕ := 4

/-- The cost of each game -/
def game_cost : ℕ := 2

/-- Theorem stating that the money Frank made mowing lawns is correct -/
theorem frank_lawn_money :
  lawn_money = blade_cost + num_games * game_cost :=
by sorry

end NUMINAMATH_CALUDE_frank_lawn_money_l608_60894


namespace NUMINAMATH_CALUDE_pipe_stack_height_l608_60831

/-- The height of a stack of three pipes in an isosceles triangular configuration -/
theorem pipe_stack_height (d : ℝ) (h : d = 12) : 
  let r := d / 2
  let base_center_distance := 2 * r
  let triangle_height := Real.sqrt (base_center_distance ^ 2 - (base_center_distance / 2) ^ 2)
  let total_height := triangle_height + 2 * r
  total_height = 12 + 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_pipe_stack_height_l608_60831


namespace NUMINAMATH_CALUDE_price_reduction_l608_60890

theorem price_reduction (price_2010 : ℝ) (price_2011 price_2012 : ℝ) :
  price_2011 = price_2010 * (1 + 0.25) →
  price_2012 = price_2010 * (1 + 0.10) →
  price_2012 = price_2011 * (1 - 0.12) :=
by
  sorry

end NUMINAMATH_CALUDE_price_reduction_l608_60890


namespace NUMINAMATH_CALUDE_smallest_interesting_number_l608_60827

theorem smallest_interesting_number :
  ∃ (n : ℕ), n = 1800 ∧
  (∀ m : ℕ, m < n →
    ¬(∃ k : ℕ, 2 * m = k ^ 2) ∨
    ¬(∃ l : ℕ, 15 * m = l ^ 3)) ∧
  (∃ k : ℕ, 2 * n = k ^ 2) ∧
  (∃ l : ℕ, 15 * n = l ^ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_interesting_number_l608_60827


namespace NUMINAMATH_CALUDE_even_polynomial_iff_composition_l608_60851

open Polynomial

theorem even_polynomial_iff_composition (P : Polynomial ℝ) :
  (∀ x, P.eval (-x) = P.eval x) ↔ 
  ∃ Q : Polynomial ℝ, P = Q.comp (X ^ 2) :=
sorry

end NUMINAMATH_CALUDE_even_polynomial_iff_composition_l608_60851


namespace NUMINAMATH_CALUDE_max_b_for_inequality_solution_l608_60847

theorem max_b_for_inequality_solution (b : ℝ) : 
  (∃ x : ℝ, b * (b ^ (1/2)) * (x^2 - 10*x + 25) + (b ^ (1/2)) / (x^2 - 10*x + 25) ≤ 
    (1/5) * (b ^ (3/4)) * |Real.sin (π * x / 10)|) 
  → b ≤ (1/10000) :=
sorry

end NUMINAMATH_CALUDE_max_b_for_inequality_solution_l608_60847


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l608_60856

theorem tracy_art_fair_sales (total_customers : ℕ) (first_group : ℕ) (second_group : ℕ) (last_group : ℕ)
  (first_group_purchases : ℕ) (second_group_purchases : ℕ) (total_sales : ℕ) :
  total_customers = first_group + second_group + last_group →
  first_group = 4 →
  second_group = 12 →
  last_group = 4 →
  first_group_purchases = 2 →
  second_group_purchases = 1 →
  total_sales = 36 →
  ∃ (last_group_purchases : ℕ),
    total_sales = first_group * first_group_purchases +
                  second_group * second_group_purchases +
                  last_group * last_group_purchases ∧
    last_group_purchases = 4 :=
by sorry

end NUMINAMATH_CALUDE_tracy_art_fair_sales_l608_60856


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l608_60899

/-- The focus of the parabola y = 2x^2 has coordinates (0, 1/8) -/
theorem parabola_focus_coordinates :
  let f : ℝ → ℝ := λ x => 2 * x^2
  ∃ (focus : ℝ × ℝ), focus = (0, 1/8) ∧
    ∀ (x y : ℝ), y = f x → 
      (x - focus.1)^2 + (y - focus.2)^2 = (y - focus.2 + 1/4)^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l608_60899


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l608_60897

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 2 → (x + 1) * (x - 2) > 0) ∧
  (∃ x, (x + 1) * (x - 2) > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l608_60897


namespace NUMINAMATH_CALUDE_parallel_intersection_false_l608_60854

-- Define the types for planes and lines
variable (α β : Plane) (m n : Line)

-- Define the parallel and intersection relations
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem parallel_intersection_false :
  ¬(∀ α β m n,
    (parallel m α ∧ intersection α β = n) → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_intersection_false_l608_60854


namespace NUMINAMATH_CALUDE_right_triangle_base_length_l608_60852

theorem right_triangle_base_length 
  (area : ℝ) 
  (hypotenuse : ℝ) 
  (side : ℝ) 
  (h_area : area = 24) 
  (h_hypotenuse : hypotenuse = 10) 
  (h_side : side = 8) : 
  ∃ (base height : ℝ), 
    area = (1/2) * base * height ∧ 
    hypotenuse^2 = base^2 + height^2 ∧ 
    (base = side ∨ height = side) ∧ 
    base = 8 := by
  sorry

#check right_triangle_base_length

end NUMINAMATH_CALUDE_right_triangle_base_length_l608_60852


namespace NUMINAMATH_CALUDE_a_in_P_and_b_in_Q_l608_60882

-- Define the sets P and Q
def P : Set ℤ := {x | ∃ m : ℤ, x = 2 * m + 1}
def Q : Set ℤ := {y | ∃ n : ℤ, y = 2 * n}

-- Define the theorem
theorem a_in_P_and_b_in_Q (x₀ y₀ : ℤ) (hx : x₀ ∈ P) (hy : y₀ ∈ Q) :
  let a := x₀ + y₀
  let b := x₀ * y₀
  a ∈ P ∧ b ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_a_in_P_and_b_in_Q_l608_60882


namespace NUMINAMATH_CALUDE_regular_octagon_diagonal_l608_60825

theorem regular_octagon_diagonal (s : ℝ) (h : s = 12) : 
  let diagonal := s * Real.sqrt 2
  diagonal = 12 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_regular_octagon_diagonal_l608_60825


namespace NUMINAMATH_CALUDE_gum_pieces_per_package_l608_60875

/-- The number of packages of gum Robin has -/
def num_packages : ℕ := 5

/-- The number of extra pieces of gum Robin has -/
def extra_pieces : ℕ := 6

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := 41

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := (total_pieces - extra_pieces) / num_packages

theorem gum_pieces_per_package : pieces_per_package = 7 := by
  sorry

end NUMINAMATH_CALUDE_gum_pieces_per_package_l608_60875


namespace NUMINAMATH_CALUDE_andrews_age_l608_60815

/-- Andrew's age problem -/
theorem andrews_age :
  ∀ (a g : ℚ),
  g = 10 * a →
  g - a = 60 →
  a = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l608_60815


namespace NUMINAMATH_CALUDE_game_result_l608_60822

-- Define the point function
def g (n : Nat) : Nat :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

-- Define Allie's rolls
def allie_rolls : List Nat := [3, 6, 5, 2, 4]

-- Define Betty's rolls
def betty_rolls : List Nat := [2, 6, 1, 4]

-- Calculate total points for a list of rolls
def total_points (rolls : List Nat) : Nat :=
  rolls.map g |>.sum

-- Theorem statement
theorem game_result :
  (total_points allie_rolls) * (total_points betty_rolls) = 308 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l608_60822


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l608_60863

theorem units_digit_of_7_power_2023 : (7^2023 : ℕ) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l608_60863


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l608_60895

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℤ) -- The arithmetic sequence
  (d : ℤ) -- The common difference
  (h1 : a 0 = 23) -- First term is 23
  (h2 : ∀ n, a (n + 1) = a n + d) -- Arithmetic sequence definition
  (h3 : ∀ n, n < 6 → a n > 0) -- First 6 terms are positive
  (h4 : ∀ n, n ≥ 6 → a n < 0) -- Terms from 7th onward are negative
  : d = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l608_60895


namespace NUMINAMATH_CALUDE_magnitude_of_Z_l608_60884

theorem magnitude_of_Z (Z : ℂ) (h : (1 - Complex.I) * Z = 1 + Complex.I) : Complex.abs Z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_Z_l608_60884


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l608_60872

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 5*x + 6 > 0 ↔ x < 2 ∨ x > 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l608_60872


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l608_60850

/-- Proves the volume of fuel A in a partially filled tank -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ) :
  tank_capacity = 214 →
  ethanol_a = 0.12 →
  ethanol_b = 0.16 →
  total_ethanol = 30 →
  ∃ (volume_a : ℝ), 
    volume_a + (tank_capacity - volume_a) = tank_capacity ∧
    ethanol_a * volume_a + ethanol_b * (tank_capacity - volume_a) = total_ethanol ∧
    volume_a = 106 := by
  sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l608_60850


namespace NUMINAMATH_CALUDE_no_14_consecutive_integers_exists_21_consecutive_integers_l608_60862

-- Define the set of primes for part 1
def primes1 : Set ℕ := {2, 3, 5, 7, 11}

-- Define the set of primes for part 2
def primes2 : Set ℕ := {2, 3, 5, 7, 11, 13}

-- Define a function to check if a number is divisible by any prime in a given set
def divisibleByAnyPrime (n : ℕ) (primes : Set ℕ) : Prop :=
  ∃ p ∈ primes, n % p = 0

-- Part 1: No set of 14 consecutive integers satisfies the condition
theorem no_14_consecutive_integers : 
  ¬∃ n : ℕ, ∀ k ∈ Finset.range 14, divisibleByAnyPrime (n + k) primes1 := by
sorry

-- Part 2: There exists a set of 21 consecutive integers that satisfies the condition
theorem exists_21_consecutive_integers : 
  ∃ n : ℕ, ∀ k ∈ Finset.range 21, divisibleByAnyPrime (n + k) primes2 := by
sorry

end NUMINAMATH_CALUDE_no_14_consecutive_integers_exists_21_consecutive_integers_l608_60862


namespace NUMINAMATH_CALUDE_bd_length_is_twelve_l608_60800

-- Define the triangle ABC
def triangle_ABC : Type := Unit

-- Define point D
def point_D : Type := Unit

-- Define that B is a right angle
def B_is_right_angle (t : triangle_ABC) : Prop := sorry

-- Define that a circle with diameter BC intersects AC at D
def circle_intersects_AC (t : triangle_ABC) (d : point_D) : Prop := sorry

-- Define the area of triangle ABC
def area_ABC (t : triangle_ABC) : ℝ := 120

-- Define the length of AC
def length_AC (t : triangle_ABC) : ℝ := 20

-- Define the length of BD
def length_BD (t : triangle_ABC) (d : point_D) : ℝ := sorry

-- Theorem statement
theorem bd_length_is_twelve (t : triangle_ABC) (d : point_D) :
  B_is_right_angle t →
  circle_intersects_AC t d →
  length_BD t d = 12 :=
sorry

end NUMINAMATH_CALUDE_bd_length_is_twelve_l608_60800


namespace NUMINAMATH_CALUDE_problem_statement_l608_60838

theorem problem_statement (a b c : ℝ) : 
  a < b →
  (∀ x : ℝ, (x - a) * (x - b) / (x - c) ≤ 0 ↔ (x < -1 ∨ |x - 10| ≤ 2)) →
  a + 2 * b + 3 * c = 29 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l608_60838


namespace NUMINAMATH_CALUDE_probability_of_selecting_two_specific_elements_l608_60833

theorem probability_of_selecting_two_specific_elements 
  (total_elements : Nat) 
  (elements_to_select : Nat) 
  (h1 : total_elements = 6) 
  (h2 : elements_to_select = 2) :
  (1 : ℚ) / (Nat.choose total_elements elements_to_select) = 1 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_selecting_two_specific_elements_l608_60833


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l608_60896

theorem cubic_equation_solution (a : ℝ) (h : a^2 - a - 1 = 0) : 
  a^3 - a^2 - a + 2023 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l608_60896


namespace NUMINAMATH_CALUDE_orange_boxes_l608_60868

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 2650) (h2 : oranges_per_box = 10) :
  total_oranges / oranges_per_box = 265 := by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_l608_60868


namespace NUMINAMATH_CALUDE_ceiling_of_3_7_l608_60857

theorem ceiling_of_3_7 : ⌈(3.7 : ℝ)⌉ = 4 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_3_7_l608_60857


namespace NUMINAMATH_CALUDE_stating_gray_cube_count_gray_cube_count_3x3x3_gray_cube_count_5x5x5_l608_60879

/-- 
Represents the number of 1x1x1 cubes with a specific number of gray faces 
in an nxnxn cube where all outer faces are painted gray.
-/
def grayCubes (n : ℕ) : ℕ × ℕ :=
  (6 * (n - 2)^2, (n - 2)^3)

/-- 
Theorem stating the correct number of 1x1x1 cubes with exactly one gray face 
and with no gray faces in an nxnxn cube where all outer faces are painted gray.
-/
theorem gray_cube_count (n : ℕ) (h : n ≥ 3) : 
  grayCubes n = (6 * (n - 2)^2, (n - 2)^3) := by
  sorry

/-- 
Corollary for the specific case of a 3x3x3 cube, giving the number of cubes 
with exactly one gray face and exactly two gray faces.
-/
theorem gray_cube_count_3x3x3 : 
  (grayCubes 3).1 = 6 ∧ 12 = 12 := by
  sorry

/-- 
Corollary for the specific case of a 5x5x5 cube, giving the number of cubes 
with exactly one gray face and with no gray faces.
-/
theorem gray_cube_count_5x5x5 : 
  (grayCubes 5).1 = 54 ∧ (grayCubes 5).2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_stating_gray_cube_count_gray_cube_count_3x3x3_gray_cube_count_5x5x5_l608_60879


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l608_60835

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ
  diagonals : ℕ

/-- Calculate the number of diagonals in a polygon with n sides -/
def diagonalsCount (n : ℕ) : ℕ :=
  n * (n - 3) / 2

/-- Calculate the perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ :=
  p.sides * p.sideLength

theorem regular_polygon_properties :
  ∃ (p : RegularPolygon),
    p.diagonals = 15 ∧
    p.sideLength = 6 ∧
    p.sides = 7 ∧
    perimeter p = 42 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l608_60835


namespace NUMINAMATH_CALUDE_doodads_produced_l608_60836

/-- Represents the production rate of gizmos per worker per hour -/
def gizmo_rate (workers : ℕ) (hours : ℕ) (gizmos : ℕ) : ℚ :=
  (gizmos : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Represents the production rate of doodads per worker per hour -/
def doodad_rate (workers : ℕ) (hours : ℕ) (doodads : ℕ) : ℚ :=
  (doodads : ℚ) / ((workers : ℚ) * (hours : ℚ))

/-- Theorem stating the number of doodads produced by 40 workers in 4 hours -/
theorem doodads_produced
  (h1 : gizmo_rate 80 2 160 = gizmo_rate 70 3 210)
  (h2 : doodad_rate 80 2 240 = doodad_rate 70 3 420)
  (h3 : gizmo_rate 40 4 160 = gizmo_rate 80 2 160) :
  (doodad_rate 80 2 240 * (40 : ℚ) * 4) = 320 := by
  sorry

end NUMINAMATH_CALUDE_doodads_produced_l608_60836


namespace NUMINAMATH_CALUDE_remainder_theorem_l608_60818

def polynomial (x : ℝ) : ℝ := 5*x^5 - 12*x^4 + 3*x^3 - 7*x^2 + x - 30

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), 
    polynomial x = (divisor x) * (q x) + 234 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l608_60818


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l608_60814

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_subtraction_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (2, 0))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a - 2 • b))^2 + (Prod.snd (a - 2 • b))^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l608_60814


namespace NUMINAMATH_CALUDE_pizza_buffet_theorem_l608_60837

theorem pizza_buffet_theorem (A B C : ℕ+) :
  (A : ℚ) = 1.8 * B ∧
  (B : ℚ) = C / 8 ∧
  A ≥ 2 ∧ B ≥ 1 ∧ C ≥ 8 →
  A = 2 ∧ B = 1 ∧ C = 8 := by
sorry

end NUMINAMATH_CALUDE_pizza_buffet_theorem_l608_60837


namespace NUMINAMATH_CALUDE_sophie_cookies_l608_60891

/-- Represents the number of cookies Sophie bought -/
def num_cookies : ℕ := sorry

/-- The cost of a single cupcake -/
def cupcake_cost : ℚ := 2

/-- The cost of a single doughnut -/
def doughnut_cost : ℚ := 1

/-- The cost of a single slice of apple pie -/
def apple_pie_slice_cost : ℚ := 2

/-- The cost of a single cookie -/
def cookie_cost : ℚ := 6/10

/-- The total amount Sophie spent -/
def total_spent : ℚ := 33

/-- The number of cupcakes Sophie bought -/
def num_cupcakes : ℕ := 5

/-- The number of doughnuts Sophie bought -/
def num_doughnuts : ℕ := 6

/-- The number of apple pie slices Sophie bought -/
def num_apple_pie_slices : ℕ := 4

theorem sophie_cookies :
  num_cookies = 15 ∧
  (num_cupcakes : ℚ) * cupcake_cost +
  (num_doughnuts : ℚ) * doughnut_cost +
  (num_apple_pie_slices : ℚ) * apple_pie_slice_cost +
  (num_cookies : ℚ) * cookie_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_sophie_cookies_l608_60891


namespace NUMINAMATH_CALUDE_min_correct_answers_to_advance_l608_60892

/-- Represents a math competition with specified rules -/
structure MathCompetition where
  total_questions : ℕ
  points_correct : ℕ
  points_incorrect : ℕ
  min_score : ℕ

/-- Calculates the score for a given number of correct answers in the competition -/
def calculate_score (comp : MathCompetition) (correct_answers : ℕ) : ℤ :=
  (correct_answers * comp.points_correct : ℤ) - 
  ((comp.total_questions - correct_answers) * comp.points_incorrect : ℤ)

/-- Theorem stating the minimum number of correct answers needed to advance -/
theorem min_correct_answers_to_advance (comp : MathCompetition) 
  (h1 : comp.total_questions = 25)
  (h2 : comp.points_correct = 4)
  (h3 : comp.points_incorrect = 1)
  (h4 : comp.min_score = 60) :
  ∃ (n : ℕ), n = 17 ∧ 
    (∀ (m : ℕ), m ≥ n → calculate_score comp m ≥ comp.min_score) ∧
    (∀ (m : ℕ), m < n → calculate_score comp m < comp.min_score) :=
  sorry

end NUMINAMATH_CALUDE_min_correct_answers_to_advance_l608_60892


namespace NUMINAMATH_CALUDE_math_eng_only_is_five_l608_60871

structure SubjectDistribution where
  total : ℕ
  mathEngOnly : ℕ
  mathHistOnly : ℕ
  engHistOnly : ℕ
  allThree : ℕ
  mathOnly : ℕ
  engOnly : ℕ
  histOnly : ℕ

def isValidDistribution (d : SubjectDistribution) : Prop :=
  d.total = 228 ∧
  d.mathEngOnly + d.mathHistOnly + d.engHistOnly + d.allThree + d.mathOnly + d.engOnly + d.histOnly = d.total ∧
  d.mathEngOnly = d.mathOnly ∧
  d.engOnly = 0 ∧
  d.histOnly = 0 ∧
  d.mathHistOnly = 6 ∧
  d.engHistOnly = 5 * d.allThree ∧
  d.allThree > 0 ∧
  d.allThree % 2 = 0

theorem math_eng_only_is_five (d : SubjectDistribution) (h : isValidDistribution d) : 
  d.mathEngOnly = 5 := by
  sorry

end NUMINAMATH_CALUDE_math_eng_only_is_five_l608_60871


namespace NUMINAMATH_CALUDE_rita_butterfly_hours_l608_60889

theorem rita_butterfly_hours : ∀ (total_required hours_backstroke hours_breaststroke monthly_freestyle_sidestroke months : ℕ),
  total_required = 1500 →
  hours_backstroke = 50 →
  hours_breaststroke = 9 →
  monthly_freestyle_sidestroke = 220 →
  months = 6 →
  total_required - (hours_backstroke + hours_breaststroke + monthly_freestyle_sidestroke * months) = 121 :=
by
  sorry

#check rita_butterfly_hours

end NUMINAMATH_CALUDE_rita_butterfly_hours_l608_60889


namespace NUMINAMATH_CALUDE_joe_initial_cars_l608_60821

/-- Given that Joe will have 62 cars after getting 12 more, prove that he initially had 50 cars. -/
theorem joe_initial_cars : 
  ∀ (initial_cars : ℕ), 
  (initial_cars + 12 = 62) → 
  initial_cars = 50 := by
sorry

end NUMINAMATH_CALUDE_joe_initial_cars_l608_60821


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l608_60826

open Real

theorem trigonometric_simplification (α x : ℝ) :
  ((sin (π - α) * cos (3*π - α) * tan (-α - π) * tan (α - 2*π)) / 
   (tan (4*π - α) * sin (5*π + α)) = sin α) ∧
  ((sin (3*π - x) / tan (5*π - x)) * 
   (1 / (tan (5*π/2 - x) * tan (4.5*π - x))) * 
   (cos (2*π - x) / sin (-x)) = sin x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l608_60826


namespace NUMINAMATH_CALUDE_probability_non_adjacent_rational_terms_l608_60829

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def non_adjacent_arrangements (irrational_terms rational_terms : ℕ) : ℕ :=
  Nat.factorial irrational_terms * Nat.choose (irrational_terms + 1) rational_terms

theorem probability_non_adjacent_rational_terms 
  (total_terms : ℕ) 
  (irrational_terms : ℕ) 
  (rational_terms : ℕ) 
  (h1 : total_terms = irrational_terms + rational_terms) 
  (h2 : irrational_terms = 6) 
  (h3 : rational_terms = 3) :
  (non_adjacent_arrangements irrational_terms rational_terms : ℚ) / 
  (total_arrangements total_terms : ℚ) = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_adjacent_rational_terms_l608_60829


namespace NUMINAMATH_CALUDE_age_difference_l608_60849

/-- Given three people A, B, and C, where the total age of A and B is more than
    the total age of B and C, and C is 13 years younger than A, prove that the
    difference between (A + B) and (B + C) is 13 years. -/
theorem age_difference (A B C : ℕ) 
  (h1 : A + B > B + C) 
  (h2 : C = A - 13) : 
  (A + B) - (B + C) = 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l608_60849


namespace NUMINAMATH_CALUDE_special_triangle_angle_exists_l608_60839

/-- A triangle with a circumcircle where one altitude is tangent to the circumcircle -/
structure SpecialTriangle where
  /-- The triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The circumcircle of the triangle -/
  circumcircle : Set (ℝ × ℝ)
  /-- An altitude of the triangle -/
  altitude : Set (ℝ × ℝ)
  /-- The altitude is tangent to the circumcircle -/
  is_tangent : altitude ∩ circumcircle ≠ ∅

/-- The angles of a triangle -/
def angles (t : SpecialTriangle) : Set ℝ := sorry

/-- Theorem: In a SpecialTriangle, there exists an angle greater than 90° and less than 135° -/
theorem special_triangle_angle_exists (t : SpecialTriangle) :
  ∃ θ ∈ angles t, 90 < θ ∧ θ < 135 := by sorry

end NUMINAMATH_CALUDE_special_triangle_angle_exists_l608_60839


namespace NUMINAMATH_CALUDE_exists_k_for_prime_divisor_inequality_l608_60810

/-- The largest prime divisor of a positive integer greater than 1 -/
def largest_prime_divisor (n : ℕ) : ℕ :=
  sorry

/-- Theorem: For any odd prime q, there exists a positive integer k such that
    the largest prime divisor of (q^(2^k) - 1) is less than q, and
    q is less than the largest prime divisor of (q^(2^k) + 1) -/
theorem exists_k_for_prime_divisor_inequality (q : ℕ) (hq : q.Prime) (hq_odd : q % 2 = 1) :
  ∃ k : ℕ, k > 0 ∧
    largest_prime_divisor (q^(2^k) - 1) < q ∧
    q < largest_prime_divisor (q^(2^k) + 1) :=
  sorry

end NUMINAMATH_CALUDE_exists_k_for_prime_divisor_inequality_l608_60810


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l608_60830

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_a4 : a 4 = 8) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) - a n = d) ∧ d = 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l608_60830


namespace NUMINAMATH_CALUDE_binomial_sum_abs_coefficients_l608_60859

theorem binomial_sum_abs_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ),
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 2187 :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_abs_coefficients_l608_60859


namespace NUMINAMATH_CALUDE_sara_peaches_l608_60802

theorem sara_peaches (initial_peaches additional_peaches : ℝ) 
  (h1 : initial_peaches = 61.0) 
  (h2 : additional_peaches = 24.0) : 
  initial_peaches + additional_peaches = 85.0 := by
  sorry

end NUMINAMATH_CALUDE_sara_peaches_l608_60802


namespace NUMINAMATH_CALUDE_integer_triplet_sum_product_l608_60855

theorem integer_triplet_sum_product (a b c : ℤ) : 
  a < 4 ∧ b < 4 ∧ c < 4 ∧ 
  a < b ∧ b < c ∧ 
  a + b + c = a * b * c →
  ((a, b, c) = (1, 2, 3) ∨ 
   (a, b, c) = (-3, -2, -1) ∨ 
   (a, b, c) = (-1, 0, 1) ∨ 
   (a, b, c) = (-2, 0, 2) ∨ 
   (a, b, c) = (-3, 0, 3)) :=
by sorry

end NUMINAMATH_CALUDE_integer_triplet_sum_product_l608_60855


namespace NUMINAMATH_CALUDE_exists_divisible_sum_squares_l608_60834

/-- For any polynomial P with real coefficients and positive integer n,
    there exists a polynomial Q with real coefficients such that
    (1+x^2)^n divides P(x)^2 + Q(x)^2. -/
theorem exists_divisible_sum_squares (P : Polynomial ℝ) (n : ℕ) (hn : n > 0) :
  ∃ Q : Polynomial ℝ, (1 + X^2)^n ∣ P^2 + Q^2 := by
  sorry

#check exists_divisible_sum_squares

end NUMINAMATH_CALUDE_exists_divisible_sum_squares_l608_60834


namespace NUMINAMATH_CALUDE_missing_number_proof_l608_60860

def known_numbers : List ℤ := [744, 745, 747, 748, 752, 752, 753, 755, 755]

theorem missing_number_proof (total_count : ℕ) (average : ℤ) (missing_number : ℤ) :
  total_count = 10 →
  average = 750 →
  missing_number = 1549 →
  (List.sum known_numbers + missing_number) / total_count = average :=
by sorry

end NUMINAMATH_CALUDE_missing_number_proof_l608_60860


namespace NUMINAMATH_CALUDE_sqrt_5_minus_1_bounds_l608_60870

theorem sqrt_5_minus_1_bounds : 1 < Real.sqrt 5 - 1 ∧ Real.sqrt 5 - 1 < 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_minus_1_bounds_l608_60870


namespace NUMINAMATH_CALUDE_quadrupled_bonus_remainder_l608_60885

/-- Represents the bonus pool and its division among employees -/
structure BonusPool :=
  (total : ℕ)
  (employees : ℕ)
  (remainder : ℕ)

/-- Theorem stating the relationship between the original and quadrupled bonus pools -/
theorem quadrupled_bonus_remainder
  (original : BonusPool)
  (h1 : original.employees = 8)
  (h2 : original.remainder = 5)
  (quadrupled : BonusPool)
  (h3 : quadrupled.employees = original.employees)
  (h4 : quadrupled.total = 4 * original.total) :
  quadrupled.remainder = 4 := by
sorry

end NUMINAMATH_CALUDE_quadrupled_bonus_remainder_l608_60885


namespace NUMINAMATH_CALUDE_total_students_present_l608_60820

/-- Calculates the total number of students present across four kindergarten sessions -/
theorem total_students_present
  (morning_registered : ℕ) (morning_absent : ℕ)
  (early_afternoon_registered : ℕ) (early_afternoon_absent : ℕ)
  (late_afternoon_registered : ℕ) (late_afternoon_absent : ℕ)
  (evening_registered : ℕ) (evening_absent : ℕ)
  (h1 : morning_registered = 25) (h2 : morning_absent = 3)
  (h3 : early_afternoon_registered = 24) (h4 : early_afternoon_absent = 4)
  (h5 : late_afternoon_registered = 30) (h6 : late_afternoon_absent = 5)
  (h7 : evening_registered = 35) (h8 : evening_absent = 7) :
  (morning_registered - morning_absent) +
  (early_afternoon_registered - early_afternoon_absent) +
  (late_afternoon_registered - late_afternoon_absent) +
  (evening_registered - evening_absent) = 95 :=
by sorry

end NUMINAMATH_CALUDE_total_students_present_l608_60820


namespace NUMINAMATH_CALUDE_complex_number_properties_l608_60845

open Complex

theorem complex_number_properties (z : ℂ) (h : I * (z + 1) = -2 + 2*I) : 
  z.im = 2 ∧ abs (z / (1 - 2*I)) ^ 2015 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l608_60845


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l608_60867

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a - b = -10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l608_60867


namespace NUMINAMATH_CALUDE_quadratic_points_order_l608_60880

/-- Given a quadratic function f(x) = x² - 4x - m, prove that the y-coordinates
    of the points (-1, y₃), (3, y₂), and (2, y₁) on this function satisfy y₃ > y₂ > y₁ -/
theorem quadratic_points_order (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x - m
  let y₁ : ℝ := f 2
  let y₂ : ℝ := f 3
  let y₃ : ℝ := f (-1)
  y₃ > y₂ ∧ y₂ > y₁ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_points_order_l608_60880


namespace NUMINAMATH_CALUDE_monthly_spending_is_99_l608_60823

def original_price : ℚ := 50
def price_increase_percent : ℚ := 10
def discount_percent : ℚ := 10
def monthly_purchase : ℚ := 2

def calculate_monthly_spending (original_price price_increase_percent discount_percent monthly_purchase : ℚ) : ℚ :=
  let new_price := original_price * (1 + price_increase_percent / 100)
  let discounted_price := new_price * (1 - discount_percent / 100)
  discounted_price * monthly_purchase

theorem monthly_spending_is_99 :
  calculate_monthly_spending original_price price_increase_percent discount_percent monthly_purchase = 99 := by
  sorry

end NUMINAMATH_CALUDE_monthly_spending_is_99_l608_60823


namespace NUMINAMATH_CALUDE_min_value_constraint_l608_60886

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

-- Define the theorem
theorem min_value_constraint (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 2, f a x ≥ 3) ∧ (∃ x ∈ Set.Icc 0 2, f a x = 3) ↔ 
  a = 1 - Real.sqrt 2 ∨ a = 5 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_constraint_l608_60886


namespace NUMINAMATH_CALUDE_ben_picking_peas_l608_60861

/-- Given that Ben can pick 56 sugar snap peas in 7 minutes, 
    prove that it takes 9 minutes to pick 72 sugar snap peas. -/
theorem ben_picking_peas (rate : ℚ) (h1 : rate = 56 / 7) : 72 / rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_ben_picking_peas_l608_60861


namespace NUMINAMATH_CALUDE_gala_trees_count_l608_60866

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  fuji : ℕ
  gala : ℕ
  cross_pollinated : ℕ

/-- The conditions of the orchard problem -/
def orchard_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.fuji + o.cross_pollinated = 204 ∧
  o.fuji = 3 * o.total / 4 ∧
  o.total = o.fuji + o.gala + o.cross_pollinated

theorem gala_trees_count (o : Orchard) (h : orchard_conditions o) : o.gala = 60 := by
  sorry

end NUMINAMATH_CALUDE_gala_trees_count_l608_60866


namespace NUMINAMATH_CALUDE_cheese_arrangement_count_l608_60807

/-- Represents a cheese flavor -/
inductive Flavor
| Paprika
| BearsGarlic

/-- Represents a cheese slice -/
structure CheeseSlice :=
  (flavor : Flavor)

/-- Represents a box of cheese slices -/
structure CheeseBox :=
  (slices : List CheeseSlice)

/-- Represents an arrangement of cheese slices in two boxes -/
structure CheeseArrangement :=
  (box1 : CheeseBox)
  (box2 : CheeseBox)

/-- Checks if two arrangements are equivalent under rotation -/
def areEquivalentUnderRotation (arr1 arr2 : CheeseArrangement) : Prop :=
  sorry

/-- Counts the number of distinct arrangements -/
def countDistinctArrangements (arrangements : List CheeseArrangement) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem cheese_arrangement_count :
  let totalSlices := 16
  let paprikaSlices := 8
  let bearsGarlicSlices := 8
  let allArrangements := sorry -- List of all possible arrangements
  countDistinctArrangements allArrangements = 234 :=
sorry

end NUMINAMATH_CALUDE_cheese_arrangement_count_l608_60807


namespace NUMINAMATH_CALUDE_some_fast_animals_are_pets_l608_60811

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (Wolf FastAnimal Pet : U → Prop)

-- State the theorem
theorem some_fast_animals_are_pets
  (h1 : ∀ x, Wolf x → FastAnimal x)
  (h2 : ∃ x, Pet x ∧ Wolf x) :
  ∃ x, FastAnimal x ∧ Pet x :=
sorry

end NUMINAMATH_CALUDE_some_fast_animals_are_pets_l608_60811


namespace NUMINAMATH_CALUDE_hazel_eyed_brunettes_l608_60873

/-- Represents the characteristics of students in a class -/
structure ClassCharacteristics where
  total_students : ℕ
  green_eyed_blondes : ℕ
  brunettes : ℕ
  hazel_eyed : ℕ

/-- Theorem: Number of hazel-eyed brunettes in the class -/
theorem hazel_eyed_brunettes (c : ClassCharacteristics) 
  (h1 : c.total_students = 60)
  (h2 : c.green_eyed_blondes = 20)
  (h3 : c.brunettes = 35)
  (h4 : c.hazel_eyed = 25) :
  c.total_students - (c.brunettes + c.green_eyed_blondes) = c.hazel_eyed - (c.total_students - c.brunettes) :=
by sorry

#check hazel_eyed_brunettes

end NUMINAMATH_CALUDE_hazel_eyed_brunettes_l608_60873


namespace NUMINAMATH_CALUDE_sector_central_angle_l608_60805

theorem sector_central_angle (arc_length : Real) (area : Real) :
  arc_length = 2 * Real.pi ∧ area = 5 * Real.pi →
  ∃ (central_angle : Real),
    central_angle = 72 ∧
    central_angle * Real.pi / 180 = 2 * Real.pi * Real.pi / (5 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l608_60805


namespace NUMINAMATH_CALUDE_power_three_2023_mod_10_l608_60876

theorem power_three_2023_mod_10 : 3^2023 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_three_2023_mod_10_l608_60876


namespace NUMINAMATH_CALUDE_inequality_problem_l608_60816

theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) :
  (1/a > 1/b) ∧ (abs a > -b) ∧ (Real.sqrt (-a) > Real.sqrt (-b)) ∧ ¬(1/(a-b) > 1/a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l608_60816


namespace NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l608_60877

theorem five_fourths_of_twelve_fifths : (5 / 4 : ℚ) * (12 / 5 : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_twelve_fifths_l608_60877


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l608_60881

/-- Given a triangle with side lengths a, b, and c, the sum of the ratios of each side length
    to the square root of twice the sum of squares of the other two sides minus the square
    of the current side is greater than or equal to the square root of 3. -/
theorem triangle_inequality_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a / Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) +
  (b / Real.sqrt (2 * c^2 + 2 * a^2 - b^2)) +
  (c / Real.sqrt (2 * a^2 + 2 * b^2 - c^2)) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l608_60881


namespace NUMINAMATH_CALUDE_square_of_recurring_third_l608_60808

/-- The repeating decimal 0.333... --/
def recurring_third : ℚ := 1/3

/-- Theorem: The square of 0.333... is equal to 1/9 --/
theorem square_of_recurring_third : recurring_third ^ 2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_square_of_recurring_third_l608_60808


namespace NUMINAMATH_CALUDE_jessica_pie_count_l608_60841

theorem jessica_pie_count (apples_per_serving : ℝ) (num_guests : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℝ)
  (h1 : apples_per_serving = 1.5)
  (h2 : num_guests = 12)
  (h3 : servings_per_pie = 8)
  (h4 : apples_per_guest = 3) :
  (num_guests * apples_per_guest / apples_per_serving) / servings_per_pie = 3 := by
  sorry

end NUMINAMATH_CALUDE_jessica_pie_count_l608_60841


namespace NUMINAMATH_CALUDE_rope_ratio_proof_l608_60878

theorem rope_ratio_proof (total_length shorter_length longer_length : ℝ) 
  (h1 : total_length = 60)
  (h2 : shorter_length = 20)
  (h3 : longer_length = total_length - shorter_length) :
  longer_length / shorter_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_rope_ratio_proof_l608_60878


namespace NUMINAMATH_CALUDE_probability_three_teachers_same_gate_l608_60832

-- Define the number of teachers and gates
def num_teachers : ℕ := 12
def num_gates : ℕ := 3
def teachers_per_gate : ℕ := 4

-- Define the probability function
noncomputable def probability_same_gate : ℚ :=
  3 / 55

-- Theorem statement
theorem probability_three_teachers_same_gate :
  probability_same_gate = 3 / 55 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_three_teachers_same_gate_l608_60832


namespace NUMINAMATH_CALUDE_cubic_properties_l608_60817

theorem cubic_properties :
  (∀ x : ℝ, x^3 > 0 → x > 0) ∧
  (∀ x : ℝ, x < 1 → x^3 < x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_properties_l608_60817


namespace NUMINAMATH_CALUDE_largest_valid_number_l608_60846

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → (∃! p, 10^p * d ≤ n ∧ n < 10^(p+1) * d)) ∧
  n % 11 = 0

theorem largest_valid_number : 
  (∀ n : ℕ, is_valid_number n → n ≤ 987652413) ∧ 
  is_valid_number 987652413 := by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l608_60846


namespace NUMINAMATH_CALUDE_garden_perimeter_is_72_l608_60806

/-- A rectangular garden with specific properties -/
structure Garden where
  /-- The shorter side of the garden -/
  short_side : ℝ
  /-- The longer side of the garden -/
  long_side : ℝ
  /-- The diagonal of the garden is 34 meters -/
  diagonal_eq : short_side ^ 2 + long_side ^ 2 = 34 ^ 2
  /-- The area of the garden is 240 square meters -/
  area_eq : short_side * long_side = 240
  /-- The longer side is three times the shorter side -/
  side_ratio : long_side = 3 * short_side

/-- The perimeter of a rectangular garden -/
def perimeter (g : Garden) : ℝ :=
  2 * (g.short_side + g.long_side)

/-- Theorem stating that the perimeter of the garden is 72 meters -/
theorem garden_perimeter_is_72 (g : Garden) : perimeter g = 72 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_is_72_l608_60806


namespace NUMINAMATH_CALUDE_class_average_problem_l608_60840

theorem class_average_problem (n1 n2 : ℕ) (avg2 avg_all : ℝ) (h1 : n1 = 30) (h2 : n2 = 50) 
  (h3 : avg2 = 60) (h4 : avg_all = 56.25) : 
  (n1 + n2 : ℝ) * avg_all = n1 * ((n1 + n2 : ℝ) * avg_all - n2 * avg2) / n1 + n2 * avg2 := by
  sorry

#check class_average_problem

end NUMINAMATH_CALUDE_class_average_problem_l608_60840


namespace NUMINAMATH_CALUDE_investment_profit_l608_60842

/-- The daily price increase rate of the shares -/
def daily_increase : ℝ := 1.1

/-- The amount spent on shares each day in rubles -/
def daily_investment : ℝ := 1000

/-- The number of days the businessman buys shares -/
def investment_days : ℕ := 3

/-- The number of days until the shares are sold -/
def total_days : ℕ := 4

/-- Calculate the total profit from the share investment -/
def calculate_profit : ℝ :=
  let total_investment := daily_investment * investment_days
  let total_value := daily_investment * (daily_increase^3 + daily_increase^2 + daily_increase)
  total_value - total_investment

theorem investment_profit :
  calculate_profit = 641 := by sorry

end NUMINAMATH_CALUDE_investment_profit_l608_60842


namespace NUMINAMATH_CALUDE_minimum_packages_shipped_minimum_packages_value_l608_60887

def sarahs_load : ℕ := 18
def ryans_load : ℕ := 11

theorem minimum_packages_shipped (n : ℕ) :
  (n % sarahs_load = 0) ∧ (n % ryans_load = 0) →
  n ≥ Nat.lcm sarahs_load ryans_load :=
by sorry

theorem minimum_packages_value :
  Nat.lcm sarahs_load ryans_load = 198 :=
by sorry

end NUMINAMATH_CALUDE_minimum_packages_shipped_minimum_packages_value_l608_60887


namespace NUMINAMATH_CALUDE_beetle_speed_l608_60865

/-- Beetle's speed in km/h given ant's speed and relative distance -/
theorem beetle_speed (ant_distance : ℝ) (time_minutes : ℝ) (beetle_relative_distance : ℝ) :
  ant_distance = 1000 →
  time_minutes = 30 →
  beetle_relative_distance = 0.9 →
  (ant_distance * beetle_relative_distance / time_minutes) * (60 / 1000) = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_beetle_speed_l608_60865


namespace NUMINAMATH_CALUDE_waiter_new_customers_l608_60828

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 33) 
  (h2 : customers_left = 31) 
  (h3 : final_customers = 28) : 
  final_customers - (initial_customers - customers_left) = 26 := by
  sorry

end NUMINAMATH_CALUDE_waiter_new_customers_l608_60828


namespace NUMINAMATH_CALUDE_sum_of_solutions_equation_l608_60824

theorem sum_of_solutions_equation (x : ℝ) :
  (x ≠ 1 ∧ x ≠ -1) →
  ((-12 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 8 / (x - 1)) →
  ∃ (y : ℝ), (y ≠ 1 ∧ y ≠ -1) ∧
    ((-12 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 8 / (y - 1)) ∧
    (x + y = 10 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equation_l608_60824


namespace NUMINAMATH_CALUDE_shopping_cost_difference_l608_60848

theorem shopping_cost_difference (shirt wallet food : ℝ) 
  (shirt_cost : shirt = wallet / 3)
  (wallet_more_expensive : wallet > food)
  (food_cost : food = 30)
  (total_spent : shirt + wallet + food = 150) :
  wallet - food = 60 := by
  sorry

end NUMINAMATH_CALUDE_shopping_cost_difference_l608_60848


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l608_60864

theorem least_sum_of_bases (a b : ℕ+) : 
  (7 * a.val + 8 = 8 * b.val + 7) → 
  (∀ c d : ℕ+, (7 * c.val + 8 = 8 * d.val + 7) → (c.val + d.val ≥ a.val + b.val)) →
  a.val + b.val = 17 := by
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l608_60864


namespace NUMINAMATH_CALUDE_thirtiethDigitOf_1_11_plus_1_13_l608_60883

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in the sum of the decimal representations of two rational numbers -/
def nthDigitAfterDecimal (q₁ q₂ : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 30th digit after the decimal point in the sum of 1/11 and 1/13 is 2 -/
theorem thirtiethDigitOf_1_11_plus_1_13 : 
  nthDigitAfterDecimal (1/11) (1/13) 30 = 2 := by sorry

end NUMINAMATH_CALUDE_thirtiethDigitOf_1_11_plus_1_13_l608_60883


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l608_60869

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p, p < 20 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 667 ∧ has_no_prime_factors_less_than_20 667) ∧
  (∀ m : ℕ, m < 667 → ¬(is_composite m ∧ has_no_prime_factors_less_than_20 m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l608_60869


namespace NUMINAMATH_CALUDE_quadratic_inequality_l608_60844

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_inequality (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ x < 0 ∨ x > 2) →
  (a = -2 ∧ b = 0) ∧
  (∀ m : ℝ,
    (m = 0 → ∀ x, ¬(f a b x < m^2 - 1)) ∧
    (m > 0 → ∀ x, f a b x < m^2 - 1 ↔ 1 - m < x ∧ x < 1 + m) ∧
    (m < 0 → ∀ x, f a b x < m^2 - 1 ↔ 1 + m < x ∧ x < 1 - m)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l608_60844
