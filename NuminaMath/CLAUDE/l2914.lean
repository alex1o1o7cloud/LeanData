import Mathlib

namespace NUMINAMATH_CALUDE_power_multiplication_l2914_291494

theorem power_multiplication (m : ℝ) : m^3 * m^2 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2914_291494


namespace NUMINAMATH_CALUDE_triangle_rectangle_ratio_l2914_291477

theorem triangle_rectangle_ratio : 
  ∀ (t w : ℝ), 
  t > 0 → w > 0 →
  3 * t = 24 →
  2 * (2 * w) + 2 * w = 24 →
  t / w = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_rectangle_ratio_l2914_291477


namespace NUMINAMATH_CALUDE_gcd_45345_34534_l2914_291482

theorem gcd_45345_34534 : Nat.gcd 45345 34534 = 71 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45345_34534_l2914_291482


namespace NUMINAMATH_CALUDE_correct_multiplication_result_l2914_291459

theorem correct_multiplication_result (result : ℕ) (wrong_digits : List ℕ) :
  result = 867559827931 ∧
  wrong_digits = [8, 6, 7, 5, 2, 7, 9] ∧
  (∃ n : ℕ, n * 98765 = result) →
  ∃ m : ℕ, m * 98765 = 888885 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_result_l2914_291459


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2914_291465

theorem least_common_multiple_first_ten : ∃ n : ℕ+, 
  (∀ k : ℕ+, k ≤ 10 → k ∣ n) ∧ 
  (∀ m : ℕ+, (∀ k : ℕ+, k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 := by
sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l2914_291465


namespace NUMINAMATH_CALUDE_number_of_subsets_complement_union_l2914_291497

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 3}

-- Define set B
def B : Finset Nat := {1, 2, 4}

-- Theorem statement
theorem number_of_subsets_complement_union (U A B : Finset Nat) : 
  Finset.card (Finset.powerset ((U \ B) ∪ A)) = 8 :=
sorry

end NUMINAMATH_CALUDE_number_of_subsets_complement_union_l2914_291497


namespace NUMINAMATH_CALUDE_stating_discount_calculation_l2914_291448

/-- Represents the profit percentage after discount -/
def profit_after_discount : ℝ := 25

/-- Represents the profit percentage without discount -/
def profit_without_discount : ℝ := 38.89

/-- Represents the discount percentage -/
def discount_percentage : ℝ := 10

/-- 
Theorem stating that given the profit percentages with and without discount, 
the discount percentage is 10%
-/
theorem discount_calculation (cost : ℝ) (cost_positive : cost > 0) :
  let selling_price := cost * (1 + profit_after_discount / 100)
  let marked_price := cost * (1 + profit_without_discount / 100)
  selling_price = marked_price * (1 - discount_percentage / 100) :=
by
  sorry


end NUMINAMATH_CALUDE_stating_discount_calculation_l2914_291448


namespace NUMINAMATH_CALUDE_bread_slices_left_l2914_291472

/-- The number of slices of bread Tony uses per sandwich -/
def slices_per_sandwich : ℕ := 2

/-- The number of sandwiches Tony made from Monday to Friday -/
def weekday_sandwiches : ℕ := 5

/-- The number of sandwiches Tony made on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The total number of slices in the loaf Tony started with -/
def initial_slices : ℕ := 22

/-- Theorem stating the number of bread slices left after Tony made sandwiches for the week -/
theorem bread_slices_left : 
  initial_slices - (slices_per_sandwich * (weekday_sandwiches + saturday_sandwiches)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_bread_slices_left_l2914_291472


namespace NUMINAMATH_CALUDE_solution_system_equations_l2914_291407

theorem solution_system_equations (w x y z : ℝ) 
  (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : w + x + y + z = 12)
  (h2 : w * x * y * z = w * x + w * y + w * z + x * y + x * z + y * z + 27) :
  w = 3 ∧ x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_system_equations_l2914_291407


namespace NUMINAMATH_CALUDE_certain_number_problem_l2914_291453

theorem certain_number_problem (x : ℝ) : 0.7 * x = (4/5 * 25) + 8 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2914_291453


namespace NUMINAMATH_CALUDE_complex_calculation_l2914_291422

theorem complex_calculation : (1 - Complex.I)^2 - (4 + 2 * Complex.I) / (1 - 2 * Complex.I) = -4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l2914_291422


namespace NUMINAMATH_CALUDE_potato_sack_problem_l2914_291471

theorem potato_sack_problem (original_potatoes : ℕ) : 
  original_potatoes - 69 - (2 * 69) - ((2 * 69) / 3) = 47 → 
  original_potatoes = 300 := by
sorry

end NUMINAMATH_CALUDE_potato_sack_problem_l2914_291471


namespace NUMINAMATH_CALUDE_corn_acres_l2914_291441

theorem corn_acres (total_land : ℕ) (beans_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : beans_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (beans_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acres_l2914_291441


namespace NUMINAMATH_CALUDE_miles_trumpets_l2914_291423

-- Define the number of body parts (as per typical human attributes)
def hands : Nat := 2
def head : Nat := 1
def fingers : Nat := 10

-- Define the number of each instrument based on the conditions
def guitars : Nat := hands + 2
def trombones : Nat := head + 2
def french_horns : Nat := guitars - 1
def trumpets : Nat := fingers - 3

-- Define the total number of instruments
def total_instruments : Nat := 17

-- Theorem to prove
theorem miles_trumpets :
  guitars + trombones + french_horns + trumpets = total_instruments ∧ trumpets = 7 := by
  sorry

end NUMINAMATH_CALUDE_miles_trumpets_l2914_291423


namespace NUMINAMATH_CALUDE_path_equivalence_arrow_sequence_equivalence_l2914_291444

/-- Represents the cyclic pattern of points in the path -/
def cycle_length : ℕ := 5

/-- Maps a point to its equivalent position in the cycle -/
def cycle_position (n : ℕ) : ℕ := n % cycle_length

/-- Theorem: The path from point 520 to 523 is equivalent to 0 to 3 in the cycle -/
theorem path_equivalence : 
  (cycle_position 520 = cycle_position 0) ∧ 
  (cycle_position 523 = cycle_position 3) := by
  sorry

/-- The sequence of arrows from 520 to 523 is the same as 0 to 3 -/
theorem arrow_sequence_equivalence : 
  ∀ (i : ℕ), i < 3 → 
  cycle_position (520 + i) = cycle_position i := by
  sorry

end NUMINAMATH_CALUDE_path_equivalence_arrow_sequence_equivalence_l2914_291444


namespace NUMINAMATH_CALUDE_unique_m_for_solution_set_minimum_a_for_inequality_l2914_291408

-- Define the function f
def f (x : ℝ) := |2 * x - 1|

-- Part 1
theorem unique_m_for_solution_set :
  ∃! m : ℝ, m > 0 ∧
  (∀ x : ℝ, |2 * (x + 1/2) - 1| ≤ 2 * m + 1 ↔ x ≤ -2 ∨ x ≥ 2) ∧
  m = 3/2 :=
sorry

-- Part 2
theorem minimum_a_for_inequality :
  ∃ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧
  (∀ a' : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a'/(2^y) + |2*x + 3|) → a ≤ a') ∧
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_m_for_solution_set_minimum_a_for_inequality_l2914_291408


namespace NUMINAMATH_CALUDE_number_exists_l2914_291492

theorem number_exists : ∃ x : ℝ, 0.75 * x = 0.3 * 1000 + 250 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l2914_291492


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l2914_291491

theorem complementary_angles_difference (a b : Real) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of angles is 5:4
  (max a b - min a b) = 10 :=  -- positive difference is 10
by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l2914_291491


namespace NUMINAMATH_CALUDE_f_neg_two_eq_eleven_l2914_291421

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 1

/-- Theorem: f(-2) = 11 -/
theorem f_neg_two_eq_eleven : f (-2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_eleven_l2914_291421


namespace NUMINAMATH_CALUDE_shopping_theorem_l2914_291464

def shopping_calculation (initial_amount : ℝ) 
  (baguette_cost : ℝ) (baguette_quantity : ℕ)
  (water_cost : ℝ) (water_quantity : ℕ)
  (chocolate_cost : ℝ) (chocolate_quantity : ℕ)
  (milk_cost : ℝ) (milk_discount : ℝ)
  (chips_cost : ℝ) (chips_discount : ℝ)
  (sales_tax : ℝ) : ℝ :=
  let baguette_total := baguette_cost * baguette_quantity
  let water_total := water_cost * water_quantity
  let chocolate_total := (chocolate_cost * 2) * 0.8 * (1 + sales_tax)
  let milk_total := milk_cost * (1 - milk_discount)
  let chips_total := (chips_cost + chips_cost * chips_discount) * (1 + sales_tax)
  let total_cost := baguette_total + water_total + chocolate_total + milk_total + chips_total
  initial_amount - total_cost

theorem shopping_theorem : 
  shopping_calculation 50 2 2 1 2 1.5 3 3.5 0.1 2.5 0.5 0.08 = 34.208 := by
  sorry

end NUMINAMATH_CALUDE_shopping_theorem_l2914_291464


namespace NUMINAMATH_CALUDE_function_derivative_existence_l2914_291463

open Set

theorem function_derivative_existence (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : 0 < a) (h2 : a < b)
  (h3 : ContinuousOn f (Icc a b))
  (h4 : DifferentiableOn ℝ f (Ioo a b)) :
  ∃ c ∈ Ioo a b, deriv f c = 1 / (a - c) + 1 / (b - c) + 1 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_existence_l2914_291463


namespace NUMINAMATH_CALUDE_sequence_formula_l2914_291467

/-- A sequence where S_n = n^2 * a_n for n ≥ 2, and a_1 = 1 -/
def sequence_a (n : ℕ) : ℚ := sorry

/-- Sum of the first n terms of the sequence -/
def S (n : ℕ) : ℚ := sorry

theorem sequence_formula :
  ∀ n : ℕ, n ≥ 1 →
  (∀ k : ℕ, k ≥ 2 → S k = k^2 * sequence_a k) →
  sequence_a 1 = 1 →
  sequence_a n = 1 / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l2914_291467


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2914_291432

/-- Two arithmetic sequences a and b with their respective sums A and B -/
def arithmetic_sequences (a b : ℕ → ℚ) (A B : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, 
    (∃ d₁ d₂ : ℚ, ∀ k : ℕ, a (k + 1) = a k + d₁ ∧ b (k + 1) = b k + d₂) ∧
    A n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1) ∧
    B n = n * b 1 + n * (n - 1) / 2 * (b 2 - b 1)

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) (A B : ℕ → ℚ) 
  (h : arithmetic_sequences a b A B) 
  (h_ratio : ∀ n : ℕ, A n / B n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, a n / b n = (4 * n - 3) / (6 * n - 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2914_291432


namespace NUMINAMATH_CALUDE_solve_equations_l2914_291470

theorem solve_equations :
  (∀ x : ℝ, 4 * x^2 = 9 ↔ x = 3/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, (1 - 2*x)^3 = 8 ↔ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equations_l2914_291470


namespace NUMINAMATH_CALUDE_units_digit_of_factorial_sum_l2914_291480

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

def units_digit (n : ℕ) : ℕ := sorry

theorem units_digit_of_factorial_sum :
  units_digit (sum_factorials 15) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_factorial_sum_l2914_291480


namespace NUMINAMATH_CALUDE_soup_feeding_theorem_l2914_291437

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remaining_adults_fed (total_cans : ℕ) (can_capacity : SoupCan) (children_fed : ℕ) : ℕ :=
  let cans_used_for_children := children_fed / can_capacity.children
  let remaining_cans := total_cans - cans_used_for_children
  remaining_cans * can_capacity.adults

/-- Theorem: Given 7 cans of soup, where each can feeds 4 adults or 7 children,
    if 21 children are fed, the remaining soup can feed 16 adults -/
theorem soup_feeding_theorem :
  let can_capacity : SoupCan := { adults := 4, children := 7 }
  let total_cans : ℕ := 7
  let children_fed : ℕ := 21
  remaining_adults_fed total_cans can_capacity children_fed = 16 := by
  sorry


end NUMINAMATH_CALUDE_soup_feeding_theorem_l2914_291437


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_c_value_l2914_291476

/-- If 2x+5 is a factor of 4x^3 + 19x^2 + cx + 45, then c = 40.5 -/
theorem polynomial_factor_implies_c_value :
  ∀ c : ℝ,
  (∀ x : ℝ, (2 * x + 5) ∣ (4 * x^3 + 19 * x^2 + c * x + 45)) →
  c = 40.5 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_c_value_l2914_291476


namespace NUMINAMATH_CALUDE_lcm_24_30_40_50_l2914_291442

theorem lcm_24_30_40_50 : Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 50)) = 600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_30_40_50_l2914_291442


namespace NUMINAMATH_CALUDE_trouser_original_price_l2914_291433

theorem trouser_original_price (sale_price : ℝ) (discount_percentage : ℝ) (original_price : ℝ) :
  sale_price = 10 →
  discount_percentage = 90 →
  sale_price = original_price * (1 - discount_percentage / 100) →
  original_price = 100 :=
by sorry

end NUMINAMATH_CALUDE_trouser_original_price_l2914_291433


namespace NUMINAMATH_CALUDE_paths_in_7x6_grid_l2914_291462

/-- The number of paths in a grid with specified horizontal and vertical steps --/
def numPaths (horizontal vertical : ℕ) : ℕ :=
  Nat.choose (horizontal + vertical) vertical

/-- Theorem stating that the number of paths in a 7x6 grid is 1716 --/
theorem paths_in_7x6_grid :
  numPaths 7 6 = 1716 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_7x6_grid_l2914_291462


namespace NUMINAMATH_CALUDE_complex_modulus_example_l2914_291468

theorem complex_modulus_example : 
  let z : ℂ := 1 - 2*I
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l2914_291468


namespace NUMINAMATH_CALUDE_fermat_number_prime_factor_l2914_291434

theorem fermat_number_prime_factor (n : ℕ) (hn : n ≥ 3) :
  ∃ p : ℕ, Prime p ∧ p ∣ (2^(2^n) + 1) ∧ p > 2^(n+2) * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_prime_factor_l2914_291434


namespace NUMINAMATH_CALUDE_school_bus_seats_l2914_291499

/-- Proves that the number of seats on each school bus is 9, given the conditions of the field trip. -/
theorem school_bus_seats (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 45) (h2 : num_buses = 5) (h3 : total_students % num_buses = 0) :
  total_students / num_buses = 9 := by
sorry

end NUMINAMATH_CALUDE_school_bus_seats_l2914_291499


namespace NUMINAMATH_CALUDE_lcm_perfect_square_l2914_291405

theorem lcm_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a*b) % (a*b*(a - b)) = 0) : 
  ∃ k : ℕ, Nat.lcm a b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_perfect_square_l2914_291405


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2914_291413

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a * x^2 + b * x + c = 0) → (
    (|r₁ - r₂| = 1 ∧ max r₁ r₂ = 4) ↔ (a = 1 ∧ b = -7 ∧ c = 12)
  ) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2914_291413


namespace NUMINAMATH_CALUDE_wage_restoration_l2914_291411

theorem wage_restoration (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.7 * original_wage
  let raise_percentage := 100 * (1 / 0.7 - 1)
  reduced_wage * (1 + raise_percentage / 100) = original_wage := by
sorry

end NUMINAMATH_CALUDE_wage_restoration_l2914_291411


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2914_291493

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2914_291493


namespace NUMINAMATH_CALUDE_binomial_12_9_l2914_291447

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l2914_291447


namespace NUMINAMATH_CALUDE_route_down_length_l2914_291401

/-- Proves that the length of the route down the mountain is 12 miles given the specified conditions. -/
theorem route_down_length (time_up time_down : ℝ) (rate_up : ℝ) (rate_down_factor : ℝ) :
  time_up = time_down →
  rate_down_factor = 1.5 →
  rate_up = 4 →
  time_up = 2 →
  rate_up * time_up = 8 →
  rate_down_factor * rate_up * time_down = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_route_down_length_l2914_291401


namespace NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l2914_291469

def first_four_composites : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_first_four_composites :
  units_digit (product_of_list first_four_composites) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_first_four_composites_l2914_291469


namespace NUMINAMATH_CALUDE_triangle_inequality_l2914_291419

theorem triangle_inequality (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_sum_angles : A + B + C = Real.pi)
  (h_area : S = (1/2) * a * b * Real.sin C)
  (h_side_a : a = b * Real.sin C / Real.sin A)
  (h_side_b : b = c * Real.sin A / Real.sin B)
  (h_side_c : c = a * Real.sin B / Real.sin C) :
  a^2 * Real.tan (A/2) + b^2 * Real.tan (B/2) + c^2 * Real.tan (C/2) ≥ 4 * S :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2914_291419


namespace NUMINAMATH_CALUDE_number_of_values_l2914_291484

theorem number_of_values (initial_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) (correct_mean : ℚ) :
  initial_mean = 180 →
  incorrect_value = 135 →
  correct_value = 155 →
  correct_mean = 180 + 2/3 →
  ∃ n : ℕ, 
    n * initial_mean = n * correct_mean - (correct_value - incorrect_value) ∧
    n = 60 :=
by sorry

end NUMINAMATH_CALUDE_number_of_values_l2914_291484


namespace NUMINAMATH_CALUDE_shadow_length_ratio_l2914_291458

theorem shadow_length_ratio (α β : Real) 
  (h1 : Real.tan (α - β) = 1 / 3)
  (h2 : Real.tan β = 1) :
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_ratio_l2914_291458


namespace NUMINAMATH_CALUDE_product_equality_l2914_291478

theorem product_equality : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l2914_291478


namespace NUMINAMATH_CALUDE_car_travel_time_l2914_291475

/-- Proves that a car with given specifications travels for 5 hours -/
theorem car_travel_time (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (fuel_used_ratio : ℝ) :
  speed = 50 →
  fuel_efficiency = 30 →
  tank_capacity = 10 →
  fuel_used_ratio = 0.8333333333333334 →
  (fuel_used_ratio * tank_capacity * fuel_efficiency) / speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_car_travel_time_l2914_291475


namespace NUMINAMATH_CALUDE_ring_price_calculation_l2914_291430

def total_revenue : ℕ := 80
def necklace_price : ℕ := 12
def necklaces_sold : ℕ := 4
def rings_sold : ℕ := 8

theorem ring_price_calculation : 
  ∃ (ring_price : ℕ), 
    necklaces_sold * necklace_price + rings_sold * ring_price = total_revenue ∧ 
    ring_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_ring_price_calculation_l2914_291430


namespace NUMINAMATH_CALUDE_college_student_count_l2914_291429

/-- Represents the number of students in each category -/
structure StudentCount where
  boys : ℕ
  girls : ℕ
  nonBinary : ℕ

/-- Calculates the total number of students -/
def totalStudents (s : StudentCount) : ℕ :=
  s.boys + s.girls + s.nonBinary

/-- Theorem: Given the ratio and number of girls, prove the total number of students -/
theorem college_student_count :
  ∀ (s : StudentCount),
    s.boys * 5 = s.girls * 8 →
    s.nonBinary * 5 = s.girls * 3 →
    s.girls = 400 →
    totalStudents s = 1280 := by
  sorry


end NUMINAMATH_CALUDE_college_student_count_l2914_291429


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2914_291449

-- Define the line equation
def line (k x y : ℝ) : Prop := 2 * k * x - y + 1 = 0

-- Define the ellipse equation
def ellipse (x y m : ℝ) : Prop := (x^2 / 9) + (y^2 / m) = 1

-- State the theorem
theorem ellipse_line_intersection_range (k : ℝ) :
  (∀ m : ℝ, (∀ x y : ℝ, line k x y → ellipse x y m → (∃ x' y' : ℝ, line k x' y' ∧ ellipse x' y' m))) →
  (∃ S : Set ℝ, S = {m : ℝ | m ∈ Set.Icc 1 9 ∪ Set.Ioi 9}) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_range_l2914_291449


namespace NUMINAMATH_CALUDE_odd_function_property_l2914_291427

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (fun x ↦ f (x + 2))) 
  (h_f_neg_one : f (-1) = 1) : 
  f 2017 + f 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l2914_291427


namespace NUMINAMATH_CALUDE_cally_shorts_count_l2914_291415

/-- Represents the number of clothing items a person has. -/
structure ClothingItems where
  whiteShirts : Nat
  coloredShirts : Nat
  shorts : Nat
  pants : Nat

/-- Calculate the total number of clothing items. -/
def totalItems (items : ClothingItems) : Nat :=
  items.whiteShirts + items.coloredShirts + items.shorts + items.pants

theorem cally_shorts_count (totalWashed : Nat) (cally : ClothingItems) (danny : ClothingItems)
    (h1 : totalWashed = 58)
    (h2 : cally.whiteShirts = 10)
    (h3 : cally.coloredShirts = 5)
    (h4 : cally.pants = 6)
    (h5 : danny.whiteShirts = 6)
    (h6 : danny.coloredShirts = 8)
    (h7 : danny.shorts = 10)
    (h8 : danny.pants = 6)
    (h9 : totalWashed = totalItems cally + totalItems danny) :
  cally.shorts = 7 := by
  sorry


end NUMINAMATH_CALUDE_cally_shorts_count_l2914_291415


namespace NUMINAMATH_CALUDE_original_calculation_l2914_291474

theorem original_calculation (x : ℚ) (h : ((x * 3) + 14) * 2 = 946) : ((x / 3) + 14) * 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_original_calculation_l2914_291474


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2914_291495

theorem subtraction_preserves_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2914_291495


namespace NUMINAMATH_CALUDE_square_side_length_range_l2914_291439

theorem square_side_length_range (area : ℝ) (h : area = 15) :
  ∃ side : ℝ, side^2 = area ∧ 3 < side ∧ side < 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_range_l2914_291439


namespace NUMINAMATH_CALUDE_rockville_baseball_league_members_l2914_291412

/-- The cost of a pair of cleats in dollars -/
def cleatCost : ℕ := 6

/-- The additional cost of a jersey compared to cleats in dollars -/
def jerseyAdditionalCost : ℕ := 7

/-- The total cost for all members in dollars -/
def totalCost : ℕ := 3360

/-- The number of sets (cleats and jersey) each member needs -/
def setsPerMember : ℕ := 2

/-- The cost of one set (cleats and jersey) for a member -/
def setCost : ℕ := cleatCost + (cleatCost + jerseyAdditionalCost)

/-- The total cost for one member -/
def memberCost : ℕ := setsPerMember * setCost

/-- The number of members in the Rockville Baseball League -/
def numberOfMembers : ℕ := totalCost / memberCost

theorem rockville_baseball_league_members :
  numberOfMembers = 88 := by sorry

end NUMINAMATH_CALUDE_rockville_baseball_league_members_l2914_291412


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l2914_291435

/-- Given a person who walks a certain distance at two different speeds, 
    prove that the slower speed is 10 km/hr. -/
theorem slower_speed_calculation 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 50) 
  (h2 : faster_speed = 14) 
  (h3 : additional_distance = 20) :
  let total_distance := actual_distance + additional_distance
  let time := total_distance / faster_speed
  let slower_speed := actual_distance / time
  slower_speed = 10 := by
sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l2914_291435


namespace NUMINAMATH_CALUDE_cara_age_is_40_l2914_291436

-- Define the ages as natural numbers
def grandmother_age : ℕ := 75
def mom_age : ℕ := grandmother_age - 15
def cara_age : ℕ := mom_age - 20

-- Theorem statement
theorem cara_age_is_40 : cara_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_cara_age_is_40_l2914_291436


namespace NUMINAMATH_CALUDE_cricket_players_count_l2914_291486

/-- The number of cricket players in a game, given the numbers of other players and the total. -/
theorem cricket_players_count 
  (hockey_players : ℕ) 
  (football_players : ℕ) 
  (softball_players : ℕ) 
  (total_players : ℕ) 
  (h1 : hockey_players = 12)
  (h2 : football_players = 16)
  (h3 : softball_players = 13)
  (h4 : total_players = 51)
  (h5 : total_players = hockey_players + football_players + softball_players + cricket_players) :
  cricket_players = 10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_count_l2914_291486


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l2914_291489

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 2.5
  let d : ℝ := 5
  let aₙ : ℝ := 62.5
  (aₙ - a₁) / d + 1 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l2914_291489


namespace NUMINAMATH_CALUDE_fraction_equality_l2914_291400

theorem fraction_equality (a b : ℚ) (h : 2 * a = 3 * b) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2914_291400


namespace NUMINAMATH_CALUDE_infinite_m_exist_l2914_291431

/-- A(n) is the number of subsets of {1,2,...,n} with sum of elements divisible by p -/
def A (p : ℕ) (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem infinite_m_exist (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p)
  (h_not_div : ¬(p^2 ∣ (2^(p-1) - 1))) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ (m : ℕ), m ∈ S →
      ∀ (k : ℤ), ∃ (q : ℤ), A p m - k = p * q := by
  sorry

end NUMINAMATH_CALUDE_infinite_m_exist_l2914_291431


namespace NUMINAMATH_CALUDE_committee_formation_count_l2914_291404

/-- The number of ways to choose k elements from n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players in the basketball team --/
def total_players : ℕ := 13

/-- The size of the committee to be formed --/
def committee_size : ℕ := 4

/-- The number of players to be chosen after including player A --/
def remaining_to_choose : ℕ := committee_size - 1

/-- The number of players to choose from after excluding player A --/
def players_to_choose_from : ℕ := total_players - 1

theorem committee_formation_count :
  choose players_to_choose_from remaining_to_choose = 220 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2914_291404


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l2914_291456

/-- A parabola with directrix y = 1/2 has the standard equation x^2 = -2y -/
theorem parabola_standard_equation (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, y = 1/2 → (x^2 = -2*p*y ↔ y = -x^2/(2*p))) →
  p = 1 :=
by sorry

#check parabola_standard_equation

end NUMINAMATH_CALUDE_parabola_standard_equation_l2914_291456


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l2914_291481

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l2914_291481


namespace NUMINAMATH_CALUDE_eleven_percent_of_700_is_77_l2914_291479

theorem eleven_percent_of_700_is_77 : (11 / 100) * 700 = 77 := by
  sorry

end NUMINAMATH_CALUDE_eleven_percent_of_700_is_77_l2914_291479


namespace NUMINAMATH_CALUDE_average_of_data_set_l2914_291426

def data_set : List ℝ := [9.8, 9.9, 10, 10.1, 10.2]

theorem average_of_data_set :
  (List.sum data_set) / (List.length data_set) = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_data_set_l2914_291426


namespace NUMINAMATH_CALUDE_bells_toll_together_l2914_291445

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 5) (hb : b = 8) (hc : c = 11) (hd : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 := by
sorry

end NUMINAMATH_CALUDE_bells_toll_together_l2914_291445


namespace NUMINAMATH_CALUDE_solve_cubic_equation_l2914_291461

theorem solve_cubic_equation :
  ∃ y : ℝ, (y - 5)^3 = (1/27)⁻¹ ∧ y = 8 := by sorry

end NUMINAMATH_CALUDE_solve_cubic_equation_l2914_291461


namespace NUMINAMATH_CALUDE_elevator_max_weight_next_person_l2914_291485

/-- Given an elevator scenario with adults and children, calculate the maximum weight of the next person that can enter without overloading the elevator. -/
theorem elevator_max_weight_next_person 
  (num_adults : ℕ) 
  (avg_weight_adults : ℝ) 
  (num_children : ℕ) 
  (avg_weight_children : ℝ) 
  (max_elevator_weight : ℝ) 
  (h1 : num_adults = 7) 
  (h2 : avg_weight_adults = 150) 
  (h3 : num_children = 5) 
  (h4 : avg_weight_children = 70) 
  (h5 : max_elevator_weight = 1500) :
  max_elevator_weight - (num_adults * avg_weight_adults + num_children * avg_weight_children) = 100 := by
  sorry

end NUMINAMATH_CALUDE_elevator_max_weight_next_person_l2914_291485


namespace NUMINAMATH_CALUDE_macaroon_weight_l2914_291440

theorem macaroon_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  num_bags = 4 →
  total_macaroons % num_bags = 0 →
  (total_macaroons - total_macaroons / num_bags) * weight_per_macaroon = 45 := by
  sorry

end NUMINAMATH_CALUDE_macaroon_weight_l2914_291440


namespace NUMINAMATH_CALUDE_no_infinite_prime_sequence_l2914_291417

theorem no_infinite_prime_sequence :
  ¬∃ (p : ℕ → ℕ), (∀ n, Prime (p n)) ∧
    (∀ k > 0, p k = 2 * p (k - 1) + 1 ∨ p k = 2 * p (k - 1) - 1) ∧
    (∀ m, ∃ n > m, p n ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_prime_sequence_l2914_291417


namespace NUMINAMATH_CALUDE_profit_calculation_l2914_291402

/-- The number of pencils John needs to sell to make a profit of $120 -/
def pencils_to_sell : ℕ := 1200

/-- The cost of buying 5 pencils in dollars -/
def buy_cost : ℚ := 7

/-- The number of pencils John buys at the given cost -/
def buy_quantity : ℕ := 5

/-- The selling price of 4 pencils in dollars -/
def sell_price : ℚ := 6

/-- The number of pencils John sells at the given price -/
def sell_quantity : ℕ := 4

/-- The desired profit in dollars -/
def target_profit : ℚ := 120

/-- Theorem stating that the number of pencils John needs to sell to make a profit of $120 is correct -/
theorem profit_calculation (p : ℕ) (h : p = pencils_to_sell) :
  (p : ℚ) * (sell_price / sell_quantity - buy_cost / buy_quantity) = target_profit :=
sorry

end NUMINAMATH_CALUDE_profit_calculation_l2914_291402


namespace NUMINAMATH_CALUDE_problem_solution_l2914_291409

theorem problem_solution (x : ℝ) : 
  3.5 * ((3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5)) = 2800.0000000000005 → x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2914_291409


namespace NUMINAMATH_CALUDE_inequality_proof_l2914_291488

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_condition : a + b + c = 1) :
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2914_291488


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l2914_291487

-- Define the polar coordinate equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Define the rectangular coordinate equation
def rectangular_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Theorem stating the equivalence of the two equations
theorem polar_to_rectangular_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (polar_equation ρ θ ↔ rectangular_equation x y) :=
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l2914_291487


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2914_291490

-- Define the quadratic function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define the linear function g
def g (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem quadratic_inequality (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 1 < x ∧ x < 2) →
  (a = 3 ∧ b = 2) ∧
  (∀ c : ℝ,
    (c > -1 → ∀ x, f a b x > c * g x ↔ x > c + 2 ∨ x < 1) ∧
    (c < -1 → ∀ x, f a b x > c * g x ↔ x > 1 ∨ x < c + 2) ∧
    (c = -1 → ∀ x, f a b x > c * g x ↔ x ≠ 1)) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_inequality_l2914_291490


namespace NUMINAMATH_CALUDE_certain_value_calculation_l2914_291452

theorem certain_value_calculation (N : ℝ) : 
  (0.4 * N = 180) → ((1/4) * (1/3) * (2/5) * N = 15) := by
  sorry

end NUMINAMATH_CALUDE_certain_value_calculation_l2914_291452


namespace NUMINAMATH_CALUDE_count_pairs_sum_squares_less_than_50_l2914_291466

theorem count_pairs_sum_squares_less_than_50 :
  (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 + p.2^2 < 50)
    (Finset.product (Finset.range 50) (Finset.range 50))).card = 32 :=
by sorry

end NUMINAMATH_CALUDE_count_pairs_sum_squares_less_than_50_l2914_291466


namespace NUMINAMATH_CALUDE_at_least_one_sum_of_primes_l2914_291446

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is the sum of two primes
def isSumOfTwoPrimes (n : ℕ) : Prop :=
  ∃ p q : ℕ, isPrime p ∧ isPrime q ∧ n = p + q

-- Theorem statement
theorem at_least_one_sum_of_primes (n : ℕ) (h : n > 1) :
  isSumOfTwoPrimes (2*n) ∨ isSumOfTwoPrimes (2*n + 2) ∨ isSumOfTwoPrimes (2*n + 4) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_sum_of_primes_l2914_291446


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2914_291455

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * (x - 175) * (x - 176) + 6

/-- The transformed quadratic function -/
def g (x : ℝ) : ℝ := f x - 6

/-- The roots of the transformed function -/
def root1 : ℝ := 175
def root2 : ℝ := 176

theorem quadratic_transformation :
  (g root1 = 0) ∧ 
  (g root2 = 0) ∧ 
  (root2 - root1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2914_291455


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l2914_291457

/-- An unfair eight-sided die with given probabilities -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℝ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℝ
  /-- The probability of rolling an 8 is 3/8 -/
  h1 : prob_eight = 3/8
  /-- The probability of rolling any number from 1 to 7 is 5/56 -/
  h2 : prob_others = 5/56
  /-- The sum of all probabilities is 1 -/
  h3 : prob_eight + 7 * prob_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℝ :=
  d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.prob_eight * 8

/-- Theorem stating that the expected value of rolling the unfair die is 5.5 -/
theorem unfair_die_expected_value (d : UnfairDie) :
  expected_value d = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_l2914_291457


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2914_291410

theorem rectangle_ratio (w : ℝ) (h1 : w > 0) (h2 : 2 * w + 2 * 10 = 36) : w / 10 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2914_291410


namespace NUMINAMATH_CALUDE_equation_solution_l2914_291451

theorem equation_solution : ∃ x : ℝ, (1 / 6 + 6 / x = 15 / x + 1 / 15) ∧ x = 90 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2914_291451


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2914_291406

theorem fraction_multiplication (x : ℚ) : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5060 = 759 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2914_291406


namespace NUMINAMATH_CALUDE_julia_tuesday_kids_l2914_291425

/-- The number of kids Julia played with on Tuesday -/
def kids_on_tuesday (total kids_monday kids_wednesday : ℕ) : ℕ :=
  total - kids_monday - kids_wednesday

theorem julia_tuesday_kids :
  kids_on_tuesday 34 17 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_julia_tuesday_kids_l2914_291425


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2914_291460

/-- Parabola P with equation y = x^2 + 3x + 1 -/
def P : ℝ → ℝ := λ x => x^2 + 3*x + 1

/-- Point Q -/
def Q : ℝ × ℝ := (10, 50)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x => m*(x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

/-- Main theorem -/
theorem parabola_line_intersection :
  ∃! (r s : ℝ), (∀ m, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 46 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2914_291460


namespace NUMINAMATH_CALUDE_unique_number_theorem_l2914_291450

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Generates the three numbers obtained by replacing one digit with 1 -/
def generateReplacedNumbers (n : ThreeDigitNumber) : List Nat :=
  [100 + 10 * n.tens + n.ones,
   100 * n.hundreds + 10 + n.ones,
   100 * n.hundreds + 10 * n.tens + 1]

/-- The main theorem stating that if the sum of replaced numbers is 1243,
    then the original number must be 566 -/
theorem unique_number_theorem (n : ThreeDigitNumber) :
  (generateReplacedNumbers n).sum = 1243 → n.toNat = 566 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_theorem_l2914_291450


namespace NUMINAMATH_CALUDE_problem_solution_l2914_291418

theorem problem_solution : ∃ Y : ℚ, 
  let A : ℚ := 2010 / 3
  let B : ℚ := A / 3
  Y = A + B ∧ Y = 893 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2914_291418


namespace NUMINAMATH_CALUDE_function_1_extrema_function_2_extrema_l2914_291498

-- Function 1
theorem function_1_extrema :
  (∀ x : ℝ, 2 * Real.sin x - 3 ≤ -1) ∧
  (∃ x : ℝ, 2 * Real.sin x - 3 = -1) ∧
  (∀ x : ℝ, 2 * Real.sin x - 3 ≥ -5) ∧
  (∃ x : ℝ, 2 * Real.sin x - 3 = -5) :=
sorry

-- Function 2
theorem function_2_extrema :
  (∀ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 ≤ 2) ∧
  (∃ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 = 2) ∧
  (∀ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 ≥ -1/4) ∧
  (∃ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 = -1/4) :=
sorry

end NUMINAMATH_CALUDE_function_1_extrema_function_2_extrema_l2914_291498


namespace NUMINAMATH_CALUDE_f_minimum_value_l2914_291403

noncomputable def f (x : ℝ) : ℝ := ((2 * x - 1) * Real.exp x) / (x - 1)

theorem f_minimum_value :
  ∃ (x_min : ℝ), x_min = 3 / 2 ∧
  (∀ x : ℝ, x ≠ 1 → f x ≥ f x_min) ∧
  f x_min = 4 * Real.exp (3 / 2) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2914_291403


namespace NUMINAMATH_CALUDE_buses_meet_at_two_pm_l2914_291414

/-- Represents a bus with departure and arrival times -/
structure Bus where
  departure : ℕ
  arrival : ℕ

/-- The time when two buses meet given their schedules -/
def meeting_time (bus1 bus2 : Bus) : ℕ :=
  sorry

theorem buses_meet_at_two_pm (bus1 bus2 : Bus)
  (h1 : bus1.departure = 11 ∧ bus1.arrival = 16)
  (h2 : bus2.departure = 12 ∧ bus2.arrival = 17) :
  meeting_time bus1 bus2 = 14 :=
sorry

end NUMINAMATH_CALUDE_buses_meet_at_two_pm_l2914_291414


namespace NUMINAMATH_CALUDE_james_hives_l2914_291454

theorem james_hives (honey_per_hive : ℝ) (jar_capacity : ℝ) (jars_to_buy : ℕ) :
  honey_per_hive = 20 →
  jar_capacity = 0.5 →
  jars_to_buy = 100 →
  (honey_per_hive * (jars_to_buy : ℝ) * jar_capacity) / honey_per_hive = 5 :=
by sorry

end NUMINAMATH_CALUDE_james_hives_l2914_291454


namespace NUMINAMATH_CALUDE_vanessa_age_proof_l2914_291424

def guesses : List Nat := [32, 34, 36, 40, 42, 45, 48, 52, 55, 58]

def vanessaAge : Nat := 53

theorem vanessa_age_proof :
  -- At least half of the guesses are too low
  (guesses.filter (· < vanessaAge)).length ≥ guesses.length / 2 ∧
  -- Three guesses are off by one
  (guesses.filter (fun x => x = vanessaAge - 1 ∨ x = vanessaAge + 1)).length = 3 ∧
  -- Vanessa's age is a prime number
  Nat.Prime vanessaAge ∧
  -- One guess is exactly correct
  guesses.contains vanessaAge ∧
  -- Vanessa's age is 53
  vanessaAge = 53 := by
  sorry

#eval vanessaAge

end NUMINAMATH_CALUDE_vanessa_age_proof_l2914_291424


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2914_291473

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 + 48 / 4 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2914_291473


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l2914_291438

theorem rectangle_width_decrease (L W : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let new_L := 1.5 * L
  let new_W := W * (L / new_L)
  (W - new_W) / W = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l2914_291438


namespace NUMINAMATH_CALUDE_remainder_problem_l2914_291420

theorem remainder_problem : (55^55 + 15) % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2914_291420


namespace NUMINAMATH_CALUDE_bananas_left_l2914_291416

/-- The number of bananas initially in the jar -/
def initial_bananas : ℕ := 46

/-- The number of bananas removed from the jar -/
def removed_bananas : ℕ := 5

/-- Theorem: The number of bananas left in the jar is 41 -/
theorem bananas_left : initial_bananas - removed_bananas = 41 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l2914_291416


namespace NUMINAMATH_CALUDE_root_differences_ratio_l2914_291443

open Real

/-- Given quadratic trinomials and their root differences, prove the ratio of differences squared is 3 -/
theorem root_differences_ratio (a b : ℝ) : 
  let f₁ := fun x : ℝ => x^2 + a*x + 3
  let f₂ := fun x : ℝ => x^2 + 2*x - b
  let f₃ := fun x : ℝ => x^2 + 2*(a-1)*x + b + 6
  let f₄ := fun x : ℝ => x^2 + (4-a)*x - 2*b - 3
  let A := sqrt (a^2 - 12)
  let B := sqrt (4 + 4*b)
  let C := sqrt (4*a^2 - 8*a - 4*b - 20)
  let D := sqrt (a^2 - 8*a + 8*b + 28)
  A^2 ≠ B^2 →
  (C^2 - D^2) / (A^2 - B^2) = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_root_differences_ratio_l2914_291443


namespace NUMINAMATH_CALUDE_hash_12_6_hash_general_form_l2914_291483

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ := 
  r * s + 2 * r

-- Theorem to prove
theorem hash_12_6 : hash 12 6 = 96 := by
  sorry

-- Axioms for the # operation
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 2

-- Additional theorem to prove the general form
theorem hash_general_form (r s : ℝ) : hash r s = r * s + 2 * r := by
  sorry

end NUMINAMATH_CALUDE_hash_12_6_hash_general_form_l2914_291483


namespace NUMINAMATH_CALUDE_intersecting_lines_sum_l2914_291428

/-- Two lines intersecting at a point -/
structure IntersectingLines where
  m : ℝ
  b : ℝ
  intersect_x : ℝ
  intersect_y : ℝ
  eq1 : intersect_y = 2 * m * intersect_x + 5
  eq2 : intersect_y = 4 * intersect_x + b

/-- The sum of b and m for two intersecting lines -/
def sum_b_m (lines : IntersectingLines) : ℝ :=
  lines.b + lines.m

/-- Theorem: For two lines y = 2mx + 5 and y = 4x + b intersecting at (4, 17), b + m = 2.5 -/
theorem intersecting_lines_sum (lines : IntersectingLines)
    (h1 : lines.intersect_x = 4)
    (h2 : lines.intersect_y = 17) :
    sum_b_m lines = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_intersecting_lines_sum_l2914_291428


namespace NUMINAMATH_CALUDE_equal_positive_reals_l2914_291496

theorem equal_positive_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (x*y + 1) / (x + 1) = (y*z + 1) / (y + 1))
  (h2 : (y*z + 1) / (y + 1) = (z*x + 1) / (z + 1)) :
  x = y ∧ y = z := by sorry

end NUMINAMATH_CALUDE_equal_positive_reals_l2914_291496
