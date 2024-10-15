import Mathlib

namespace NUMINAMATH_CALUDE_division_problem_l2067_206755

theorem division_problem (n x : ℝ) (h1 : n = 4.5) (h2 : (n / x) * 12 = 9) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2067_206755


namespace NUMINAMATH_CALUDE_rationalize_and_minimize_sum_l2067_206724

theorem rationalize_and_minimize_sum : ∃ (A B C D : ℕ),
  (D > 0) ∧
  (∀ (p : ℕ), Prime p → ¬(p^2 ∣ B)) ∧
  ((A : ℝ) * Real.sqrt B + C) / D = (Real.sqrt 32) / (Real.sqrt 16 - Real.sqrt 2) ∧
  (∀ (A' B' C' D' : ℕ),
    (D' > 0) →
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ B')) →
    ((A' : ℝ) * Real.sqrt B' + C') / D' = (Real.sqrt 32) / (Real.sqrt 16 - Real.sqrt 2) →
    A + B + C + D ≤ A' + B' + C' + D') ∧
  A + B + C + D = 21 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_and_minimize_sum_l2067_206724


namespace NUMINAMATH_CALUDE_perimeter_of_equilateral_triangle_with_base_8_l2067_206704

-- Define an equilateral triangle
structure EquilateralTriangle where
  base : ℝ
  is_positive : base > 0

-- Define the perimeter of an equilateral triangle
def perimeter (t : EquilateralTriangle) : ℝ := 3 * t.base

-- Theorem statement
theorem perimeter_of_equilateral_triangle_with_base_8 :
  ∀ t : EquilateralTriangle, t.base = 8 → perimeter t = 24 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_equilateral_triangle_with_base_8_l2067_206704


namespace NUMINAMATH_CALUDE_work_completion_time_l2067_206707

-- Define the work rates and time
def work_rate_B : ℚ := 1 / 18
def work_rate_A : ℚ := 2 * work_rate_B
def time_together : ℚ := 6

-- State the theorem
theorem work_completion_time :
  (work_rate_A = 2 * work_rate_B) →
  (work_rate_B = 1 / 18) →
  (time_together * (work_rate_A + work_rate_B) = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2067_206707


namespace NUMINAMATH_CALUDE_find_t_l2067_206746

-- Define a decreasing function f on ℝ
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define the property that f passes through (0, 5) and (3, -1)
def passes_through_points (f : ℝ → ℝ) : Prop :=
  f 0 = 5 ∧ f 3 = -1

-- Define the solution set of |f(x+t)-2|<3
def solution_set (f : ℝ → ℝ) (t : ℝ) : Set ℝ :=
  {x : ℝ | |f (x + t) - 2| < 3}

-- State the theorem
theorem find_t (f : ℝ → ℝ) (t : ℝ) :
  is_decreasing f →
  passes_through_points f →
  solution_set f t = Set.Ioo (-1) 2 →
  t = 1 := by sorry

end NUMINAMATH_CALUDE_find_t_l2067_206746


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2067_206735

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - m) : ℂ) + m * I).re = 0 → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2067_206735


namespace NUMINAMATH_CALUDE_terms_before_negative17_l2067_206752

/-- An arithmetic sequence with first term 103 and common difference -7 -/
def arithmeticSequence (n : ℕ) : ℤ := 103 - 7 * (n - 1)

/-- The position of -17 in the sequence -/
def positionOfNegative17 : ℕ := 18

theorem terms_before_negative17 :
  (∀ k < positionOfNegative17 - 1, arithmeticSequence k > -17) ∧
  arithmeticSequence positionOfNegative17 = -17 :=
sorry

end NUMINAMATH_CALUDE_terms_before_negative17_l2067_206752


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2067_206778

/-- Given an arithmetic sequence where the 20th term is 17 and the 21st term is 20,
    prove that the 5th term is -28. -/
theorem arithmetic_sequence_fifth_term
  (a : ℤ) -- First term of the sequence
  (d : ℤ) -- Common difference
  (h1 : a + 19 * d = 17) -- 20th term is 17
  (h2 : a + 20 * d = 20) -- 21st term is 20
  : a + 4 * d = -28 := by -- 5th term is -28
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2067_206778


namespace NUMINAMATH_CALUDE_smallest_greater_discount_l2067_206764

def discount_sequence_1 (x : ℝ) : ℝ := (1 - 0.2) * (1 - 0.1) * x
def discount_sequence_2 (x : ℝ) : ℝ := (1 - 0.08)^3 * x
def discount_sequence_3 (x : ℝ) : ℝ := (1 - 0.15) * (1 - 0.12) * x

def effective_discount_1 (x : ℝ) : ℝ := x - discount_sequence_1 x
def effective_discount_2 (x : ℝ) : ℝ := x - discount_sequence_2 x
def effective_discount_3 (x : ℝ) : ℝ := x - discount_sequence_3 x

theorem smallest_greater_discount : 
  ∀ x > 0, 
    effective_discount_1 x / x < 0.29 ∧ 
    effective_discount_2 x / x < 0.29 ∧ 
    effective_discount_3 x / x < 0.29 ∧
    ∀ n : ℕ, n < 29 → 
      (effective_discount_1 x / x > n / 100 ∨ 
       effective_discount_2 x / x > n / 100 ∨ 
       effective_discount_3 x / x > n / 100) :=
by sorry

end NUMINAMATH_CALUDE_smallest_greater_discount_l2067_206764


namespace NUMINAMATH_CALUDE_principal_amount_is_16065_l2067_206799

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (simple_interest : ℚ) (rate : ℚ) (time : ℕ) : ℚ :=
  (simple_interest * 100) / (rate * time)

/-- Theorem: Given the specified conditions, the principal amount is 16065 -/
theorem principal_amount_is_16065 :
  let simple_interest : ℚ := 4016.25
  let rate : ℚ := 5
  let time : ℕ := 5
  calculate_principal simple_interest rate time = 16065 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_is_16065_l2067_206799


namespace NUMINAMATH_CALUDE_fourth_square_area_l2067_206769

theorem fourth_square_area (PQ PR PS QR RS : ℝ) : 
  PQ^2 = 25 → QR^2 = 64 → RS^2 = 49 → 
  (PQ^2 + QR^2 = PR^2) → (PR^2 + RS^2 = PS^2) → 
  PS^2 = 138 := by sorry

end NUMINAMATH_CALUDE_fourth_square_area_l2067_206769


namespace NUMINAMATH_CALUDE_set_intersection_equality_l2067_206711

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

-- State the theorem
theorem set_intersection_equality : M ∩ N = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l2067_206711


namespace NUMINAMATH_CALUDE_pizza_pepperoni_count_l2067_206741

theorem pizza_pepperoni_count :
  ∀ (pepperoni ham sausage : ℕ),
    ham = 2 * pepperoni →
    sausage = pepperoni + 12 →
    pepperoni + ham + sausage = 22 * 6 →
    pepperoni = 30 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pepperoni_count_l2067_206741


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l2067_206713

/-- Given two points M and N in a plane, and P as the midpoint of MN, 
    prove that P has the specified coordinates. -/
theorem midpoint_coordinates (M N P : ℝ × ℝ) : 
  M = (3, -2) → N = (-5, -1) → P = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) → 
  P = (-1, -3/2) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_l2067_206713


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2067_206709

/-- The equation of a line perpendicular to 2x + y - 5 = 0 and passing through (3, 0) -/
theorem perpendicular_line_through_point (x y : ℝ) :
  (2 : ℝ) * x + y - 5 = 0 →  -- Given line
  (∃ c : ℝ, x - 2 * y + c = 0 ∧  -- General form of perpendicular line
            3 - 2 * 0 + c = 0 ∧  -- Passes through (3, 0)
            x - 2 * y - 3 = 0) :=  -- The specific equation we want to prove
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2067_206709


namespace NUMINAMATH_CALUDE_product_and_reciprocal_relation_l2067_206782

theorem product_and_reciprocal_relation (x y : ℝ) :
  x > 0 ∧ y > 0 ∧ x * y = 12 ∧ 1 / x = 3 * (1 / y) → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_and_reciprocal_relation_l2067_206782


namespace NUMINAMATH_CALUDE_power_function_property_l2067_206701

/-- A power function with a specific property -/
def PowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x > 0, f x = x ^ α

theorem power_function_property (f : ℝ → ℝ) (h1 : PowerFunction f) (h2 : f 4 / f 2 = 3) :
  f (1/2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l2067_206701


namespace NUMINAMATH_CALUDE_elsas_marbles_l2067_206773

theorem elsas_marbles (initial : ℕ) (lost_breakfast : ℕ) (given_lunch : ℕ) (received_mom : ℕ) : 
  initial = 40 →
  lost_breakfast = 3 →
  given_lunch = 5 →
  received_mom = 12 →
  initial - lost_breakfast - given_lunch + received_mom + 2 * given_lunch = 54 := by
  sorry

end NUMINAMATH_CALUDE_elsas_marbles_l2067_206773


namespace NUMINAMATH_CALUDE_log_equation_solution_l2067_206779

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log x / Real.log 9 = 5 → x = (3^10)^(1/3) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2067_206779


namespace NUMINAMATH_CALUDE_miles_driven_equals_365_l2067_206791

/-- Calculates the total miles driven given car efficiencies and gas usage --/
def total_miles_driven (highway_mpg city_mpg : ℚ) (total_gas : ℚ) (highway_city_diff : ℚ) : ℚ :=
  let city_miles := (total_gas * highway_mpg * city_mpg - city_mpg * highway_city_diff) / (highway_mpg + city_mpg)
  let highway_miles := city_miles + highway_city_diff
  city_miles + highway_miles

/-- Theorem stating the total miles driven under given conditions --/
theorem miles_driven_equals_365 :
  total_miles_driven 37 30 11 5 = 365 := by
  sorry

end NUMINAMATH_CALUDE_miles_driven_equals_365_l2067_206791


namespace NUMINAMATH_CALUDE_sweater_shirt_price_difference_l2067_206727

/-- Given the total price and quantity of shirts and sweaters, 
    prove that the average price of a sweater exceeds that of a shirt by $4 -/
theorem sweater_shirt_price_difference 
  (shirt_quantity : ℕ) 
  (shirt_total_price : ℚ)
  (sweater_quantity : ℕ)
  (sweater_total_price : ℚ)
  (h_shirt_quantity : shirt_quantity = 25)
  (h_shirt_price : shirt_total_price = 400)
  (h_sweater_quantity : sweater_quantity = 75)
  (h_sweater_price : sweater_total_price = 1500) :
  sweater_total_price / sweater_quantity - shirt_total_price / shirt_quantity = 4 := by
sorry

end NUMINAMATH_CALUDE_sweater_shirt_price_difference_l2067_206727


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2067_206784

theorem trigonometric_simplification (α : ℝ) :
  (Real.sin (π - α) / Real.cos (π + α)) *
  (Real.cos (-α) * Real.cos (2*π - α)) /
  Real.sin (π/2 + α) = -Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2067_206784


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l2067_206747

theorem product_of_roots_cubic_equation : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 4 * x^2 + x - 5
  ∃ a b c : ℝ, (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ a * b * c = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l2067_206747


namespace NUMINAMATH_CALUDE_intersection_implies_x_value_l2067_206732

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, 2*x, x^2}

theorem intersection_implies_x_value :
  ∀ x : ℝ, A x ∩ B x = {1, 4} → x = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_x_value_l2067_206732


namespace NUMINAMATH_CALUDE_product_sum_relation_l2067_206725

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 1 → b = 7 → b - a = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l2067_206725


namespace NUMINAMATH_CALUDE_min_side_length_is_optimal_l2067_206733

/-- The minimum side length of a square satisfying the given conditions -/
def min_side_length : ℝ := 25

/-- The condition that the area of the square is at least 625 square feet -/
def area_condition (s : ℝ) : Prop := s^2 ≥ 625

/-- The condition that there exists a smaller square inside with side length equal to half the side length of the larger square -/
def inner_square_condition (s : ℝ) : Prop := ∃ (inner_s : ℝ), inner_s = s / 2

/-- Theorem stating that the minimum side length satisfies both conditions and is minimal -/
theorem min_side_length_is_optimal :
  area_condition min_side_length ∧
  inner_square_condition min_side_length ∧
  ∀ s, s < min_side_length → ¬(area_condition s ∧ inner_square_condition s) :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_is_optimal_l2067_206733


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l2067_206785

theorem largest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a + b = 105 →      -- Sum of two angles is 7/6 of a right angle (90° * 7/6 = 105°)
  b = a + 40 →       -- One angle is 40° larger than the other
  max a (max b c) = 75 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l2067_206785


namespace NUMINAMATH_CALUDE_part1_part2_l2067_206745

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |2*x + a|

-- Part 1: Prove that for a=1, f(x) + |x-1| ≥ 3 for all x
theorem part1 : ∀ x : ℝ, f 1 x + |x - 1| ≥ 3 := by sorry

-- Part 2: Prove that the minimum value of f(x) is 2 if and only if a = 2 or a = -6
theorem part2 : (∃ x : ℝ, f a x = 2) ∧ (∀ y : ℝ, f a y ≥ 2) ↔ a = 2 ∨ a = -6 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2067_206745


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l2067_206749

/-- A curve represented by the equation mx^2 + ny^2 = 1 is an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem mn_positive_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → m * n > 0) ∧
  (∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l2067_206749


namespace NUMINAMATH_CALUDE_distribute_6_4_l2067_206712

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 4 indistinguishable boxes is 262 -/
theorem distribute_6_4 : distribute 6 4 = 262 := by sorry

end NUMINAMATH_CALUDE_distribute_6_4_l2067_206712


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2067_206768

theorem pizza_toppings_combinations (n m : ℕ) (h1 : n = 8) (h2 : m = 5) : 
  Nat.choose n m = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2067_206768


namespace NUMINAMATH_CALUDE_percentage_less_l2067_206708

theorem percentage_less (x y : ℝ) (h : x = 3 * y) : 
  (x - y) / x * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_less_l2067_206708


namespace NUMINAMATH_CALUDE_smallest_n_for_congruence_l2067_206743

/-- Concatenation of powers of 2 -/
def A (n : ℕ) : ℕ :=
  -- We define A as a placeholder function, as the actual implementation is complex
  sorry

/-- The main theorem -/
theorem smallest_n_for_congruence : 
  (∀ k : ℕ, 3 ≤ k → k < 14 → ¬(A k ≡ 2^(10*k) [MOD 2^170])) ∧ 
  (A 14 ≡ 2^(10*14) [MOD 2^170]) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_congruence_l2067_206743


namespace NUMINAMATH_CALUDE_book_spending_is_correct_l2067_206721

def allowance : ℚ := 50

def game_fraction : ℚ := 1/4
def snack_fraction : ℚ := 1/5
def toy_fraction : ℚ := 2/5

def book_spending : ℚ := allowance - (allowance * game_fraction + allowance * snack_fraction + allowance * toy_fraction)

theorem book_spending_is_correct : book_spending = 7.5 := by sorry

end NUMINAMATH_CALUDE_book_spending_is_correct_l2067_206721


namespace NUMINAMATH_CALUDE_smallest_integer_side_of_triangle_l2067_206738

theorem smallest_integer_side_of_triangle (s : ℕ) : 
  (4 : ℝ) ≤ s ∧ 
  (7.8 : ℝ) + s > 11 ∧ 
  (7.8 : ℝ) + 11 > s ∧ 
  11 + s > (7.8 : ℝ) ∧
  ∀ (t : ℕ), t < s → 
    ((7.8 : ℝ) + (t : ℝ) ≤ 11 ∨ 
     (7.8 : ℝ) + 11 ≤ (t : ℝ) ∨ 
     11 + (t : ℝ) ≤ (7.8 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_side_of_triangle_l2067_206738


namespace NUMINAMATH_CALUDE_mod_equivalence_l2067_206719

theorem mod_equivalence (n : ℕ) : 
  (179 * 933 / 7) % 50 = n ∧ 0 ≤ n ∧ n < 50 → n = 1 := by sorry

end NUMINAMATH_CALUDE_mod_equivalence_l2067_206719


namespace NUMINAMATH_CALUDE_parabola_properties_l2067_206788

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 1)^2 - 2

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  (∀ x y : ℝ, f x = y → ∃ a : ℝ, a > 0 ∧ y = a * (x - 1)^2 - 2) ∧ 
  (∀ x y : ℝ, f x = y → f (2 - x) = y) ∧
  (f 1 = -2 ∧ ∀ x : ℝ, f x ≥ -2) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 ∧ x₁ > x₂ → f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2067_206788


namespace NUMINAMATH_CALUDE_shopping_cost_calculation_l2067_206715

-- Define the prices and quantities
def carrot_price : ℚ := 2
def carrot_quantity : ℕ := 7
def milk_price : ℚ := 3
def milk_quantity : ℕ := 4
def pineapple_price : ℚ := 5
def pineapple_quantity : ℕ := 3
def pineapple_discount : ℚ := 0.5
def flour_price : ℚ := 8
def flour_quantity : ℕ := 1
def cookie_price : ℚ := 10
def cookie_quantity : ℕ := 1

-- Define the store's discount conditions
def store_discount_threshold : ℚ := 40
def store_discount_rate : ℚ := 0.1

-- Define the coupon conditions
def coupon_value : ℚ := 5
def coupon_threshold : ℚ := 25

-- Calculate the total cost before discounts
def total_before_discounts : ℚ :=
  carrot_price * carrot_quantity +
  milk_price * milk_quantity +
  pineapple_price * pineapple_quantity * (1 - pineapple_discount) +
  flour_price * flour_quantity +
  cookie_price * cookie_quantity

-- Apply store discount if applicable
def after_store_discount : ℚ :=
  if total_before_discounts > store_discount_threshold then
    total_before_discounts * (1 - store_discount_rate)
  else
    total_before_discounts

-- Apply coupon if applicable
def final_cost : ℚ :=
  if after_store_discount > coupon_threshold then
    after_store_discount - coupon_value
  else
    after_store_discount

-- Theorem to prove
theorem shopping_cost_calculation :
  final_cost = 41.35 := by sorry

end NUMINAMATH_CALUDE_shopping_cost_calculation_l2067_206715


namespace NUMINAMATH_CALUDE_quadratic_ratio_l2067_206748

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 800*x + 2400

-- Define the completed square form
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

-- Theorem statement
theorem quadratic_ratio :
  ∃ (b c : ℝ), (∀ x, f x = g x b c) ∧ (c / b = -394) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l2067_206748


namespace NUMINAMATH_CALUDE_base_n_problem_l2067_206780

theorem base_n_problem (n : ℕ+) (d : ℕ) (h1 : d < 10) 
  (h2 : 4 * n ^ 2 + 2 * n + d = 347)
  (h3 : 4 * n ^ 2 + 2 * n + 9 = 1 * 7 ^ 3 + 2 * 7 ^ 2 + d * 7 + 2) :
  n + d = 11 := by
sorry

end NUMINAMATH_CALUDE_base_n_problem_l2067_206780


namespace NUMINAMATH_CALUDE_right_triangle_area_thrice_hypotenuse_l2067_206723

theorem right_triangle_area_thrice_hypotenuse : ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive sides
  c^2 = a^2 + b^2 ∧        -- Pythagorean theorem
  (1/2) * a * b = 3 * c    -- Area equals thrice the hypotenuse
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_thrice_hypotenuse_l2067_206723


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l2067_206703

/-- Given a complex equation, prove that the point is in the first quadrant -/
theorem point_in_first_quadrant (x y : ℝ) (h : x + y + (x - y) * Complex.I = 3 - Complex.I) :
  x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l2067_206703


namespace NUMINAMATH_CALUDE_sheila_tuesday_thursday_hours_l2067_206730

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hours_tt : ℕ   -- Hours worked on Tuesday, Thursday
  hourly_rate : ℕ -- Hourly rate in dollars
  weekly_earnings : ℕ -- Total weekly earnings in dollars

/-- Calculates the total hours worked in a week --/
def total_hours (s : WorkSchedule) : ℕ :=
  3 * s.hours_mwf + 2 * s.hours_tt

/-- Calculates the total earnings based on hours worked and hourly rate --/
def calculated_earnings (s : WorkSchedule) : ℕ :=
  s.hourly_rate * (total_hours s)

/-- Theorem stating that Sheila works 6 hours on Tuesday and Thursday --/
theorem sheila_tuesday_thursday_hours (s : WorkSchedule) 
  (h1 : s.hours_mwf = 8)
  (h2 : s.hourly_rate = 11)
  (h3 : s.weekly_earnings = 396)
  (h4 : calculated_earnings s = s.weekly_earnings) :
  s.hours_tt = 6 := by
  sorry

end NUMINAMATH_CALUDE_sheila_tuesday_thursday_hours_l2067_206730


namespace NUMINAMATH_CALUDE_alyssas_attended_games_l2067_206792

theorem alyssas_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 31) 
  (h2 : missed_games = 18) : 
  total_games - missed_games = 13 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_attended_games_l2067_206792


namespace NUMINAMATH_CALUDE_order_of_magnitude_l2067_206776

theorem order_of_magnitude (a b : ℝ) (ha : a > 0) (hb : b < 0) (hab : |a| < |b|) :
  -b > a ∧ a > -a ∧ -a > b := by
  sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l2067_206776


namespace NUMINAMATH_CALUDE_intersection_at_midpoint_l2067_206796

/-- A line with equation x - y = c intersects the line segment from (1, 4) to (3, 8) at its midpoint -/
theorem intersection_at_midpoint (c : ℝ) : 
  (∃ (x y : ℝ), x - y = c ∧ 
    x = (1 + 3) / 2 ∧ 
    y = (4 + 8) / 2 ∧ 
    (x, y) = ((1 + 3) / 2, (4 + 8) / 2)) → 
  c = -4 := by
sorry

end NUMINAMATH_CALUDE_intersection_at_midpoint_l2067_206796


namespace NUMINAMATH_CALUDE_zach_stadium_goal_years_l2067_206765

/-- The number of years required to save enough money to visit all major league baseball stadiums. -/
def years_to_visit_stadiums (num_stadiums : ℕ) (cost_per_stadium : ℕ) (annual_savings : ℕ) : ℕ :=
  (num_stadiums * cost_per_stadium) / annual_savings

/-- Theorem stating that it takes 18 years to save enough money to visit all 30 major league baseball stadiums
    given an average cost of $900 per stadium and annual savings of $1,500. -/
theorem zach_stadium_goal_years :
  years_to_visit_stadiums 30 900 1500 = 18 := by
  sorry

end NUMINAMATH_CALUDE_zach_stadium_goal_years_l2067_206765


namespace NUMINAMATH_CALUDE_periodic_function_decomposition_l2067_206771

-- Define the type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define the property of being 2π-periodic
def isPeriodic2Pi (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x + 2 * Real.pi) = f x

-- Define the property of being an even function
def isEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the property of being π-periodic
def isPeriodicPi (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (x + Real.pi) = f x

theorem periodic_function_decomposition (f : RealFunction) (h : isPeriodic2Pi f) :
  ∃ (f₁ f₂ f₃ f₄ : RealFunction),
    (∀ i ∈ [f₁, f₂, f₃, f₄], isEven i ∧ isPeriodicPi i) ∧
    (∀ x : ℝ, f x = f₁ x + f₂ x * Real.cos x + f₃ x * Real.sin x + f₄ x * Real.sin (2 * x)) :=
sorry

end NUMINAMATH_CALUDE_periodic_function_decomposition_l2067_206771


namespace NUMINAMATH_CALUDE_spring_properties_l2067_206717

-- Define the spring's properties
def initial_length : ℝ := 18
def extension_rate : ℝ := 2

-- Define the relationship between mass and length
def spring_length (mass : ℝ) : ℝ := initial_length + extension_rate * mass

theorem spring_properties :
  (spring_length 4 = 26) ∧
  (∀ x y, x < y → spring_length x < spring_length y) ∧
  (∀ x, spring_length x = 2 * x + 18) ∧
  (spring_length 12 = 42) := by
  sorry

end NUMINAMATH_CALUDE_spring_properties_l2067_206717


namespace NUMINAMATH_CALUDE_division_problem_l2067_206798

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 52 → 
  divisor = 3 → 
  remainder = 4 → 
  dividend = divisor * quotient + remainder →
  quotient = 16 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2067_206798


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l2067_206793

/-- Given an angle α = -51°, this theorem states that all angles with the same terminal side as α
    can be represented as k · 360° - 51°, where k is an integer. -/
theorem same_terminal_side_angles (α : ℝ) (h : α = -51) :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 - 51) ↔ (∃ n : ℤ, θ = α + n * 360) :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angles_l2067_206793


namespace NUMINAMATH_CALUDE_tan_sum_product_equality_l2067_206763

theorem tan_sum_product_equality (α β γ : ℝ) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_sum : α + β + γ = π / 2) : 
  Real.tan α * Real.tan β + Real.tan α * Real.tan γ + Real.tan β * Real.tan γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_equality_l2067_206763


namespace NUMINAMATH_CALUDE_dragon_legs_correct_l2067_206797

/-- Represents the number of legs of a three-headed dragon -/
def dragon_legs : ℕ := 14

/-- Represents the number of centipedes -/
def num_centipedes : ℕ := 5

/-- Represents the number of three-headed dragons -/
def num_dragons : ℕ := 7

/-- The total number of heads in the herd -/
def total_heads : ℕ := 26

/-- The total number of legs in the herd -/
def total_legs : ℕ := 298

/-- Each centipede has one head -/
def centipede_heads : ℕ := 1

/-- Each centipede has 40 legs -/
def centipede_legs : ℕ := 40

/-- Each dragon has three heads -/
def dragon_heads : ℕ := 3

theorem dragon_legs_correct :
  (num_centipedes * centipede_heads + num_dragons * dragon_heads = total_heads) ∧
  (num_centipedes * centipede_legs + num_dragons * dragon_legs = total_legs) :=
by sorry

end NUMINAMATH_CALUDE_dragon_legs_correct_l2067_206797


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l2067_206739

theorem min_value_x_plus_2y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (heq : x + 2*y + 2*x*y = 3) : 
  ∀ z, x + 2*y ≥ z → z ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l2067_206739


namespace NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l2067_206740

/-- A quadratic function y = kx^2 - 7x - 7 intersects the x-axis if and only if k ≥ -7/4 and k ≠ 0 -/
theorem quadratic_intersects_x_axis (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ (k ≥ -7/4 ∧ k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersects_x_axis_l2067_206740


namespace NUMINAMATH_CALUDE_quarter_circle_sum_approaches_semi_circumference_l2067_206705

/-- The sum of quarter-circle arc lengths approaches the semi-circumference as n approaches infinity --/
theorem quarter_circle_sum_approaches_semi_circumference (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |2 * n * (π * D / (4 * n)) - π * D / 2| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circle_sum_approaches_semi_circumference_l2067_206705


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2067_206700

/-- Given a rhombus with diagonal sum L and area S, its side length is (√(L² - 4S)) / 2 -/
theorem rhombus_side_length (L S : ℝ) (h1 : L > 0) (h2 : S > 0) (h3 : L^2 ≥ 4*S) :
  ∃ (side_length : ℝ), side_length = (Real.sqrt (L^2 - 4*S)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l2067_206700


namespace NUMINAMATH_CALUDE_total_snakes_l2067_206706

/-- Given information about pet ownership, prove the total number of snakes. -/
theorem total_snakes (total_people : ℕ) (only_dogs : ℕ) (only_cats : ℕ) (only_snakes : ℕ)
  (dogs_and_cats : ℕ) (cats_and_snakes : ℕ) (dogs_and_snakes : ℕ) (all_three : ℕ)
  (h1 : total_people = 120)
  (h2 : only_dogs = 30)
  (h3 : only_cats = 25)
  (h4 : only_snakes = 12)
  (h5 : dogs_and_cats = 15)
  (h6 : cats_and_snakes = 10)
  (h7 : dogs_and_snakes = 8)
  (h8 : all_three = 5) :
  only_snakes + cats_and_snakes + dogs_and_snakes + all_three = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_snakes_l2067_206706


namespace NUMINAMATH_CALUDE_cube_remainder_l2067_206731

theorem cube_remainder (n : ℤ) : n % 6 = 3 → n^3 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_remainder_l2067_206731


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_one_l2067_206729

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_slopes_equal {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The lines ax + 2y - 2 = 0 and x + (a+1)y + 1 = 0 are parallel if and only if a = 1 -/
theorem lines_parallel_iff_a_eq_one (a : ℝ) :
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, 
    (a * x + 2 * y - 2 = 0 ↔ y = (-a/2) * x + b1) ∧ 
    (x + (a+1) * y + 1 = 0 ↔ y = (-1/(a+1)) * x + b2)) ↔ 
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_one_l2067_206729


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2067_206718

-- Define the inequality
def inequality (x : ℝ) : Prop := (2 - x) / (x + 4) > 1

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < -1}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2067_206718


namespace NUMINAMATH_CALUDE_recipe_flour_cups_l2067_206777

/-- The number of cups of sugar required in the recipe -/
def sugar_cups : ℕ := 9

/-- The number of cups of flour Mary has already put in -/
def flour_cups_added : ℕ := 4

/-- The total number of cups of flour required in the recipe -/
def total_flour_cups : ℕ := sugar_cups + 1

theorem recipe_flour_cups : total_flour_cups = 10 := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_cups_l2067_206777


namespace NUMINAMATH_CALUDE_survey_sample_size_l2067_206781

/-- Represents a survey conducted on students -/
structure Survey where
  numSelected : ℕ

/-- Definition of sample size for a survey -/
def sampleSize (s : Survey) : ℕ := s.numSelected

/-- Theorem stating that the sample size of the survey is 200 -/
theorem survey_sample_size :
  ∃ (s : Survey), s.numSelected = 200 ∧ sampleSize s = 200 := by
  sorry

end NUMINAMATH_CALUDE_survey_sample_size_l2067_206781


namespace NUMINAMATH_CALUDE_linear_iff_m_neq_neg_six_l2067_206757

/-- A function f is linear if there exist constants a and b such that f(x) = ax + b for all x, and a ≠ 0 -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The function f_m for a given m -/
def f_m (m : ℝ) : ℝ → ℝ := λ x ↦ (m + 2) * x + 4 * x - 5

theorem linear_iff_m_neq_neg_six (m : ℝ) :
  IsLinearFunction (f_m m) ↔ m ≠ -6 := by
  sorry

end NUMINAMATH_CALUDE_linear_iff_m_neq_neg_six_l2067_206757


namespace NUMINAMATH_CALUDE_students_6_to_8_hours_l2067_206767

/-- Represents a frequency distribution histogram for study times -/
structure StudyTimeHistogram where
  total_students : ℕ
  freq_6_to_8 : ℕ
  -- Other fields for other time intervals could be added here

/-- Theorem stating that in a given histogram of 100 students, 30 studied for 6 to 8 hours -/
theorem students_6_to_8_hours (h : StudyTimeHistogram) 
  (h_total : h.total_students = 100) : h.freq_6_to_8 = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_6_to_8_hours_l2067_206767


namespace NUMINAMATH_CALUDE_quotient_reciprocal_sum_l2067_206753

theorem quotient_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hsum : x + y = 45) (hprod : x * y = 500) : 
  (x / y) + (y / x) = 41 / 20 := by
  sorry

end NUMINAMATH_CALUDE_quotient_reciprocal_sum_l2067_206753


namespace NUMINAMATH_CALUDE_isabela_spent_2800_l2067_206774

/-- The total amount Isabela spent on cucumbers and pencils -/
def total_spent (cucumber_price : ℝ) (pencil_price : ℝ) (cucumber_count : ℕ) 
  (pencil_discount : ℝ) : ℝ :=
  let pencil_count := cucumber_count / 2
  let pencil_cost := pencil_count * pencil_price * (1 - pencil_discount)
  let cucumber_cost := cucumber_count * cucumber_price
  pencil_cost + cucumber_cost

/-- Theorem stating that Isabela spent $2800 on cucumbers and pencils -/
theorem isabela_spent_2800 : 
  total_spent 20 20 100 0.2 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_isabela_spent_2800_l2067_206774


namespace NUMINAMATH_CALUDE_chord_intersection_l2067_206726

theorem chord_intersection (a : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x + y - a - 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - 2)^2 = 4}
  let chord_length := 2 * Real.sqrt 2
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ p ∈ circle ∧ q ∈ line ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) → 
  (a = 1 ∨ a = 5) :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_l2067_206726


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l2067_206758

/-- Given two orthonormal vectors e₁ and e₂, and vectors AC and BD defined in terms of e₁ and e₂,
    prove that the area of quadrilateral ABCD is 10. -/
theorem area_of_quadrilateral (e₁ e₂ : ℝ × ℝ) 
    (h_orthonormal : e₁ • e₁ = 1 ∧ e₂ • e₂ = 1 ∧ e₁ • e₂ = 0) 
    (AC : ℝ × ℝ) (h_AC : AC = 3 • e₁ - e₂)
    (BD : ℝ × ℝ) (h_BD : BD = 2 • e₁ + 6 • e₂) : 
  Real.sqrt ((AC.1^2 + AC.2^2) * (BD.1^2 + BD.2^2)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l2067_206758


namespace NUMINAMATH_CALUDE_new_person_weight_l2067_206714

/-- Calculates the weight of a new person given the following conditions:
  * There are 6 people initially
  * Replacing one person weighing 69 kg with a new person increases the average weight by 1.8 kg
-/
theorem new_person_weight (num_people : Nat) (weight_increase : Real) (replaced_weight : Real) :
  num_people = 6 →
  weight_increase = 1.8 →
  replaced_weight = 69 →
  ∃ (new_weight : Real), new_weight = 79.8 ∧
    new_weight = replaced_weight + num_people * weight_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2067_206714


namespace NUMINAMATH_CALUDE_constant_remainder_condition_l2067_206761

-- Define the polynomials
def dividend (a : ℝ) (x : ℝ) : ℝ := 12 * x^3 - 9 * x^2 + a * x + 8
def divisor (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

-- Define the theorem
theorem constant_remainder_condition (a : ℝ) :
  (∃ (r : ℝ), ∀ (x : ℝ), ∃ (q : ℝ), dividend a x = divisor x * q + r) ↔ a = -7 := by
  sorry

end NUMINAMATH_CALUDE_constant_remainder_condition_l2067_206761


namespace NUMINAMATH_CALUDE_cookie_jar_problem_l2067_206702

theorem cookie_jar_problem (initial_cookies : ℕ) 
  (cookies_removed : ℕ) (cookies_added : ℕ) : 
  initial_cookies = 7 → 
  cookies_removed = 1 → 
  cookies_added = 5 → 
  initial_cookies - cookies_removed = (initial_cookies + cookies_added) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_problem_l2067_206702


namespace NUMINAMATH_CALUDE_quadratic_sum_l2067_206759

/-- Given a quadratic expression 4x^2 - 8x + 5, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals 6 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (4 * x^2 - 8 * x + 5 = a * (x - h)^2 + k) ∧ (a + h + k = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2067_206759


namespace NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2067_206775

theorem remainder_sum_mod_seven (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 7 = 2 →
  (3 * c) % 7 = 1 →
  (4 * b) % 7 = (2 + b) % 7 →
  (a + b + c) % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_seven_l2067_206775


namespace NUMINAMATH_CALUDE_intersection_A_B_l2067_206789

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2067_206789


namespace NUMINAMATH_CALUDE_taxi_overtakes_bus_l2067_206772

/-- 
Given a taxi and a bus with the following conditions:
- The taxi travels at 45 mph
- The bus travels 30 mph slower than the taxi
- The taxi starts 4 hours after the bus
This theorem proves that the taxi will overtake the bus in 2 hours.
-/
theorem taxi_overtakes_bus (taxi_speed : ℝ) (bus_speed : ℝ) (head_start : ℝ) 
  (overtake_time : ℝ) :
  taxi_speed = 45 →
  bus_speed = taxi_speed - 30 →
  head_start = 4 →
  overtake_time = 2 →
  taxi_speed * overtake_time = bus_speed * (overtake_time + head_start) :=
by
  sorry

#check taxi_overtakes_bus

end NUMINAMATH_CALUDE_taxi_overtakes_bus_l2067_206772


namespace NUMINAMATH_CALUDE_shoe_ratio_problem_l2067_206790

/-- Proof of the shoe ratio problem -/
theorem shoe_ratio_problem (brian_shoes edward_shoes jacob_shoes : ℕ) : 
  (edward_shoes = 3 * brian_shoes) →
  (brian_shoes = 22) →
  (jacob_shoes + edward_shoes + brian_shoes = 121) →
  (jacob_shoes : ℚ) / edward_shoes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_shoe_ratio_problem_l2067_206790


namespace NUMINAMATH_CALUDE_black_balls_count_l2067_206750

theorem black_balls_count (red_balls : ℕ) (prob_red : ℚ) (black_balls : ℕ) : 
  red_balls = 3 → prob_red = 1/4 → black_balls = 9 → 
  (red_balls : ℚ) / (red_balls + black_balls : ℚ) = prob_red :=
by sorry

end NUMINAMATH_CALUDE_black_balls_count_l2067_206750


namespace NUMINAMATH_CALUDE_quiz_competition_l2067_206720

theorem quiz_competition (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 100)
  (h2 : correct_score = 10)
  (h3 : incorrect_score = -5)
  (h4 : total_score = 850) :
  ∃ (incorrect : ℕ), 
    incorrect = 10 ∧ 
    (total_questions - incorrect : ℤ) * correct_score + incorrect * incorrect_score = total_score :=
by sorry

end NUMINAMATH_CALUDE_quiz_competition_l2067_206720


namespace NUMINAMATH_CALUDE_parabola_properties_given_parabola_properties_l2067_206744

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  D : ℝ × ℝ

/-- Given conditions for the parabola -/
def given_parabola : Parabola where
  p := 2  -- This is derived from the solution, not given directly
  equation := λ x y => y^2 = 2 * 2 * x
  focus := (1, 0)
  D := (2, 0)

/-- Theorem stating the main results -/
theorem parabola_properties (C : Parabola) 
  (h1 : C.p > 0)
  (h2 : C.D = (C.p, 0))
  (h3 : ∃ (M : ℝ × ℝ), C.equation M.1 M.2 ∧ 
        (M.2 - C.D.2) / (M.1 - C.D.1) = 0 ∧ 
        Real.sqrt ((M.1 - C.focus.1)^2 + (M.2 - C.focus.2)^2) = 3) :
  (C.equation = λ x y => y^2 = 4*x) ∧
  (∃ (A B : ℝ × ℝ), 
    C.equation A.1 A.2 ∧ 
    C.equation B.1 B.2 ∧
    (B.2 - A.2) / (B.1 - A.1) = -1/Real.sqrt 2 ∧
    A.1 - Real.sqrt 2 * A.2 - 4 = 0) :=
by sorry

/-- Applying the theorem to the given parabola -/
theorem given_parabola_properties : 
  (given_parabola.equation = λ x y => y^2 = 4*x) ∧
  (∃ (A B : ℝ × ℝ), 
    given_parabola.equation A.1 A.2 ∧ 
    given_parabola.equation B.1 B.2 ∧
    (B.2 - A.2) / (B.1 - A.1) = -1/Real.sqrt 2 ∧
    A.1 - Real.sqrt 2 * A.2 - 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_given_parabola_properties_l2067_206744


namespace NUMINAMATH_CALUDE_team_selection_count_l2067_206770

def people : Finset Char := {'a', 'b', 'c', 'd', 'e'}

theorem team_selection_count :
  let all_selections := (people.powerset.filter (fun s => s.card = 2)).card
  let invalid_selections := (people.erase 'a').card
  all_selections - invalid_selections = 16 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2067_206770


namespace NUMINAMATH_CALUDE_range_of_x_range_of_a_l2067_206722

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 3*a*x + 2*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x (x : ℝ) (h : p x 2 ∨ q x) : 2 < x ∧ x < 4 :=
sorry

-- Part 2
theorem range_of_a (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h' : ∃ x, ¬(p x a) ∧ q x) : 3/2 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_a_l2067_206722


namespace NUMINAMATH_CALUDE_factorization_equality_l2067_206737

theorem factorization_equality (a b : ℝ) : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2067_206737


namespace NUMINAMATH_CALUDE_no_tie_in_total_hr_l2067_206762

/-- Represents the months of the baseball season -/
inductive Month
| Mar
| Apr
| May
| Jun
| Jul
| Aug
| Sep

/-- Returns the number of home runs hit by Johnson in a given month -/
def johnson_hr (m : Month) : ℕ :=
  match m with
  | Month.Mar => 2
  | Month.Apr => 11
  | Month.May => 15
  | Month.Jun => 9
  | Month.Jul => 7
  | Month.Aug => 12
  | Month.Sep => 14

/-- Returns the number of home runs hit by Carter in a given month -/
def carter_hr (m : Month) : ℕ :=
  match m with
  | Month.Mar => 0
  | Month.Apr => 5
  | Month.May => 8
  | Month.Jun => 18
  | Month.Jul => 6
  | Month.Aug => 15
  | Month.Sep => 10

/-- Calculates the cumulative home runs for a player up to and including a given month -/
def cumulative_hr (hr_func : Month → ℕ) (m : Month) : ℕ :=
  match m with
  | Month.Mar => hr_func Month.Mar
  | Month.Apr => hr_func Month.Mar + hr_func Month.Apr
  | Month.May => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May
  | Month.Jun => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun
  | Month.Jul => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul
  | Month.Aug => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul + hr_func Month.Aug
  | Month.Sep => hr_func Month.Mar + hr_func Month.Apr + hr_func Month.May + hr_func Month.Jun + hr_func Month.Jul + hr_func Month.Aug + hr_func Month.Sep

theorem no_tie_in_total_hr : ∀ m : Month, cumulative_hr johnson_hr m ≠ cumulative_hr carter_hr m := by
  sorry

end NUMINAMATH_CALUDE_no_tie_in_total_hr_l2067_206762


namespace NUMINAMATH_CALUDE_cubic_real_root_l2067_206728

/-- Given a cubic polynomial ax^3 + 3x^2 + bx - 65 = 0 where a and b are real numbers,
    and -2 - 3i is one of its roots, the real root of this polynomial is 5/2. -/
theorem cubic_real_root (a b : ℝ) :
  (∃ (z : ℂ), z = -2 - 3*I ∧ a * z^3 + 3 * z^2 + b * z - 65 = 0) →
  (∃ (x : ℝ), a * x^3 + 3 * x^2 + b * x - 65 = 0 ∧ x = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l2067_206728


namespace NUMINAMATH_CALUDE_diagonals_15_gon_l2067_206787

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex 15-gon is 90 -/
theorem diagonals_15_gon : num_diagonals 15 = 90 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_15_gon_l2067_206787


namespace NUMINAMATH_CALUDE_number_thought_of_l2067_206751

theorem number_thought_of (x : ℝ) : (x / 5 + 8 = 61) → x = 265 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l2067_206751


namespace NUMINAMATH_CALUDE_circle_radius_is_sqrt_two_l2067_206766

/-- A circle inside a right angle with specific properties -/
structure CircleInRightAngle where
  /-- The radius of the circle -/
  R : ℝ
  /-- The length of chord AB -/
  AB : ℝ
  /-- The length of chord CD -/
  CD : ℝ
  /-- The circle is inside a right angle -/
  inside_right_angle : True
  /-- The circle is tangent to one side of the angle -/
  tangent_to_side : True
  /-- The circle intersects the other side at points A and B -/
  intersects_side : True
  /-- The circle intersects the angle bisector at points C and D -/
  intersects_bisector : True
  /-- AB = √6 -/
  h_AB : AB = Real.sqrt 6
  /-- CD = √7 -/
  h_CD : CD = Real.sqrt 7

/-- The theorem stating that the radius of the circle is √2 -/
theorem circle_radius_is_sqrt_two (c : CircleInRightAngle) : c.R = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_sqrt_two_l2067_206766


namespace NUMINAMATH_CALUDE_difference_of_31st_terms_l2067_206734

def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem difference_of_31st_terms : 
  let C := arithmeticSequence 50 12
  let D := arithmeticSequence 50 (-8)
  |C 31 - D 31| = 600 := by sorry

end NUMINAMATH_CALUDE_difference_of_31st_terms_l2067_206734


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2067_206756

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    4 * π * r^2 = 324 * π →
    (4 / 3) * π * r^3 = 972 * π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2067_206756


namespace NUMINAMATH_CALUDE_first_half_rate_is_5_4_l2067_206736

/-- Represents a cricket game with two halves --/
structure CricketGame where
  total_overs : ℕ
  target_runs : ℕ
  second_half_rate : ℚ

/-- Calculates the run rate for the first half of the game --/
def first_half_run_rate (game : CricketGame) : ℚ :=
  let first_half_overs : ℚ := game.total_overs / 2
  let second_half_runs : ℚ := game.second_half_rate * first_half_overs
  let first_half_runs : ℚ := game.target_runs - second_half_runs
  first_half_runs / first_half_overs

/-- Theorem stating the first half run rate for the given game conditions --/
theorem first_half_rate_is_5_4 (game : CricketGame) 
    (h1 : game.total_overs = 50)
    (h2 : game.target_runs = 400)
    (h3 : game.second_half_rate = 53 / 5) : 
  first_half_run_rate game = 27 / 5 := by
  sorry

#eval (53 : ℚ) / 5  -- Outputs 10.6
#eval (27 : ℚ) / 5  -- Outputs 5.4

end NUMINAMATH_CALUDE_first_half_rate_is_5_4_l2067_206736


namespace NUMINAMATH_CALUDE_candy_eating_problem_l2067_206794

/-- Represents the number of candies eaten by each person -/
structure CandyEaten where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the rates at which each person eats candies -/
structure EatingRates where
  andrey : ℚ
  boris : ℚ
  denis : ℚ

/-- The theorem statement based on the given problem -/
theorem candy_eating_problem (rates : EatingRates) 
  (h1 : rates.andrey * 4 = rates.boris * 3)
  (h2 : rates.denis * 6 = rates.andrey * 7)
  (h3 : rates.andrey + rates.boris + rates.denis = 70) :
  ∃ (eaten : CandyEaten), 
    eaten.andrey = 24 ∧ 
    eaten.boris = 18 ∧ 
    eaten.denis = 28 ∧
    eaten.andrey + eaten.boris + eaten.denis = 70 := by
  sorry

end NUMINAMATH_CALUDE_candy_eating_problem_l2067_206794


namespace NUMINAMATH_CALUDE_quadratic_root_ratio_l2067_206795

theorem quadratic_root_ratio (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_ratio_l2067_206795


namespace NUMINAMATH_CALUDE_hexagon_pentagon_angle_sum_l2067_206783

theorem hexagon_pentagon_angle_sum : 
  let hexagon_angle := 180 * (6 - 2) / 6
  let pentagon_angle := 180 * (5 - 2) / 5
  hexagon_angle + pentagon_angle = 228 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_pentagon_angle_sum_l2067_206783


namespace NUMINAMATH_CALUDE_tic_tac_toe_strategy_l2067_206786

/-- Represents a 10x10 tic-tac-toe board -/
def Board := Fin 10 → Fin 10 → Bool

/-- Counts the number of sets of five consecutive marks for a player -/
def count_sets (b : Board) (player : Bool) : ℕ := sorry

/-- Calculates the score for the first player (X) -/
def score (b : Board) : ℤ :=
  (count_sets b true : ℤ) - (count_sets b false : ℤ)

/-- A strategy for a player -/
def Strategy := Board → Fin 10 × Fin 10

/-- Applies a strategy to a board, returning the updated board -/
def apply_strategy (b : Board) (s : Strategy) (player : Bool) : Board := sorry

/-- Represents a full game play -/
def play_game (s1 s2 : Strategy) : Board := sorry

theorem tic_tac_toe_strategy :
  (∃ (s : Strategy), ∀ (s2 : Strategy), score (play_game s s2) ≥ 0) ∧
  (¬ ∃ (s : Strategy), ∀ (s2 : Strategy), score (play_game s s2) > 0) :=
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_strategy_l2067_206786


namespace NUMINAMATH_CALUDE_chessboard_decomposition_l2067_206760

/-- Represents a square on a chessboard -/
structure Square where
  side : ℕ
  area : ℕ
  area_eq : area = side * side

/-- Represents a chessboard -/
structure Chessboard where
  side : ℕ
  area : ℕ
  area_eq : area = side * side

/-- Represents a decomposition of a chessboard into squares -/
structure Decomposition (board : Chessboard) where
  squares : List Square
  piece_count : ℕ
  valid : piece_count = 6 ∧ squares.length = 3
  area_sum : (squares.map (·.area)).sum = board.area

/-- The main theorem: A 7x7 chessboard can be decomposed into 6 pieces 
    that form three squares of sizes 6x6, 3x3, and 2x2 -/
theorem chessboard_decomposition :
  ∃ (d : Decomposition ⟨7, 49, rfl⟩),
    d.squares = [⟨6, 36, rfl⟩, ⟨3, 9, rfl⟩, ⟨2, 4, rfl⟩] := by
  sorry

end NUMINAMATH_CALUDE_chessboard_decomposition_l2067_206760


namespace NUMINAMATH_CALUDE_square_root_of_square_negative_two_l2067_206710

theorem square_root_of_square_negative_two : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_negative_two_l2067_206710


namespace NUMINAMATH_CALUDE_part_one_part_two_l2067_206716

/-- Condition p: (x - a)(x - 3a) < 0 -/
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

/-- Condition q: (x - 3) / (x - 2) ≤ 0 -/
def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

/-- Part 1: When a = 1 and p ∧ q is true, then 2 < x < 3 -/
theorem part_one (x : ℝ) (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

/-- Part 2: When p is necessary but not sufficient for q, and a > 0, then 1 ≤ a ≤ 2 -/
theorem part_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x, q x → p x a) 
  (h3 : ∃ x, p x a ∧ ¬q x) : 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2067_206716


namespace NUMINAMATH_CALUDE_exponent_division_l2067_206742

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^12 / a^4 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2067_206742


namespace NUMINAMATH_CALUDE_system_solution_l2067_206754

theorem system_solution (a b c d e f : ℝ) 
  (eq1 : 4 * a = (b + c + d + e)^4)
  (eq2 : 4 * b = (c + d + e + f)^4)
  (eq3 : 4 * c = (d + e + f + a)^4)
  (eq4 : 4 * d = (e + f + a + b)^4)
  (eq5 : 4 * e = (f + a + b + c)^4)
  (eq6 : 4 * f = (a + b + c + d)^4) :
  a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4 ∧ e = 1/4 ∧ f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2067_206754
