import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_proof_l4050_405078

theorem smallest_number_proof (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : a + b + c = 73)
  (h4 : c - b = 5)
  (h5 : b - a = 6) :
  a = 56 / 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l4050_405078


namespace NUMINAMATH_CALUDE_largest_operation_l4050_405088

theorem largest_operation : ∀ a b c d e : ℝ,
  a = 15432 + 1 / 3241 →
  b = 15432 - 1 / 3241 →
  c = 15432 * (1 / 3241) →
  d = 15432 / (1 / 3241) →
  e = 15432.3241 →
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_operation_l4050_405088


namespace NUMINAMATH_CALUDE_simplify_negative_x_powers_l4050_405054

theorem simplify_negative_x_powers (x : ℝ) : (-x)^3 * (-x)^2 = -x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_negative_x_powers_l4050_405054


namespace NUMINAMATH_CALUDE_expansion_coefficient_l4050_405090

/-- The coefficient of x^3 in the expansion of ((ax-1)^6) -/
def coefficient_x3 (a : ℝ) : ℝ := -20 * a^3

/-- The theorem states that if the coefficient of x^3 in the expansion of ((ax-1)^6) is 20, then a = -1 -/
theorem expansion_coefficient (a : ℝ) : coefficient_x3 a = 20 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l4050_405090


namespace NUMINAMATH_CALUDE_number_division_puzzle_l4050_405074

theorem number_division_puzzle (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a / b = (a + b) / (2 * a) ∧ a / b ≠ 1 → a / b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_number_division_puzzle_l4050_405074


namespace NUMINAMATH_CALUDE_tshirt_pricing_theorem_l4050_405061

/-- Represents the cost and pricing information for two batches of T-shirts --/
structure TShirtBatches where
  first_batch_cost : ℕ
  second_batch_cost : ℕ
  quantity_ratio : ℚ
  price_difference : ℕ
  first_batch_selling_price : ℕ
  min_total_profit : ℕ

/-- Calculates the cost price of each T-shirt in the first batch --/
def cost_price_first_batch (b : TShirtBatches) : ℚ :=
  sorry

/-- Calculates the minimum selling price for the second batch --/
def min_selling_price_second_batch (b : TShirtBatches) : ℕ :=
  sorry

/-- Theorem stating the correct cost price and minimum selling price --/
theorem tshirt_pricing_theorem (b : TShirtBatches) 
  (h1 : b.first_batch_cost = 4000)
  (h2 : b.second_batch_cost = 5400)
  (h3 : b.quantity_ratio = 3/2)
  (h4 : b.price_difference = 5)
  (h5 : b.first_batch_selling_price = 70)
  (h6 : b.min_total_profit = 4060) :
  cost_price_first_batch b = 50 ∧ 
  min_selling_price_second_batch b = 66 :=
  sorry

end NUMINAMATH_CALUDE_tshirt_pricing_theorem_l4050_405061


namespace NUMINAMATH_CALUDE_tan_ratio_max_tan_A_l4050_405071

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.b^2 + 3*t.a^2 = t.c^2

-- Theorem 1: tan(C) / tan(B) = -2
theorem tan_ratio (t : Triangle) (h : triangle_condition t) :
  Real.tan t.C / Real.tan t.B = -2 := by sorry

-- Theorem 2: Maximum value of tan(A) is √2/4
theorem max_tan_A (t : Triangle) (h : triangle_condition t) :
  ∃ (max_tan_A : ℝ), (∀ (t' : Triangle), triangle_condition t' → Real.tan t'.A ≤ max_tan_A) ∧ max_tan_A = Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_tan_ratio_max_tan_A_l4050_405071


namespace NUMINAMATH_CALUDE_smallest_B_for_divisibility_l4050_405066

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number (B C : ℕ) : ℕ := 4000000 + 100000 * B + 80000 + 3000 + 900 + 90 + C

theorem smallest_B_for_divisibility :
  ∃ (C : ℕ), is_digit C ∧ number 0 C % 3 = 0 ∧
  ∀ (B : ℕ), is_digit B → (∃ (C : ℕ), is_digit C ∧ number B C % 3 = 0) → B ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_B_for_divisibility_l4050_405066


namespace NUMINAMATH_CALUDE_expression_equivalence_l4050_405019

/-- Prove that the given expression is equivalent to 4xy(x^2 + y^2)/(x^4 + y^4) -/
theorem expression_equivalence (x y : ℝ) :
  let P := x^2 + y^2
  let Q := x*y
  ((P + Q) / (P - Q)) - ((P - Q) / (P + Q)) = (4*x*y*(x^2 + y^2)) / (x^4 + y^4) := by
sorry

end NUMINAMATH_CALUDE_expression_equivalence_l4050_405019


namespace NUMINAMATH_CALUDE_det_A_l4050_405011

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, -6, 6; 0, 6, -2; 3, -1, 2]

theorem det_A : Matrix.det A = -52 := by
  sorry

end NUMINAMATH_CALUDE_det_A_l4050_405011


namespace NUMINAMATH_CALUDE_equation_solutions_l4050_405064

theorem equation_solutions :
  ∀ a b c : ℕ+,
  (1 : ℚ) / a + (2 : ℚ) / b - (3 : ℚ) / c = 1 ↔
  (∃ n : ℕ+, a = 1 ∧ b = 2 * n ∧ c = 3 * n) ∨
  (a = 2 ∧ b = 1 ∧ c = 2) ∨
  (a = 2 ∧ b = 3 ∧ c = 18) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l4050_405064


namespace NUMINAMATH_CALUDE_product_mod_25_l4050_405016

theorem product_mod_25 : (43 * 67 * 92) % 25 = 2 := by
  sorry

#check product_mod_25

end NUMINAMATH_CALUDE_product_mod_25_l4050_405016


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l4050_405044

/-- Given an inverse proportion function y = k/x passing through (-2, 3), prove k = -6 -/
theorem inverse_proportion_k_value : ∀ k : ℝ, 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → f x = k / x) ∧ f (-2) = 3) → 
  k = -6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l4050_405044


namespace NUMINAMATH_CALUDE_complex_power_equivalence_l4050_405031

theorem complex_power_equivalence :
  (Complex.exp (Complex.I * Real.pi * (35 / 180)))^100 = Complex.exp (Complex.I * Real.pi * (20 / 180)) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_equivalence_l4050_405031


namespace NUMINAMATH_CALUDE_divisibility_of_10_pow_6_minus_1_l4050_405003

theorem divisibility_of_10_pow_6_minus_1 :
  ∃ (a b c d : ℕ), 10^6 - 1 = 7 * a ∧ 10^6 - 1 = 13 * b ∧ 10^6 - 1 = 91 * c ∧ 10^6 - 1 = 819 * d :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_10_pow_6_minus_1_l4050_405003


namespace NUMINAMATH_CALUDE_turtle_difference_l4050_405045

/-- Given the following conditions about turtle ownership:
  1. Trey has 9 times as many turtles as Kris
  2. Kris has 1/3 as many turtles as Kristen
  3. Layla has twice as many turtles as Trey
  4. Tim has half as many turtles as Kristen
  5. Kristen has 18 turtles

  Prove that Trey has 45 more turtles than Tim. -/
theorem turtle_difference (kristen tim trey kris layla : ℕ) : 
  kristen = 18 →
  kris = kristen / 3 →
  trey = 9 * kris →
  layla = 2 * trey →
  tim = kristen / 2 →
  trey - tim = 45 := by
sorry

end NUMINAMATH_CALUDE_turtle_difference_l4050_405045


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_3_l4050_405062

theorem no_solution_implies_a_leq_3 (a : ℝ) : 
  (∀ x : ℝ, ¬(x ≥ 3 ∧ x < a)) → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_3_l4050_405062


namespace NUMINAMATH_CALUDE_orange_cost_calculation_l4050_405021

def initial_amount : ℕ := 95
def apple_cost : ℕ := 25
def candy_cost : ℕ := 6
def amount_left : ℕ := 50

theorem orange_cost_calculation : 
  initial_amount - amount_left - apple_cost - candy_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_calculation_l4050_405021


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l4050_405025

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 4) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l4050_405025


namespace NUMINAMATH_CALUDE_nursing_home_milk_distribution_l4050_405027

theorem nursing_home_milk_distribution (elderly : ℕ) (milk : ℕ) : 
  (2 * elderly + 16 = milk) ∧ (4 * elderly = milk + 12) → 
  (elderly = 14 ∧ milk = 44) :=
by sorry

end NUMINAMATH_CALUDE_nursing_home_milk_distribution_l4050_405027


namespace NUMINAMATH_CALUDE_equation_solutions_l4050_405086

theorem equation_solutions :
  (∃ x : ℝ, x * (x + 10) = -9 ↔ x = -9 ∨ x = -1) ∧
  (∃ x : ℝ, x * (2 * x + 3) = 8 * x + 12 ↔ x = -3/2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l4050_405086


namespace NUMINAMATH_CALUDE_unique_integer_solution_l4050_405049

/-- The function f(x) = -x^2 + x + m + 2 -/
def f (x m : ℝ) : ℝ := -x^2 + x + m + 2

/-- The solution set of f(x) ≥ |x| -/
def solution_set (m : ℝ) : Set ℝ := {x : ℝ | f x m ≥ |x|}

/-- The set of integers in the solution set -/
def integer_solutions (m : ℝ) : Set ℤ := {i : ℤ | (i : ℝ) ∈ solution_set m}

theorem unique_integer_solution (m : ℝ) :
  (∃! (i : ℤ), (i : ℝ) ∈ solution_set m) → -2 < m ∧ m < -1 :=
sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l4050_405049


namespace NUMINAMATH_CALUDE_range_of_a_l4050_405093

def f (x : ℝ) : ℝ := -x^5 - 3*x^3 - 5*x + 3

theorem range_of_a (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4050_405093


namespace NUMINAMATH_CALUDE_ellipse_properties_l4050_405014

/-- Properties of the ellipse y²/25 + x²/16 = 1 -/
theorem ellipse_properties :
  let ellipse := (fun (x y : ℝ) => y^2 / 25 + x^2 / 16 = 1)
  ∃ (a b c : ℝ),
    -- Major and minor axis lengths
    a = 5 ∧ b = 4 ∧
    -- Vertices
    ellipse (-4) 0 ∧ ellipse 4 0 ∧ ellipse 0 5 ∧ ellipse 0 (-5) ∧
    -- Foci
    c = 3 ∧
    -- Eccentricity
    c / a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4050_405014


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l4050_405096

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l4050_405096


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l4050_405001

/-- The volume of a rectangular parallelepiped with given conditions -/
theorem rectangular_parallelepiped_volume :
  ∀ (length width height : ℝ),
  length > 0 →
  width > 0 →
  height > 0 →
  length = width →
  2 * (length + width) = 32 →
  height = 9 →
  length * width * height = 576 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l4050_405001


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l4050_405082

theorem smallest_number_divisible (n : ℕ) : n ≥ 58 →
  (∃ k : ℕ, n - 10 = 24 * k) →
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, m - 10 = 24 * k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l4050_405082


namespace NUMINAMATH_CALUDE_factorization_problem_1_l4050_405079

theorem factorization_problem_1 (x y : ℝ) : x * y - x + y - 1 = (x + 1) * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l4050_405079


namespace NUMINAMATH_CALUDE_cos_two_pi_seventh_inequality_l4050_405039

theorem cos_two_pi_seventh_inequality (a : ℝ) :
  a = Real.cos ((2 * Real.pi) / 7) →
  0 < (1 : ℝ) / 2 ∧ (1 : ℝ) / 2 < a ∧ a < Real.sqrt 2 / 2 ∧ Real.sqrt 2 / 2 < 1 →
  2^(a - 1/2) < 2 * a := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_seventh_inequality_l4050_405039


namespace NUMINAMATH_CALUDE_curve_area_range_l4050_405051

theorem curve_area_range (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*m*x + 2 = 0) →
  (∀ (x y : ℝ), x^2 + y^2 + 2*m*x + 2 = 0 → π * ((x + m)^2 + y^2) ≥ 4 * π) →
  m ≤ -Real.sqrt 6 ∨ m ≥ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_curve_area_range_l4050_405051


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4050_405067

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 5 * x + c = 0 ↔ x = (-5 + Real.sqrt 21) / 4 ∨ x = (-5 - Real.sqrt 21) / 4) →
  c = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4050_405067


namespace NUMINAMATH_CALUDE_bird_feet_count_l4050_405015

theorem bird_feet_count (num_birds : ℕ) (feet_per_bird : ℕ) (h1 : num_birds = 46) (h2 : feet_per_bird = 2) :
  num_birds * feet_per_bird = 92 := by
  sorry

end NUMINAMATH_CALUDE_bird_feet_count_l4050_405015


namespace NUMINAMATH_CALUDE_average_age_when_youngest_born_l4050_405098

theorem average_age_when_youngest_born (total_people : ℕ) (current_average_age : ℝ) (youngest_age : ℝ) :
  total_people = 7 →
  current_average_age = 30 →
  youngest_age = 7 →
  (total_people * current_average_age - (total_people - 1) * youngest_age) / (total_people - 1) = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_when_youngest_born_l4050_405098


namespace NUMINAMATH_CALUDE_negation_of_proposition_cubic_inequality_negation_l4050_405038

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem cubic_inequality_negation : 
  (¬∀ x : ℝ, x^3 + 2 < 0) ↔ (∃ x : ℝ, x^3 + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_cubic_inequality_negation_l4050_405038


namespace NUMINAMATH_CALUDE_cool_drink_solution_volume_l4050_405091

/-- Represents the cool-drink solution problem --/
theorem cool_drink_solution_volume 
  (initial_jasmine_percent : Real)
  (added_jasmine : Real)
  (added_water : Real)
  (final_jasmine_percent : Real)
  (h1 : initial_jasmine_percent = 0.05)
  (h2 : added_jasmine = 8)
  (h3 : added_water = 2)
  (h4 : final_jasmine_percent = 0.125)
  : ∃ (initial_volume : Real),
    initial_volume * initial_jasmine_percent + added_jasmine = 
    (initial_volume + added_jasmine + added_water) * final_jasmine_percent ∧
    initial_volume = 90 := by
  sorry

end NUMINAMATH_CALUDE_cool_drink_solution_volume_l4050_405091


namespace NUMINAMATH_CALUDE_three_intersections_iff_l4050_405024

/-- The number of intersection points between the curves x^2 + y^2 = a^2 and y = x^2 + a -/
def num_intersections (a : ℝ) : ℕ :=
  sorry

/-- Theorem stating the condition for exactly 3 intersection points -/
theorem three_intersections_iff (a : ℝ) : 
  num_intersections a = 3 ↔ a < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_three_intersections_iff_l4050_405024


namespace NUMINAMATH_CALUDE_coefficient_sum_y_terms_l4050_405060

theorem coefficient_sum_y_terms (x y : ℝ) : 
  let expanded := (5*x + 3*y - 4) * (2*x - 3*y + 6)
  let coeff_y := -9  -- Coefficient of xy
  let coeff_y2 := -9 -- Coefficient of y^2
  let coeff_y1 := 30 -- Coefficient of y
  coeff_y + coeff_y2 + coeff_y1 = 12 := by sorry

end NUMINAMATH_CALUDE_coefficient_sum_y_terms_l4050_405060


namespace NUMINAMATH_CALUDE_gcd_of_90_and_405_l4050_405032

theorem gcd_of_90_and_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_90_and_405_l4050_405032


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4050_405037

theorem modulus_of_complex_fraction (i : ℂ) (h : i * i = -1) :
  Complex.abs (i / (2 - i)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l4050_405037


namespace NUMINAMATH_CALUDE_circle_area_with_complex_conditions_l4050_405076

theorem circle_area_with_complex_conditions (z₁ z₂ : ℂ) 
  (h1 : z₁^2 - 4*z₁*z₂ + 4*z₂^2 = 0)
  (h2 : Complex.abs z₂ = 2) :
  Real.pi * (Complex.abs z₁ / 2)^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_complex_conditions_l4050_405076


namespace NUMINAMATH_CALUDE_jerome_has_zero_left_l4050_405058

/-- Represents Jerome's financial transactions --/
def jerome_transactions (initial_euros : ℝ) (exchange_rate : ℝ) (meg_amount : ℝ) : ℝ := by
  -- Convert initial amount to dollars
  let initial_dollars := initial_euros * exchange_rate
  -- Subtract Meg's amount
  let after_meg := initial_dollars - meg_amount
  -- Subtract Bianca's amount (thrice Meg's)
  let after_bianca := after_meg - (3 * meg_amount)
  -- Give all remaining money to Nathan
  exact 0

/-- Theorem stating that Jerome has $0 left after transactions --/
theorem jerome_has_zero_left : 
  ∀ (initial_euros : ℝ) (exchange_rate : ℝ) (meg_amount : ℝ),
  initial_euros > 0 ∧ exchange_rate > 0 ∧ meg_amount > 0 →
  jerome_transactions initial_euros exchange_rate meg_amount = 0 := by
  sorry

#check jerome_has_zero_left

end NUMINAMATH_CALUDE_jerome_has_zero_left_l4050_405058


namespace NUMINAMATH_CALUDE_train_meeting_time_l4050_405026

/-- Calculates the time for two trains to meet given their speeds, lengths, and the platform length --/
theorem train_meeting_time (length_A length_B platform_length : ℝ)
                           (speed_A speed_B : ℝ)
                           (h1 : length_A = 120)
                           (h2 : length_B = 150)
                           (h3 : platform_length = 180)
                           (h4 : speed_A = 90 * 1000 / 3600)
                           (h5 : speed_B = 72 * 1000 / 3600) :
  (length_A + length_B + platform_length) / (speed_A + speed_B) = 10 := by
  sorry

#check train_meeting_time

end NUMINAMATH_CALUDE_train_meeting_time_l4050_405026


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l4050_405057

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | x < -2 ∨ x > 5}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l4050_405057


namespace NUMINAMATH_CALUDE_specialIntegers_infinite_l4050_405023

/-- A function that converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- A predicate that checks if a list of digits contains only 1 and 2 -/
def containsOnly1And2 (digits : List ℕ) : Prop :=
  sorry

/-- The set of positive integers n such that n^2 in base 4 contains only digits 1 and 2 -/
def specialIntegers : Set ℕ :=
  {n : ℕ | n > 0 ∧ containsOnly1And2 (toBase4 (n^2))}

/-- The main theorem stating that the set of special integers is infinite -/
theorem specialIntegers_infinite : Set.Infinite specialIntegers :=
  sorry

end NUMINAMATH_CALUDE_specialIntegers_infinite_l4050_405023


namespace NUMINAMATH_CALUDE_rudolph_encountered_two_stop_signs_per_mile_l4050_405097

/-- Rudolph's car trip across town -/
def rudolph_trip (miles : ℕ) (stop_signs : ℕ) : Prop :=
  miles = 5 + 2 ∧ stop_signs = 17 - 3

/-- The number of stop signs per mile -/
def stop_signs_per_mile (miles : ℕ) (stop_signs : ℕ) : ℚ :=
  stop_signs / miles

/-- Theorem: Rudolph encountered 2 stop signs per mile -/
theorem rudolph_encountered_two_stop_signs_per_mile :
  ∀ (miles : ℕ) (stop_signs : ℕ),
  rudolph_trip miles stop_signs →
  stop_signs_per_mile miles stop_signs = 2 := by
  sorry

end NUMINAMATH_CALUDE_rudolph_encountered_two_stop_signs_per_mile_l4050_405097


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4050_405041

theorem complex_equation_solution :
  ∃ (z : ℂ), 5 + 2 * I * z = 3 - 5 * I * z ∧ z = (2 * I) / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4050_405041


namespace NUMINAMATH_CALUDE_range_of_f_l4050_405052

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l4050_405052


namespace NUMINAMATH_CALUDE_inverse_log_property_l4050_405063

noncomputable section

variable (a : ℝ)
variable (a_pos : a > 0)
variable (a_ne_one : a ≠ 1)

def f (x : ℝ) := Real.log x / Real.log a

def f_inverse (x : ℝ) := a ^ x

theorem inverse_log_property (h : f_inverse a 2 = 9) : f a 9 + f a 6 = 2 := by
  sorry

#check inverse_log_property

end NUMINAMATH_CALUDE_inverse_log_property_l4050_405063


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l4050_405034

-- Define set A
def A : Set ℝ := {x | x^2 + x - 2 = 0}

-- Define set B
def B : Set ℝ := {x | x ≥ 0 ∧ x < 1}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x | x = -2 ∨ (0 ≤ x ∧ x < 1)} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l4050_405034


namespace NUMINAMATH_CALUDE_multiples_of_12_between_30_and_200_l4050_405073

theorem multiples_of_12_between_30_and_200 : 
  (Finset.filter (fun n => n % 12 = 0 ∧ n ≥ 30 ∧ n ≤ 200) (Finset.range 201)).card = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_12_between_30_and_200_l4050_405073


namespace NUMINAMATH_CALUDE_boys_score_in_class_l4050_405046

theorem boys_score_in_class (boy_percentage : ℝ) (girl_percentage : ℝ) 
  (girl_score : ℝ) (class_average : ℝ) : 
  boy_percentage = 40 →
  girl_percentage = 100 - boy_percentage →
  girl_score = 90 →
  class_average = 86 →
  (boy_percentage * boy_score + girl_percentage * girl_score) / 100 = class_average →
  boy_score = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_score_in_class_l4050_405046


namespace NUMINAMATH_CALUDE_profit_achieved_min_lemons_optimal_l4050_405095

/-- The number of lemons bought in one purchase -/
def lemons_bought : ℕ := 4

/-- The cost in cents for buying lemons_bought lemons -/
def buying_cost : ℕ := 25

/-- The number of lemons sold in one sale -/
def lemons_sold : ℕ := 7

/-- The revenue in cents from selling lemons_sold lemons -/
def selling_revenue : ℕ := 50

/-- The desired profit in cents -/
def desired_profit : ℕ := 150

/-- The minimum number of lemons needed to be sold to achieve the desired profit -/
def min_lemons_to_sell : ℕ := 169

theorem profit_achieved (n : ℕ) : n ≥ min_lemons_to_sell →
  (n * selling_revenue / lemons_sold - n * buying_cost / lemons_bought) ≥ desired_profit :=
by sorry

theorem min_lemons_optimal : 
  ∀ m : ℕ, m < min_lemons_to_sell →
  (m * selling_revenue / lemons_sold - m * buying_cost / lemons_bought) < desired_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_achieved_min_lemons_optimal_l4050_405095


namespace NUMINAMATH_CALUDE_distance_between_harper_and_jack_l4050_405018

/-- The distance between two runners at the end of a race --/
def distance_between_runners (race_length : ℕ) (jack_position : ℕ) : ℕ :=
  race_length - jack_position

/-- Theorem: The distance between Harper and Jack when Harper finished the race is 848 meters --/
theorem distance_between_harper_and_jack :
  let race_length_meters : ℕ := 1000  -- 1 km = 1000 meters
  let jack_position : ℕ := 152
  distance_between_runners race_length_meters jack_position = 848 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_harper_and_jack_l4050_405018


namespace NUMINAMATH_CALUDE_chess_team_arrangement_l4050_405017

/-- The number of boys on the chess team -/
def num_boys : ℕ := 3

/-- The number of girls on the chess team -/
def num_girls : ℕ := 2

/-- The total number of students on the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange the chess team in a row with a girl at each end and boys in the middle -/
def num_arrangements : ℕ := num_girls.factorial * num_boys.factorial

theorem chess_team_arrangement :
  num_arrangements = 12 :=
sorry

end NUMINAMATH_CALUDE_chess_team_arrangement_l4050_405017


namespace NUMINAMATH_CALUDE_bamboo_problem_l4050_405042

def arithmetic_sequence (a : ℚ → ℚ) (n : ℕ) : Prop :=
  ∃ (a₁ d : ℚ), ∀ k, a k = a₁ + (k - 1) * d

theorem bamboo_problem (a : ℚ → ℚ) :
  arithmetic_sequence a 9 →
  (a 1 + a 2 + a 3 + a 4 = 3) →
  (a 7 + a 8 + a 9 = 4) →
  a 5 = 67 / 66 := by
sorry

end NUMINAMATH_CALUDE_bamboo_problem_l4050_405042


namespace NUMINAMATH_CALUDE_polynomial_not_perfect_square_l4050_405047

theorem polynomial_not_perfect_square (a b c d : ℤ) (n : ℕ+) :
  ∃ (S : Finset ℕ), 
    S.card ≥ n / 4 ∧ 
    ∀ m ∈ S, m ≤ n ∧ 
    ¬∃ (k : ℤ), (m^5 : ℤ) + d*m^4 + c*m^3 + b*m^2 + 2023*m + a = k^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_perfect_square_l4050_405047


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4050_405022

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a line in 3D space
structure Line3D where
  point1 : Point3D
  point2 : Point3D

-- Define the property of non-coplanar points
def nonCoplanar (E F G H : Point3D) : Prop := sorry

-- Define the property of non-intersecting lines
def nonIntersecting (l1 l2 : Line3D) : Prop := sorry

theorem sufficient_but_not_necessary 
  (E F G H : Point3D) 
  (EF : Line3D) 
  (GH : Line3D) 
  (h_EF : EF.point1 = E ∧ EF.point2 = F) 
  (h_GH : GH.point1 = G ∧ GH.point2 = H) :
  (nonCoplanar E F G H → nonIntersecting EF GH) ∧ 
  ∃ E' F' G' H' : Point3D, ∃ EF' GH' : Line3D, 
    (EF'.point1 = E' ∧ EF'.point2 = F') ∧ 
    (GH'.point1 = G' ∧ GH'.point2 = H') ∧ 
    nonIntersecting EF' GH' ∧ 
    ¬(nonCoplanar E' F' G' H') := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l4050_405022


namespace NUMINAMATH_CALUDE_basketball_spectators_l4050_405092

theorem basketball_spectators (total : ℕ) (children : ℕ) 
  (h1 : total = 10000)
  (h2 : children = 2500)
  (h3 : children = 5 * (total - children - (total - children - children) / 5)) :
  total - children - (total - children - children) / 5 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_basketball_spectators_l4050_405092


namespace NUMINAMATH_CALUDE_boys_to_total_ratio_l4050_405010

theorem boys_to_total_ratio (boys girls : ℕ) (h1 : boys > 0) (h2 : girls > 0) : 
  let total := boys + girls
  let prob_boy := boys / total
  let prob_girl := girls / total
  prob_boy = (1 / 4 : ℚ) * prob_girl →
  (boys : ℚ) / total = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_boys_to_total_ratio_l4050_405010


namespace NUMINAMATH_CALUDE_total_cotton_yield_l4050_405099

/-- 
Given two cotton fields:
- Field 1 has m hectares and produces an average of a kilograms per hectare
- Field 2 has n hectares and produces an average of b kilograms per hectare
This theorem proves that the total cotton yield is am + bn kilograms
-/
theorem total_cotton_yield 
  (m n a b : ℝ) 
  (h1 : m ≥ 0) 
  (h2 : n ≥ 0) 
  (h3 : a ≥ 0) 
  (h4 : b ≥ 0) : 
  m * a + n * b = m * a + n * b := by
  sorry

end NUMINAMATH_CALUDE_total_cotton_yield_l4050_405099


namespace NUMINAMATH_CALUDE_equal_squares_in_5x8_grid_l4050_405030

/-- A rectangular grid with alternating light and dark squares -/
structure AlternatingGrid where
  rows : ℕ
  cols : ℕ

/-- Count of dark squares in an AlternatingGrid -/
def dark_squares (grid : AlternatingGrid) : ℕ :=
  sorry

/-- Count of light squares in an AlternatingGrid -/
def light_squares (grid : AlternatingGrid) : ℕ :=
  sorry

/-- Theorem: In a 5 × 8 grid with alternating squares, the number of dark squares equals the number of light squares -/
theorem equal_squares_in_5x8_grid :
  let grid : AlternatingGrid := ⟨5, 8⟩
  dark_squares grid = light_squares grid :=
by sorry

end NUMINAMATH_CALUDE_equal_squares_in_5x8_grid_l4050_405030


namespace NUMINAMATH_CALUDE_quadratic_radicals_theorem_l4050_405081

-- Define the condition that the radicals can be combined
def radicals_can_combine (a : ℝ) : Prop := 3 * a - 8 = 17 - 2 * a

-- Define the range of x that makes √(4a-2x) meaningful
def valid_x_range (a x : ℝ) : Prop := 4 * a - 2 * x ≥ 0

-- Theorem statement
theorem quadratic_radicals_theorem (a x : ℝ) :
  radicals_can_combine a → (∃ a, radicals_can_combine a ∧ a = 5) →
  (valid_x_range a x ↔ x ≤ 10) :=
sorry

end NUMINAMATH_CALUDE_quadratic_radicals_theorem_l4050_405081


namespace NUMINAMATH_CALUDE_xiaoding_distance_l4050_405029

/-- Represents the distance to school for each student in meters -/
structure SchoolDistances where
  xiaoding : ℕ
  xiaowang : ℕ
  xiaocheng : ℕ
  xiaozhang : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (d : SchoolDistances) : Prop :=
  d.xiaowang + d.xiaoding + d.xiaocheng + d.xiaozhang = 705 ∧
  d.xiaowang = 4 * d.xiaoding ∧
  d.xiaocheng = d.xiaowang / 2 + 20 ∧
  d.xiaozhang = 2 * d.xiaocheng - 15

/-- The theorem to be proved -/
theorem xiaoding_distance (d : SchoolDistances) :
  satisfiesConditions d → d.xiaoding = 60 := by
  sorry


end NUMINAMATH_CALUDE_xiaoding_distance_l4050_405029


namespace NUMINAMATH_CALUDE_sum_of_star_equation_l4050_405020

/-- Custom operation ★ -/
def star (a b : ℕ) : ℕ := a^b + a + b

theorem sum_of_star_equation {a b : ℕ} (ha : a ≥ 2) (hb : b ≥ 2) (heq : star a b = 20) :
  a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_star_equation_l4050_405020


namespace NUMINAMATH_CALUDE_printer_Z_time_l4050_405053

/-- The time it takes for printer Z to do the job alone -/
def T_Z : ℝ := 18

/-- The time it takes for printer X to do the job alone -/
def T_X : ℝ := 15

/-- The time it takes for printer Y to do the job alone -/
def T_Y : ℝ := 12

/-- The ratio of X's time to Y and Z's combined time -/
def ratio : ℝ := 2.0833333333333335

theorem printer_Z_time :
  T_Z = 18 ∧
  T_X = 15 ∧
  T_Y = 12 ∧
  ratio = 15 / (1 / (1 / T_Y + 1 / T_Z)) :=
by sorry

end NUMINAMATH_CALUDE_printer_Z_time_l4050_405053


namespace NUMINAMATH_CALUDE_percentage_difference_l4050_405002

theorem percentage_difference : 
  (0.80 * 170 : ℝ) - (0.35 * 300 : ℝ) = 31 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l4050_405002


namespace NUMINAMATH_CALUDE_total_students_l4050_405040

-- Define the score groups
inductive ScoreGroup
| Low : ScoreGroup    -- [20, 40)
| Medium : ScoreGroup -- [40, 60)
| High : ScoreGroup   -- [60, 80)
| VeryHigh : ScoreGroup -- [80, 100]

-- Define the frequency distribution
def FrequencyDistribution := ScoreGroup → ℕ

-- Theorem statement
theorem total_students (freq : FrequencyDistribution) 
  (below_60 : freq ScoreGroup.Low + freq ScoreGroup.Medium = 15) :
  freq ScoreGroup.Low + freq ScoreGroup.Medium + 
  freq ScoreGroup.High + freq ScoreGroup.VeryHigh = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_total_students_l4050_405040


namespace NUMINAMATH_CALUDE_square_sum_given_cube_sum_and_product_l4050_405000

theorem square_sum_given_cube_sum_and_product (x y : ℝ) : 
  (x + y)^3 = 8 → x * y = 5 → x^2 + y^2 = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_cube_sum_and_product_l4050_405000


namespace NUMINAMATH_CALUDE_girls_in_class_l4050_405083

/-- Proves the number of girls in a class with a given ratio and total students -/
theorem girls_in_class (total : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ) :
  total = 63 ∧ girls_ratio = 4 ∧ boys_ratio = 3 →
  (girls_ratio * total) / (girls_ratio + boys_ratio) = 36 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_class_l4050_405083


namespace NUMINAMATH_CALUDE_max_sum_abs_coords_ellipse_l4050_405012

def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

theorem max_sum_abs_coords_ellipse :
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ x y : ℝ, ellipse x y → |x| + |y| ≤ M) ∧
  (∃ x y : ℝ, ellipse x y ∧ |x| + |y| = M) :=
sorry

end NUMINAMATH_CALUDE_max_sum_abs_coords_ellipse_l4050_405012


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l4050_405036

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 8
  let θ : ℝ := 5 * π / 4
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (-2 * Real.sqrt 2, -2 * Real.sqrt 2, 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l4050_405036


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l4050_405007

theorem sun_radius_scientific_notation : 369000 = 3.69 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l4050_405007


namespace NUMINAMATH_CALUDE_negative_y_positive_l4050_405033

theorem negative_y_positive (y : ℝ) (h : y < 0) : -y > 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_y_positive_l4050_405033


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4050_405005

theorem inequality_solution_set (x : ℝ) : (x - 3) / (x - 5) ≥ 3 ↔ x ∈ Set.Ioo 5 6 ∪ {6} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4050_405005


namespace NUMINAMATH_CALUDE_combined_job_time_l4050_405085

def job_time_A : ℝ := 8
def job_time_B : ℝ := 12

theorem combined_job_time : 
  let rate_A := 1 / job_time_A
  let rate_B := 1 / job_time_B
  let combined_rate := rate_A + rate_B
  1 / combined_rate = 4.8 := by sorry

end NUMINAMATH_CALUDE_combined_job_time_l4050_405085


namespace NUMINAMATH_CALUDE_probability_problem_l4050_405035

-- Define the sample space and events
def Ω : Type := Unit
def A₁ : Set Ω := sorry
def A₂ : Set Ω := sorry
def A₃ : Set Ω := sorry
def B : Set Ω := sorry

-- Define the probability measure
def P : Set Ω → ℝ := sorry

-- Theorem statement
theorem probability_problem :
  -- 1. A₁, A₂, and A₃ are pairwise mutually exclusive
  (A₁ ∩ A₂ = ∅ ∧ A₁ ∩ A₃ = ∅ ∧ A₂ ∩ A₃ = ∅) ∧
  -- 2. P(B|A₁) = 1/3
  P B / P A₁ = 1/3 ∧
  -- 3. P(B) = 19/48
  P B = 19/48 ∧
  -- 4. A₂ and B are not independent events
  P (A₂ ∩ B) ≠ P A₂ * P B :=
by sorry

end NUMINAMATH_CALUDE_probability_problem_l4050_405035


namespace NUMINAMATH_CALUDE_unique_intersection_condition_l4050_405094

/-- A function f(x) = kx^2 + 2(k+1)x + k-1 has only one intersection point with the x-axis if and only if k = 0 or k = -1/3 -/
theorem unique_intersection_condition (k : ℝ) : 
  (∃! x, k * x^2 + 2*(k+1)*x + (k-1) = 0) ↔ (k = 0 ∨ k = -1/3) := by
sorry

end NUMINAMATH_CALUDE_unique_intersection_condition_l4050_405094


namespace NUMINAMATH_CALUDE_complex_cube_root_l4050_405056

theorem complex_cube_root (a b : ℕ+) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (↑a + ↑b * Complex.I) ^ 3 = (2 : ℂ) + 11 * Complex.I →
  ↑a + ↑b * Complex.I = (2 : ℂ) + Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_root_l4050_405056


namespace NUMINAMATH_CALUDE_correct_matching_probability_l4050_405084

/-- The number of celebrities, recent photos, and baby photos -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities to their recent photos -/
def prob_recent : ℚ := 1 / (n.factorial : ℚ)

/-- The probability of correctly matching all recent photos to baby photos -/
def prob_baby : ℚ := 1 / (n.factorial : ℚ)

/-- The overall probability of correctly matching all celebrities to their baby photos through recent photos -/
def prob_total : ℚ := prob_recent * prob_baby

theorem correct_matching_probability :
  prob_total = 1 / 576 := by sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l4050_405084


namespace NUMINAMATH_CALUDE_teachers_pizza_fraction_l4050_405013

theorem teachers_pizza_fraction (teachers : ℕ) (staff : ℕ) (staff_pizza_fraction : ℚ) (non_pizza_eaters : ℕ) :
  teachers = 30 →
  staff = 45 →
  staff_pizza_fraction = 4/5 →
  non_pizza_eaters = 19 →
  (teachers : ℚ) * (2/3) + (staff : ℚ) * staff_pizza_fraction = (teachers + staff : ℚ) - non_pizza_eaters := by
  sorry

end NUMINAMATH_CALUDE_teachers_pizza_fraction_l4050_405013


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l4050_405070

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l4050_405070


namespace NUMINAMATH_CALUDE_max_correct_answers_l4050_405087

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (blank_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 50 →
  correct_points = 4 →
  blank_points = 0 →
  incorrect_points = -1 →
  total_score = 99 →
  ∃ (max_correct : ℕ), 
    max_correct ≤ total_questions ∧
    (∀ (correct blank incorrect : ℕ),
      correct + blank + incorrect = total_questions →
      correct_points * correct + blank_points * blank + incorrect_points * incorrect = total_score →
      correct ≤ max_correct) ∧
    max_correct = 29 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l4050_405087


namespace NUMINAMATH_CALUDE_person_age_puzzle_l4050_405028

theorem person_age_puzzle (x : ℝ) : 4 * (x + 3) - 4 * (x - 3) = x ↔ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l4050_405028


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4050_405077

theorem complex_fraction_simplification :
  1007 * ((7/4 / (3/4) + 3 / (9/4) + 1/3) / ((1+2+3+4+5) * 5 - 22)) / 19 = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4050_405077


namespace NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_95_minus_107_l4050_405006

-- Define the number
def n : ℕ := 95

-- Define the function to calculate the number
def f (n : ℕ) : ℤ := 10^n - 107

-- Define the function to calculate the sum of digits
def sum_of_digits (z : ℤ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_10_pow_95_minus_107 :
  sum_of_digits (f n) = 849 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_10_pow_95_minus_107_l4050_405006


namespace NUMINAMATH_CALUDE_tonys_monthly_rent_l4050_405059

/-- Calculates the monthly rent for a cottage given its room sizes and cost per square foot. -/
def calculate_monthly_rent (master_area : ℕ) (guest_bedroom_area : ℕ) (num_guest_bedrooms : ℕ) (other_areas : ℕ) (cost_per_sqft : ℕ) : ℕ :=
  let total_area := master_area + (guest_bedroom_area * num_guest_bedrooms) + other_areas
  total_area * cost_per_sqft

/-- Theorem stating that Tony's monthly rent is $3000 given the specified conditions. -/
theorem tonys_monthly_rent : 
  calculate_monthly_rent 500 200 2 600 2 = 3000 := by
  sorry

#eval calculate_monthly_rent 500 200 2 600 2

end NUMINAMATH_CALUDE_tonys_monthly_rent_l4050_405059


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l4050_405048

theorem cos_2alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.cos (2*α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l4050_405048


namespace NUMINAMATH_CALUDE_square_perimeter_l4050_405072

theorem square_perimeter (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h1 : rectangle_length = 4 * rectangle_width) 
  (h2 : 28 * rectangle_width = 56) : 
  4 * (rectangle_width + rectangle_length) = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l4050_405072


namespace NUMINAMATH_CALUDE_h_range_l4050_405065

/-- A quadratic function passing through three points with specific y-value relationships -/
structure QuadraticFunction where
  h : ℝ
  k : ℝ
  y₁ : ℝ
  y₂ : ℝ
  y₃ : ℝ
  eq_p₁ : y₁ = ((-3) - h)^2 + k
  eq_p₂ : y₂ = ((-1) - h)^2 + k
  eq_p₃ : y₃ = (1 - h)^2 + k
  y_order : y₂ < y₁ ∧ y₁ < y₃

/-- The range of h for the quadratic function -/
theorem h_range (f : QuadraticFunction) : -2 < f.h ∧ f.h < -1 := by
  sorry

end NUMINAMATH_CALUDE_h_range_l4050_405065


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_fourth_l4050_405043

theorem tan_thirteen_pi_fourth : Real.tan (13 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_fourth_l4050_405043


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l4050_405055

theorem perfect_square_trinomial_m_values (m : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-3)*x + 16 = (a*x + b)^2) → 
  (m = 7 ∨ m = -1) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_values_l4050_405055


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4050_405075

theorem complex_equation_solution : ∃ (a b : ℝ) (z : ℂ), 
  a > 0 ∧ b > 0 ∧ 
  z = a + b * I ∧
  z * (z + I) * (z + 3 * I) * (z - 2) = 180 * I ∧
  a = Real.sqrt 180 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4050_405075


namespace NUMINAMATH_CALUDE_min_product_of_three_min_product_exists_l4050_405004

def S : Finset Int := {-10, -5, -3, 0, 2, 4, 6, 8}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≥ -480 := by sorry

theorem min_product_exists :
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = -480 := by sorry

end NUMINAMATH_CALUDE_min_product_of_three_min_product_exists_l4050_405004


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l4050_405069

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 6 * x + 18 = 0) ∧ 
  (∀ x y : ℝ, 3 * x^2 - m * x - 6 * x + 18 = 0 ∧ 3 * y^2 - m * y - 6 * y + 18 = 0 → x = y) ∧
  ((m + 6) / 3 < -2) →
  m = -6 - 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l4050_405069


namespace NUMINAMATH_CALUDE_length_PF_is_16_over_3_l4050_405080

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*(x+2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 0)

-- Define the line through the focus
def line_through_focus (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the intersection points A and B (we don't calculate them explicitly)
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line_through_focus A.1 A.2 ∧ line_through_focus B.1 B.2

-- Define point P on x-axis
def point_P (P : ℝ × ℝ) : Prop := P.2 = 0

-- Main theorem
theorem length_PF_is_16_over_3 
  (A B P : ℝ × ℝ) 
  (h_intersect : intersection_points A B)
  (h_P : point_P P)
  (h_perpendicular : sorry) -- Additional hypothesis for P being on the perpendicular bisector
  : ‖P - focus‖ = 16/3 :=
sorry

end NUMINAMATH_CALUDE_length_PF_is_16_over_3_l4050_405080


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l4050_405009

theorem sphere_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * r^3) / (1 / 2 * 4 / 3 * Real.pi * (3 * r)^3) = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l4050_405009


namespace NUMINAMATH_CALUDE_jacket_transaction_profit_jacket_transaction_profit_is_10_l4050_405089

theorem jacket_transaction_profit (selling_price : ℝ) 
  (profit_percent : ℝ) (loss_percent : ℝ) : ℝ :=
  let cost_profit_jacket := selling_price / (1 + profit_percent)
  let cost_loss_jacket := selling_price / (1 - loss_percent)
  let total_cost := cost_profit_jacket + cost_loss_jacket
  let total_revenue := 2 * selling_price
  total_revenue - total_cost

theorem jacket_transaction_profit_is_10 :
  jacket_transaction_profit 80 0.6 0.2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_jacket_transaction_profit_jacket_transaction_profit_is_10_l4050_405089


namespace NUMINAMATH_CALUDE_canvas_bag_break_even_trips_eq_300_l4050_405050

/-- The number of shopping trips required for a canvas bag to become the lower-carbon solution compared to plastic bags. -/
def canvas_bag_break_even_trips (canvas_bag_co2_pounds : ℕ) (plastic_bag_co2_ounces : ℕ) (bags_per_trip : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  (canvas_bag_co2_pounds * ounces_per_pound) / (plastic_bag_co2_ounces * bags_per_trip)

/-- Theorem stating that 300 shopping trips are required for the canvas bag to become the lower-carbon solution. -/
theorem canvas_bag_break_even_trips_eq_300 :
  canvas_bag_break_even_trips 600 4 8 16 = 300 := by
  sorry

end NUMINAMATH_CALUDE_canvas_bag_break_even_trips_eq_300_l4050_405050


namespace NUMINAMATH_CALUDE_young_inequality_l4050_405068

theorem young_inequality (a b p q : ℝ) 
  (ha : a > 0) (hb : b > 0) (hp : p > 1) (hq : q > 1) (hpq : 1/p + 1/q = 1) :
  a^(1/p) * b^(1/q) ≤ a/p + b/q := by
  sorry

end NUMINAMATH_CALUDE_young_inequality_l4050_405068


namespace NUMINAMATH_CALUDE_mixture_composition_l4050_405008

theorem mixture_composition 
  (p_carbonated : ℝ) 
  (q_carbonated : ℝ) 
  (mixture_carbonated : ℝ) 
  (h1 : p_carbonated = 0.80) 
  (h2 : q_carbonated = 0.55) 
  (h3 : mixture_carbonated = 0.72) :
  let p := (mixture_carbonated - q_carbonated) / (p_carbonated - q_carbonated)
  p = 0.68 := by sorry

end NUMINAMATH_CALUDE_mixture_composition_l4050_405008
