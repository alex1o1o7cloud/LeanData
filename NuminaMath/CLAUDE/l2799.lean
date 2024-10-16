import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l2799_279943

theorem problem_solution :
  -- Part 1(i)
  (∀ a b : ℝ, a + b = 13 ∧ a * b = 36 → (a - b)^2 = 25) ∧
  -- Part 1(ii)
  (∀ a b : ℝ, a^2 + a*b = 8 ∧ b^2 + a*b = 1 → 
    (a = 8/3 ∧ b = 1/3) ∨ (a = -8/3 ∧ b = -1/3)) ∧
  -- Part 2
  (∀ a b x y : ℝ, 
    a*x + b*y = 3 ∧ 
    a*x^2 + b*y^2 = 7 ∧ 
    a*x^3 + b*y^3 = 16 ∧ 
    a*x^4 + b*y^4 = 42 → 
    x + y = -14) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2799_279943


namespace NUMINAMATH_CALUDE_dagger_example_l2799_279967

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (n / q)

-- Theorem statement
theorem dagger_example : dagger (5/9) (12/4) = 135 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l2799_279967


namespace NUMINAMATH_CALUDE_ribbon_used_parts_l2799_279919

-- Define the total length of the ribbon
def total_length : ℕ := 30

-- Define the number of parts the ribbon is cut into
def num_parts : ℕ := 6

-- Define the length of unused ribbon
def unused_length : ℕ := 10

-- Theorem to prove
theorem ribbon_used_parts : 
  ∃ (part_length : ℕ) (unused_parts : ℕ),
    part_length * num_parts = total_length ∧
    unused_parts * part_length = unused_length ∧
    num_parts - unused_parts = 4 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_used_parts_l2799_279919


namespace NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2799_279980

/-- Given a segment with midpoint (10, -5) and one endpoint (15, 10),
    the sum of the coordinates of the other endpoint is -15. -/
theorem endpoint_coordinate_sum :
  let midpoint : ℝ × ℝ := (10, -5)
  let endpoint1 : ℝ × ℝ := (15, 10)
  let endpoint2 : ℝ × ℝ := (2 * midpoint.1 - endpoint1.1, 2 * midpoint.2 - endpoint1.2)
  endpoint2.1 + endpoint2.2 = -15 := by
sorry

end NUMINAMATH_CALUDE_endpoint_coordinate_sum_l2799_279980


namespace NUMINAMATH_CALUDE_inequality_proof_l2799_279956

theorem inequality_proof (a : ℝ) (h : a > 0) : 
  Real.sqrt (a + 1/a) - Real.sqrt 2 ≥ Real.sqrt a + 1/(Real.sqrt a) - 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2799_279956


namespace NUMINAMATH_CALUDE_no_cube_sum_equals_cube_l2799_279985

theorem no_cube_sum_equals_cube : ∀ m n : ℕ+, m^3 + 11^3 ≠ n^3 := by
  sorry

end NUMINAMATH_CALUDE_no_cube_sum_equals_cube_l2799_279985


namespace NUMINAMATH_CALUDE_max_integer_for_inequality_l2799_279936

theorem max_integer_for_inequality : 
  (∀ a : ℕ+, a ≤ 12 → Real.sqrt 3 + Real.sqrt 8 > 1 + Real.sqrt (a : ℝ)) ∧
  (Real.sqrt 3 + Real.sqrt 8 ≤ 1 + Real.sqrt 13) :=
by sorry

end NUMINAMATH_CALUDE_max_integer_for_inequality_l2799_279936


namespace NUMINAMATH_CALUDE_extra_apples_l2799_279930

theorem extra_apples (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : students = 21)
  (h4 : ∀ s, s ≤ students → s = 1) : 
  red_apples + green_apples - students = 35 := by
  sorry

end NUMINAMATH_CALUDE_extra_apples_l2799_279930


namespace NUMINAMATH_CALUDE_function_sum_property_l2799_279924

/-- Given a function f(x) = ax^7 - bx^5 + cx^3 + 2, where f(-5) = 3, prove that f(5) + f(-5) = 4 -/
theorem function_sum_property (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^7 - b * x^5 + c * x^3 + 2
  (f (-5) = 3) → (f 5 + f (-5) = 4) := by
  sorry

end NUMINAMATH_CALUDE_function_sum_property_l2799_279924


namespace NUMINAMATH_CALUDE_exists_nonzero_digits_multiple_of_power_of_two_l2799_279942

/-- Returns true if all digits of n in decimal representation are non-zero -/
def allDigitsNonZero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0

/-- For every positive integer power of 2, there exists a multiple of it 
    such that all the digits (in decimal) are non-zero -/
theorem exists_nonzero_digits_multiple_of_power_of_two :
  ∀ k : ℕ+, ∃ n : ℕ, (2^k.val ∣ n) ∧ allDigitsNonZero n :=
sorry

end NUMINAMATH_CALUDE_exists_nonzero_digits_multiple_of_power_of_two_l2799_279942


namespace NUMINAMATH_CALUDE_equivalence_of_congruences_l2799_279921

theorem equivalence_of_congruences (p x y z : ℕ) (hp : Prime p) (hx : 0 < x) (hy : x < y) (hz : y < z) (hzp : z < p) :
  (x^3 ≡ y^3 [MOD p] ∧ x^3 ≡ z^3 [MOD p]) ↔ (y^2 ≡ z*x [MOD p] ∧ z^2 ≡ x*y [MOD p]) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_of_congruences_l2799_279921


namespace NUMINAMATH_CALUDE_classroom_lights_l2799_279907

theorem classroom_lights (num_lamps : ℕ) (h : num_lamps = 4) : 
  (2^num_lamps : ℕ) - 1 = 15 := by
  sorry

#check classroom_lights

end NUMINAMATH_CALUDE_classroom_lights_l2799_279907


namespace NUMINAMATH_CALUDE_sin_75_degrees_l2799_279966

theorem sin_75_degrees : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_degrees_l2799_279966


namespace NUMINAMATH_CALUDE_polynomial_roots_l2799_279935

theorem polynomial_roots : 
  let p : ℝ → ℝ := fun x ↦ x^3 - 4*x^2 - 7*x + 10
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 5 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2799_279935


namespace NUMINAMATH_CALUDE_least_with_12_factors_l2799_279972

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 12 positive factors -/
def has_12_factors (n : ℕ+) : Prop := num_factors n = 12

/-- Theorem: 72 is the least positive integer with exactly 12 positive factors -/
theorem least_with_12_factors :
  (∀ m : ℕ+, m < 72 → ¬(has_12_factors m)) ∧ has_12_factors 72 := by sorry

end NUMINAMATH_CALUDE_least_with_12_factors_l2799_279972


namespace NUMINAMATH_CALUDE_rabbit_speed_l2799_279941

def rabbit_speed_equation (x : ℝ) : Prop :=
  2 * (2 * x + 4) = 188

theorem rabbit_speed : ∃ x : ℝ, rabbit_speed_equation x ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l2799_279941


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2799_279958

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2799_279958


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2799_279900

def A : Set ℝ := {x | x^2 - x + 1 ≥ 0}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x ≤ 1 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2799_279900


namespace NUMINAMATH_CALUDE_inequality_proof_l2799_279933

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2799_279933


namespace NUMINAMATH_CALUDE_t_formula_l2799_279915

theorem t_formula (S₁ S₂ t u : ℝ) (hu : u ≠ 0) (heq : u = (S₁ - S₂) / (t - 1)) :
  t = (S₁ - S₂ + u) / u :=
sorry

end NUMINAMATH_CALUDE_t_formula_l2799_279915


namespace NUMINAMATH_CALUDE_binomial_expectation_l2799_279999

/-- The number of trials -/
def n : ℕ := 3

/-- The probability of drawing a red ball -/
def p : ℚ := 3/5

/-- The expected value of a binomial distribution -/
def expected_value (n : ℕ) (p : ℚ) : ℚ := n * p

theorem binomial_expectation :
  expected_value n p = 9/5 := by sorry

end NUMINAMATH_CALUDE_binomial_expectation_l2799_279999


namespace NUMINAMATH_CALUDE_three_good_sets_l2799_279993

-- Define the "good set" property
def is_good_set (C : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ C, ∃ p₂ ∈ C, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the sets
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}
def C₂ : Set (ℝ × ℝ) := {p | p.1^2 - p.2^2 = 9}
def C₃ : Set (ℝ × ℝ) := {p | 2*p.1^2 + p.2^2 = 9}
def C₄ : Set (ℝ × ℝ) := {p | p.1^2 + p.2 = 9}

-- Theorem statement
theorem three_good_sets : 
  (is_good_set C₁ ∧ is_good_set C₃ ∧ is_good_set C₄ ∧ ¬is_good_set C₂) := by
  sorry

end NUMINAMATH_CALUDE_three_good_sets_l2799_279993


namespace NUMINAMATH_CALUDE_log_product_one_l2799_279925

theorem log_product_one : Real.log 5 / Real.log 2 * Real.log 2 / Real.log 3 * Real.log 3 / Real.log 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_one_l2799_279925


namespace NUMINAMATH_CALUDE_equation_solution_l2799_279984

theorem equation_solution : 
  ∃ x : ℝ, (x / (x - 1) = (x - 3) / (2*x - 2)) ∧ (x = -3) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2799_279984


namespace NUMINAMATH_CALUDE_f_properties_l2799_279976

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x - Real.cos x

theorem f_properties :
  ∃ (period amplitude : ℝ) (range : Set ℝ),
    (period = 2 * Real.pi) ∧
    (amplitude = 2) ∧
    (range = Set.Icc (-1) 2) ∧
    (∀ x ∈ Set.Icc 0 Real.pi, f x ∈ range) ∧
    (∀ y ∈ range, ∃ x ∈ Set.Icc 0 Real.pi, f x = y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2799_279976


namespace NUMINAMATH_CALUDE_simplify_fraction_l2799_279928

theorem simplify_fraction : (5^4 + 5^2) / (5^3 - 5) = 65 / 12 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2799_279928


namespace NUMINAMATH_CALUDE_frank_one_dollar_bills_l2799_279971

/-- Represents the number of bills Frank has of each denomination --/
structure Bills :=
  (ones : ℕ)
  (fives : ℕ)
  (tens : ℕ)
  (twenties : ℕ)

/-- Calculates the total value of bills --/
def totalValue (b : Bills) : ℕ :=
  b.ones + 5 * b.fives + 10 * b.tens + 20 * b.twenties

theorem frank_one_dollar_bills :
  ∃ (b : Bills),
    b.fives = 4 ∧
    b.tens = 2 ∧
    b.twenties = 1 ∧
    (∃ (peanutsPounds : ℕ),
      3 * peanutsPounds + 4 = 10 ∧  -- Cost of peanuts plus $4 change equals $10
      totalValue b + 4 = 54) →      -- Total money including change is $54
    b.ones = 4 := by
  sorry

end NUMINAMATH_CALUDE_frank_one_dollar_bills_l2799_279971


namespace NUMINAMATH_CALUDE_evaluate_expression_l2799_279988

theorem evaluate_expression (b : ℝ) : 
  let x : ℝ := b + 9
  (x - b + 5) = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2799_279988


namespace NUMINAMATH_CALUDE_trapezium_height_l2799_279914

theorem trapezium_height (a b area : ℝ) (ha : a = 30) (hb : b = 12) (harea : area = 336) :
  (area * 2) / (a + b) = 16 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_height_l2799_279914


namespace NUMINAMATH_CALUDE_expression_simplification_l2799_279951

theorem expression_simplification (x y : ℝ) : (x - 2*y) * (x + 2*y) - x * (x - y) = -4*y^2 + x*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2799_279951


namespace NUMINAMATH_CALUDE_minimize_y_l2799_279903

/-- The function to be minimized -/
def y (x a b : ℝ) : ℝ := 3 * (x - a)^2 + (x - b)^2

/-- The derivative of y with respect to x -/
def y_derivative (x a b : ℝ) : ℝ := 8 * x - 6 * a - 2 * b

/-- The second derivative of y with respect to x -/
def y_second_derivative : ℝ := 8

theorem minimize_y (a b : ℝ) :
  ∃ (x : ℝ), (∀ (z : ℝ), y z a b ≥ y x a b) ∧ x = (3 * a + b) / 4 := by
  sorry

#check minimize_y

end NUMINAMATH_CALUDE_minimize_y_l2799_279903


namespace NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l2799_279918

theorem common_number_in_overlapping_sets (numbers : List ℝ) : 
  numbers.length = 9 →
  (numbers.take 5).sum / 5 = 7 →
  (numbers.drop 4).sum / 5 = 10 →
  numbers.sum / 9 = 74 / 9 →
  ∃ x ∈ numbers.take 5 ∩ numbers.drop 4, x = 11 :=
by sorry

end NUMINAMATH_CALUDE_common_number_in_overlapping_sets_l2799_279918


namespace NUMINAMATH_CALUDE_max_pieces_is_seven_l2799_279962

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  List.length digits = List.length (List.dedup digits)

theorem max_pieces_is_seven :
  (∃ (max : ℕ), 
    (∀ (n : ℕ), ∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * n → n ≤ max) ∧
    (∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * max)) ∧
  (∀ (m : ℕ), 
    (∀ (n : ℕ), ∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * n → n ≤ m) ∧
    (∃ (P Q : ℕ), is_five_digit P ∧ is_five_digit Q ∧ has_distinct_digits P ∧ P = Q * m) → 
    m ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_is_seven_l2799_279962


namespace NUMINAMATH_CALUDE_expression_value_when_x_is_two_l2799_279960

theorem expression_value_when_x_is_two :
  let x : ℝ := 2
  (x + 2 - x) * (2 - x - 2) = -4 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_when_x_is_two_l2799_279960


namespace NUMINAMATH_CALUDE_animal_shelter_dogs_l2799_279922

theorem animal_shelter_dogs (initial_dogs : ℕ) (adoption_rate : ℚ) (returned_dogs : ℕ) : 
  initial_dogs = 80 → 
  adoption_rate = 2/5 →
  returned_dogs = 5 →
  initial_dogs - (initial_dogs * adoption_rate).floor + returned_dogs = 53 := by
sorry

end NUMINAMATH_CALUDE_animal_shelter_dogs_l2799_279922


namespace NUMINAMATH_CALUDE_sam_need_change_probability_l2799_279986

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 9

/-- The price of Sam's favorite toy in half-dollars -/
def favorite_toy_price : ℕ := 5

/-- The number of half-dollar coins Sam has -/
def sam_coins : ℕ := 10

/-- The probability of Sam needing to break the twenty-dollar bill -/
def probability_need_change : ℚ := 55 / 63

/-- Theorem stating the probability of Sam needing to break the twenty-dollar bill -/
theorem sam_need_change_probability :
  let total_arrangements := (num_toys.factorial : ℚ)
  let favorable_outcomes := ((num_toys - 1).factorial : ℚ) + ((num_toys - 2).factorial : ℚ) + ((num_toys - 3).factorial : ℚ)
  (1 - favorable_outcomes / total_arrangements) = probability_need_change := by
  sorry


end NUMINAMATH_CALUDE_sam_need_change_probability_l2799_279986


namespace NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l2799_279945

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_13th_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_5th : a 5 = 1)
  (h_sum : a 8 + a 10 = 16) :
  a 13 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l2799_279945


namespace NUMINAMATH_CALUDE_smaller_field_area_l2799_279992

/-- Given a field of 500 hectares divided into two parts, where the difference
    of the areas is one-fifth of their average, the area of the smaller part
    is 225 hectares. -/
theorem smaller_field_area (x y : ℝ) (h1 : x + y = 500)
    (h2 : y - x = (1 / 5) * ((x + y) / 2)) : x = 225 := by
  sorry

end NUMINAMATH_CALUDE_smaller_field_area_l2799_279992


namespace NUMINAMATH_CALUDE_square_difference_l2799_279931

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2799_279931


namespace NUMINAMATH_CALUDE_sum_and_simplification_l2799_279955

theorem sum_and_simplification : 
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (7 : ℚ) / 8 + (11 : ℚ) / 12 = (n : ℚ) / d ∧ 
  (∀ (k : ℕ), k > 1 → ¬(k ∣ n ∧ k ∣ d)) :=
by sorry

end NUMINAMATH_CALUDE_sum_and_simplification_l2799_279955


namespace NUMINAMATH_CALUDE_thomson_savings_l2799_279926

/-- Calculates the amount saved by Mrs. Thomson given her spending pattern -/
def amount_saved (X : ℝ) : ℝ :=
  let after_food := X * (1 - 0.375)
  let after_clothes := after_food * (1 - 0.22)
  let after_household := after_clothes * (1 - 0.15)
  let after_stocks := after_household * (1 - 0.30)
  let after_tuition := after_stocks * (1 - 0.40)
  after_tuition

theorem thomson_savings (X : ℝ) : amount_saved X = 0.1740375 * X := by
  sorry

end NUMINAMATH_CALUDE_thomson_savings_l2799_279926


namespace NUMINAMATH_CALUDE_largest_non_representable_l2799_279902

def is_representable (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = 15 * x + 18 * y + 20 * z

theorem largest_non_representable : 
  (∀ m > 97, is_representable m) ∧ ¬(is_representable 97) :=
sorry

end NUMINAMATH_CALUDE_largest_non_representable_l2799_279902


namespace NUMINAMATH_CALUDE_kyles_money_after_snowboarding_l2799_279970

/-- Calculates Kyle's remaining money after snowboarding -/
def kyles_remaining_money (daves_money : ℕ) : ℕ :=
  let kyles_initial_money := 3 * daves_money - 12
  let snowboarding_cost := kyles_initial_money / 3
  kyles_initial_money - snowboarding_cost

/-- Proves that Kyle has $84 left after snowboarding -/
theorem kyles_money_after_snowboarding :
  kyles_remaining_money 46 = 84 := by
  sorry

#eval kyles_remaining_money 46

end NUMINAMATH_CALUDE_kyles_money_after_snowboarding_l2799_279970


namespace NUMINAMATH_CALUDE_smallest_integer_c_l2799_279947

theorem smallest_integer_c (x : ℕ) (h : x = 8 * 3) : 
  (∃ c : ℕ, 27 ^ c > 3 ^ x ∧ ∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) → 
  (∃ c : ℕ, 27 ^ c > 3 ^ x ∧ ∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) ∧ 
  (∀ c : ℕ, 27 ^ c > 3 ^ x ∧ (∀ k : ℕ, 27 ^ k > 3 ^ x → c ≤ k) → c = 9) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_c_l2799_279947


namespace NUMINAMATH_CALUDE_inequality_solution_l2799_279917

-- Define the solution set for x^2 - ax - b < 0
def solution_set (a b : ℝ) : Set ℝ := {x | 2 < x ∧ x < 3}

-- Theorem statement
theorem inequality_solution :
  ∀ a b : ℝ, (solution_set a b = {x | 2 < x ∧ x < 3}) →
  (a = 5 ∧ b = -6) ∧
  ({x : ℝ | b * x^2 - a * x - 1 > 0} = {x : ℝ | -1/2 < x ∧ x < -1/3}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2799_279917


namespace NUMINAMATH_CALUDE_fixed_point_values_l2799_279950

/-- A function has exactly one fixed point if and only if
    the equation f(x) = x has exactly one solution. -/
def has_exactly_one_fixed_point (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = x

/-- The quadratic function f(x) = ax² + (2a-3)x + 1 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + (2*a - 3) * x + 1

/-- The set of values for a such that f has exactly one fixed point -/
def A : Set ℝ := {a | has_exactly_one_fixed_point (f a)}

theorem fixed_point_values :
  A = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_fixed_point_values_l2799_279950


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l2799_279995

theorem smallest_undefined_inverse (a : ℕ) : 
  (∀ b < a, (Nat.gcd b 70 = 1 ∨ Nat.gcd b 84 = 1)) → 
  (Nat.gcd a 70 > 1 ∧ Nat.gcd a 84 > 1) → 
  a = 14 := by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l2799_279995


namespace NUMINAMATH_CALUDE_special_quadrilateral_is_square_l2799_279920

/-- A quadrilateral with perpendicular diagonals of equal length -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perpendicular_diagonals : Bool
  /-- The quadrilateral has diagonals of equal length -/
  equal_length_diagonals : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Prop :=
  q.perpendicular_diagonals ∧ q.equal_length_diagonals

/-- Theorem: A quadrilateral with perpendicular diagonals of equal length is always a square -/
theorem special_quadrilateral_is_square (q : SpecialQuadrilateral) 
  (h1 : q.perpendicular_diagonals = true) 
  (h2 : q.equal_length_diagonals = true) : 
  is_square q := by
  sorry

#check special_quadrilateral_is_square

end NUMINAMATH_CALUDE_special_quadrilateral_is_square_l2799_279920


namespace NUMINAMATH_CALUDE_apartment_units_per_floor_l2799_279934

theorem apartment_units_per_floor (total_units : ℕ) (first_floor_units : ℕ) (num_buildings : ℕ) (num_floors : ℕ) :
  total_units = 34 →
  first_floor_units = 2 →
  num_buildings = 2 →
  num_floors = 4 →
  ∃ (other_floor_units : ℕ),
    total_units = num_buildings * (first_floor_units + (num_floors - 1) * other_floor_units) ∧
    other_floor_units = 5 :=
by sorry

end NUMINAMATH_CALUDE_apartment_units_per_floor_l2799_279934


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2799_279965

-- Define the quadratic function
def f (x : ℝ) := -2 * x^2 + x + 1

-- Define the solution set
def solution_set := {x : ℝ | -1/2 < x ∧ x < 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2799_279965


namespace NUMINAMATH_CALUDE_abs_inequality_l2799_279905

theorem abs_inequality (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l2799_279905


namespace NUMINAMATH_CALUDE_sin_two_a_value_l2799_279974

theorem sin_two_a_value (a : ℝ) (h : Real.sin a - Real.cos a = 4/3) : 
  Real.sin (2 * a) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_a_value_l2799_279974


namespace NUMINAMATH_CALUDE_lynne_spent_75_l2799_279901

/-- The total amount Lynne spent on books and magazines -/
def total_spent (cat_books solar_books magazines book_price magazine_price : ℕ) : ℕ :=
  (cat_books + solar_books) * book_price + magazines * magazine_price

/-- Theorem stating that Lynne spent $75 in total -/
theorem lynne_spent_75 : 
  total_spent 7 2 3 7 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_lynne_spent_75_l2799_279901


namespace NUMINAMATH_CALUDE_problem_solution_l2799_279910

theorem problem_solution (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) : 
  (∃ k : ℕ, m + 1 = 4 * k - 1 ∧ Nat.Prime (m + 1)) →
  (∃ p a : ℕ, Nat.Prime p ∧ (m^(2^n - 1) - 1) / (m - 1) = m^n + p^a) →
  (∃ p : ℕ, Nat.Prime p ∧ p = 4 * (p / 4) - 1 ∧ m = p - 1 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2799_279910


namespace NUMINAMATH_CALUDE_absolute_difference_41st_terms_l2799_279923

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem absolute_difference_41st_terms :
  let A := arithmetic_sequence 50 6
  let B := arithmetic_sequence 100 (-15)
  abs (A 41 - B 41) = 790 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_41st_terms_l2799_279923


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l2799_279981

/-- A linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

/-- Point P₁ on the graph of f -/
def P₁ : ℝ × ℝ := (-3, f (-3))

/-- Point P₂ on the graph of f -/
def P₂ : ℝ × ℝ := (2, f 2)

/-- y₁ is the y-coordinate of P₁ -/
def y₁ : ℝ := (P₁.2)

/-- y₂ is the y-coordinate of P₂ -/
def y₂ : ℝ := (P₂.2)

theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l2799_279981


namespace NUMINAMATH_CALUDE_coin_toss_frequency_l2799_279904

/-- Given a coin tossed 10 times with 6 heads, prove that the frequency of heads is 3/5 -/
theorem coin_toss_frequency :
  ∀ (total_tosses : ℕ) (heads : ℕ),
  total_tosses = 10 →
  heads = 6 →
  (heads : ℚ) / total_tosses = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_coin_toss_frequency_l2799_279904


namespace NUMINAMATH_CALUDE_pen_buyers_difference_l2799_279912

theorem pen_buyers_difference (pen_cost : ℕ+) 
  (h1 : 178 % pen_cost.val = 0)
  (h2 : 252 % pen_cost.val = 0)
  (h3 : 35 * pen_cost.val ≤ 252) :
  252 / pen_cost.val - 178 / pen_cost.val = 5 := by
  sorry

end NUMINAMATH_CALUDE_pen_buyers_difference_l2799_279912


namespace NUMINAMATH_CALUDE_pi_shape_points_for_10cm_square_l2799_279957

/-- The number of unique points on a "П" shape formed from a square --/
def unique_points_on_pi_shape (square_side : ℕ) (point_spacing : ℕ) : ℕ :=
  let points_per_side := square_side / point_spacing + 1
  let total_points := points_per_side * 3
  let corner_points := 3
  total_points - (corner_points - 1)

/-- Theorem stating that for a square with side 10 cm and points placed every 1 cm,
    the number of unique points on the "П" shape is 31 --/
theorem pi_shape_points_for_10cm_square :
  unique_points_on_pi_shape 10 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_pi_shape_points_for_10cm_square_l2799_279957


namespace NUMINAMATH_CALUDE_common_difference_is_negative_two_l2799_279977

def arithmetic_sequence (n : ℕ) : ℤ := 3 - 2 * n

theorem common_difference_is_negative_two :
  ∀ n : ℕ, arithmetic_sequence (n + 1) - arithmetic_sequence n = -2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_negative_two_l2799_279977


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2799_279953

theorem consecutive_integers_sum (x : ℤ) : x * (x + 1) = 440 → x + (x + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2799_279953


namespace NUMINAMATH_CALUDE_systematic_sampling_relation_third_group_sample_l2799_279961

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  group_size : ℕ
  last_group_sample : ℕ

/-- Theorem stating the relationship between samples from different groups -/
theorem systematic_sampling_relation (s : SystematicSampling)
  (h1 : s.total_students = 180)
  (h2 : s.num_groups = 20)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.last_group_sample = 176) :
  s.last_group_sample = (s.num_groups - 1) * s.group_size + (s.group_size + 5) :=
by sorry

/-- Corollary: The sample from the 3rd group is 23 -/
theorem third_group_sample (s : SystematicSampling)
  (h1 : s.total_students = 180)
  (h2 : s.num_groups = 20)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.last_group_sample = 176) :
  s.group_size + 5 = 23 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_relation_third_group_sample_l2799_279961


namespace NUMINAMATH_CALUDE_money_division_l2799_279929

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total ∧ 
  3 * p = 7 * q ∧ 
  7 * q = 12 * r ∧ 
  r - q = 3500 → 
  q - p = 2800 := by sorry

end NUMINAMATH_CALUDE_money_division_l2799_279929


namespace NUMINAMATH_CALUDE_find_constant_k_l2799_279978

theorem find_constant_k : ∃ k : ℝ, ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4) → k = -16 := by
  sorry

end NUMINAMATH_CALUDE_find_constant_k_l2799_279978


namespace NUMINAMATH_CALUDE_expression_value_l2799_279940

theorem expression_value (x y : ℤ) (hx : x = -5) (hy : y = 8) :
  2 * (x - y)^2 - x * y = 378 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2799_279940


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2799_279927

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, (x < -1 → (x < -1 ∨ x > 1)) ∧ ¬((x < -1 ∨ x > 1) → x < -1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2799_279927


namespace NUMINAMATH_CALUDE_Q_on_circle_25_line_AB_equation_l2799_279991

-- Define the circle P
def circle_P (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Define that Q is outside circle P
def Q_outside_P (a b : ℝ) : Prop := a^2 + b^2 > 16

-- Define circle M with diameter PQ intersecting circle P at A and B
def circle_M_intersects_P (a b : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧ 
  ((x1 - a)^2 + (y1 - b)^2 = (x1^2 + y1^2) / 4) ∧
  ((x2 - a)^2 + (y2 - b)^2 = (x2^2 + y2^2) / 4)

-- Theorem 1: When QA = QB = 3, Q lies on x^2 + y^2 = 25
theorem Q_on_circle_25 (a b : ℝ) 
  (h1 : Q_outside_P a b) 
  (h2 : circle_M_intersects_P a b) 
  (h3 : ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧
        (x1 - a)^2 + (y1 - b)^2 = 9 ∧ (x2 - a)^2 + (y2 - b)^2 = 9) :
  a^2 + b^2 = 25 :=
sorry

-- Theorem 2: When Q(4, 6), the equation of line AB is 2x + 3y - 8 = 0
theorem line_AB_equation 
  (h1 : Q_outside_P 4 6) 
  (h2 : circle_M_intersects_P 4 6) :
  ∃ (x1 y1 x2 y2 : ℝ), circle_P x1 y1 ∧ circle_P x2 y2 ∧
  2 * x1 + 3 * y1 - 8 = 0 ∧ 2 * x2 + 3 * y2 - 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_Q_on_circle_25_line_AB_equation_l2799_279991


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l2799_279983

-- Define the curve C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 = 4 * p.2}

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -2}

-- Define a function to get tangent points on C from a point on l
def tangentPoints (E : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ C ∧ ∃ m : ℝ, (p.2 - E.2) = m * (p.1 - E.1) ∧ p.1 = 2 * m}

-- Theorem statement
theorem tangent_line_intersection (E : ℝ × ℝ) (hE : E ∈ l) :
  ∃ A B : ℝ × ℝ, A ∈ tangentPoints E ∧ B ∈ tangentPoints E ∧ A ≠ B ∧
  ∃ t : ℝ, (1 - t) • A + t • B = (0, 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l2799_279983


namespace NUMINAMATH_CALUDE_clothing_sales_theorem_l2799_279964

/-- Represents the sales data for a clothing store --/
structure SalesData where
  typeA_sold : ℕ
  typeB_sold : ℕ
  total_sales : ℕ

/-- Represents the pricing and sales increase data --/
structure ClothingData where
  typeA_price : ℕ
  typeB_price : ℕ
  typeA_increase : ℚ
  typeB_increase : ℚ

def store_A : SalesData := ⟨60, 15, 3600⟩
def store_B : SalesData := ⟨40, 60, 4400⟩

theorem clothing_sales_theorem (d : ClothingData) :
  d.typeA_price = 50 ∧ 
  d.typeB_price = 40 ∧ 
  d.typeA_increase = 1/5 ∧
  d.typeB_increase = 1/2 →
  (store_A.typeA_sold * d.typeA_price + store_A.typeB_sold * d.typeB_price = store_A.total_sales) ∧
  (store_B.typeA_sold * d.typeA_price + store_B.typeB_sold * d.typeB_price = store_B.total_sales) ∧
  ((store_A.typeA_sold + store_B.typeA_sold) * d.typeA_price * (1 + d.typeA_increase) : ℚ) / 
  ((store_A.typeB_sold + store_B.typeB_sold) * d.typeB_price * (1 + d.typeB_increase) : ℚ) = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_clothing_sales_theorem_l2799_279964


namespace NUMINAMATH_CALUDE_arithmetic_sum_specific_l2799_279938

/-- The sum of an arithmetic sequence with given parameters -/
def arithmetic_sum (a₁ aₙ : Int) (d : Int) : Int :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Theorem: The sum of the arithmetic sequence from -39 to -1 with common difference 2 is -400 -/
theorem arithmetic_sum_specific : arithmetic_sum (-39) (-1) 2 = -400 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_specific_l2799_279938


namespace NUMINAMATH_CALUDE_min_value_expression_l2799_279969

theorem min_value_expression (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) :
  (5 * r) / (3 * p + q) + (5 * p) / (q + 3 * r) + (4 * q) / (2 * p + 2 * r) ≥ 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2799_279969


namespace NUMINAMATH_CALUDE_frog_eggs_difference_l2799_279975

/-- Represents the number of eggs laid by a frog over 4 days -/
def FrogEggs : Type :=
  { eggs : Fin 4 → ℕ // 
    eggs 0 = 50 ∧ 
    eggs 1 = 2 * eggs 0 ∧ 
    eggs 3 = 2 * (eggs 0 + eggs 1 + eggs 2) ∧
    eggs 0 + eggs 1 + eggs 2 + eggs 3 = 810 }

/-- The difference between eggs laid on the third day and second day is 20 -/
theorem frog_eggs_difference (e : FrogEggs) : e.val 2 - e.val 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_frog_eggs_difference_l2799_279975


namespace NUMINAMATH_CALUDE_campsite_theorem_l2799_279997

/-- Represents the distances between tents and dining hall -/
structure CampSite where
  p : ℝ  -- Distance from Peter's tent to dining hall
  m : ℝ  -- Distance from Michael's tent to dining hall
  s : ℝ  -- Distance between Peter's and Michael's tents

/-- The campsite satisfies the given conditions -/
def satisfies_conditions (c : CampSite) : Prop :=
  c.s + c.m = c.p + 20 ∧ c.s + c.p = c.m + 16

theorem campsite_theorem (c : CampSite) 
  (h : satisfies_conditions c) : c.s = 18 ∧ c.m = c.p + 2 := by
  sorry

#check campsite_theorem

end NUMINAMATH_CALUDE_campsite_theorem_l2799_279997


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_l2799_279944

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The theorem states that if a * cos(A) = b * cos(B) in a triangle,
    then the triangle is either isosceles (A = B) or right-angled (A + B = π/2). -/
theorem triangle_isosceles_or_right (t : Triangle) 
  (h : t.a * Real.cos t.A = t.b * Real.cos t.B) :
  t.A = t.B ∨ t.A + t.B = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_l2799_279944


namespace NUMINAMATH_CALUDE_john_took_six_pink_l2799_279954

def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def initial_yellow : ℕ := 24
def carl_took : ℕ := 4
def total_remaining : ℕ := 43

def john_took_pink (p : ℕ) : Prop :=
  initial_pink - carl_took - p +
  initial_green - 2 * p +
  initial_yellow = total_remaining

theorem john_took_six_pink : john_took_pink 6 := by sorry

end NUMINAMATH_CALUDE_john_took_six_pink_l2799_279954


namespace NUMINAMATH_CALUDE_fifty_third_term_is_2_to_53_l2799_279989

def double_sequence : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * double_sequence n

theorem fifty_third_term_is_2_to_53 :
  double_sequence 52 = 2^53 := by
  sorry

end NUMINAMATH_CALUDE_fifty_third_term_is_2_to_53_l2799_279989


namespace NUMINAMATH_CALUDE_journey_distance_l2799_279990

/-- Represents the man's journey with different speeds and times for each segment -/
structure Journey where
  flat_walk_speed : ℝ
  downhill_run_speed : ℝ
  hilly_walk_speed : ℝ
  hilly_run_speed : ℝ
  flat_walk_time : ℝ
  downhill_run_time : ℝ
  hilly_walk_time : ℝ
  hilly_run_time : ℝ

/-- Calculates the total distance traveled during the journey -/
def total_distance (j : Journey) : ℝ :=
  j.flat_walk_speed * j.flat_walk_time +
  j.downhill_run_speed * j.downhill_run_time +
  j.hilly_walk_speed * j.hilly_walk_time +
  j.hilly_run_speed * j.hilly_run_time

/-- Theorem stating that the total distance traveled is 90 km -/
theorem journey_distance :
  let j : Journey := {
    flat_walk_speed := 8,
    downhill_run_speed := 24,
    hilly_walk_speed := 6,
    hilly_run_speed := 18,
    flat_walk_time := 3,
    downhill_run_time := 1.5,
    hilly_walk_time := 2,
    hilly_run_time := 1
  }
  total_distance j = 90 := by sorry

end NUMINAMATH_CALUDE_journey_distance_l2799_279990


namespace NUMINAMATH_CALUDE_height_from_bisected_hypotenuse_l2799_279906

/-- 
Given a right-angled triangle where the bisector of the right angle divides 
the hypotenuse into segments p and q, the height m corresponding to the 
hypotenuse is equal to (p + q) * p * q / (p^2 + q^2).
-/
theorem height_from_bisected_hypotenuse (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∃ m : ℝ, m = (p + q) * p * q / (p^2 + q^2) ∧ 
  m = Real.sqrt ((p + q)^2 * p^2 * q^2 / (p^2 + q^2)^2) :=
by sorry

end NUMINAMATH_CALUDE_height_from_bisected_hypotenuse_l2799_279906


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l2799_279909

theorem min_sum_absolute_values :
  ∃ (min : ℝ), min = 4 ∧ 
  (∀ x : ℝ, |x + 3| + |x + 5| + |x + 7| ≥ min) ∧
  (∃ x : ℝ, |x + 3| + |x + 5| + |x + 7| = min) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l2799_279909


namespace NUMINAMATH_CALUDE_bark_ratio_is_two_to_one_l2799_279939

/-- The number of times the terrier's owner says "hush" -/
def hush_count : ℕ := 6

/-- The number of times the poodle barks -/
def poodle_barks : ℕ := 24

/-- The number of times the terrier barks before being hushed -/
def terrier_barks_per_hush : ℕ := 2

/-- Calculates the total number of times the terrier barks -/
def total_terrier_barks : ℕ := hush_count * terrier_barks_per_hush

/-- The ratio of poodle barks to terrier barks -/
def bark_ratio : ℚ := poodle_barks / total_terrier_barks

theorem bark_ratio_is_two_to_one : bark_ratio = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_bark_ratio_is_two_to_one_l2799_279939


namespace NUMINAMATH_CALUDE_xiao_ming_banknote_combinations_l2799_279987

def is_valid_combination (x y z : ℕ) : Prop :=
  x + 2*y + 5*z = 18 ∧ x + y + z ≤ 10 ∧ (x > 0 ∧ y > 0) ∨ (x > 0 ∧ z > 0) ∨ (y > 0 ∧ z > 0)

def count_valid_combinations : ℕ := sorry

theorem xiao_ming_banknote_combinations : count_valid_combinations = 9 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_banknote_combinations_l2799_279987


namespace NUMINAMATH_CALUDE_min_sum_theorem_l2799_279916

-- Define the equation
def equation (x y : ℝ) : Prop := -x^2 + 7*x + y - 10 = 0

-- Define the sum function
def sum (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem min_sum_theorem :
  ∃ (min : ℝ), min = 1 ∧ 
  (∀ x y : ℝ, equation x y → sum x y ≥ min) ∧
  (∃ x y : ℝ, equation x y ∧ sum x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_sum_theorem_l2799_279916


namespace NUMINAMATH_CALUDE_circle_center_trajectory_l2799_279937

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*(m+3)*x + 2*(1-4*m^2) + 16*m^4 + 9 = 0

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  y = 4*(x-3)^2 - 1

-- Theorem statement
theorem circle_center_trajectory :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) →
  ∃ x y : ℝ, trajectory_equation x y ∧ 20/7 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_circle_center_trajectory_l2799_279937


namespace NUMINAMATH_CALUDE_words_per_page_l2799_279994

theorem words_per_page (total_pages : Nat) (words_mod : Nat) (mod_value : Nat) :
  total_pages = 150 →
  words_mod = 210 →
  mod_value = 221 →
  ∃ (words_per_page : Nat),
    words_per_page ≤ 120 ∧
    (total_pages * words_per_page) % mod_value = words_mod ∧
    words_per_page = 195 := by
  sorry

end NUMINAMATH_CALUDE_words_per_page_l2799_279994


namespace NUMINAMATH_CALUDE_solution_count_l2799_279963

/-- The greatest integer function (floor function) -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The number of solutions to x^2 - ⌊x⌋^2 = (x - ⌊x⌋)^2 in [1, n] -/
def num_solutions (n : ℕ) : ℕ :=
  n^2 - n + 1

/-- Theorem stating the number of solutions to the equation -/
theorem solution_count (n : ℕ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ n →
    x^2 - (floor x)^2 = (x - floor x)^2) →
  num_solutions n = n^2 - n + 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_count_l2799_279963


namespace NUMINAMATH_CALUDE_earnings_difference_theorem_l2799_279973

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  inv_ratio_a : ℕ
  inv_ratio_b : ℕ
  inv_ratio_c : ℕ
  ret_ratio_a : ℕ
  ret_ratio_b : ℕ
  ret_ratio_c : ℕ

/-- Calculates the earnings difference between investors b and a -/
def earnings_difference (data : InvestmentData) (total_earnings : ℕ) : ℕ :=
  let total_ratio := data.inv_ratio_a * data.ret_ratio_a + 
                     data.inv_ratio_b * data.ret_ratio_b + 
                     data.inv_ratio_c * data.ret_ratio_c
  let unit_earning := total_earnings / total_ratio
  (data.inv_ratio_b * data.ret_ratio_b - data.inv_ratio_a * data.ret_ratio_a) * unit_earning

/-- Theorem: Given the investment ratios 3:4:5, return ratios 6:5:4, and total earnings 10150,
    the earnings difference between b and a is 350 -/
theorem earnings_difference_theorem : 
  let data : InvestmentData := {
    inv_ratio_a := 3, inv_ratio_b := 4, inv_ratio_c := 5,
    ret_ratio_a := 6, ret_ratio_b := 5, ret_ratio_c := 4
  }
  earnings_difference data 10150 = 350 := by
  sorry


end NUMINAMATH_CALUDE_earnings_difference_theorem_l2799_279973


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2799_279949

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x | x^2 < 10}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2799_279949


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l2799_279959

theorem sequence_sum_problem (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ = 5)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ = 14)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ = 30)
  (eq4 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ = 70) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ = 130 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_problem_l2799_279959


namespace NUMINAMATH_CALUDE_fixed_point_of_shifted_function_l2799_279908

theorem fixed_point_of_shifted_function 
  (f : ℝ → ℝ) 
  (h : f 1 = 1) :
  ∃ x : ℝ, f (x + 2) = x ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_shifted_function_l2799_279908


namespace NUMINAMATH_CALUDE_impossibility_theorem_l2799_279932

/-- Represents a pile of chips -/
structure Pile :=
  (chips : ℕ)

/-- Represents the state of all piles -/
def State := List Pile

/-- The i-th prime number -/
def ithPrime (i : ℕ) : ℕ := sorry

/-- Initial state with 2018 piles, where the i-th pile has p_i chips (p_i is the i-th prime) -/
def initialState : State := 
  List.range 2018 |>.map (fun i => Pile.mk (ithPrime (i + 1)))

/-- Splits a pile into two piles and adds one chip to one of the new piles -/
def splitPile (s : State) (i : ℕ) (j k : ℕ) : State := sorry

/-- Merges two piles and adds one chip to the resulting pile -/
def mergePiles (s : State) (i j : ℕ) : State := sorry

/-- The target state with 2018 piles, each containing 2018 chips -/
def targetState : State := 
  List.replicate 2018 (Pile.mk 2018)

/-- Predicate to check if a given state is reachable from the initial state -/
def isReachable (s : State) : Prop := sorry

theorem impossibility_theorem : ¬ isReachable targetState := by
  sorry

end NUMINAMATH_CALUDE_impossibility_theorem_l2799_279932


namespace NUMINAMATH_CALUDE_matrix_multiplication_l2799_279968

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -2; -1, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 3; 2, -2]

theorem matrix_multiplication :
  A * B = !![(-4), 16; 10, (-13)] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l2799_279968


namespace NUMINAMATH_CALUDE_trig_expression_max_value_l2799_279913

theorem trig_expression_max_value (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) *
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_trig_expression_max_value_l2799_279913


namespace NUMINAMATH_CALUDE_green_balloons_l2799_279979

theorem green_balloons (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by
  sorry

end NUMINAMATH_CALUDE_green_balloons_l2799_279979


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l2799_279948

/-- The coordinates of the foci for the hyperbola x^2 - 4y^2 = 4 are (±√5, 0) -/
theorem hyperbola_foci_coordinates :
  let h : ℝ → ℝ → Prop := λ x y => x^2 - 4*y^2 = 4
  ∃ c : ℝ, c^2 = 5 ∧ 
    (∀ x y, h x y ↔ (x/2)^2 - y^2 = 1) ∧
    (∀ x y, h x y → (x = c ∨ x = -c) ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l2799_279948


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2799_279952

/-- A hyperbola with given asymptotes and minimum distance to a point -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are x ± 2y = 0 -/
  asymptotes : ∀ (x y : ℝ), x = 2*y ∨ x = -2*y
  /-- The minimum distance from any point on the hyperbola to A(5,0) is √6 -/
  min_distance : ∀ (x y : ℝ), (x - 5)^2 + y^2 ≥ 6

/-- The equation of the hyperbola satisfies one of two forms -/
theorem hyperbola_equation (h : Hyperbola) :
  (∀ (x y : ℝ), x^2 - 4*y^2 = (5 + Real.sqrt 6)^2) ∨
  (∀ (x y : ℝ), 4*y^2 - x^2 = 4) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2799_279952


namespace NUMINAMATH_CALUDE_rectangular_field_shortcut_l2799_279998

theorem rectangular_field_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_shortcut_l2799_279998


namespace NUMINAMATH_CALUDE_not_perfect_cube_l2799_279911

theorem not_perfect_cube (n : ℕ+) (h : ∃ p : ℕ, n^5 + n^3 + 2*n^2 + 2*n + 2 = p^3) :
  ¬∃ q : ℕ, 2*n^2 + n + 2 = q^3 := by
sorry

end NUMINAMATH_CALUDE_not_perfect_cube_l2799_279911


namespace NUMINAMATH_CALUDE_square_root_scaling_l2799_279946

theorem square_root_scaling (a b : ℝ) (ha : 0 < a) (hb : b = 100 * a) :
  Real.sqrt b = 10 * Real.sqrt a := by sorry

end NUMINAMATH_CALUDE_square_root_scaling_l2799_279946


namespace NUMINAMATH_CALUDE_sum_of_y_values_is_0_04_l2799_279996

-- Define the function g
def g (x : ℝ) : ℝ := (5*x)^2 - 5*x + 2

-- State the theorem
theorem sum_of_y_values_is_0_04 :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ g y₁ = 12 ∧ g y₂ = 12 ∧ y₁ + y₂ = 0.04 ∧
  ∀ y : ℝ, g y = 12 → y = y₁ ∨ y = y₂ := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_values_is_0_04_l2799_279996


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l2799_279982

theorem max_value_of_fraction (x : ℝ) (hx : x < 0) :
  (1 + x^2) / x ≤ -2 ∧ ((1 + x^2) / x = -2 ↔ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l2799_279982
