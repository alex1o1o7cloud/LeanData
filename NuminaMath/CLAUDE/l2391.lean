import Mathlib

namespace NUMINAMATH_CALUDE_fraction_of_decimals_equals_300_l2391_239109

theorem fraction_of_decimals_equals_300 : (0.3 ^ 4) / (0.03 ^ 3) = 300 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_decimals_equals_300_l2391_239109


namespace NUMINAMATH_CALUDE_complex_number_real_l2391_239154

theorem complex_number_real (a : ℝ) : 
  (∃ (r : ℝ), Complex.mk r 0 = Complex.mk 0 2 - (Complex.I * a) / (1 - Complex.I)) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_l2391_239154


namespace NUMINAMATH_CALUDE_age_difference_is_fifty_l2391_239186

/-- Represents the ages of family members in the year 2000 -/
structure FamilyAges where
  daughter : ℕ
  son : ℕ
  mother : ℕ
  father : ℕ

/-- Conditions given in the problem -/
def familyConditions (ages : FamilyAges) : Prop :=
  ages.mother = 4 * ages.daughter ∧
  ages.father = 6 * ages.son ∧
  ages.son = (3 * ages.daughter) / 2 ∧
  ages.father + 10 = 2 * (ages.mother + 10)

/-- The theorem to be proved -/
theorem age_difference_is_fifty (ages : FamilyAges) 
  (h : familyConditions ages) : ages.father - ages.mother = 50 := by
  sorry

#check age_difference_is_fifty

end NUMINAMATH_CALUDE_age_difference_is_fifty_l2391_239186


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2391_239162

theorem inequality_solution_range (b : ℝ) : 
  (∃ x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
   (∀ z : ℤ, z < 0 → (z - b > 0 ↔ z = x ∨ z = y))) → 
  -3 ≤ b ∧ b < -2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2391_239162


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l2391_239145

theorem multiply_and_simplify (y : ℝ) (h : y ≠ 0) :
  (18 * y^3) * (9 * y^2) * (1 / (6*y)^2) = (9/2) * y^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l2391_239145


namespace NUMINAMATH_CALUDE_shortest_side_of_special_triangle_l2391_239144

theorem shortest_side_of_special_triangle :
  ∀ (a b c : ℕ),
    a = 17 →
    a + b + c = 50 →
    (∃ A : ℕ, A^2 = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c) / 16) →
    b ≥ 13 ∧ c ≥ 13 :=
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_special_triangle_l2391_239144


namespace NUMINAMATH_CALUDE_expression_value_l2391_239157

theorem expression_value (x y : ℝ) (h : x - 2*y = -4) :
  (2*y - x)^2 - 2*x + 4*y - 1 = 23 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2391_239157


namespace NUMINAMATH_CALUDE_tom_final_balance_l2391_239185

def calculate_final_balance (initial_allowance : ℚ) (extra_earning : ℚ) (final_spending : ℚ) : ℚ :=
  let week1_balance := initial_allowance - initial_allowance / 3
  let week2_balance := week1_balance - week1_balance / 4
  let week3_balance_before_spending := week2_balance + extra_earning
  let week3_balance_after_spending := week3_balance_before_spending / 2
  week3_balance_after_spending - final_spending

theorem tom_final_balance :
  calculate_final_balance 12 5 3 = (5/2 : ℚ) := by sorry

end NUMINAMATH_CALUDE_tom_final_balance_l2391_239185


namespace NUMINAMATH_CALUDE_max_cupcakes_eaten_l2391_239148

/-- Given 30 cupcakes shared among three people, where one person eats twice as much as the first
    and the same as the second, the maximum number of cupcakes the first person could have eaten is 6. -/
theorem max_cupcakes_eaten (total : ℕ) (ben charles diana : ℕ) : 
  total = 30 →
  diana = 2 * ben →
  diana = charles →
  total = ben + charles + diana →
  ben ≤ 6 ∧ ∃ ben', ben' = 6 ∧ 
    ∃ charles' diana', 
      diana' = 2 * ben' ∧ 
      diana' = charles' ∧ 
      total = ben' + charles' + diana' :=
by sorry

end NUMINAMATH_CALUDE_max_cupcakes_eaten_l2391_239148


namespace NUMINAMATH_CALUDE_cube_plus_inverse_cube_l2391_239124

theorem cube_plus_inverse_cube (a : ℝ) (h : (a + 1 / (3 * a))^2 = 3) : 
  27 * a^3 + 1 / a^3 = 54 * Real.sqrt 3 ∨ 27 * a^3 + 1 / a^3 = -54 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_plus_inverse_cube_l2391_239124


namespace NUMINAMATH_CALUDE_non_pen_pencil_sales_l2391_239117

/-- The percentage of June sales for pens -/
def pen_sales : ℝ := 42

/-- The percentage of June sales for pencils -/
def pencil_sales : ℝ := 27

/-- The total percentage of all sales -/
def total_sales : ℝ := 100

/-- Theorem: The combined percentage of June sales that were not pens or pencils is 31% -/
theorem non_pen_pencil_sales : 
  total_sales - (pen_sales + pencil_sales) = 31 := by sorry

end NUMINAMATH_CALUDE_non_pen_pencil_sales_l2391_239117


namespace NUMINAMATH_CALUDE_g_of_3_equals_2_l2391_239102

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem statement
theorem g_of_3_equals_2 : g 3 = 2 := by sorry

end NUMINAMATH_CALUDE_g_of_3_equals_2_l2391_239102


namespace NUMINAMATH_CALUDE_at_least_one_less_than_two_l2391_239151

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + y) / x) ((1 + x) / y) < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_two_l2391_239151


namespace NUMINAMATH_CALUDE_rooster_weight_problem_l2391_239137

theorem rooster_weight_problem (price_per_kg : ℝ) (weight_rooster1 : ℝ) (total_earnings : ℝ) :
  price_per_kg = 0.5 →
  weight_rooster1 = 30 →
  total_earnings = 35 →
  ∃ weight_rooster2 : ℝ,
    weight_rooster2 = 40 ∧
    total_earnings = price_per_kg * (weight_rooster1 + weight_rooster2) :=
by sorry

end NUMINAMATH_CALUDE_rooster_weight_problem_l2391_239137


namespace NUMINAMATH_CALUDE_new_student_weight_l2391_239156

theorem new_student_weight
  (initial_students : ℕ)
  (initial_avg_weight : ℝ)
  (new_avg_weight : ℝ)
  (h1 : initial_students = 19)
  (h2 : initial_avg_weight = 15)
  (h3 : new_avg_weight = 14.8) :
  (initial_students + 1) * new_avg_weight - initial_students * initial_avg_weight = 11 :=
by sorry

end NUMINAMATH_CALUDE_new_student_weight_l2391_239156


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l2391_239183

-- Define the arithmetic sequence
def arithmeticSequence (a₁ aₙ d : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (λ i => a₁ + i * d)

-- Define the sum of a list of natural numbers
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

-- State the theorem
theorem arithmetic_sequence_sum_divisibility :
  let seq := arithmeticSequence 3 251 8
  let sum := sumList seq
  sum % 8 = 0 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l2391_239183


namespace NUMINAMATH_CALUDE_final_price_after_discounts_arun_paid_price_l2391_239167

/-- Calculates the final price of an article after applying two consecutive discounts -/
theorem final_price_after_discounts (original_price : ℝ) 
  (standard_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  let price_after_standard := original_price * (1 - standard_discount)
  let final_price := price_after_standard * (1 - additional_discount)
  final_price

/-- Proves that the final price of an article originally priced at 2000, 
    after a 30% standard discount and a 20% additional discount, is 1120 -/
theorem arun_paid_price : 
  final_price_after_discounts 2000 0.3 0.2 = 1120 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_arun_paid_price_l2391_239167


namespace NUMINAMATH_CALUDE_total_ways_eq_7464_l2391_239163

def num_oreo_flavors : ℕ := 6
def num_milk_flavors : ℕ := 4
def total_products : ℕ := 5

def ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

def alpha_choices (k : ℕ) : ℕ := ways_to_choose (num_oreo_flavors + num_milk_flavors) k

def beta_choices (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then num_oreo_flavors
  else if k = 2 then ways_to_choose num_oreo_flavors 2 + num_oreo_flavors
  else if k = 3 then ways_to_choose num_oreo_flavors 3 + num_oreo_flavors * (num_oreo_flavors - 1) + num_oreo_flavors
  else if k = 4 then ways_to_choose num_oreo_flavors 4 + num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 1 + num_oreo_flavors
  else ways_to_choose num_oreo_flavors 5 + num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 1 + 
       num_oreo_flavors * ways_to_choose (num_oreo_flavors - 1) 2 + num_oreo_flavors

def total_ways : ℕ := 
  (Finset.range (total_products + 1)).sum (λ k => alpha_choices k * beta_choices (total_products - k))

theorem total_ways_eq_7464 : total_ways = 7464 := by sorry

end NUMINAMATH_CALUDE_total_ways_eq_7464_l2391_239163


namespace NUMINAMATH_CALUDE_treasure_chest_value_l2391_239197

def base7_to_base10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem treasure_chest_value : 
  let coins := [6, 4, 3, 5]
  let gems := [1, 2, 5, 6]
  let maps := [0, 2, 3]
  base7_to_base10 coins + base7_to_base10 gems + base7_to_base10 maps = 4305 := by
sorry

#eval base7_to_base10 [6, 4, 3, 5] + base7_to_base10 [1, 2, 5, 6] + base7_to_base10 [0, 2, 3]

end NUMINAMATH_CALUDE_treasure_chest_value_l2391_239197


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_132_l2391_239115

theorem complex_arithmetic_expression_equals_132 :
  10 * 9 * 8 + 7 * 6 * 5 + 6 * 5 * 4 + 3 * 2 * 1 - 9 * 8 * 7 - 8 * 7 * 6 - 5 * 4 * 3 - 4 * 3 * 2 = 132 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_132_l2391_239115


namespace NUMINAMATH_CALUDE_apple_count_l2391_239174

theorem apple_count (blue_apples : ℕ) (yellow_apples : ℕ) : 
  blue_apples = 5 →
  yellow_apples = 2 * blue_apples →
  (blue_apples + yellow_apples) - ((blue_apples + yellow_apples) / 5) = 12 := by
sorry

end NUMINAMATH_CALUDE_apple_count_l2391_239174


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l2391_239169

theorem average_of_six_numbers (numbers : List ℕ) :
  numbers = [12, 412, 812, 1212, 1612, 2012] →
  (numbers.sum / numbers.length : ℚ) = 1012 := by
sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l2391_239169


namespace NUMINAMATH_CALUDE_pyramid_cross_section_distance_l2391_239127

/-- Given a right octagonal pyramid with two cross sections parallel to the base,
    this theorem proves the distance of the larger cross section from the apex. -/
theorem pyramid_cross_section_distance
  (area_small area_large : ℝ)
  (height_diff : ℝ)
  (h_area_small : area_small = 256 * Real.sqrt 2)
  (h_area_large : area_large = 576 * Real.sqrt 2)
  (h_height_diff : height_diff = 12) :
  ∃ (h : ℝ), h = 36 ∧ 
    (area_small / area_large = (2/3)^2) ∧
    (h - 2/3 * h = height_diff) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_cross_section_distance_l2391_239127


namespace NUMINAMATH_CALUDE_orange_savings_percentage_l2391_239195

/-- Calculates the percentage of money saved when receiving free items instead of buying them -/
theorem orange_savings_percentage 
  (family_size : ℕ) 
  (planned_spending : ℝ) 
  (orange_price : ℝ) : 
  family_size = 4 → 
  planned_spending = 15 → 
  orange_price = 1.5 → 
  (family_size * orange_price / planned_spending) * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_orange_savings_percentage_l2391_239195


namespace NUMINAMATH_CALUDE_estimate_expression_range_l2391_239125

theorem estimate_expression_range : 
  5 < (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1/5) ∧
  (2 * Real.sqrt 5 + 5 * Real.sqrt 2) * Real.sqrt (1/5) < 6 := by
  sorry

end NUMINAMATH_CALUDE_estimate_expression_range_l2391_239125


namespace NUMINAMATH_CALUDE_existence_of_good_subset_l2391_239120

def M : Finset ℕ := Finset.range 20

def is_valid_function (f : Finset ℕ → ℕ) : Prop :=
  ∀ S : Finset ℕ, S ⊆ M → S.card = 9 → f S ∈ M

theorem existence_of_good_subset (f : Finset ℕ → ℕ) (h : is_valid_function f) :
  ∃ T : Finset ℕ, T ⊆ M ∧ T.card = 10 ∧ ∀ k ∈ T, f (T \ {k}) ≠ k := by sorry

end NUMINAMATH_CALUDE_existence_of_good_subset_l2391_239120


namespace NUMINAMATH_CALUDE_quotient_relation_l2391_239193

theorem quotient_relation : ∃ (k l : ℝ), k ≠ l ∧ (64 / k = 4 * (64 / l)) := by
  sorry

end NUMINAMATH_CALUDE_quotient_relation_l2391_239193


namespace NUMINAMATH_CALUDE_michael_has_270_eggs_l2391_239153

/-- Calculates the number of eggs Michael has after buying and giving away crates. -/
def michaels_eggs (initial_crates : ℕ) (given_crates : ℕ) (bought_crates : ℕ) (eggs_per_crate : ℕ) : ℕ :=
  (initial_crates - given_crates + bought_crates) * eggs_per_crate

/-- Proves that Michael has 270 eggs after his transactions. -/
theorem michael_has_270_eggs :
  michaels_eggs 6 2 5 30 = 270 := by
  sorry

end NUMINAMATH_CALUDE_michael_has_270_eggs_l2391_239153


namespace NUMINAMATH_CALUDE_solve_coin_problem_l2391_239161

def coin_problem (total : ℕ) (coin1 : ℕ) (coin2 : ℕ) : Prop :=
  ∃ (max min : ℕ),
    (∃ (a : ℕ), a * coin1 = total ∧ a = max) ∧
    (∃ (b c : ℕ), b * coin1 + c * coin2 = total ∧ b + c = min) ∧
    max - min = 2

theorem solve_coin_problem :
  coin_problem 45 10 25 := by sorry

end NUMINAMATH_CALUDE_solve_coin_problem_l2391_239161


namespace NUMINAMATH_CALUDE_headmaster_retirement_l2391_239150

/-- Represents the months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Returns the month that is n months after the given month -/
def monthsAfter (start : Month) (n : ℕ) : Month :=
  match n with
  | 0 => start
  | n + 1 => monthsAfter (match start with
    | Month.January => Month.February
    | Month.February => Month.March
    | Month.March => Month.April
    | Month.April => Month.May
    | Month.May => Month.June
    | Month.June => Month.July
    | Month.July => Month.August
    | Month.August => Month.September
    | Month.September => Month.October
    | Month.October => Month.November
    | Month.November => Month.December
    | Month.December => Month.January
  ) n

theorem headmaster_retirement (start_month : Month) (duration : ℕ) :
  start_month = Month.March → duration = 3 →
  monthsAfter start_month duration = Month.May :=
by
  sorry

end NUMINAMATH_CALUDE_headmaster_retirement_l2391_239150


namespace NUMINAMATH_CALUDE_f_range_l2391_239152

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the range
def range : Set ℝ := {y | ∃ x ∈ domain, f x = y}

-- Theorem statement
theorem f_range : range = {y | -1 ≤ y ∧ y ≤ 3} := by sorry

end NUMINAMATH_CALUDE_f_range_l2391_239152


namespace NUMINAMATH_CALUDE_solve_system_l2391_239133

theorem solve_system (C D : ℚ) 
  (eq1 : 3 * C - 4 * D = 18)
  (eq2 : C = 2 * D - 5) : 
  C = 28 ∧ D = 33 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2391_239133


namespace NUMINAMATH_CALUDE_simplify_expression_solve_quadratic_equation_l2391_239155

-- Part 1
theorem simplify_expression :
  Real.sqrt 18 / Real.sqrt 9 - Real.sqrt (1/4) * 2 * Real.sqrt 2 + Real.sqrt 32 = 4 * Real.sqrt 2 := by
  sorry

-- Part 2
theorem solve_quadratic_equation :
  ∀ x : ℝ, x^2 - 2*x = 3 ↔ x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_solve_quadratic_equation_l2391_239155


namespace NUMINAMATH_CALUDE_probability_yellow_chalk_l2391_239158

/-- The number of yellow chalks in the box -/
def yellow_chalks : ℕ := 3

/-- The number of red chalks in the box -/
def red_chalks : ℕ := 2

/-- The total number of chalks in the box -/
def total_chalks : ℕ := yellow_chalks + red_chalks

/-- The probability of selecting a yellow chalk -/
def prob_yellow : ℚ := yellow_chalks / total_chalks

theorem probability_yellow_chalk :
  prob_yellow = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_yellow_chalk_l2391_239158


namespace NUMINAMATH_CALUDE_sqrt_seven_fraction_l2391_239141

theorem sqrt_seven_fraction (p q : ℝ) (hp : p > 0) (hq : q > 0) (h : Real.sqrt 7 = p / q) :
  Real.sqrt 7 = (7 * q - 2 * p) / (p - 2 * q) ∧ p - 2 * q > 0 ∧ p - 2 * q < q := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_fraction_l2391_239141


namespace NUMINAMATH_CALUDE_hcf_of_three_numbers_l2391_239140

theorem hcf_of_three_numbers (a b c : ℕ) (h_lcm : Nat.lcm (Nat.lcm a b) c = 2^4 * 3^2 * 17 * 7)
  (h_a : a = 136) (h_b : b = 144) (h_c : c = 168) : Nat.gcd (Nat.gcd a b) c = 8 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_three_numbers_l2391_239140


namespace NUMINAMATH_CALUDE_triangle_solution_l2391_239199

noncomputable def triangle_problem (a b c A B C : ℝ) : Prop :=
  (2 * b - c) * Real.cos A = a * Real.cos C ∧
  a = Real.sqrt 13 ∧
  (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3

theorem triangle_solution (a b c A B C : ℝ) 
  (h : triangle_problem a b c A B C) :
  A = π / 3 ∧ a + b + c = 7 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_solution_l2391_239199


namespace NUMINAMATH_CALUDE_sqrt_a_3a_sqrt_a_l2391_239119

theorem sqrt_a_3a_sqrt_a (a : ℝ) (ha : a > 0) :
  Real.sqrt (a * 3 * a * Real.sqrt a) = a ^ (3/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_3a_sqrt_a_l2391_239119


namespace NUMINAMATH_CALUDE_probability_theorem_l2391_239146

def is_valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b

def satisfies_condition (a b : ℕ) : Prop :=
  ∃ m : ℕ, a * b + a + b = 7 * m - 2

def count_valid_pairs : ℕ := Nat.choose 100 2

def count_satisfying_pairs : ℕ := 1295

theorem probability_theorem :
  (count_satisfying_pairs : ℚ) / count_valid_pairs = 259 / 990 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l2391_239146


namespace NUMINAMATH_CALUDE_sixth_term_is_three_l2391_239143

/-- An arithmetic sequence with 10 terms where the sum of even-numbered terms is 15 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m, a (n + 1) - a n = a (m + 1) - a m) ∧ 
  (a 2 + a 4 + a 6 + a 8 + a 10 = 15)

/-- The 6th term of the arithmetic sequence is 3 -/
theorem sixth_term_is_three (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  a 6 = 3 := by sorry

end NUMINAMATH_CALUDE_sixth_term_is_three_l2391_239143


namespace NUMINAMATH_CALUDE_houses_with_garage_count_l2391_239130

/-- Represents the number of houses with various features in a development --/
structure Development where
  total : ℕ
  withPool : ℕ
  withBoth : ℕ
  withNeither : ℕ

/-- Calculates the number of houses with a two-car garage --/
def housesWithGarage (d : Development) : ℕ :=
  d.total + d.withBoth - d.withPool - d.withNeither

/-- Theorem stating that in the given development, 75 houses have a two-car garage --/
theorem houses_with_garage_count (d : Development) 
  (h1 : d.total = 85)
  (h2 : d.withPool = 40)
  (h3 : d.withBoth = 35)
  (h4 : d.withNeither = 30) :
  housesWithGarage d = 75 := by
  sorry

#eval housesWithGarage ⟨85, 40, 35, 30⟩

end NUMINAMATH_CALUDE_houses_with_garage_count_l2391_239130


namespace NUMINAMATH_CALUDE_inequality_proof_l2391_239166

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : a * (c^2 - 1) = b * (b^2 + c^2)) 
  (h_d : d ≤ 1) : 
  d * (a * Real.sqrt (1 - d^2) + b^2 * Real.sqrt (1 + d^2)) ≤ (a + b) * c / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2391_239166


namespace NUMINAMATH_CALUDE_sum_of_intersection_points_l2391_239132

/-- A type representing a line in a plane -/
structure Line :=
  (id : ℕ)

/-- A type representing an intersection point of two lines -/
structure IntersectionPoint :=
  (line1 : Line)
  (line2 : Line)

/-- A configuration of lines in a plane -/
structure LineConfiguration :=
  (lines : Finset Line)
  (intersections : Finset IntersectionPoint)
  (distinct_lines : lines.card = 5)
  (no_triple_intersections : ∀ p q r : Line, p ∈ lines → q ∈ lines → r ∈ lines → 
    p ≠ q → q ≠ r → p ≠ r → 
    ¬∃ i : IntersectionPoint, i ∈ intersections ∧ 
      (i.line1 = p ∧ i.line2 = q) ∧
      (i.line1 = q ∧ i.line2 = r) ∧
      (i.line1 = p ∧ i.line2 = r))

/-- The theorem to be proved -/
theorem sum_of_intersection_points (config : LineConfiguration) :
  (Finset.range 11).sum (λ n => n * (Finset.filter (λ c : LineConfiguration => c.intersections.card = n) {config}).card) = 54 :=
sorry

end NUMINAMATH_CALUDE_sum_of_intersection_points_l2391_239132


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l2391_239149

/-- The parabola is defined by the equation y = x^2 - 3x - 4 -/
def parabola (x y : ℝ) : Prop := y = x^2 - 3*x - 4

/-- The y-axis is defined by x = 0 -/
def y_axis (x : ℝ) : Prop := x = 0

/-- Theorem: The intersection point of the parabola y = x^2 - 3x - 4 with the y-axis has coordinates (0, -4) -/
theorem parabola_y_axis_intersection :
  ∃ (x y : ℝ), parabola x y ∧ y_axis x ∧ x = 0 ∧ y = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l2391_239149


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2391_239182

theorem candy_bar_cost (total_bars : ℕ) (dave_bars : ℕ) (john_paid : ℚ) : 
  total_bars = 20 → 
  dave_bars = 6 → 
  john_paid = 21 → 
  (john_paid / (total_bars - dave_bars : ℚ)) = 1.5 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2391_239182


namespace NUMINAMATH_CALUDE_trains_crossing_time_trains_crossing_time_approx_9_seconds_l2391_239188

/-- Time for trains to cross each other -/
theorem trains_crossing_time (train1_length train2_length : ℝ) 
  (train1_speed train2_speed : ℝ) : ℝ :=
  let total_length := train1_length + train2_length
  let relative_speed_kmh := train1_speed + train2_speed
  let relative_speed_ms := relative_speed_kmh * 1000 / 3600
  total_length / relative_speed_ms

/-- Proof that the time for the trains to cross is approximately 9 seconds -/
theorem trains_crossing_time_approx_9_seconds : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |trains_crossing_time 120.00001 380.03999 120 80 - 9| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_trains_crossing_time_trains_crossing_time_approx_9_seconds_l2391_239188


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2391_239107

/-- If x^2 + 6x + k^2 is a perfect square polynomial, then k = ± 3 -/
theorem perfect_square_condition (k : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, x^2 + 6*x + k^2 = (p x)^2) → 
  k = 3 ∨ k = -3 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2391_239107


namespace NUMINAMATH_CALUDE_expression_evaluation_l2391_239198

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -3
  (15 * x^3 * y - 10 * x^2 * y^2) / (5 * x * y) - (3*x + y) * (x - 3*y) = 18 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2391_239198


namespace NUMINAMATH_CALUDE_flashlight_visibility_difference_l2391_239170

/-- Flashlight visibility problem -/
theorem flashlight_visibility_difference (veronica_visibility : ℝ) :
  veronica_visibility = 1000 →
  let freddie_visibility := 3 * veronica_visibility
  let velma_visibility := 5 * freddie_visibility - 2000
  let daphne_visibility := (veronica_visibility + freddie_visibility + velma_visibility) / 3
  let total_visibility := veronica_visibility + freddie_visibility + velma_visibility + daphne_visibility
  total_visibility = 40000 →
  ∃ ε > 0, |velma_visibility - daphne_visibility - 7666.67| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_flashlight_visibility_difference_l2391_239170


namespace NUMINAMATH_CALUDE_system_solution_conditions_l2391_239103

theorem system_solution_conditions (a b x y z : ℝ) : 
  (x + y + z = a) →
  (x^2 + y^2 + z^2 = b^2) →
  (x * y = z^2) →
  (x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z) ↔ 
  (3 * b^2 > a^2 ∧ a^2 > b^2 ∧ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_conditions_l2391_239103


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l2391_239108

/-- Given Jake's current weight and the condition about his weight relative to his sister's,
    prove that their combined weight is 212 pounds. -/
theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 152 →
  jake_weight - 32 = 2 * sister_weight →
  jake_weight + sister_weight = 212 := by
  sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l2391_239108


namespace NUMINAMATH_CALUDE_max_value_of_f_l2391_239111

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x - 4 * x^3

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ ∀ x, x ∈ Set.Icc 0 1 → f x ≤ f c ∧ f c = 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2391_239111


namespace NUMINAMATH_CALUDE_five_fourths_of_fifteen_fourths_l2391_239177

theorem five_fourths_of_fifteen_fourths (x : ℚ) : 
  x = 15 / 4 → (5 / 4 : ℚ) * x = 75 / 16 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_fifteen_fourths_l2391_239177


namespace NUMINAMATH_CALUDE_consecutive_points_distance_l2391_239104

/-- Given 5 consecutive points on a straight line, prove that ac = 11 -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 8) →            -- de = 8
  (b - a = 5) →            -- ab = 5
  (e - a = 21) →           -- ae = 21
  (c - a = 11) :=          -- ac = 11
by sorry

end NUMINAMATH_CALUDE_consecutive_points_distance_l2391_239104


namespace NUMINAMATH_CALUDE_percentage_failed_hindi_l2391_239112

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) 
  (h1 : failed_english = 45)
  (h2 : failed_both = 20)
  (h3 : passed_both = 40) :
  ∃ (failed_hindi : ℝ), failed_hindi = 35 := by
sorry

end NUMINAMATH_CALUDE_percentage_failed_hindi_l2391_239112


namespace NUMINAMATH_CALUDE_total_coins_l2391_239189

/-- Represents a 3x3 grid of cells --/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The sum of coins in the corner cells --/
def corner_sum (g : Grid) : ℕ :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The number of coins in the center cell --/
def center_value (g : Grid) : ℕ :=
  g 1 1

/-- Theorem stating the total number of coins in the grid --/
theorem total_coins (g : Grid) 
  (h_corner : corner_sum g = 8) 
  (h_center : center_value g = 3) : 
  ∃ (total : ℕ), total = 8 :=
sorry

end NUMINAMATH_CALUDE_total_coins_l2391_239189


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2391_239190

theorem geometric_arithmetic_sequence_problem (a b c : ℝ) : 
  a + b + c = 114 →
  b / a = c / b →
  b / a ≠ 1 →
  b - a = c - b →
  c - a = 24 * (b - a) →
  a = 2 ∧ b = 14 ∧ c = 98 := by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2391_239190


namespace NUMINAMATH_CALUDE_units_digit_47_power_47_l2391_239176

theorem units_digit_47_power_47 : 47^47 ≡ 3 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_units_digit_47_power_47_l2391_239176


namespace NUMINAMATH_CALUDE_combined_stock_cost_value_l2391_239142

/-- Calculate the final cost of a stock given its initial parameters -/
def calculate_stock_cost (initial_price discount brokerage tax_rate transaction_fee : ℚ) : ℚ :=
  let discounted_price := initial_price * (1 - discount)
  let brokerage_fee := discounted_price * brokerage
  let net_purchase_price := discounted_price + brokerage_fee
  let tax := net_purchase_price * tax_rate
  net_purchase_price + tax + transaction_fee

/-- The combined cost of three stocks with given parameters -/
def combined_stock_cost : ℚ :=
  calculate_stock_cost 100 (4/100) (1/500) (12/100) 2 +
  calculate_stock_cost 200 (6/100) (1/400) (10/100) 3 +
  calculate_stock_cost 150 (3/100) (1/200) (15/100) 1

/-- Theorem stating the combined cost of the three stocks -/
theorem combined_stock_cost_value : 
  combined_stock_cost = 489213665/1000000 := by sorry

end NUMINAMATH_CALUDE_combined_stock_cost_value_l2391_239142


namespace NUMINAMATH_CALUDE_subcommittee_count_l2391_239164

/-- The number of members in the planning committee -/
def total_members : ℕ := 12

/-- The number of teachers in the planning committee -/
def teacher_count : ℕ := 5

/-- The size of the subcommittee to be formed -/
def subcommittee_size : ℕ := 4

/-- The minimum number of teachers required in the subcommittee -/
def min_teachers : ℕ := 2

/-- Calculates the number of valid subcommittees -/
def valid_subcommittees : ℕ := 285

theorem subcommittee_count :
  (Nat.choose total_members subcommittee_size) -
  (Nat.choose (total_members - teacher_count) subcommittee_size) -
  (Nat.choose teacher_count 1 * Nat.choose (total_members - teacher_count) (subcommittee_size - 1)) =
  valid_subcommittees :=
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l2391_239164


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2391_239101

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -3)  -- vertex at (1, -3)
  (h2 : f a b c 2 = -5/2)  -- passes through (2, -5/2)
  (h3 : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a b c x1 = f a b c x2 ∧ |x1 - x2| = 6)  -- intersects y = m at two points 6 units apart
  : 
  (∀ x, f a b c x = 1/2 * x^2 - x - 5/2) ∧  -- Part 1
  (∃ m : ℝ, m = 3/2 ∧ ∀ x, f a b c x = m → (∃ y, f a b c y = m ∧ |x - y| = 6)) ∧  -- Part 2
  (∀ x, -3 < x → x < 3 → -3 ≤ f a b c x ∧ f a b c x < 5)  -- Part 3
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2391_239101


namespace NUMINAMATH_CALUDE_first_divisor_problem_l2391_239173

theorem first_divisor_problem (x : ℕ) : x = 7 ↔ 
  x > 0 ∧ 
  x ≠ 15 ∧ 
  184 % x = 2 ∧ 
  184 % 15 = 4 ∧ 
  ∀ y : ℕ, y > 0 ∧ y < x ∧ y ≠ 15 → 184 % y ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_first_divisor_problem_l2391_239173


namespace NUMINAMATH_CALUDE_equation_solution_l2391_239178

theorem equation_solution (x y : ℝ) (h : x / (x - 1) = (y^2 + 2*y - 1) / (y^2 + 2*y - 2)) :
  x = y^2 + 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2391_239178


namespace NUMINAMATH_CALUDE_smallest_n_for_314_fraction_l2391_239131

def is_relatively_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

def contains_314 (q : ℚ) : Prop :=
  ∃ k : ℕ, (10^k * q - (10^k * q).floor) * 1000 ≥ 314 ∧
            (10^k * q - (10^k * q).floor) * 1000 < 315

theorem smallest_n_for_314_fraction :
  ∃ (m n : ℕ), 
    n = 159 ∧
    m < n ∧
    is_relatively_prime m n ∧
    contains_314 (m / n) ∧
    (∀ (m' n' : ℕ), n' < 159 → m' < n' → is_relatively_prime m' n' → ¬contains_314 (m' / n')) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_314_fraction_l2391_239131


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2391_239105

theorem inequality_and_equality_condition 
  (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) 
  (hab : a + b = 1) : 
  (1 / (a / x + b / y) ≤ a * x + b * y) ∧ 
  (1 / (a / x + b / y) = a * x + b * y ↔ a * y = b * x) :=
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2391_239105


namespace NUMINAMATH_CALUDE_fred_onions_l2391_239181

theorem fred_onions (sara_onions sally_onions total_onions : ℕ) 
  (h1 : sara_onions = 4)
  (h2 : sally_onions = 5)
  (h3 : total_onions = 18) :
  total_onions - (sara_onions + sally_onions) = 9 := by
sorry

end NUMINAMATH_CALUDE_fred_onions_l2391_239181


namespace NUMINAMATH_CALUDE_mia_sixth_game_shots_l2391_239135

-- Define the initial conditions
def initial_shots : ℕ := 50
def initial_made : ℕ := 20
def new_shots : ℕ := 15

-- Define the function to calculate the new shooting average
def new_average (x : ℕ) : ℚ :=
  (initial_made + x : ℚ) / (initial_shots + new_shots : ℚ)

-- Theorem statement
theorem mia_sixth_game_shots :
  ∃ x : ℕ, x ≤ new_shots ∧ new_average x = 45 / 100 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_mia_sixth_game_shots_l2391_239135


namespace NUMINAMATH_CALUDE_carpet_fits_both_rooms_l2391_239159

-- Define the carpet and room dimensions
def carpet_width : ℝ := 25
def carpet_length : ℝ := 50
def room1_width : ℝ := 38
def room1_length : ℝ := 55
def room2_width : ℝ := 50
def room2_length : ℝ := 55

-- Define a function to check if the carpet fits in a room
def carpet_fits_room (carpet_w carpet_l room_w room_l : ℝ) : Prop :=
  carpet_w^2 + carpet_l^2 = room_w^2 + room_l^2

-- Theorem statement
theorem carpet_fits_both_rooms :
  carpet_fits_room carpet_width carpet_length room1_width room1_length ∧
  carpet_fits_room carpet_width carpet_length room2_width room2_length :=
by sorry

end NUMINAMATH_CALUDE_carpet_fits_both_rooms_l2391_239159


namespace NUMINAMATH_CALUDE_machine_profit_percentage_l2391_239110

/-- Calculates the profit percentage given the purchase price, repair cost, transportation charges, and selling price of a machine. -/
def profit_percentage (purchase_price repair_cost transport_charges selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost + transport_charges
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem: The profit percentage for the given machine transaction is 50%. -/
theorem machine_profit_percentage :
  profit_percentage 13000 5000 1000 28500 = 50 := by
  sorry

end NUMINAMATH_CALUDE_machine_profit_percentage_l2391_239110


namespace NUMINAMATH_CALUDE_triangle_xy_length_l2391_239179

-- Define the triangle
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  -- Right angle at X
  (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0 ∧
  -- 45° angle at Y
  (Z.1 - Y.1) * (X.1 - Y.1) + (Z.2 - Y.2) * (X.2 - Y.2) = 
    Real.sqrt ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2) * Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) / 2 ∧
  -- XZ = 12√2
  (Z.1 - X.1)^2 + (Z.2 - X.2)^2 = 288

-- Theorem statement
theorem triangle_xy_length (X Y Z : ℝ × ℝ) (h : Triangle X Y Z) :
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_triangle_xy_length_l2391_239179


namespace NUMINAMATH_CALUDE_u_closed_under_multiplication_l2391_239175

def u : Set ℕ := {n : ℕ | ∃ m : ℕ, n = m * m ∧ m > 0}

theorem u_closed_under_multiplication :
  ∀ x y : ℕ, x ∈ u → y ∈ u → (x * y) ∈ u :=
by
  sorry

end NUMINAMATH_CALUDE_u_closed_under_multiplication_l2391_239175


namespace NUMINAMATH_CALUDE_parallelepiped_volume_solution_l2391_239196

/-- The volume of a parallelepiped defined by vectors (3,4,5), (2,k,3), and (2,3,k) -/
def parallelepipedVolume (k : ℝ) : ℝ := |3 * k^2 - 15 * k + 27|

/-- Theorem stating that k = 5 is the positive solution for the parallelepiped volume equation -/
theorem parallelepiped_volume_solution :
  ∃! k : ℝ, k > 0 ∧ parallelepipedVolume k = 27 ∧ k = 5 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_solution_l2391_239196


namespace NUMINAMATH_CALUDE_one_in_A_l2391_239165

def A : Set ℝ := {x : ℝ | x ≥ -1}

theorem one_in_A : (1 : ℝ) ∈ A := by
  sorry

end NUMINAMATH_CALUDE_one_in_A_l2391_239165


namespace NUMINAMATH_CALUDE_probability_of_meeting_theorem_l2391_239121

/-- Represents the practice schedule of a person --/
structure PracticeSchedule where
  start_time : ℝ
  duration : ℝ

/-- Represents the practice schedules of two people over multiple days --/
structure PracticeScenario where
  your_schedule : PracticeSchedule
  friend_schedule : PracticeSchedule
  num_days : ℕ

/-- Calculates the probability of meeting given two practice schedules --/
def probability_of_meeting (s : PracticeScenario) : ℝ :=
  sorry

/-- Calculates the probability of meeting on at least k days out of n days --/
def probability_of_meeting_at_least (s : PracticeScenario) (k : ℕ) : ℝ :=
  sorry

theorem probability_of_meeting_theorem :
  let s : PracticeScenario := {
    your_schedule := { start_time := 0, duration := 3 },
    friend_schedule := { start_time := 5, duration := 1 },
    num_days := 5
  }
  probability_of_meeting_at_least s 2 = 232 / 243 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_meeting_theorem_l2391_239121


namespace NUMINAMATH_CALUDE_infinitely_many_very_good_pairs_l2391_239180

/-- A pair of natural numbers is "good" if they consist of the same prime divisors, possibly in different powers. -/
def isGoodPair (m n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)

/-- A pair of natural numbers is "very good" if both the pair and their successors form "good" pairs. -/
def isVeryGoodPair (m n : ℕ) : Prop :=
  isGoodPair m n ∧ isGoodPair (m + 1) (n + 1) ∧ m ≠ n

/-- There exist infinitely many "very good" pairs of natural numbers. -/
theorem infinitely_many_very_good_pairs :
  ∀ k : ℕ, ∃ m n : ℕ, m > k ∧ n > k ∧ isVeryGoodPair m n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_very_good_pairs_l2391_239180


namespace NUMINAMATH_CALUDE_lilith_water_bottles_l2391_239122

/-- The number of water bottles Lilith originally had -/
def num_bottles : ℕ := 60

/-- The original selling price per bottle in dollars -/
def original_price : ℚ := 2

/-- The reduced selling price per bottle in dollars -/
def reduced_price : ℚ := 185/100

theorem lilith_water_bottles :
  (original_price * num_bottles : ℚ) - (reduced_price * num_bottles) = 9 :=
sorry

end NUMINAMATH_CALUDE_lilith_water_bottles_l2391_239122


namespace NUMINAMATH_CALUDE_tuesday_distance_l2391_239168

/-- Proves that the distance driven on Tuesday is 18 miles -/
theorem tuesday_distance (monday_distance : ℝ) (wednesday_distance : ℝ) (average_distance : ℝ) (num_days : ℕ) :
  monday_distance = 12 →
  wednesday_distance = 21 →
  average_distance = 17 →
  num_days = 3 →
  (monday_distance + wednesday_distance + (num_days * average_distance - monday_distance - wednesday_distance)) / num_days = average_distance →
  num_days * average_distance - monday_distance - wednesday_distance = 18 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_distance_l2391_239168


namespace NUMINAMATH_CALUDE_f_of_5_equals_92_l2391_239106

/-- Given a function f(x) = 2x^2 + y where f(2) = 50, prove that f(5) = 92 -/
theorem f_of_5_equals_92 (f : ℝ → ℝ) (y : ℝ) 
  (h1 : ∀ x, f x = 2 * x^2 + y) 
  (h2 : f 2 = 50) : 
  f 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_equals_92_l2391_239106


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l2391_239136

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 7x + 3 and y = (3c)x + 5 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 7 * x + 3 ↔ y = (3 * c) * x + 5) → c = 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l2391_239136


namespace NUMINAMATH_CALUDE_probability_calculation_l2391_239116

def total_silverware : ℕ := 8 + 7 + 5

def probability_2forks_1spoon_1knife (forks spoons knives total : ℕ) : ℚ :=
  let favorable_outcomes := Nat.choose forks 2 * Nat.choose spoons 1 * Nat.choose knives 1
  let total_outcomes := Nat.choose total 4
  favorable_outcomes / total_outcomes

theorem probability_calculation :
  probability_2forks_1spoon_1knife 8 7 5 total_silverware = 196 / 969 := by
  sorry

end NUMINAMATH_CALUDE_probability_calculation_l2391_239116


namespace NUMINAMATH_CALUDE_water_transfer_result_l2391_239129

/-- Represents the volume of water transferred on a given day -/
def transfer (day : ℕ) : ℚ :=
  if day % 2 = 1 then day else day + 1

/-- Calculates the sum of transfers for odd or even days up to n days -/
def sumTransfers (n : ℕ) (isOdd : Bool) : ℚ :=
  let start := if isOdd then 1 else 2
  let count := n / 2
  count * (2 * start + (count - 1) * 4) / 2

/-- The initial volume of water in each jar (in ml) -/
def initialVolume : ℚ := 1000

/-- The number of days for which transfers occur -/
def totalDays : ℕ := 200

/-- The final volume in Maria's jar after the transfers -/
def finalVolume : ℚ := initialVolume + sumTransfers totalDays true - sumTransfers totalDays false

theorem water_transfer_result :
  finalVolume = 900 := by sorry

end NUMINAMATH_CALUDE_water_transfer_result_l2391_239129


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l2391_239134

theorem quadratic_non_real_roots (b : ℝ) : 
  (∀ x : ℂ, x^2 + b*x + 16 = 0 → x.im ≠ 0) ↔ -8 < b ∧ b < 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l2391_239134


namespace NUMINAMATH_CALUDE_horner_evaluation_f_5_l2391_239191

def f (x : ℝ) : ℝ := 2*x^7 - 9*x^6 + 5*x^5 - 49*x^4 - 5*x^3 + 2*x^2 + x + 1

theorem horner_evaluation_f_5 : f 5 = 56 := by sorry

end NUMINAMATH_CALUDE_horner_evaluation_f_5_l2391_239191


namespace NUMINAMATH_CALUDE_sum_of_series_equals_two_l2391_239184

/-- The sum of the infinite series ∑(n=1 to ∞) (4n-1)/3^n is equal to 2 -/
theorem sum_of_series_equals_two :
  let series := fun n : ℕ => (4 * n - 1) / (3 ^ n : ℝ)
  (∑' n, series n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_two_l2391_239184


namespace NUMINAMATH_CALUDE_square_of_negative_cube_l2391_239100

theorem square_of_negative_cube (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_cube_l2391_239100


namespace NUMINAMATH_CALUDE_frog_climb_time_l2391_239113

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℝ
  climb_distance : ℝ
  slip_distance : ℝ
  slip_time_ratio : ℝ

/-- Calculates the time taken for the frog to climb the well -/
def climb_time (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the frog takes 22 minutes to climb the well -/
theorem frog_climb_time :
  let f : FrogClimb := {
    well_depth := 12,
    climb_distance := 3,
    slip_distance := 1,
    slip_time_ratio := 1/3
  }
  climb_time f = 22 := by sorry

end NUMINAMATH_CALUDE_frog_climb_time_l2391_239113


namespace NUMINAMATH_CALUDE_barry_sotter_magic_barry_sotter_days_l2391_239147

theorem barry_sotter_magic (n : ℕ) : (3/2 : ℝ)^n ≥ 50 ↔ n ≥ 10 := by sorry

theorem barry_sotter_days : ∃ (n : ℕ), (∀ (m : ℕ), (3/2 : ℝ)^m ≥ 50 → n ≤ m) ∧ (3/2 : ℝ)^n ≥ 50 :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_barry_sotter_days_l2391_239147


namespace NUMINAMATH_CALUDE_probability_two_kings_or_at_least_two_aces_l2391_239172

def standard_deck : ℕ := 52
def num_aces : ℕ := 4
def num_kings : ℕ := 4
def cards_drawn : ℕ := 3

def prob_two_kings : ℚ := (Nat.choose num_kings 2 * Nat.choose (standard_deck - num_kings) 1) / Nat.choose standard_deck cards_drawn

def prob_two_aces : ℚ := (Nat.choose num_aces 2 * Nat.choose (standard_deck - num_aces) 1) / Nat.choose standard_deck cards_drawn

def prob_three_aces : ℚ := Nat.choose num_aces 3 / Nat.choose standard_deck cards_drawn

def prob_at_least_two_aces : ℚ := prob_two_aces + prob_three_aces

theorem probability_two_kings_or_at_least_two_aces :
  prob_two_kings + prob_at_least_two_aces = 1090482 / 40711175 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_kings_or_at_least_two_aces_l2391_239172


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l2391_239123

/-- Fuel efficiency of a car on highway and in city -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank : ℝ     -- Tank capacity in gallons
  h_positive : highway > 0
  c_positive : city > 0
  t_positive : tank > 0
  city_less : city = highway - 6

/-- Theorem stating the car's fuel efficiency in the city is 18 mpg -/
theorem city_fuel_efficiency 
  (car : CarFuelEfficiency)
  (h_highway : car.highway * car.tank = 448)
  (h_city : car.city * car.tank = 336) :
  car.city = 18 := by
  sorry

end NUMINAMATH_CALUDE_city_fuel_efficiency_l2391_239123


namespace NUMINAMATH_CALUDE_probability_at_least_one_unqualified_l2391_239160

/-- The number of products -/
def total_products : ℕ := 5

/-- The number of qualified products -/
def qualified_products : ℕ := 3

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products inspected -/
def inspected_products : ℕ := 2

/-- The probability of selecting at least one unqualified product -/
def prob_at_least_one_unqualified : ℚ := 7/10

/-- Theorem stating the probability of selecting at least one unqualified product -/
theorem probability_at_least_one_unqualified :
  let total_ways := Nat.choose total_products inspected_products
  let qualified_ways := Nat.choose qualified_products inspected_products
  1 - (qualified_ways : ℚ) / (total_ways : ℚ) = prob_at_least_one_unqualified :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_unqualified_l2391_239160


namespace NUMINAMATH_CALUDE_price_reduction_l2391_239138

theorem price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_reduction := 1 - 0.08
  let second_reduction := 1 - 0.10
  let final_price := original_price * first_reduction * second_reduction
  final_price / original_price = 0.828 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_l2391_239138


namespace NUMINAMATH_CALUDE_program_schedule_arrangements_l2391_239128

theorem program_schedule_arrangements (n : ℕ) (h : n = 6) : 
  (n + 1).choose 1 * (n + 2).choose 1 = 56 := by
  sorry

end NUMINAMATH_CALUDE_program_schedule_arrangements_l2391_239128


namespace NUMINAMATH_CALUDE_ryan_project_average_funding_l2391_239187

/-- The average amount each person funds to Ryan's project -/
def average_funding (total_goal : ℕ) (people : ℕ) (initial_funds : ℕ) : ℚ :=
  (total_goal - initial_funds : ℚ) / people

/-- Theorem: The average funding per person for Ryan's project is $10 -/
theorem ryan_project_average_funding :
  average_funding 1000 80 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ryan_project_average_funding_l2391_239187


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2391_239126

theorem gain_percent_calculation (cost_price selling_price : ℝ) : 
  cost_price = 600 → 
  selling_price = 1080 → 
  (selling_price - cost_price) / cost_price * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2391_239126


namespace NUMINAMATH_CALUDE_angle_measure_proof_l2391_239139

theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles are supplementary
  C = 9 * D →    -- C is 9 times D
  C = 162 :=     -- The measure of angle C is 162 degrees
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l2391_239139


namespace NUMINAMATH_CALUDE_amount_increases_to_approx_87030_l2391_239171

/-- The amount after two years given an initial amount and yearly increase rate. -/
def amount_after_two_years (initial_amount : ℝ) (yearly_increase_rate : ℝ) : ℝ :=
  initial_amount * (1 + yearly_increase_rate)^2

/-- Theorem stating that given an initial amount of 64000 that increases by 1/6th each year,
    the amount after two years is approximately 87030.40. -/
theorem amount_increases_to_approx_87030 :
  let initial_amount := 64000
  let yearly_increase_rate := 1 / 6
  let final_amount := amount_after_two_years initial_amount yearly_increase_rate
  ∃ ε > 0, |final_amount - 87030.40| < ε :=
sorry

end NUMINAMATH_CALUDE_amount_increases_to_approx_87030_l2391_239171


namespace NUMINAMATH_CALUDE_inequality_proof_l2391_239194

theorem inequality_proof (x y z t : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0) (non_neg_t : t ≥ 0)
  (sum_condition : x + y + z + t = 7) : 
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (t^2 + 64) + Real.sqrt (z^2 + t^2) ≥ 17 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2391_239194


namespace NUMINAMATH_CALUDE_marble_probability_l2391_239114

theorem marble_probability (green yellow white : ℕ) 
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 6) :
  (green + yellow : ℚ) / (green + yellow + white) = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l2391_239114


namespace NUMINAMATH_CALUDE_seashell_ratio_l2391_239118

def seashells_day1 : ℕ := 5
def seashells_day2 : ℕ := 7
def total_seashells : ℕ := 36

def seashells_first_two_days : ℕ := seashells_day1 + seashells_day2
def seashells_day3 : ℕ := total_seashells - seashells_first_two_days

theorem seashell_ratio :
  seashells_day3 / seashells_first_two_days = 2 := by sorry

end NUMINAMATH_CALUDE_seashell_ratio_l2391_239118


namespace NUMINAMATH_CALUDE_adlai_animal_legs_l2391_239192

/-- The number of legs for each animal type --/
def dogsLegs : ℕ := 4
def chickenLegs : ℕ := 2
def catsLegs : ℕ := 4
def spidersLegs : ℕ := 8
def octopusLegs : ℕ := 0

/-- Adlai's animal collection --/
def numDogs : ℕ := 2
def numChickens : ℕ := 1
def numCats : ℕ := 3
def numSpiders : ℕ := 4
def numOctopuses : ℕ := 5

/-- The total number of animal legs in Adlai's collection --/
def totalLegs : ℕ := 
  numDogs * dogsLegs + 
  numChickens * chickenLegs + 
  numCats * catsLegs + 
  numSpiders * spidersLegs + 
  numOctopuses * octopusLegs

theorem adlai_animal_legs : totalLegs = 54 := by
  sorry

end NUMINAMATH_CALUDE_adlai_animal_legs_l2391_239192
