import Mathlib

namespace simplest_fraction_sum_l733_73382

theorem simplest_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / b = 0.4375 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / d = 0.4375 → a ≤ c ∧ b ≤ d →
  a + b = 23 := by
sorry

end simplest_fraction_sum_l733_73382


namespace least_integer_with_divisibility_l733_73375

def n : ℕ := 2329089562800

theorem least_integer_with_divisibility (k : ℕ) (hk : k < n) : 
  (∀ i ∈ Finset.range 18, n % (i + 1) = 0) ∧ 
  (∀ i ∈ Finset.range 10, n % (i + 21) = 0) ∧ 
  n % 19 ≠ 0 ∧ 
  n % 20 ≠ 0 → 
  ¬(∀ i ∈ Finset.range 18, k % (i + 1) = 0) ∨ 
  ¬(∀ i ∈ Finset.range 10, k % (i + 21) = 0) ∨ 
  k % 19 = 0 ∨ 
  k % 20 = 0 :=
by sorry

#check least_integer_with_divisibility

end least_integer_with_divisibility_l733_73375


namespace lives_per_player_l733_73321

theorem lives_per_player (initial_players : ℕ) (quitting_players : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8)
  (h2 : quitting_players = 5)
  (h3 : total_lives = 15)
  (h4 : initial_players > quitting_players) :
  total_lives / (initial_players - quitting_players) = 5 := by
  sorry

end lives_per_player_l733_73321


namespace pencils_purchased_l733_73383

theorem pencils_purchased (num_pens : ℕ) (total_cost : ℝ) (pencil_price : ℝ) (pen_price : ℝ) :
  num_pens = 30 →
  total_cost = 690 →
  pencil_price = 2 →
  pen_price = 18 →
  (total_cost - num_pens * pen_price) / pencil_price = 75 := by
  sorry

end pencils_purchased_l733_73383


namespace cookie_sales_value_l733_73372

theorem cookie_sales_value (total_boxes : ℝ) (plain_boxes : ℝ) 
  (choc_chip_price : ℝ) (plain_price : ℝ) 
  (h1 : total_boxes = 1585) 
  (h2 : plain_boxes = 793.125) 
  (h3 : choc_chip_price = 1.25) 
  (h4 : plain_price = 0.75) : 
  (total_boxes - plain_boxes) * choc_chip_price + plain_boxes * plain_price = 1584.6875 := by
  sorry

end cookie_sales_value_l733_73372


namespace sqrt_a_div_sqrt_b_l733_73376

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = (25*a / (73*b)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end sqrt_a_div_sqrt_b_l733_73376


namespace nitrogen_electron_count_hydrazine_N2O4_reaction_hydrazine_combustion_l733_73308

-- Define the chemical reactions and their enthalpies
def reaction1_enthalpy : ℝ := -19.5
def reaction2_enthalpy : ℝ := -534.2
def reaction3_enthalpy : ℝ := 44.0

-- Define the number of electrons in the L shell of a nitrogen atom
def nitrogen_L_shell_electrons : ℕ := 5

-- Define the enthalpy of the reaction between hydrazine and N₂O₄
def hydrazine_N2O4_reaction_enthalpy : ℝ := -1048.9

-- Define the combustion heat of hydrazine
def hydrazine_combustion_heat : ℝ := -622.2

-- Theorem statements
theorem nitrogen_electron_count :
  nitrogen_L_shell_electrons = 5 := by sorry

theorem hydrazine_N2O4_reaction :
  hydrazine_N2O4_reaction_enthalpy = 2 * reaction2_enthalpy - reaction1_enthalpy := by sorry

theorem hydrazine_combustion :
  hydrazine_combustion_heat = reaction2_enthalpy - 2 * reaction3_enthalpy := by sorry

end nitrogen_electron_count_hydrazine_N2O4_reaction_hydrazine_combustion_l733_73308


namespace determinant_equality_l733_73337

theorem determinant_equality (a x y : ℝ) : 
  Matrix.det ![![1, x^2, y], ![1, a*x + y, y^2], ![1, x^2, a*x + y]] = 
    a^2*x^2 + 2*a*x*y + y^2 - a*x^3 - x*y^2 := by sorry

end determinant_equality_l733_73337


namespace greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63_l733_73390

def reverse_number (n : ℕ) : ℕ :=
  let digits := Nat.digits 10 n
  Nat.ofDigits 10 (List.reverse digits)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63 :
  ∃ (p : ℕ),
    is_four_digit p ∧
    p % 63 = 0 ∧
    (reverse_number p) % 63 = 0 ∧
    p % 11 = 0 ∧
    ∀ (q : ℕ),
      is_four_digit q ∧
      q % 63 = 0 ∧
      (reverse_number q) % 63 = 0 ∧
      q % 11 = 0 →
      q ≤ p ∧
    p = 7623 :=
by
  sorry

end greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63_l733_73390


namespace custom_op_properties_l733_73344

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 2 * a * b

-- State the theorem
theorem custom_op_properties :
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → custom_op a b = 2 * a * b) →
  (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → custom_op a b = custom_op b a) ∧
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 → custom_op a (custom_op b c) = custom_op (custom_op a b) c) ∧
  (∀ a : ℝ, a ≠ 0 → custom_op a (1/2) = a) ∧
  (∀ a : ℝ, a ≠ 0 → custom_op a (1/(2*a)) ≠ 1/2) :=
by sorry


end custom_op_properties_l733_73344


namespace negative_expressions_count_l733_73305

theorem negative_expressions_count : 
  let expressions := [-(-5), -|(-5)|, -(5^2), (-5)^2, 1/(-5)]
  (expressions.filter (λ x => x < 0)).length = 3 := by
sorry

end negative_expressions_count_l733_73305


namespace investment_growth_l733_73312

-- Define the initial investment
def initial_investment : ℝ := 359

-- Define the interest rate
def interest_rate : ℝ := 0.12

-- Define the number of years
def years : ℕ := 3

-- Define the final amount
def final_amount : ℝ := 504.32

-- Theorem statement
theorem investment_growth :
  initial_investment * (1 + interest_rate) ^ years = final_amount := by
  sorry

end investment_growth_l733_73312


namespace quadratic_inequality_l733_73357

/-- A quadratic function of the form y = a(x-3)² + c where a < 0 -/
def quadratic_function (a c : ℝ) (h : a < 0) : ℝ → ℝ := 
  fun x => a * (x - 3)^2 + c

theorem quadratic_inequality (a c : ℝ) (h : a < 0) :
  let f := quadratic_function a c h
  let y₁ := f (Real.sqrt 5)
  let y₂ := f 0
  let y₃ := f 4
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end quadratic_inequality_l733_73357


namespace least_sum_of_bases_l733_73378

theorem least_sum_of_bases (a b : ℕ+) : 
  (4 * a.val + 7 = 7 * b.val + 4) →  -- 47 in base a equals 74 in base b
  (∀ (x y : ℕ+), (4 * x.val + 7 = 7 * y.val + 4) → (x.val + y.val ≥ a.val + b.val)) →
  (a.val + b.val = 24) :=
by sorry

end least_sum_of_bases_l733_73378


namespace lines_intersection_l733_73396

-- Define the two lines
def line1 (s : ℚ) : ℚ × ℚ := (1 + 2*s, 4 - 3*s)
def line2 (v : ℚ) : ℚ × ℚ := (-2 + 3*v, 6 - v)

-- Define the intersection point
def intersection_point : ℚ × ℚ := (17/11, 35/11)

-- Theorem statement
theorem lines_intersection :
  ∃ (s v : ℚ), line1 s = line2 v ∧ line1 s = intersection_point :=
by sorry

end lines_intersection_l733_73396


namespace shepherd_sheep_problem_l733_73328

/-- The number of sheep with the shepherd boy on the mountain -/
def x : ℕ := 20

/-- The number of sheep with the shepherd boy at the foot of the mountain -/
def y : ℕ := 12

/-- Theorem stating that the given numbers of sheep satisfy the problem conditions -/
theorem shepherd_sheep_problem :
  (x - 4 = y + 4) ∧ (x + 4 = 3 * (y - 4)) := by
  sorry

#check shepherd_sheep_problem

end shepherd_sheep_problem_l733_73328


namespace pentagon_area_form_pentagon_area_sum_l733_73350

/-- A pentagon constructed from 15 line segments of length 3 -/
structure Pentagon :=
  (F G H I J : ℝ × ℝ)
  (segments : List (ℝ × ℝ))
  (segment_length : ℝ)
  (segment_count : ℕ)
  (is_valid : segment_count = 15 ∧ segment_length = 3)

/-- The area of the pentagon -/
def pentagon_area (p : Pentagon) : ℝ := sorry

/-- The area can be expressed as √p + √q where p and q are positive integers -/
theorem pentagon_area_form (p : Pentagon) : 
  ∃ (a b : ℕ), pentagon_area p = Real.sqrt a + Real.sqrt b ∧ a > 0 ∧ b > 0 := sorry

/-- The sum of p and q is 48 -/
theorem pentagon_area_sum (p : Pentagon) :
  ∃ (a b : ℕ), pentagon_area p = Real.sqrt a + Real.sqrt b ∧ a > 0 ∧ b > 0 ∧ a + b = 48 := sorry

end pentagon_area_form_pentagon_area_sum_l733_73350


namespace als_original_portion_l733_73356

theorem als_original_portion (al betty clare : ℕ) : 
  al + betty + clare = 1200 →
  al ≠ betty →
  al ≠ clare →
  betty ≠ clare →
  al - 150 + 3 * betty + 3 * clare = 1800 →
  al = 825 := by
sorry

end als_original_portion_l733_73356


namespace intersection_at_origin_l733_73345

/-- A line in the coordinate plane --/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The origin point (0, 0) --/
def origin : ℝ × ℝ := (0, 0)

/-- Check if a point lies on a line --/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  p.2 = l.slope * p.1 + (l.point.2 - l.slope * l.point.1)

theorem intersection_at_origin 
  (k : Line)
  (l : Line)
  (hk_slope : k.slope = 1/2)
  (hk_origin : pointOnLine k origin)
  (hl_slope : l.slope = -2)
  (hl_point : l.point = (-2, 4)) :
  ∃ (p : ℝ × ℝ), pointOnLine k p ∧ pointOnLine l p ∧ p = origin :=
sorry

#check intersection_at_origin

end intersection_at_origin_l733_73345


namespace equation_holds_iff_base_ten_l733_73349

/-- Represents a digit in base k --/
def Digit (k : ℕ) := Fin k

/-- Converts a natural number to its representation in base k --/
def toBaseK (n : ℕ) (k : ℕ) : List (Digit k) :=
  sorry

/-- Adds two numbers represented in base k --/
def addBaseK (a b : List (Digit k)) : List (Digit k) :=
  sorry

/-- Converts a list of digits in base k to a natural number --/
def fromBaseK (digits : List (Digit k)) (k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the equation holds iff k = 10 --/
theorem equation_holds_iff_base_ten (k : ℕ) :
  (fromBaseK (addBaseK (toBaseK 5342 k) (toBaseK 6421 k)) k = fromBaseK (toBaseK 14163 k) k) ↔ k = 10 :=
sorry

end equation_holds_iff_base_ten_l733_73349


namespace third_person_weight_is_131_l733_73394

/-- Calculates the true weight of the third person (C) entering an elevator given the following conditions:
    - There are initially 6 people in the elevator with an average weight of 156 lbs.
    - Three people (A, B, C) enter the elevator one by one.
    - The weights of their clothing and backpacks are 18 lbs, 20 lbs, and 22 lbs respectively.
    - After each person enters, the average weight changes to 159 lbs, 162 lbs, and 161 lbs respectively. -/
def calculate_third_person_weight (initial_people : Nat) (initial_avg : Nat)
  (a_extra_weight : Nat) (b_extra_weight : Nat) (c_extra_weight : Nat)
  (avg_after_a : Nat) (avg_after_b : Nat) (avg_after_c : Nat) : Nat :=
  let total_initial := initial_people * initial_avg
  let total_after_a := (initial_people + 1) * avg_after_a
  let total_after_b := (initial_people + 2) * avg_after_b
  let total_after_c := (initial_people + 3) * avg_after_c
  total_after_c - total_after_b - c_extra_weight

/-- Theorem stating that given the conditions in the problem, 
    the true weight of the third person (C) is 131 lbs. -/
theorem third_person_weight_is_131 :
  calculate_third_person_weight 6 156 18 20 22 159 162 161 = 131 := by
  sorry

end third_person_weight_is_131_l733_73394


namespace not_prime_sum_minus_one_l733_73302

theorem not_prime_sum_minus_one (m n : ℤ) 
  (hm : m > 1) 
  (hn : n > 1) 
  (h_divides : (m + n - 1) ∣ (m^2 + n^2 - 1)) : 
  ¬(Nat.Prime (m + n - 1).natAbs) := by
sorry

end not_prime_sum_minus_one_l733_73302


namespace function_difference_equals_nine_minimum_value_minus_four_l733_73307

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 3

-- Theorem 1
theorem function_difference_equals_nine (a : ℝ) :
  f a (a + 1) - f a a = 9 → a = 2 :=
sorry

-- Theorem 2
theorem minimum_value_minus_four (a : ℝ) :
  (∃ x, f a x = -4 ∧ ∀ y, f a y ≥ -4) → (a = 1 ∨ a = -1) :=
sorry

end function_difference_equals_nine_minimum_value_minus_four_l733_73307


namespace cost_price_is_40_l733_73393

/-- Calculates the cost price per metre of cloth given the total length, 
    total selling price, and loss per metre. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Proves that the cost price per metre is 40 given the specified conditions. -/
theorem cost_price_is_40 :
  cost_price_per_metre 500 15000 10 = 40 := by
  sorry

end cost_price_is_40_l733_73393


namespace carls_marbles_l733_73306

theorem carls_marbles (initial_marbles : ℕ) : 
  (initial_marbles / 2 + 10 + 25 = 41) → initial_marbles = 12 := by
  sorry

end carls_marbles_l733_73306


namespace second_discount_percentage_l733_73346

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 400 →
  first_discount = 10 →
  final_price = 331.2 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 8 :=
by sorry

end second_discount_percentage_l733_73346


namespace perpendicular_lines_parallel_l733_73395

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end perpendicular_lines_parallel_l733_73395


namespace total_bottles_l733_73341

theorem total_bottles (regular_soda : ℕ) (diet_soda : ℕ) 
  (h1 : regular_soda = 30) (h2 : diet_soda = 8) : 
  regular_soda + diet_soda = 38 := by
  sorry

end total_bottles_l733_73341


namespace intersection_points_are_two_and_eight_l733_73369

/-- The set of k values for which |z - 4| = 3|z + 4| intersects |z| = k at exactly one point -/
def intersection_points : Set ℝ :=
  {k : ℝ | ∃! z : ℂ, Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k}

/-- Theorem stating that the intersection_points set contains only 2 and 8 -/
theorem intersection_points_are_two_and_eight :
  intersection_points = {2, 8} := by
  sorry

end intersection_points_are_two_and_eight_l733_73369


namespace prob_even_diagonals_eq_one_over_101_l733_73338

/-- Represents a 3x3 grid filled with numbers 1 to 9 --/
def Grid := Fin 9 → Fin 9

/-- Checks if a given grid has even sums on both diagonals --/
def has_even_diagonal_sums (g : Grid) : Prop :=
  (g 0 + g 4 + g 8) % 2 = 0 ∧ (g 2 + g 4 + g 6) % 2 = 0

/-- The set of all valid grids --/
def all_grids : Finset Grid :=
  sorry

/-- The set of grids with even diagonal sums --/
def even_sum_grids : Finset Grid :=
  sorry

/-- The probability of having even sums on both diagonals --/
def prob_even_diagonals : ℚ :=
  (Finset.card even_sum_grids : ℚ) / (Finset.card all_grids : ℚ)

theorem prob_even_diagonals_eq_one_over_101 : 
  prob_even_diagonals = 1 / 101 :=
sorry

end prob_even_diagonals_eq_one_over_101_l733_73338


namespace original_price_calculation_l733_73334

def discount_rate : ℝ := 0.2
def discounted_price : ℝ := 56

theorem original_price_calculation :
  ∃ (original_price : ℝ), 
    original_price * (1 - discount_rate) = discounted_price ∧ 
    original_price = 70 :=
by sorry

end original_price_calculation_l733_73334


namespace contrapositive_equivalence_l733_73397

theorem contrapositive_equivalence (f : ℝ → ℝ) (a : ℝ) :
  (a ≥ (1/2) → ∀ x ≥ 0, f x ≥ 0) ↔
  (∃ x ≥ 0, f x < 0 → a < (1/2)) :=
sorry

end contrapositive_equivalence_l733_73397


namespace complex_equation_solution_l733_73384

theorem complex_equation_solution (a b c : ℂ) (h_real : a.im = 0) 
  (h_sum : a + b + c = 5)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 5) :
  a = 4 := by sorry

end complex_equation_solution_l733_73384


namespace function_minimum_implies_parameter_range_l733_73326

/-- Given a function f(x) with parameter a > 0, if its minimum value is ln²(a) + 3ln(a) + 2,
    then a ≥ e^(-3/2) -/
theorem function_minimum_implies_parameter_range (a : ℝ) (h_a_pos : a > 0) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = a^2 * Real.exp (-2*x) + a * (2*x + 1) * Real.exp (-x) + x^2 + x) ∧
    (∀ x, f x ≥ Real.log a ^ 2 + 3 * Real.log a + 2) ∧
    (∃ x₀, f x₀ = Real.log a ^ 2 + 3 * Real.log a + 2)) →
  a ≥ Real.exp (-3/2) := by
  sorry

end function_minimum_implies_parameter_range_l733_73326


namespace arrival_time_difference_l733_73355

def distance : ℝ := 1.2
def jill_speed : ℝ := 8
def jack_speed : ℝ := 5

theorem arrival_time_difference : 
  let jill_time := distance / jill_speed
  let jack_time := distance / jack_speed
  (jack_time - jill_time) * 60 = 5.4 := by sorry

end arrival_time_difference_l733_73355


namespace pony_price_is_18_l733_73309

/-- The regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

/-- The discount rate for Pony jeans as a decimal -/
def pony_discount : ℝ := 0.1

/-- The sum of discount rates for Fox and Pony jeans as a decimal -/
def total_discount : ℝ := 0.22

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings in dollars from purchasing both types of jeans -/
def total_savings : ℝ := 9

/-- Theorem stating that the regular price of Pony jeans is $18 -/
theorem pony_price_is_18 : 
  ∃ (pony_price : ℝ), 
    pony_price = 18 ∧ 
    (fox_quantity * fox_price * (total_discount - pony_discount) + 
     pony_quantity * pony_price * pony_discount = total_savings) :=
by sorry

end pony_price_is_18_l733_73309


namespace expression_simplification_and_evaluation_expression_evaluation_at_negative_one_l733_73364

theorem expression_simplification_and_evaluation :
  ∀ a : ℝ, a ≠ 1 ∧ a ≠ 2 ∧ a ≠ 0 →
    (a - 3 + 1 / (a - 1)) / ((a^2 - 4) / (a^2 + 2*a)) * (1 / (a - 2)) = a / (a - 1) :=
by sorry

theorem expression_evaluation_at_negative_one :
  (-1 - 3 + 1 / (-1 - 1)) / (((-1)^2 - 4) / ((-1)^2 + 2*(-1))) * (1 / (-1 - 2)) = 1 / 2 :=
by sorry

end expression_simplification_and_evaluation_expression_evaluation_at_negative_one_l733_73364


namespace range_of_a_minus_b_l733_73324

theorem range_of_a_minus_b (a b : ℝ) (ha : -2 < a ∧ a < 1) (hb : 0 < b ∧ b < 4) :
  ∀ x, (∃ (a' b' : ℝ), -2 < a' ∧ a' < 1 ∧ 0 < b' ∧ b' < 4 ∧ x = a' - b') ↔ -6 < x ∧ x < 1 :=
by sorry

end range_of_a_minus_b_l733_73324


namespace range_of_H_l733_73354

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ 3 ≤ y ∧ y ≤ 5 :=
by sorry

end range_of_H_l733_73354


namespace test_question_points_l733_73318

theorem test_question_points : 
  ∀ (other_point_value : ℕ),
    (40 : ℕ) = 10 + (100 - 10 * 4) / other_point_value →
    other_point_value = 2 :=
by
  sorry

end test_question_points_l733_73318


namespace max_min_product_l733_73353

theorem max_min_product (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 2 ∧ 
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 2 :=
sorry

end max_min_product_l733_73353


namespace derivative_of_even_is_odd_l733_73335

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem derivative_of_even_is_odd
  (f : ℝ → ℝ) (hf : IsEven f) (g : ℝ → ℝ) (hg : ∀ x, HasDerivAt f (g x) x) :
  ∀ x, g (-x) = -g x :=
sorry

end derivative_of_even_is_odd_l733_73335


namespace quadratic_rational_root_even_coefficient_l733_73340

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : ∃ (x : ℚ), a * x^2 + b * x + c = 0) :
  Even a ∨ Even b ∨ Even c :=
sorry

end quadratic_rational_root_even_coefficient_l733_73340


namespace absolute_value_equation_solution_l733_73365

theorem absolute_value_equation_solution (x y : ℝ) :
  |x - Real.log (y^2)| = x + Real.log (y^2) →
  x = 0 ∧ (y = 1 ∨ y = -1) :=
by sorry

end absolute_value_equation_solution_l733_73365


namespace arithmetic_sequence_count_l733_73374

theorem arithmetic_sequence_count (a₁ a_n d : ℤ) (h1 : a₁ = 156) (h2 : a_n = 36) (h3 : d = -4) :
  (a₁ - a_n) / d + 1 = 31 := by
  sorry

end arithmetic_sequence_count_l733_73374


namespace center_is_midpoint_distance_between_foci_l733_73325

/-- The equation of an ellipse with foci at (6, -3) and (-4, 5) -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 6)^2 + (y + 3)^2) + Real.sqrt ((x + 4)^2 + (y - 5)^2) = 24

/-- The center of the ellipse -/
def center : ℝ × ℝ := (1, 1)

/-- The first focus of the ellipse -/
def focus1 : ℝ × ℝ := (6, -3)

/-- The second focus of the ellipse -/
def focus2 : ℝ × ℝ := (-4, 5)

/-- The center is the midpoint of the foci -/
theorem center_is_midpoint : center = ((focus1.1 + focus2.1) / 2, (focus1.2 + focus2.2) / 2) := by sorry

/-- The distance between the foci of the ellipse is 2√41 -/
theorem distance_between_foci : 
  Real.sqrt ((focus1.1 - focus2.1)^2 + (focus1.2 - focus2.2)^2) = 2 * Real.sqrt 41 := by sorry

end center_is_midpoint_distance_between_foci_l733_73325


namespace circular_seating_arrangement_l733_73361

/-- Given a circular seating arrangement where the 7th person is directly opposite the 27th person,
    prove that the total number of people in the circle is 40. -/
theorem circular_seating_arrangement (n : ℕ) : n = 40 := by
  sorry

end circular_seating_arrangement_l733_73361


namespace opposite_numbers_theorem_l733_73387

theorem opposite_numbers_theorem (a : ℚ) : (4 * a + 9) + (3 * a + 5) = 0 → a = -2 := by
  sorry

end opposite_numbers_theorem_l733_73387


namespace monic_polynomial_theorem_l733_73360

def is_monic_degree_7 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e f g : ℝ, ∀ x, p x = x^7 + a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + g

def satisfies_conditions (p : ℝ → ℝ) : Prop :=
  p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5 ∧ p 6 = 6 ∧ p 7 = 7

theorem monic_polynomial_theorem (p : ℝ → ℝ) 
  (h1 : is_monic_degree_7 p) 
  (h2 : satisfies_conditions p) : 
  p 8 = 5048 := by
  sorry

end monic_polynomial_theorem_l733_73360


namespace xy_value_l733_73317

theorem xy_value (x y : ℝ) (h : x * (x + 3 * y) = x^2 + 24) : 3 * x * y = 24 := by
  sorry

end xy_value_l733_73317


namespace repeating_decimal_fraction_sum_l733_73327

-- Define the repeating decimal
def repeating_decimal : ℚ := 7 + 17 / 990

-- Theorem statement
theorem repeating_decimal_fraction_sum :
  (repeating_decimal = 710 / 99) ∧
  (710 + 99 = 809) :=
by sorry

end repeating_decimal_fraction_sum_l733_73327


namespace fuel_a_amount_proof_l733_73380

-- Define the tank capacity
def tank_capacity : ℝ := 200

-- Define the ethanol content percentages
def ethanol_content_a : ℝ := 0.12
def ethanol_content_b : ℝ := 0.16

-- Define the total ethanol in the full tank
def total_ethanol : ℝ := 28

-- Define the amount of fuel A added (to be proved)
def fuel_a_added : ℝ := 100

-- Theorem statement
theorem fuel_a_amount_proof :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ 
    x ≤ tank_capacity ∧
    ethanol_content_a * x + ethanol_content_b * (tank_capacity - x) = total_ethanol ∧
    x = fuel_a_added :=
by
  sorry


end fuel_a_amount_proof_l733_73380


namespace negation_equivalence_l733_73399

variable (a : ℝ)

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) := by
  sorry

end negation_equivalence_l733_73399


namespace cylinder_minus_cones_volume_l733_73304

/-- The volume of a cylinder minus two cones -/
theorem cylinder_minus_cones_volume (r h : ℝ) (hr : r = 15) (hh : h = 30) :
  π * r^2 * h - 2 * (1/3 * π * r^2 * (h/2)) = 4500 * π := by
  sorry

#check cylinder_minus_cones_volume

end cylinder_minus_cones_volume_l733_73304


namespace circle_diameter_l733_73363

theorem circle_diameter (r : ℝ) (h : r = 4) : 2 * r = 8 := by
  sorry

end circle_diameter_l733_73363


namespace division_problem_l733_73313

theorem division_problem (divisor : ℕ) : 
  (265 / divisor = 12) ∧ (265 % divisor = 1) → divisor = 22 := by
  sorry

end division_problem_l733_73313


namespace molecular_weight_proof_l733_73310

/-- Given 4 moles of a compound with a total molecular weight of 304 g/mol,
    prove that the molecular weight of 1 mole of the compound is 76 g/mol. -/
theorem molecular_weight_proof (total_weight : ℝ) (total_moles : ℝ) 
  (h1 : total_weight = 304)
  (h2 : total_moles = 4) :
  total_weight / total_moles = 76 := by
  sorry

end molecular_weight_proof_l733_73310


namespace probability_of_white_ball_l733_73386

theorem probability_of_white_ball (P_red P_black P_yellow P_white : ℚ) : 
  P_red = 1/3 →
  P_black + P_yellow = 5/12 →
  P_yellow + P_white = 5/12 →
  P_red + P_black + P_yellow + P_white = 1 →
  P_white = 1/4 := by
sorry

end probability_of_white_ball_l733_73386


namespace sticker_distribution_l733_73385

/-- The number of stickers Mary bought initially -/
def total_stickers : ℕ := 1500

/-- Susan's share of stickers -/
def susan_share : ℕ := 300

/-- Andrew's initial share of stickers -/
def andrew_initial_share : ℕ := 300

/-- Sam's initial share of stickers -/
def sam_initial_share : ℕ := 900

/-- The amount of stickers Sam gave to Andrew -/
def sam_to_andrew : ℕ := 600

/-- Andrew's final share of stickers -/
def andrew_final_share : ℕ := 900

theorem sticker_distribution :
  -- The total is the sum of all initial shares
  total_stickers = susan_share + andrew_initial_share + sam_initial_share ∧
  -- The ratio of shares is 1:1:3
  susan_share = andrew_initial_share ∧
  sam_initial_share = 3 * andrew_initial_share ∧
  -- Sam gave Andrew two-thirds of his share
  sam_to_andrew = 2 * sam_initial_share / 3 ∧
  -- Andrew's final share is his initial plus what Sam gave him
  andrew_final_share = andrew_initial_share + sam_to_andrew :=
by sorry

end sticker_distribution_l733_73385


namespace max_table_height_specific_triangle_l733_73332

/-- Triangle ABC with sides a, b, and c -/
structure Triangle (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α

/-- The maximum height of a table constructed from a triangle -/
def maxTableHeight {α : Type*} [LinearOrderedField α] (t : Triangle α) : α :=
  sorry

/-- The theorem stating the maximum height of the table -/
theorem max_table_height_specific_triangle :
  let t : Triangle ℝ := ⟨25, 29, 32⟩
  maxTableHeight t = 84 * Real.sqrt 1547 / 57 := by
  sorry

end max_table_height_specific_triangle_l733_73332


namespace history_book_cost_l733_73314

theorem history_book_cost 
  (total_books : ℕ) 
  (math_book_cost : ℕ) 
  (total_price : ℕ) 
  (math_books : ℕ) 
  (h1 : total_books = 80) 
  (h2 : math_book_cost = 4) 
  (h3 : total_price = 368) 
  (h4 : math_books = 32) : 
  (total_price - math_books * math_book_cost) / (total_books - math_books) = 5 := by
sorry

end history_book_cost_l733_73314


namespace condition_A_right_triangle_condition_B_right_triangle_condition_C_not_right_triangle_condition_D_right_triangle_l733_73330

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define functions to calculate side lengths and angles
def side_length (p q : ℝ × ℝ) : ℝ := sorry
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a triangle is right-angled
def is_right_triangle (t : Triangle) : Prop :=
  let a := side_length t.A t.B
  let b := side_length t.B t.C
  let c := side_length t.C t.A
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

-- Theorem for condition A
theorem condition_A_right_triangle (t : Triangle) :
  side_length t.A t.B = 3 ∧ side_length t.B t.C = 4 ∧ side_length t.C t.A = 5 →
  is_right_triangle t :=
sorry

-- Theorem for condition B
theorem condition_B_right_triangle (t : Triangle) (k : ℝ) :
  side_length t.A t.B = 3*k ∧ side_length t.B t.C = 4*k ∧ side_length t.C t.A = 5*k →
  is_right_triangle t :=
sorry

-- Theorem for condition C
theorem condition_C_not_right_triangle (t : Triangle) :
  ∃ (k : ℝ), angle t.B t.A t.C = 3*k ∧ angle t.C t.B t.A = 4*k ∧ angle t.A t.C t.B = 5*k →
  ¬ is_right_triangle t :=
sorry

-- Theorem for condition D
theorem condition_D_right_triangle (t : Triangle) :
  angle t.B t.A t.C = 40 ∧ angle t.C t.B t.A = 50 →
  is_right_triangle t :=
sorry

end condition_A_right_triangle_condition_B_right_triangle_condition_C_not_right_triangle_condition_D_right_triangle_l733_73330


namespace discounted_tickets_count_l733_73323

/-- Proves the number of discounted tickets bought given the problem conditions -/
theorem discounted_tickets_count :
  ∀ (full_price discounted_price : ℚ) 
    (total_tickets : ℕ) 
    (total_spent : ℚ),
  full_price = 2 →
  discounted_price = (8 : ℚ) / 5 →
  total_tickets = 10 →
  total_spent = (92 : ℚ) / 5 →
  ∃ (full_tickets discounted_tickets : ℕ),
    full_tickets + discounted_tickets = total_tickets ∧
    full_price * full_tickets + discounted_price * discounted_tickets = total_spent ∧
    discounted_tickets = 4 :=
by
  sorry

end discounted_tickets_count_l733_73323


namespace smaller_circle_circumference_l733_73301

theorem smaller_circle_circumference :
  ∀ (r R s d : ℝ),
  s^2 = 784 →
  s = 2 * R →
  d = r + R →
  R = (7/3) * r →
  2 * π * r = 12 * π :=
by
  sorry

end smaller_circle_circumference_l733_73301


namespace zero_in_M_l733_73333

def M : Set ℕ := {0, 1, 2}

theorem zero_in_M : 0 ∈ M := by sorry

end zero_in_M_l733_73333


namespace electrocardiogram_is_line_chart_l733_73300

/-- Represents different types of charts --/
inductive ChartType
  | BarChart
  | LineChart
  | PieChart

/-- Represents a chart that can display data --/
structure Chart where
  type : ChartType
  representsChangesOverTime : Bool

/-- Defines an electrocardiogram as a chart --/
def Electrocardiogram : Chart :=
  { type := ChartType.LineChart,
    representsChangesOverTime := true }

/-- Theorem stating that an electrocardiogram is a line chart --/
theorem electrocardiogram_is_line_chart : 
  Electrocardiogram.type = ChartType.LineChart :=
by
  sorry


end electrocardiogram_is_line_chart_l733_73300


namespace x_squared_mod_20_l733_73336

theorem x_squared_mod_20 (x : ℤ) 
  (h1 : 4 * x ≡ 8 [ZMOD 20]) 
  (h2 : 3 * x ≡ 16 [ZMOD 20]) : 
  x^2 ≡ 4 [ZMOD 20] := by
sorry

end x_squared_mod_20_l733_73336


namespace fox_jeans_price_l733_73329

/-- Regular price of Pony jeans in dollars -/
def pony_price : ℝ := 18

/-- Total savings on 5 pairs of jeans (3 Fox, 2 Pony) in dollars -/
def total_savings : ℝ := 8.91

/-- Sum of discount rates for Fox and Pony jeans as a percentage -/
def total_discount_rate : ℝ := 22

/-- Discount rate on Pony jeans as a percentage -/
def pony_discount_rate : ℝ := 10.999999999999996

/-- Regular price of Fox jeans in dollars -/
def fox_price : ℝ := 15

theorem fox_jeans_price : 
  ∃ (fox_discount_rate : ℝ),
    fox_discount_rate + pony_discount_rate = total_discount_rate ∧
    3 * (fox_price * fox_discount_rate / 100) + 
    2 * (pony_price * pony_discount_rate / 100) = total_savings :=
by sorry

end fox_jeans_price_l733_73329


namespace complex_fraction_ratio_l733_73389

theorem complex_fraction_ratio (x : ℝ) : x = 200 → x / 10 = 20 := by
  sorry

end complex_fraction_ratio_l733_73389


namespace triangle_properties_l733_73347

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (6, 4)
def C : ℝ × ℝ := (4, 0)

-- Define the perpendicular bisector equation
def perpendicular_bisector (x y : ℝ) : Prop :=
  2 * x - y - 3 = 0

-- Define the circumcircle equation
def circumcircle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - (A.1 + C.1) / 2)^2 + (y - (A.2 + C.2) / 2)^2 = 
    ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 4) ∧
  (∀ x y : ℝ, circumcircle x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end triangle_properties_l733_73347


namespace matts_climbing_speed_l733_73392

/-- Prove Matt's climbing speed given Jason's speed and their height difference after 7 minutes -/
theorem matts_climbing_speed 
  (jason_speed : ℝ) 
  (time : ℝ) 
  (height_diff : ℝ) 
  (h1 : jason_speed = 12)
  (h2 : time = 7)
  (h3 : height_diff = 42) :
  ∃ (matt_speed : ℝ), 
    matt_speed = 6 ∧ 
    jason_speed * time = matt_speed * time + height_diff :=
by sorry

end matts_climbing_speed_l733_73392


namespace shares_ratio_l733_73379

/-- Represents the shares of money for three individuals -/
structure Shares where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The problem setup -/
def problem_setup (s : Shares) : Prop :=
  s.a + s.b + s.c = 700 ∧  -- Total amount
  s.a = 280 ∧              -- A's share
  ∃ x, s.a = x * (s.b + s.c) ∧  -- A's share as a fraction of B and C
  s.b = (6/9) * (s.a + s.c)     -- B's share as 6/9 of A and C

/-- The theorem to prove -/
theorem shares_ratio (s : Shares) (h : problem_setup s) : 
  s.a / (s.b + s.c) = 2/3 := by
  sorry


end shares_ratio_l733_73379


namespace x_satisfies_quadratic_l733_73368

theorem x_satisfies_quadratic (x y : ℝ) 
  (h1 : x^2 - y = 10) 
  (h2 : x + y = 14) : 
  x^2 + x - 24 = 0 := by
  sorry

end x_satisfies_quadratic_l733_73368


namespace triangle_properties_triangle_is_equilateral_l733_73367

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C
  angleA : ℝ  -- measure of angle A
  angleB : ℝ  -- measure of angle B
  angleC : ℝ  -- measure of angle C

-- Define the theorem
theorem triangle_properties (t : Triangle)
  (h1 : (t.a + t.b + t.c) * (t.a - t.b - t.c) + 3 * t.b * t.c = 0)
  (h2 : t.a = 2 * t.c * Real.cos t.angleB) :
  t.angleA = π / 3 ∧ t.angleB = t.angleC := by
  sorry

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Prove that the triangle is equilateral
theorem triangle_is_equilateral (t : Triangle)
  (h1 : (t.a + t.b + t.c) * (t.a - t.b - t.c) + 3 * t.b * t.c = 0)
  (h2 : t.a = 2 * t.c * Real.cos t.angleB) :
  is_equilateral t := by
  sorry

end triangle_properties_triangle_is_equilateral_l733_73367


namespace six_balls_four_boxes_l733_73388

/-- Represents a distribution of balls into boxes -/
def Distribution := List Nat

/-- Checks if a distribution is valid for the given number of balls and boxes -/
def is_valid_distribution (d : Distribution) (num_balls num_boxes : Nat) : Prop :=
  d.length ≤ num_boxes ∧ d.sum = num_balls ∧ d.all (· ≥ 0)

/-- Counts the number of distinct ways to distribute indistinguishable balls into indistinguishable boxes -/
def count_distributions (num_balls num_boxes : Nat) : Nat :=
  sorry

theorem six_balls_four_boxes :
  count_distributions 6 4 = 9 := by sorry

end six_balls_four_boxes_l733_73388


namespace average_age_after_leaving_l733_73352

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 5 →
  initial_avg = 30 →
  leaving_age = 18 →
  remaining_people = 4 →
  (initial_people * initial_avg - leaving_age) / remaining_people = 33 := by
  sorry

end average_age_after_leaving_l733_73352


namespace smallest_multiple_of_6_and_15_l733_73381

theorem smallest_multiple_of_6_and_15 :
  ∃ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ (x : ℕ), x > 0 ∧ 6 ∣ x ∧ 15 ∣ x → b ≤ x :=
by
  -- The proof goes here
  sorry

end smallest_multiple_of_6_and_15_l733_73381


namespace cyclists_meeting_time_l733_73358

theorem cyclists_meeting_time 
  (course_length : ℝ) 
  (speed1 speed2 : ℝ) 
  (h1 : course_length = 45)
  (h2 : speed1 = 14)
  (h3 : speed2 = 16) :
  ∃ t : ℝ, t * speed1 + t * speed2 = course_length ∧ t = 1.5 :=
by sorry

end cyclists_meeting_time_l733_73358


namespace remainder_is_x_squared_l733_73343

-- Define the polynomials
def f (x : ℝ) := x^1010
def g (x : ℝ) := (x^2 + 1) * (x + 1) * (x - 1)

-- Define the remainder function
noncomputable def remainder (x : ℝ) := f x % g x

-- Theorem statement
theorem remainder_is_x_squared :
  ∀ x : ℝ, remainder x = x^2 :=
by
  sorry

end remainder_is_x_squared_l733_73343


namespace halloween_candy_distribution_l733_73398

theorem halloween_candy_distribution (initial_candies given_away remaining_candies : ℕ) : 
  initial_candies = 60 → given_away = 40 → remaining_candies = initial_candies - given_away → remaining_candies = 20 := by
  sorry

end halloween_candy_distribution_l733_73398


namespace cos_angle_sum_diff_vectors_l733_73320

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (2, 2)

theorem cos_angle_sum_diff_vectors :
  let sum := (a.1 + b.1, a.2 + b.2)
  let diff := (a.1 - b.1, a.2 - b.2)
  (sum.1 * diff.1 + sum.2 * diff.2) / 
  (Real.sqrt (sum.1^2 + sum.2^2) * Real.sqrt (diff.1^2 + diff.2^2)) = 
  Real.sqrt 17 / 17 := by
  sorry

end cos_angle_sum_diff_vectors_l733_73320


namespace total_value_is_71_rupees_l733_73311

/-- Represents the value of a coin in paise -/
inductive CoinValue
  | paise20 : CoinValue
  | paise25 : CoinValue

/-- Calculates the total value in rupees given the number of coins and their values -/
def totalValueInRupees (totalCoins : ℕ) (coins20paise : ℕ) : ℚ :=
  let coins25paise := totalCoins - coins20paise
  let value20paise := 20 * coins20paise
  let value25paise := 25 * coins25paise
  (value20paise + value25paise : ℚ) / 100

/-- Theorem stating that the total value of the given coins is 71 rupees -/
theorem total_value_is_71_rupees :
  totalValueInRupees 334 250 = 71 := by
  sorry


end total_value_is_71_rupees_l733_73311


namespace circle_reflection_and_translation_l733_73339

def reflect_across_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

theorem circle_reflection_and_translation :
  let initial_center : ℝ × ℝ := (-3, -4)
  let reflected_center := reflect_across_x_axis initial_center
  let final_center := translate_up reflected_center 3
  final_center = (-3, 7) := by sorry

end circle_reflection_and_translation_l733_73339


namespace min_socks_for_twelve_pairs_l733_73359

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (black : ℕ)
  (yellow : ℕ)

/-- Represents the problem setup -/
def initialDrawer : SockDrawer :=
  { red := 120
  , green := 90
  , blue := 70
  , black := 50
  , yellow := 30 }

/-- The number of pairs we want to guarantee -/
def requiredPairs : ℕ := 12

/-- Function to calculate the minimum number of socks needed to guarantee the required pairs -/
def minSocksForPairs (drawer : SockDrawer) (pairs : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 28 socks are needed to guarantee 12 pairs -/
theorem min_socks_for_twelve_pairs :
  minSocksForPairs initialDrawer requiredPairs = 28 :=
sorry

end min_socks_for_twelve_pairs_l733_73359


namespace volume_of_cut_cone_l733_73351

/-- The volume of the cone cut to form a frustum, given the frustum's properties -/
theorem volume_of_cut_cone (r R h H : ℝ) : 
  (R = 3 * r) →  -- Area of one base is 9 times the other
  (H = 3 * h) →  -- Height ratio follows from radius ratio
  (π * R^2 * H / 3 - π * r^2 * h / 3 = 52) →  -- Volume of frustum is 52
  (π * r^2 * h / 3 = 54) :=  -- Volume of cut cone is 54
by sorry

end volume_of_cut_cone_l733_73351


namespace three_x_squared_y_squared_l733_73373

theorem three_x_squared_y_squared (x y : ℤ) 
  (h : y^2 + 3*x^2*y^2 = 30*x^2 + 553) : 
  3*x^2*y^2 = 2886 := by
  sorry

end three_x_squared_y_squared_l733_73373


namespace greatest_integer_a_for_quadratic_l733_73315

theorem greatest_integer_a_for_quadratic : 
  ∃ (a : ℤ), a = 6 ∧ 
  (∀ x : ℝ, x^2 + a*x + 9 ≠ -2) ∧
  (∀ b : ℤ, b > a → ∃ x : ℝ, x^2 + b*x + 9 = -2) :=
by sorry

end greatest_integer_a_for_quadratic_l733_73315


namespace quadrilateral_side_length_l733_73377

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def are_opposite (q : Quadrilateral) (v1 v2 : ℝ × ℝ) : Prop := sorry

def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_side_length 
  (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_opposite : are_opposite q q.A q.C)
  (h_BC : side_length q.B q.C = 4)
  (h_ADC : angle_measure q.A q.D q.C = π / 3)
  (h_BAD : angle_measure q.B q.A q.D = π / 2)
  (h_area : area q = (side_length q.A q.B * side_length q.C q.D + 
                      side_length q.B q.C * side_length q.A q.D) / 2) :
  side_length q.C q.D = 4 * Real.sqrt 3 := by
  sorry

end quadrilateral_side_length_l733_73377


namespace factorization_sum_l733_73391

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 64 * x^6 - 729 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 30 := by
  sorry

end factorization_sum_l733_73391


namespace remaining_money_l733_73342

def savings : ℕ := 5376
def ticket_cost : ℕ := 1350

def octal_to_decimal (n : ℕ) : ℕ := sorry

theorem remaining_money : 
  octal_to_decimal savings - ticket_cost = 1464 := by sorry

end remaining_money_l733_73342


namespace x_power_plus_reciprocal_l733_73348

theorem x_power_plus_reciprocal (θ : ℝ) (x : ℝ) (n : ℕ+) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : x + 1 / x = 2 * Real.sin θ) : 
  x^(n : ℝ) + 1 / x^(n : ℝ) = 2 * Real.cos (n * (π / 2 - θ)) := by
  sorry

end x_power_plus_reciprocal_l733_73348


namespace man_coin_value_l733_73316

/-- Represents the value of a coin in cents -/
def coin_value (is_nickel : Bool) : ℕ :=
  if is_nickel then 5 else 10

/-- Calculates the total value of coins in cents -/
def total_value (total_coins : ℕ) (nickel_count : ℕ) : ℕ :=
  (nickel_count * coin_value true) + ((total_coins - nickel_count) * coin_value false)

theorem man_coin_value :
  total_value 8 2 = 70 := by
  sorry

end man_coin_value_l733_73316


namespace number_of_divisors_of_36_l733_73366

theorem number_of_divisors_of_36 : Nat.card {d : ℕ | d > 0 ∧ 36 % d = 0} = 9 := by
  sorry

end number_of_divisors_of_36_l733_73366


namespace symmetric_point_coordinates_l733_73362

/-- Given a point P in polar coordinates, find its symmetric point with respect to the pole -/
theorem symmetric_point_coordinates (r : ℝ) (θ : ℝ) :
  let P : ℝ × ℝ := (r, θ)
  let symmetric_polar : ℝ × ℝ := (r, θ + π)
  let symmetric_cartesian : ℝ × ℝ := (r * Real.cos (θ + π), r * Real.sin (θ + π))
  P = (2, -5 * π / 3) →
  symmetric_polar = (2, -2 * π / 3) ∧
  symmetric_cartesian = (-1, -Real.sqrt 3) :=
by sorry


end symmetric_point_coordinates_l733_73362


namespace sum_of_distances_forms_ellipse_l733_73331

/-- Definition of an ellipse based on the sum of distances to two foci -/
def is_ellipse (F₁ F₂ : ℝ × ℝ) (a : ℝ) (S : Set (ℝ × ℝ)) : Prop :=
  ∃ c : ℝ, F₁ = (c, 0) ∧ F₂ = (-c, 0) ∧ a > c ∧ c > 0 ∧
  S = {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x - c)^2 + y^2) + Real.sqrt ((x + c)^2 + y^2) = 2 * a}

/-- Theorem: The set of points satisfying the sum of distances to two foci is an ellipse -/
theorem sum_of_distances_forms_ellipse (F₁ F₂ : ℝ × ℝ) (a : ℝ) (S : Set (ℝ × ℝ)) 
  (h : is_ellipse F₁ F₂ a S) : 
  ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ S = {p : ℝ × ℝ | let (x, y) := p; x^2 / a'^2 + y^2 / b'^2 = 1} :=
sorry

end sum_of_distances_forms_ellipse_l733_73331


namespace chloe_bookcase_problem_l733_73371

theorem chloe_bookcase_problem :
  let average_books_per_shelf : ℚ := 8.5
  let mystery_shelves : ℕ := 7
  let picture_shelves : ℕ := 5
  let scifi_shelves : ℕ := 3
  let history_shelves : ℕ := 2
  let total_shelves : ℕ := mystery_shelves + picture_shelves + scifi_shelves + history_shelves
  let total_books : ℚ := average_books_per_shelf * total_shelves
  ⌈total_books⌉ = 145 := by
  sorry

#check chloe_bookcase_problem

end chloe_bookcase_problem_l733_73371


namespace probability_of_one_in_20_rows_l733_73322

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) := sorry

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := sorry

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ :=
  (countOnes n : ℚ) / (totalElements n : ℚ)

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end probability_of_one_in_20_rows_l733_73322


namespace intersection_points_l733_73303

/-- The intersection points of two cubic and quadratic functions -/
theorem intersection_points
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := -a * x^3 + b * x + c
  ∃ (x₁ x₂ : ℝ), (x₁ = 0 ∧ f x₁ = g x₁) ∧ (x₂ = -1 ∧ f x₂ = g x₂) ∧
    ∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂ := by
  sorry

end intersection_points_l733_73303


namespace extended_morse_code_symbols_l733_73319

-- Define the function to calculate the number of sequences for a given length
def sequencesForLength (n : ℕ) : ℕ := 2^n

-- Define the total number of sequences for lengths 1 to 5
def totalSequences : ℕ :=
  (sequencesForLength 1) + (sequencesForLength 2) + (sequencesForLength 3) +
  (sequencesForLength 4) + (sequencesForLength 5)

-- Theorem statement
theorem extended_morse_code_symbols :
  totalSequences = 62 := by
  sorry

end extended_morse_code_symbols_l733_73319


namespace bride_groom_age_sum_l733_73370

theorem bride_groom_age_sum :
  ∀ (groom_age bride_age : ℕ),
    groom_age = 83 →
    bride_age = groom_age + 19 →
    groom_age + bride_age = 185 :=
by
  sorry

end bride_groom_age_sum_l733_73370
