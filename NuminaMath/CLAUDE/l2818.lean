import Mathlib

namespace sum_of_a_and_b_l2818_281867

theorem sum_of_a_and_b (a b : ℝ) (h : Real.sqrt (a - 4) + (b + 5)^2 = 0) : a + b = -1 := by
  sorry

end sum_of_a_and_b_l2818_281867


namespace rational_number_equation_l2818_281812

theorem rational_number_equation (A B : ℝ) (x : ℚ) :
  x = (1 / 2) * x + (1 / 5) * ((3 / 4) * (A + B) - (2 / 3) * (A + B)) →
  x = (1 / 30) * (A + B) := by
  sorry

end rational_number_equation_l2818_281812


namespace honey_servings_l2818_281875

/-- Proves that a container with 47 1/3 cups of honey contains 40 12/21 servings when each serving is 1 1/6 cups -/
theorem honey_servings (container : ℚ) (serving : ℚ) :
  container = 47 + 1 / 3 →
  serving = 1 + 1 / 6 →
  container / serving = 40 + 12 / 21 := by
sorry

end honey_servings_l2818_281875


namespace remainder_problem_l2818_281815

theorem remainder_problem (n : ℤ) (h : n % 20 = 11) : (2 * n) % 10 = 2 := by
  sorry

end remainder_problem_l2818_281815


namespace some_number_value_l2818_281842

theorem some_number_value (x : ℝ) : (45 + 23 / x) * x = 4028 → x = 89 := by
  sorry

end some_number_value_l2818_281842


namespace variance_estimation_l2818_281898

/-- Represents the data for a group of students -/
structure GroupData where
  count : ℕ
  average_score : ℝ
  variance : ℝ

/-- Calculates the estimated variance of test scores given two groups of students -/
def estimated_variance (male : GroupData) (female : GroupData) : ℝ :=
  let total_count := male.count + female.count
  let male_weight := male.count / total_count
  let female_weight := female.count / total_count
  let overall_average := male_weight * male.average_score + female_weight * female.average_score
  male_weight * (male.variance + (overall_average - male.average_score)^2) +
  female_weight * (female.variance + (female.average_score - overall_average)^2)

theorem variance_estimation (male : GroupData) (female : GroupData) :
  male.count = 400 →
  female.count = 600 →
  male.average_score = 80 →
  male.variance = 10 →
  female.average_score = 60 →
  female.variance = 20 →
  estimated_variance male female = 112 := by
  sorry

end variance_estimation_l2818_281898


namespace order_of_surds_l2818_281830

theorem order_of_surds : 
  let a : ℝ := Real.sqrt 5 - Real.sqrt 3
  let b : ℝ := Real.sqrt 3 - 1
  let c : ℝ := Real.sqrt 7 - Real.sqrt 5
  b > a ∧ a > c := by sorry

end order_of_surds_l2818_281830


namespace max_value_of_f_l2818_281861

def f (x y : ℝ) : ℝ := x * y^2 * (x^2 + x + 1) * (y^2 + y + 1)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 951625 / 256 ∧
  (∀ (x y : ℝ), x + y = 5 → f x y ≤ M) ∧
  (∃ (x y : ℝ), x + y = 5 ∧ f x y = M) := by
  sorry

end max_value_of_f_l2818_281861


namespace tan_negative_23pi_over_6_sin_75_degrees_l2818_281846

-- Part 1
theorem tan_negative_23pi_over_6 : 
  Real.tan (-23 * π / 6) = Real.sqrt 3 / 3 := by sorry

-- Part 2
theorem sin_75_degrees : 
  Real.sin (75 * π / 180) = (Real.sqrt 2 + Real.sqrt 6) / 4 := by sorry

end tan_negative_23pi_over_6_sin_75_degrees_l2818_281846


namespace compare_quadratic_expressions_range_of_linear_combination_l2818_281801

-- Part 1
theorem compare_quadratic_expressions (a : ℝ) : 
  (a - 2) * (a - 6) < (a - 3) * (a - 5) := by sorry

-- Part 2
theorem range_of_linear_combination (x y : ℝ) 
  (hx : -2 < x ∧ x < 1) (hy : 1 < y ∧ y < 2) : 
  -6 < 2 * x - y ∧ 2 * x - y < 1 := by sorry

end compare_quadratic_expressions_range_of_linear_combination_l2818_281801


namespace power_three_sum_l2818_281843

theorem power_three_sum (m n : ℕ+) (x y : ℝ) 
  (hx : 3^(m.val) = x) 
  (hy : 9^(n.val) = y) : 
  3^(m.val + 2*n.val) = x * y := by
sorry

end power_three_sum_l2818_281843


namespace sqrt_x_minus_9_meaningful_l2818_281850

theorem sqrt_x_minus_9_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 9) ↔ x ≥ 9 := by sorry

end sqrt_x_minus_9_meaningful_l2818_281850


namespace factorization_equality_l2818_281803

theorem factorization_equality (x y : ℝ) : 6*x^2*y - 3*x*y = 3*x*y*(2*x - 1) := by
  sorry

end factorization_equality_l2818_281803


namespace prime_quadratic_roots_range_l2818_281891

theorem prime_quadratic_roots_range (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ x y : ℤ, x^2 + p*x - 520*p = 0 ∧ y^2 + p*y - 520*p = 0 ∧ x ≠ y) →
  11 < p ∧ p ≤ 21 :=
by sorry

end prime_quadratic_roots_range_l2818_281891


namespace four_digit_numbers_count_l2818_281829

theorem four_digit_numbers_count : 
  (Finset.range 4001).card = (Finset.Icc 1000 5000).card := by sorry

end four_digit_numbers_count_l2818_281829


namespace total_prairie_area_l2818_281831

def prairie_size (dust_covered : ℕ) (untouched : ℕ) : ℕ :=
  dust_covered + untouched

theorem total_prairie_area : prairie_size 64535 522 = 65057 := by
  sorry

end total_prairie_area_l2818_281831


namespace pump_count_proof_l2818_281858

/-- The number of pumps in the first scenario -/
def num_pumps : ℕ := 3

/-- The number of hours worked per day in the first scenario -/
def hours_per_day_1 : ℕ := 8

/-- The number of days to empty the tank in the first scenario -/
def days_to_empty_1 : ℕ := 2

/-- The number of pumps in the second scenario -/
def num_pumps_2 : ℕ := 8

/-- The number of hours worked per day in the second scenario -/
def hours_per_day_2 : ℕ := 6

/-- The number of days to empty the tank in the second scenario -/
def days_to_empty_2 : ℕ := 1

/-- The capacity of the tank in pump-hours -/
def tank_capacity : ℕ := num_pumps_2 * hours_per_day_2 * days_to_empty_2

theorem pump_count_proof :
  num_pumps * hours_per_day_1 * days_to_empty_1 = tank_capacity :=
by sorry

end pump_count_proof_l2818_281858


namespace arithmetic_sequence_sum_l2818_281841

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 3 + a 9 + a 15 + a 21 = 8 →
  a 1 + a 23 = 4 := by
  sorry

end arithmetic_sequence_sum_l2818_281841


namespace apples_used_for_lunch_l2818_281828

theorem apples_used_for_lunch (initial : ℕ) (bought : ℕ) (final : ℕ) : 
  initial = 38 → bought = 28 → final = 46 → initial - (final - bought) = 20 := by
sorry

end apples_used_for_lunch_l2818_281828


namespace smallest_area_increase_l2818_281864

theorem smallest_area_increase (l w : ℕ) (hl : l > 0) (hw : w > 0) :
  ∃ (x : ℕ), x > 0 ∧ (w + 1) * (l - 1) - w * l = x ∧
  ∀ (y : ℕ), y > 0 → (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (b + 1) * (a - 1) - b * a = y) → y ≥ x :=
by sorry

end smallest_area_increase_l2818_281864


namespace sqrt_sin_cos_identity_l2818_281824

theorem sqrt_sin_cos_identity : 
  Real.sqrt (1 - 2 * Real.sin (π + 2) * Real.cos (π - 2)) = Real.sin 2 - Real.cos 2 := by
  sorry

end sqrt_sin_cos_identity_l2818_281824


namespace total_absent_students_l2818_281857

def total_students : ℕ := 280

def absent_third_day (total : ℕ) : ℕ := total / 7

def absent_second_day (absent_third : ℕ) : ℕ := 2 * absent_third

def present_first_day (total : ℕ) (absent_second : ℕ) : ℕ := total - absent_second

def absent_first_day (total : ℕ) (present_first : ℕ) : ℕ := total - present_first

theorem total_absent_students :
  let absent_third := absent_third_day total_students
  let absent_second := absent_second_day absent_third
  let present_first := present_first_day total_students absent_second
  let absent_first := absent_first_day total_students present_first
  absent_first + absent_second + absent_third = 200 := by sorry

end total_absent_students_l2818_281857


namespace average_weight_increase_l2818_281881

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase
  (n : ℕ)  -- number of people in the group
  (w_old : ℝ)  -- weight of the person being replaced
  (w_new : ℝ)  -- weight of the new person
  (h_n : n = 8)  -- given that there are 8 people
  (h_w_old : w_old = 67)  -- given that the old person weighs 67 kg
  (h_w_new : w_new = 87)  -- given that the new person weighs 87 kg
  : (w_new - w_old) / n = 2.5 := by
sorry


end average_weight_increase_l2818_281881


namespace intersection_equality_implies_a_greater_than_two_l2818_281880

-- Define set A
def A : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

-- Define set B
def B (a : ℝ) : Set ℝ := Set.Ioo (-3) a

-- Theorem statement
theorem intersection_equality_implies_a_greater_than_two (a : ℝ) :
  A ∩ B a = A → a > 2 := by
  sorry

end intersection_equality_implies_a_greater_than_two_l2818_281880


namespace percent_difference_l2818_281826

theorem percent_difference (w e y z : ℝ) 
  (hw : w = 0.6 * e) 
  (he : e = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  z = 1.5 * w := by
  sorry

end percent_difference_l2818_281826


namespace points_per_vegetable_l2818_281860

/-- Proves that the number of points given for each vegetable eaten is 2 --/
theorem points_per_vegetable (total_points : ℕ) (num_students : ℕ) (num_weeks : ℕ) (veggies_per_week : ℕ)
  (h1 : total_points = 200)
  (h2 : num_students = 25)
  (h3 : num_weeks = 2)
  (h4 : veggies_per_week = 2) :
  total_points / (num_students * num_weeks * veggies_per_week) = 2 := by
  sorry

end points_per_vegetable_l2818_281860


namespace fractional_parts_inequality_l2818_281838

theorem fractional_parts_inequality (q : ℕ+) (hq : ¬ ∃ (m : ℕ), m^3 = q) :
  ∃ (c : ℝ), c > 0 ∧ ∀ (n : ℕ+),
    (nq^(1/3:ℝ) - ⌊nq^(1/3:ℝ)⌋) + (nq^(2/3:ℝ) - ⌊nq^(2/3:ℝ)⌋) ≥ c * n^(-1/2:ℝ) :=
by sorry

end fractional_parts_inequality_l2818_281838


namespace polynomial_sum_l2818_281873

-- Define the polynomials
def f (x : ℝ) : ℝ := -6 * x^2 + 2 * x - 7
def g (x : ℝ) : ℝ := -4 * x^2 + 4 * x - 3
def h (x : ℝ) : ℝ := 10 * x^2 + 6 * x + 2

-- State the theorem
theorem polynomial_sum (x : ℝ) :
  f x + g x + (h x)^2 = 100 * x^4 + 120 * x^3 + 34 * x^2 + 30 * x - 6 := by
  sorry

end polynomial_sum_l2818_281873


namespace jordan_terry_income_difference_l2818_281896

/-- Calculates the difference in weekly income between two people given their daily incomes and the number of days worked per week. -/
def weekly_income_difference (terry_daily_income jordan_daily_income days_per_week : ℕ) : ℕ :=
  (jordan_daily_income * days_per_week) - (terry_daily_income * days_per_week)

/-- Proves that the difference in weekly income between Jordan and Terry is $42. -/
theorem jordan_terry_income_difference :
  weekly_income_difference 24 30 7 = 42 := by
  sorry

end jordan_terry_income_difference_l2818_281896


namespace complex_expression_equals_19_l2818_281892

-- Define lg as base 2 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

-- State the theorem
theorem complex_expression_equals_19 :
  27 ^ (2/3) - 2 ^ (lg 3) * lg (1/8) + 2 * lg (Real.sqrt (3 + Real.sqrt 5) + Real.sqrt (3 - Real.sqrt 5)) = 19 :=
by sorry

end complex_expression_equals_19_l2818_281892


namespace rearrangement_theorem_l2818_281869

def n : ℕ := 2014

theorem rearrangement_theorem (x y : Fin n → ℤ)
  (hx : ∀ i j, i ≠ j → x i % n ≠ x j % n)
  (hy : ∀ i j, i ≠ j → y i % n ≠ y j % n) :
  ∃ σ : Equiv.Perm (Fin n), ∀ i j, i ≠ j → (x i + y (σ i)) % (2 * n) ≠ (x j + y (σ j)) % (2 * n) := by
  sorry

end rearrangement_theorem_l2818_281869


namespace custom_op_solution_l2818_281876

/-- Custom operation defined for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that given the custom operation, if 11b = 110, then b = 12 -/
theorem custom_op_solution :
  ∀ b : ℤ, customOp 11 b = 110 → b = 12 := by
  sorry

end custom_op_solution_l2818_281876


namespace fraction_equivalence_l2818_281870

theorem fraction_equivalence : 
  let n : ℚ := 13/2
  (4 + n) / (7 + n) = 7 / 9 := by sorry

end fraction_equivalence_l2818_281870


namespace seventieth_pair_is_4_9_l2818_281807

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- Calculates the sum of the numbers in a pair -/
def pairSum (p : IntPair) : ℕ := p.first + p.second

/-- Generates the nth pair in the sequence -/
def nthPair (n : ℕ) : IntPair :=
  sorry

/-- Calculates the total number of pairs up to and including pairs with sum k -/
def totalPairsUpToSum (k : ℕ) : ℕ :=
  sorry

theorem seventieth_pair_is_4_9 : nthPair 70 = IntPair.mk 4 9 := by
  sorry

end seventieth_pair_is_4_9_l2818_281807


namespace min_lines_same_quadrant_l2818_281854

/-- A line in a Cartesian coordinate system --/
structure Line where
  k : ℝ
  b : ℝ
  k_nonzero : k ≠ 0

/-- A family of lines in a Cartesian coordinate system --/
def LineFamily := Set Line

/-- The minimum number of lines needed to guarantee at least two lines in the same quadrant --/
def minLinesForSameQuadrant (family : LineFamily) : ℕ := 7

/-- Theorem stating that 7 is the minimum number of lines needed --/
theorem min_lines_same_quadrant (family : LineFamily) :
  minLinesForSameQuadrant family = 7 :=
sorry

end min_lines_same_quadrant_l2818_281854


namespace difference_of_squares_l2818_281882

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end difference_of_squares_l2818_281882


namespace abs_neg_two_neg_two_pow_zero_l2818_281810

-- Prove that the absolute value of -2 is equal to 2
theorem abs_neg_two : |(-2 : ℤ)| = 2 := by sorry

-- Prove that -2 raised to the power of 0 is equal to 1
theorem neg_two_pow_zero : (-2 : ℤ) ^ (0 : ℕ) = 1 := by sorry

end abs_neg_two_neg_two_pow_zero_l2818_281810


namespace alexander_new_galleries_l2818_281849

/-- Represents the number of pictures Alexander draws for the first gallery -/
def first_gallery_pictures : ℕ := 9

/-- Represents the number of pictures Alexander draws for each new gallery -/
def new_gallery_pictures : ℕ := 2

/-- Represents the number of pencils Alexander needs for each picture -/
def pencils_per_picture : ℕ := 4

/-- Represents the number of pencils Alexander needs for signing at each exhibition -/
def pencils_for_signing : ℕ := 2

/-- Represents the total number of pencils Alexander uses for all exhibitions -/
def total_pencils : ℕ := 88

/-- Calculates the number of new galleries Alexander drew for -/
def new_galleries : ℕ :=
  let first_gallery_pencils := first_gallery_pictures * pencils_per_picture + pencils_for_signing
  let remaining_pencils := total_pencils - first_gallery_pencils
  let pencils_per_new_gallery := new_gallery_pictures * pencils_per_picture + pencils_for_signing
  remaining_pencils / pencils_per_new_gallery

theorem alexander_new_galleries :
  new_galleries = 5 := by sorry

end alexander_new_galleries_l2818_281849


namespace total_pizzas_ordered_l2818_281848

/-- Represents the number of pizzas ordered for a group of students. -/
def pizzas_ordered (num_boys : ℕ) (num_girls : ℕ) : ℚ :=
  22 + (22 / num_boys) * (num_girls / 2)

/-- Theorem stating the total number of pizzas ordered is 33. -/
theorem total_pizzas_ordered :
  ∃ (num_boys : ℕ),
    num_boys > 13 ∧
    pizzas_ordered num_boys 13 = 33 ∧
    (∃ (n : ℕ), pizzas_ordered num_boys 13 = n) :=
sorry

end total_pizzas_ordered_l2818_281848


namespace final_stamp_collection_l2818_281845

def initial_stamps : ℕ := 3000
def mikes_gift : ℕ := 17
def damaged_stamps : ℕ := 37

def harrys_gift (mikes_gift : ℕ) : ℕ := 2 * mikes_gift + 10
def sarahs_gift (mikes_gift : ℕ) : ℕ := 3 * mikes_gift - 5

def total_gift_stamps (mikes_gift : ℕ) : ℕ :=
  mikes_gift + harrys_gift mikes_gift + sarahs_gift mikes_gift

def final_stamp_count (initial_stamps mikes_gift damaged_stamps : ℕ) : ℕ :=
  initial_stamps + total_gift_stamps mikes_gift - damaged_stamps

theorem final_stamp_collection :
  final_stamp_count initial_stamps mikes_gift damaged_stamps = 3070 :=
by sorry

end final_stamp_collection_l2818_281845


namespace leroy_payment_l2818_281884

/-- The amount LeRoy must pay to equalize costs on a shared trip -/
theorem leroy_payment (A B C : ℝ) (h1 : A < B) (h2 : B < C) : 
  (B + C - 2 * A) / 3 = (A + B + C) / 3 - A := by
  sorry

#check leroy_payment

end leroy_payment_l2818_281884


namespace four_valid_m_l2818_281893

/-- The number of positive integers m for which 2310 / (m^2 - 4) is a positive integer -/
def count_valid_m : ℕ := 4

/-- Predicate to check if 2310 / (m^2 - 4) is a positive integer -/
def is_valid (m : ℕ) : Prop :=
  m > 0 ∧ ∃ k : ℕ+, k * (m^2 - 4) = 2310

/-- Theorem stating that there are exactly 4 positive integers m satisfying the condition -/
theorem four_valid_m :
  (∃! (s : Finset ℕ), s.card = count_valid_m ∧ ∀ m, m ∈ s ↔ is_valid m) :=
sorry

end four_valid_m_l2818_281893


namespace log_calculation_l2818_281818

theorem log_calculation : Real.log 25 / Real.log 10 + 
  (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) + 
  (Real.log 2 / Real.log 10)^2 = 2 := by
  sorry

end log_calculation_l2818_281818


namespace fifth_number_21st_row_is_809_l2818_281862

/-- The number of odd numbers in the nth row of the pattern -/
def oddNumbersInRow (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of odd numbers in the first n rows -/
def sumOddNumbersInRows (n : ℕ) : ℕ :=
  (oddNumbersInRow n + 1) * n / 2

/-- The nth positive odd number -/
def nthPositiveOdd (n : ℕ) : ℕ := 2 * n - 1

theorem fifth_number_21st_row_is_809 :
  let totalPreviousRows := sumOddNumbersInRows 20
  let positionInSequence := totalPreviousRows + 5
  nthPositiveOdd positionInSequence = 809 := by sorry

end fifth_number_21st_row_is_809_l2818_281862


namespace not_necessary_condition_l2818_281839

theorem not_necessary_condition : ¬(∀ x y : ℝ, x * y = 0 → x^2 + y^2 = 0) := by sorry

end not_necessary_condition_l2818_281839


namespace cars_for_sale_l2818_281897

theorem cars_for_sale 
  (num_salespeople : ℕ)
  (cars_per_salesperson_per_month : ℕ)
  (num_months : ℕ)
  (h1 : num_salespeople = 10)
  (h2 : cars_per_salesperson_per_month = 10)
  (h3 : num_months = 5) :
  num_salespeople * cars_per_salesperson_per_month * num_months = 500 := by
  sorry

end cars_for_sale_l2818_281897


namespace arithmetic_sequence_property_l2818_281853

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 4 = 5) : 
  2 * (a 1) - (a 5) + (a 11) = 10 := by
sorry

end arithmetic_sequence_property_l2818_281853


namespace min_value_expression_l2818_281883

theorem min_value_expression (x y : ℝ) (h : x ≥ 4) :
  x^2 + y^2 - 8*x + 6*y + 26 ≥ 1 := by
  sorry

end min_value_expression_l2818_281883


namespace larger_number_problem_l2818_281866

theorem larger_number_problem (x y : ℤ) 
  (sum_is_62 : x + y = 62) 
  (y_is_larger : y = x + 12) : 
  y = 37 := by
  sorry

end larger_number_problem_l2818_281866


namespace P_sufficient_not_necessary_for_Q_l2818_281844

def P (x : ℝ) : Prop := |x - 2| ≤ 3

def Q (x : ℝ) : Prop := x ≥ -1 ∨ x ≤ 5

theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬(P x)) := by sorry

end P_sufficient_not_necessary_for_Q_l2818_281844


namespace solution_characterization_l2818_281855

def solution_set : Set (ℝ × ℝ) :=
  {(0, 0),
   (Real.sqrt (2 + Real.sqrt 2) / 2, Real.sqrt (2 - Real.sqrt 2) / 2),
   (-Real.sqrt (2 - Real.sqrt 2) / 2, Real.sqrt (2 + Real.sqrt 2) / 2),
   (-Real.sqrt (2 + Real.sqrt 2) / 2, -Real.sqrt (2 - Real.sqrt 2) / 2),
   (Real.sqrt (2 - Real.sqrt 2) / 2, -Real.sqrt (2 + Real.sqrt 2) / 2)}

def satisfies_equations (p : ℝ × ℝ) : Prop :=
  let x := p.1
  let y := p.2
  x = 3 * x^2 * y - y^3 ∧ y = x^3 - 3 * x * y^2

theorem solution_characterization :
  ∀ p : ℝ × ℝ, satisfies_equations p ↔ p ∈ solution_set := by
  sorry

end solution_characterization_l2818_281855


namespace f_difference_at_5_and_neg_5_l2818_281837

-- Define the function f
def f (x : ℝ) : ℝ := x^4 + x^2 + 5*x

-- State the theorem
theorem f_difference_at_5_and_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end f_difference_at_5_and_neg_5_l2818_281837


namespace a_18_value_l2818_281886

def equal_sum_sequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, a n + a (n + 1) = k

theorem a_18_value (a : ℕ → ℝ) (h1 : equal_sum_sequence a) (h2 : a 1 = 2) (h3 : ∃ k : ℝ, k = 5 ∧ ∀ n : ℕ, a n + a (n + 1) = k) :
  a 18 = 3 := by
  sorry

end a_18_value_l2818_281886


namespace solution_set_x_squared_minus_one_l2818_281840

theorem solution_set_x_squared_minus_one (x : ℝ) : 
  {x : ℝ | x^2 - 1 = 0} = {-1, 1} := by sorry

end solution_set_x_squared_minus_one_l2818_281840


namespace farm_has_six_cows_l2818_281821

/-- Represents the number of animals of each type on the farm -/
structure FarmAnimals where
  cows : ℕ
  chickens : ℕ
  sheep : ℕ

/-- Calculates the total number of legs for given farm animals -/
def totalLegs (animals : FarmAnimals) : ℕ :=
  4 * animals.cows + 2 * animals.chickens + 4 * animals.sheep

/-- Calculates the total number of heads for given farm animals -/
def totalHeads (animals : FarmAnimals) : ℕ :=
  animals.cows + animals.chickens + animals.sheep

/-- Theorem stating that the farm with the given conditions has 6 cows -/
theorem farm_has_six_cows :
  ∃ (animals : FarmAnimals),
    totalLegs animals = 100 ∧
    totalLegs animals = 3 * totalHeads animals + 20 ∧
    animals.cows = 6 := by
  sorry


end farm_has_six_cows_l2818_281821


namespace pitcher_juice_distribution_l2818_281817

theorem pitcher_juice_distribution (pitcher_capacity : ℝ) (num_cups : ℕ) :
  pitcher_capacity > 0 →
  num_cups = 8 →
  let juice_amount := pitcher_capacity / 2
  let juice_per_cup := juice_amount / num_cups
  juice_per_cup / pitcher_capacity = 1 / 16 := by
  sorry

end pitcher_juice_distribution_l2818_281817


namespace fruit_juice_mixture_proof_l2818_281808

/-- Proves that adding 0.4 liters of pure fruit juice to a 2-liter mixture
    that is 10% pure fruit juice results in a final mixture that is 25% pure fruit juice. -/
theorem fruit_juice_mixture_proof :
  let initial_volume : ℝ := 2
  let initial_concentration : ℝ := 0.1
  let target_concentration : ℝ := 0.25
  let added_juice : ℝ := 0.4
  let final_volume : ℝ := initial_volume + added_juice
  let final_juice_amount : ℝ := initial_volume * initial_concentration + added_juice
  final_juice_amount / final_volume = target_concentration :=
by sorry

end fruit_juice_mixture_proof_l2818_281808


namespace rationalize_denominator_l2818_281874

theorem rationalize_denominator : (3 : ℝ) / Real.sqrt 75 = Real.sqrt 3 / 5 := by
  sorry

end rationalize_denominator_l2818_281874


namespace isabel_earnings_l2818_281800

def bead_necklaces : ℕ := 3
def gemstone_necklaces : ℕ := 3
def bead_price : ℚ := 4
def gemstone_price : ℚ := 8
def sales_tax_rate : ℚ := 0.05
def discount_rate : ℚ := 0.10

def total_earned : ℚ :=
  let total_before_tax := bead_necklaces * bead_price + gemstone_necklaces * gemstone_price
  let tax_amount := total_before_tax * sales_tax_rate
  let total_after_tax := total_before_tax + tax_amount
  let discount_amount := total_after_tax * discount_rate
  total_after_tax - discount_amount

theorem isabel_earnings : total_earned = 34.02 := by
  sorry

end isabel_earnings_l2818_281800


namespace jonathan_phone_time_l2818_281827

/-- 
Given that Jonathan spends some hours on his phone daily, half of which is spent on social media,
and he spends 28 hours on social media in a week, prove that he spends 8 hours on his phone daily.
-/
theorem jonathan_phone_time (x : ℝ) 
  (daily_phone_time : x > 0) 
  (social_media_half : x / 2 * 7 = 28) : 
  x = 8 := by
sorry

end jonathan_phone_time_l2818_281827


namespace proposition_logic_l2818_281833

theorem proposition_logic : 
  let p := (3 : ℝ) ≥ 3
  let q := (3 : ℝ) > 4
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by sorry

end proposition_logic_l2818_281833


namespace rooks_knight_move_theorem_l2818_281851

/-- Represents a position on a chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a knight's move -/
structure KnightMove :=
  (drow : Int)
  (dcol : Int)

/-- Checks if a move is a valid knight's move -/
def isValidKnightMove (km : KnightMove) : Prop :=
  (km.drow.natAbs = 2 ∧ km.dcol.natAbs = 1) ∨ 
  (km.drow.natAbs = 1 ∧ km.dcol.natAbs = 2)

/-- Applies a knight's move to a position -/
def applyMove (p : Position) (km : KnightMove) : Position :=
  ⟨p.row + km.drow, p.col + km.dcol⟩

/-- Checks if two positions are non-attacking for rooks -/
def nonAttacking (p1 p2 : Position) : Prop :=
  p1.row ≠ p2.row ∧ p1.col ≠ p2.col

/-- The main theorem -/
theorem rooks_knight_move_theorem 
  (initial_positions : Fin 8 → Position)
  (h_initial_non_attacking : ∀ i j, i ≠ j → 
    nonAttacking (initial_positions i) (initial_positions j)) :
  ∃ (moves : Fin 8 → KnightMove),
    (∀ i, isValidKnightMove (moves i)) ∧
    (∀ i j, i ≠ j → 
      nonAttacking 
        (applyMove (initial_positions i) (moves i))
        (applyMove (initial_positions j) (moves j))) :=
  sorry


end rooks_knight_move_theorem_l2818_281851


namespace perpendicular_vectors_x_value_l2818_281836

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem stating that if (1,x) and (-2,3) are perpendicular, then x = 2/3 -/
theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular (1, x) (-2, 3) → x = 2/3 := by
  sorry

#check perpendicular_vectors_x_value

end perpendicular_vectors_x_value_l2818_281836


namespace evaluate_expression_l2818_281822

theorem evaluate_expression (a : ℝ) : 
  let x := a + 9
  (x - a + 3)^2 = 144 := by sorry

end evaluate_expression_l2818_281822


namespace complex_number_equality_l2818_281819

theorem complex_number_equality : (((1 : ℂ) + I) * ((3 : ℂ) + 4*I)) / I = (7 : ℂ) + I := by
  sorry

end complex_number_equality_l2818_281819


namespace horner_method_v3_l2818_281868

def f (x : ℝ) : ℝ := 2*x^5 - x + 3*x^2 + x + 1

def horner_v3 (x : ℝ) : ℝ := 
  let v0 := 2
  let v1 := v0 * x + 0
  let v2 := v1 * x - 1
  v2 * x + 3

theorem horner_method_v3 : horner_v3 3 = 54 := by sorry

end horner_method_v3_l2818_281868


namespace supplement_of_angle_with_complement_50_l2818_281805

def angle_with_complement_50 (θ : ℝ) : Prop :=
  90 - θ = 50

theorem supplement_of_angle_with_complement_50 (θ : ℝ) 
  (h : angle_with_complement_50 θ) : 180 - θ = 140 := by
  sorry

end supplement_of_angle_with_complement_50_l2818_281805


namespace negation_of_existence_l2818_281863

theorem negation_of_existence (p : Prop) :
  (¬ ∃ (x : ℤ), x^2 ≥ x) ↔ (∀ (x : ℤ), x^2 < x) := by
  sorry

end negation_of_existence_l2818_281863


namespace circle_equation_from_chord_l2818_281825

/-- Given a circle with center at the origin and a chord of length 8 cut by the line 3x + 4y + 15 = 0,
    prove that the equation of the circle is x^2 + y^2 = 25 -/
theorem circle_equation_from_chord (x y : ℝ) :
  let center := (0 : ℝ × ℝ)
  let chord_line := {(x, y) | 3 * x + 4 * y + 15 = 0}
  let chord_length := 8
  ∃ (r : ℝ), r > 0 ∧
    (∀ (p : ℝ × ℝ), p ∈ chord_line → dist center p ≤ r) ∧
    (∃ (p q : ℝ × ℝ), p ∈ chord_line ∧ q ∈ chord_line ∧ p ≠ q ∧ dist p q = chord_length) →
  x^2 + y^2 = 25 :=
sorry

end circle_equation_from_chord_l2818_281825


namespace line_passes_through_fixed_point_l2818_281811

theorem line_passes_through_fixed_point (m : ℝ) :
  (m + 1) * 1 + (2 * m - 1) * (-1) + m - 2 = 0 := by
  sorry

end line_passes_through_fixed_point_l2818_281811


namespace cupboard_sale_percentage_below_cost_l2818_281872

def cost_price : ℕ := 3750
def additional_amount : ℕ := 1200
def profit_percentage : ℚ := 16 / 100

def selling_price_with_profit : ℚ := cost_price + profit_percentage * cost_price
def actual_selling_price : ℚ := selling_price_with_profit - additional_amount

theorem cupboard_sale_percentage_below_cost (cost_price : ℕ) (additional_amount : ℕ) 
  (profit_percentage : ℚ) (selling_price_with_profit : ℚ) (actual_selling_price : ℚ) :
  (cost_price - actual_selling_price) / cost_price = 16 / 100 :=
by sorry

end cupboard_sale_percentage_below_cost_l2818_281872


namespace equation_solution_l2818_281878

theorem equation_solution : ∃ n : ℤ, 5^3 - 7 = 6^2 + n ∧ n = 82 := by
  sorry

end equation_solution_l2818_281878


namespace no_solutions_in_interval_l2818_281835

theorem no_solutions_in_interval (a : ℤ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 7, (x - 2 * (a : ℝ) + 1)^2 - 2*x + 4*(a : ℝ) - 10 ≠ 0) ↔ 
  (a ≤ -3 ∨ a ≥ 6) :=
sorry

end no_solutions_in_interval_l2818_281835


namespace prob_ice_skating_given_skiing_l2818_281816

/-- The probability that a randomly selected student likes ice skating -/
def P_ice_skating : ℝ := 0.6

/-- The probability that a randomly selected student likes skiing -/
def P_skiing : ℝ := 0.5

/-- The probability that a randomly selected student likes either ice skating or skiing -/
def P_ice_skating_or_skiing : ℝ := 0.7

/-- Theorem stating that the probability of a student liking ice skating given that they like skiing is 0.8 -/
theorem prob_ice_skating_given_skiing :
  (P_ice_skating + P_skiing - P_ice_skating_or_skiing) / P_skiing = 0.8 := by
  sorry

end prob_ice_skating_given_skiing_l2818_281816


namespace algebraic_expression_value_l2818_281890

theorem algebraic_expression_value :
  let x : ℤ := -2
  let y : ℤ := -4
  2 * x^2 - y + 3 = 15 := by sorry

end algebraic_expression_value_l2818_281890


namespace train_platform_crossing_time_l2818_281859

/-- Given a train of length 1400 m that crosses a tree in 100 sec,
    prove that it takes 150 sec to pass a platform of length 700 m. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1400)
  (h2 : tree_crossing_time = 100)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 150 := by
  sorry

end train_platform_crossing_time_l2818_281859


namespace dan_picked_nine_apples_l2818_281879

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The total number of apples picked by Benny and Dan -/
def total_apples : ℕ := 11

/-- The number of apples Dan picked -/
def dan_apples : ℕ := total_apples - benny_apples

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end dan_picked_nine_apples_l2818_281879


namespace hyperbola_asymptote_slopes_l2818_281877

/-- Given a hyperbola with equation x^2 - y^2 = 3, 
    if k₁ and k₂ are the slopes of its two asymptotes, 
    then k₁k₂ = -1 -/
theorem hyperbola_asymptote_slopes (k₁ k₂ : ℝ) : 
  (∀ x y : ℝ, x^2 - y^2 = 3 → 
    (∃ a b : ℝ, (y = k₁ * x + a ∨ y = k₂ * x + b) ∧ 
      (∀ ε > 0, ∃ x₀ > 0, ∀ x > x₀, 
        |y - k₁ * x| < ε ∨ |y - k₂ * x| < ε))) →
  k₁ * k₂ = -1 := by
sorry

end hyperbola_asymptote_slopes_l2818_281877


namespace area_increase_is_204_l2818_281834

/-- Represents the increase in vegetables from last year to this year -/
structure VegetableIncrease where
  broccoli : ℕ
  cauliflower : ℕ
  cabbage : ℕ

/-- Calculates the total increase in area given the increase in vegetables -/
def totalAreaIncrease (v : VegetableIncrease) : ℝ :=
  v.broccoli * 1 + v.cauliflower * 2 + v.cabbage * 1.5

/-- The theorem stating that the total increase in area is 204 square feet -/
theorem area_increase_is_204 (v : VegetableIncrease) 
  (h1 : v.broccoli = 79)
  (h2 : v.cauliflower = 25)
  (h3 : v.cabbage = 50) : 
  totalAreaIncrease v = 204 := by
  sorry

#eval totalAreaIncrease { broccoli := 79, cauliflower := 25, cabbage := 50 }

end area_increase_is_204_l2818_281834


namespace arithmetic_progression_bound_l2818_281899

theorem arithmetic_progression_bound :
  ∃ (C : ℝ), C > 1 ∧
  ∀ (n : ℕ) (a : ℕ → ℕ),
    n > 1 →
    (∀ i j, i < j ∧ j ≤ n → a i < a j) →
    (∃ (d : ℚ), ∀ i j, i ≤ n ∧ j ≤ n → (1 : ℚ) / a i - (1 : ℚ) / a j = d * (i - j)) →
    (a 0 : ℝ) > C^n :=
by sorry

end arithmetic_progression_bound_l2818_281899


namespace power_of_five_equality_l2818_281856

theorem power_of_five_equality (n : ℕ) : 5^n = 5 * 25^3 * 625^2 → n = 15 := by
  sorry

end power_of_five_equality_l2818_281856


namespace fundraising_solution_correct_l2818_281889

/-- Represents the prices and quantities of basketballs and soccer balls -/
structure BallPurchase where
  basketball_price : ℕ
  soccer_price : ℕ
  basketball_qty : ℕ
  soccer_qty : ℕ

/-- Represents the fundraising conditions -/
structure FundraisingConditions where
  original_budget : ℕ
  original_total_items : ℕ
  actual_raised : ℕ
  new_total_items : ℕ

/-- Checks if a purchase satisfies the original plan -/
def satisfies_original_plan (purchase : BallPurchase) (conditions : FundraisingConditions) : Prop :=
  purchase.basketball_qty + purchase.soccer_qty = conditions.original_total_items ∧
  purchase.basketball_price * purchase.basketball_qty + purchase.soccer_price * purchase.soccer_qty = conditions.original_budget

/-- Checks if a purchase is valid under the new conditions -/
def is_valid_new_purchase (purchase : BallPurchase) (conditions : FundraisingConditions) : Prop :=
  purchase.basketball_qty + purchase.soccer_qty = conditions.new_total_items ∧
  purchase.basketball_price * purchase.basketball_qty + purchase.soccer_price * purchase.soccer_qty ≤ conditions.actual_raised

/-- Theorem stating the correctness of the solution -/
theorem fundraising_solution_correct 
  (purchase : BallPurchase) 
  (conditions : FundraisingConditions) 
  (h_basketball_price : purchase.basketball_price = 100)
  (h_soccer_price : purchase.soccer_price = 80)
  (h_original_budget : conditions.original_budget = 5600)
  (h_original_total_items : conditions.original_total_items = 60)
  (h_actual_raised : conditions.actual_raised = 6890)
  (h_new_total_items : conditions.new_total_items = 80) :
  (satisfies_original_plan purchase conditions ∧ purchase.basketball_qty = 40 ∧ purchase.soccer_qty = 20) ∧
  (∀ new_purchase : BallPurchase, is_valid_new_purchase new_purchase conditions → new_purchase.basketball_qty ≤ 24) :=
by sorry

end fundraising_solution_correct_l2818_281889


namespace jamies_mean_is_88_5_l2818_281820

/-- Represents a test score series for two students -/
structure TestScores where
  scores : List Nat
  alex_count : Nat
  jamie_count : Nat
  alex_mean : Rat

/-- Calculates Jamie's mean score given the test scores -/
def jamies_mean (ts : TestScores) : Rat :=
  let total_sum := ts.scores.sum
  let alex_sum := ts.alex_mean * ts.alex_count
  let jamie_sum := total_sum - alex_sum
  jamie_sum / ts.jamie_count

/-- Theorem: Jamie's mean score is 88.5 given the conditions -/
theorem jamies_mean_is_88_5 (ts : TestScores) 
  (h1 : ts.scores = [75, 80, 85, 90, 92, 97])
  (h2 : ts.alex_count = 4)
  (h3 : ts.jamie_count = 2)
  (h4 : ts.alex_mean = 85.5)
  : jamies_mean ts = 88.5 := by
  sorry

end jamies_mean_is_88_5_l2818_281820


namespace expression_factorization_l2818_281852

theorem expression_factorization (b : ℝ) :
  (3 * b^4 + 66 * b^3 - 14) - (-4 * b^4 + 2 * b^3 - 14) = b^3 * (7 * b + 64) := by
  sorry

end expression_factorization_l2818_281852


namespace book_arrangement_count_l2818_281894

/-- The number of different math books -/
def num_math_books : ℕ := 4

/-- The number of different history books -/
def num_history_books : ℕ := 6

/-- The number of ways to arrange the books under the given conditions -/
def arrangement_count : ℕ := num_math_books * (num_math_books - 1) * Nat.factorial (num_math_books + num_history_books - 3)

theorem book_arrangement_count :
  arrangement_count = 60480 :=
sorry

end book_arrangement_count_l2818_281894


namespace min_marked_cells_13x13_board_l2818_281865

/-- Represents a rectangular board --/
structure Board :=
  (rows : Nat)
  (cols : Nat)

/-- Represents a rectangle that can be placed on the board --/
structure Rectangle :=
  (length : Nat)
  (width : Nat)

/-- Function to calculate the minimum number of cells to mark --/
def min_marked_cells (b : Board) (r : Rectangle) : Nat :=
  sorry

/-- Theorem stating the minimum number of cells to mark for the given problem --/
theorem min_marked_cells_13x13_board (b : Board) (r : Rectangle) :
  b.rows = 13 ∧ b.cols = 13 ∧ r.length = 6 ∧ r.width = 1 →
  min_marked_cells b r = 84 :=
sorry

end min_marked_cells_13x13_board_l2818_281865


namespace race_time_difference_l2818_281806

theorem race_time_difference (apple_rate mac_rate : ℝ) (race_distance : ℝ) : 
  apple_rate = 3 ∧ mac_rate = 4 ∧ race_distance = 24 → 
  (race_distance / apple_rate - race_distance / mac_rate) * 60 = 120 := by
  sorry

end race_time_difference_l2818_281806


namespace trigonometric_expression_value_l2818_281823

theorem trigonometric_expression_value (α : Real) (h : α = -35 * π / 6) :
  (2 * Real.sin (π + α) * Real.cos (π - α) - Real.cos (π + α)) /
  (1 + Real.sin α ^ 2 + Real.sin (π - α) - Real.cos (π + α) ^ 2) = Real.sqrt 3 := by
  sorry

end trigonometric_expression_value_l2818_281823


namespace freds_marbles_l2818_281832

/-- Given Fred's marble collection, prove the number of dark blue marbles. -/
theorem freds_marbles (total : ℕ) (red : ℕ) (green : ℕ) (blue : ℕ) : 
  total = 63 →
  red = 38 →
  green = red / 2 →
  total = red + green + blue →
  blue = 6 := by sorry

end freds_marbles_l2818_281832


namespace original_painting_height_l2818_281813

/-- Proves that given a painting with width 15 inches and a print of the painting with width 37.5 inches and height 25 inches, the height of the original painting is 10 inches. -/
theorem original_painting_height
  (original_width : ℝ)
  (print_width : ℝ)
  (print_height : ℝ)
  (h_original_width : original_width = 15)
  (h_print_width : print_width = 37.5)
  (h_print_height : print_height = 25) :
  print_height / (print_width / original_width) = 10 := by
  sorry


end original_painting_height_l2818_281813


namespace square_area_with_five_equal_rectangles_l2818_281895

theorem square_area_with_five_equal_rectangles (s : ℝ) (x : ℝ) (y : ℝ) : 
  s > 0 →  -- side length of square is positive
  x > 0 →  -- width of central rectangle is positive
  y > 0 →  -- height of bottom rectangle is positive
  s = 5 + 2 * y →  -- relationship between side length and rectangles
  x * (s / 2) = 5 * y →  -- equal area condition
  s^2 = 400 := by
  sorry

end square_area_with_five_equal_rectangles_l2818_281895


namespace arithmetic_mean_of_first_four_primes_reciprocals_l2818_281804

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  (reciprocals.sum / reciprocals.length : ℚ) = 247 / 840 := by
  sorry

end arithmetic_mean_of_first_four_primes_reciprocals_l2818_281804


namespace shirts_not_washed_l2818_281887

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) 
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : washed = 29) :
  short_sleeve + long_sleeve - washed = 1 := by
  sorry

end shirts_not_washed_l2818_281887


namespace slower_train_speed_l2818_281847

/-- Proves that the speed of the slower train is 36 km/hr given the conditions of the problem -/
theorem slower_train_speed (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 100 →
  faster_speed = 46 →
  passing_time = 72 →
  ∃ (slower_speed : ℝ),
    slower_speed > 0 ∧
    slower_speed < faster_speed ∧
    (faster_speed - slower_speed) * passing_time * (1000 / 3600) = 2 * train_length ∧
    slower_speed = 36 :=
by sorry

end slower_train_speed_l2818_281847


namespace convex_broken_line_in_triangle_l2818_281809

/-- A convex broken line in 2D space -/
structure ConvexBrokenLine where
  points : List (Real × Real)
  is_convex : sorry
  length : Real

/-- An equilateral triangle in 2D space -/
structure EquilateralTriangle where
  center : Real × Real
  side_length : Real

/-- A function to check if a broken line is enclosed within a triangle -/
def is_enclosed (line : ConvexBrokenLine) (triangle : EquilateralTriangle) : Prop :=
  sorry

theorem convex_broken_line_in_triangle 
  (line : ConvexBrokenLine) 
  (triangle : EquilateralTriangle) : 
  line.length = 1 → 
  triangle.side_length = 1 → 
  is_enclosed line triangle :=
sorry

end convex_broken_line_in_triangle_l2818_281809


namespace circle_trajectory_and_max_area_l2818_281802

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 49
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

-- Define the property of point Q
def Q_property (x y : ℝ) : Prop := C x y ∧ y ≠ 0

-- Define the line MN parallel to OQ and passing through F₂
def MN_parallel_OQ (m : ℝ) (x y : ℝ) : Prop := x = m * y + 2

-- Define the distinct intersection points M and N
def distinct_intersections (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
  C x₁ y₁ ∧ C x₂ y₂ ∧
  MN_parallel_OQ m x₁ y₁ ∧ MN_parallel_OQ m x₂ y₂

-- Theorem statement
theorem circle_trajectory_and_max_area :
  (∀ x y, C x y → (∃ R, (∀ x' y', F₁ x' y' → (x - x')^2 + (y - y')^2 = (7 - R)^2) ∧
                      (∀ x' y', F₂ x' y' → (x - x')^2 + (y - y')^2 = (R - 1)^2))) ∧
  (∀ m, distinct_intersections m →
    ∃ x₃ y₃, Q_property x₃ y₃ ∧
    (∀ A, (∃ x₁ y₁ x₂ y₂, C x₁ y₁ ∧ C x₂ y₂ ∧ MN_parallel_OQ m x₁ y₁ ∧ MN_parallel_OQ m x₂ y₂ ∧
           A = (1/2) * abs ((x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁))) →
    A ≤ 10/3)) :=
sorry

end circle_trajectory_and_max_area_l2818_281802


namespace uncle_age_l2818_281871

/-- Given Bud's age and the relationship to his uncle's age, calculate the uncle's age -/
theorem uncle_age (bud_age : ℕ) (h : bud_age = 8) : 
  3 * bud_age = 24 := by
  sorry

end uncle_age_l2818_281871


namespace window_treatment_cost_l2818_281885

/-- The number of windows Laura needs to buy window treatments for -/
def num_windows : ℕ := 3

/-- The cost of sheers for one window in cents -/
def sheer_cost : ℕ := 4000

/-- The cost of drapes for one window in cents -/
def drape_cost : ℕ := 6000

/-- The total cost for all windows in cents -/
def total_cost : ℕ := 30000

/-- Theorem stating that the number of windows is correct given the costs -/
theorem window_treatment_cost : 
  (sheer_cost + drape_cost) * num_windows = total_cost := by
  sorry


end window_treatment_cost_l2818_281885


namespace min_value_theorem_l2818_281888

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∃ (min : ℝ), min = 6 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → 1/x + 1/y = 1 → 1/(x-1) + 9/(y-1) ≥ min :=
by sorry

end min_value_theorem_l2818_281888


namespace arc_length_for_specific_circle_l2818_281814

/-- Given a circle with radius π and a central angle of 120°, the arc length is (2π²)/3 -/
theorem arc_length_for_specific_circle :
  let r : ℝ := Real.pi
  let θ : ℝ := 120
  let l : ℝ := (θ / 180) * Real.pi * r
  l = (2 * Real.pi^2) / 3 := by sorry

end arc_length_for_specific_circle_l2818_281814
