import Mathlib

namespace boxes_left_to_sell_l1245_124523

/-- Represents the number of cookie boxes sold to each customer --/
structure CustomerSales where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- Calculates the total number of boxes sold --/
def totalSold (sales : CustomerSales) : ℕ :=
  sales.first + sales.second + sales.third + sales.fourth + sales.fifth

/-- Represents Jill's cookie sales --/
def jillSales : CustomerSales where
  first := 5
  second := 4 * 5
  third := (4 * 5) / 2
  fourth := 3 * ((4 * 5) / 2)
  fifth := 10

/-- Jill's sales goal --/
def salesGoal : ℕ := 150

/-- Theorem stating that Jill has 75 boxes left to sell to reach her goal --/
theorem boxes_left_to_sell : salesGoal - totalSold jillSales = 75 := by
  sorry

end boxes_left_to_sell_l1245_124523


namespace number_equal_nine_l1245_124531

theorem number_equal_nine : ∃ x : ℝ, x^6 = 3^12 ∧ x = 9 := by
  sorry

end number_equal_nine_l1245_124531


namespace dance_steps_proof_l1245_124591

def total_steps (jason_steps nancy_ratio : ℕ) : ℕ :=
  jason_steps + nancy_ratio * jason_steps

theorem dance_steps_proof (jason_steps nancy_ratio : ℕ) 
  (h1 : jason_steps = 8) 
  (h2 : nancy_ratio = 3) : 
  total_steps jason_steps nancy_ratio = 32 := by
  sorry

end dance_steps_proof_l1245_124591


namespace range_of_x_when_a_is_neg_one_range_of_a_when_not_p_necessary_not_sufficient_for_not_q_l1245_124524

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - 6*x - 72 ≤ 0 ∧ x^2 + x - 6 > 0

-- Part 1
theorem range_of_x_when_a_is_neg_one :
  (∀ x : ℝ, p x (-1) → q x) →
  ∀ x : ℝ, (x ∈ Set.Icc (-6) (-3) ∪ Set.Ioc 1 12) ↔ (p x (-1) ∨ q x) :=
sorry

-- Part 2
theorem range_of_a_when_not_p_necessary_not_sufficient_for_not_q :
  (∀ x a : ℝ, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x a : ℝ, ¬(p x a) ∧ q x) →
  ∀ a : ℝ, a ∈ Set.Icc (-4) (-2) ↔ (∃ x : ℝ, p x a ∧ q x) :=
sorry

end range_of_x_when_a_is_neg_one_range_of_a_when_not_p_necessary_not_sufficient_for_not_q_l1245_124524


namespace min_value_expression_l1245_124556

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (a + 1/b - 1009) + (b + 1/a) * (b + 1/a - 1009) ≥ -509004.5 := by
  sorry

end min_value_expression_l1245_124556


namespace geometric_sequence_property_l1245_124533

/-- A geometric sequence with a_2 = 2 and a_10 = 8 has a_6 = 4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 2 = 2 →
  a 10 = 8 →
  a 6 = 4 := by
sorry

end geometric_sequence_property_l1245_124533


namespace grape_candy_count_l1245_124570

/-- Represents the number of cherry candies -/
def cherry : ℕ := sorry

/-- Represents the number of grape candies -/
def grape : ℕ := 3 * cherry

/-- Represents the number of apple candies -/
def apple : ℕ := 2 * grape

/-- The cost of each candy in cents -/
def cost_per_candy : ℕ := 250

/-- The total cost of all candies in cents -/
def total_cost : ℕ := 20000

theorem grape_candy_count :
  grape = 24 ∧
  cherry + grape + apple = total_cost / cost_per_candy :=
sorry

end grape_candy_count_l1245_124570


namespace product_from_lcm_and_gcd_l1245_124522

theorem product_from_lcm_and_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h_lcm : Nat.lcm a b = 48) (h_gcd : Nat.gcd a b = 8) : a * b = 384 := by
  sorry

end product_from_lcm_and_gcd_l1245_124522


namespace no_multiple_of_five_l1245_124550

theorem no_multiple_of_five (C : ℕ) : 
  (100 ≤ 100 + 10 * C + 4) ∧ (100 + 10 * C + 4 < 1000) ∧ (C < 10) →
  ¬(∃ k : ℕ, 100 + 10 * C + 4 = 5 * k) := by
sorry

end no_multiple_of_five_l1245_124550


namespace no_integer_solutions_l1245_124527

def ω : ℂ := Complex.I

theorem no_integer_solutions : ∀ a b : ℤ, (Complex.abs (a • ω + b) ≠ Real.sqrt 2) := by
  sorry

end no_integer_solutions_l1245_124527


namespace vertical_pairwise_sets_l1245_124509

-- Definition of a vertical pairwise set
def is_vertical_pairwise_set (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (p₁ : ℝ × ℝ), p₁ ∈ M → ∃ (p₂ : ℝ × ℝ), p₂ ∈ M ∧ p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

-- Define the four sets
def M₁ : Set (ℝ × ℝ) := {p | p.2 = 1 / p.1 ∧ p.1 ≠ 0}
def M₂ : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1 ∧ p.1 > 0}
def M₃ : Set (ℝ × ℝ) := {p | p.2 = Real.exp p.1 - 2}
def M₄ : Set (ℝ × ℝ) := {p | p.2 = Real.sin p.1 + 1}

-- Theorem stating which sets are vertical pairwise sets
theorem vertical_pairwise_sets :
  ¬(is_vertical_pairwise_set M₁) ∧
  ¬(is_vertical_pairwise_set M₂) ∧
  (is_vertical_pairwise_set M₃) ∧
  (is_vertical_pairwise_set M₄) := by
  sorry

end vertical_pairwise_sets_l1245_124509


namespace cubic_polynomial_inequality_l1245_124513

/-- 
A cubic polynomial with real coefficients that has three real roots 
satisfies the inequality 6a^3 + 10(a^2 - 2b)^(3/2) - 12ab ≥ 27c, 
with equality if and only if b = 0, c = -4/27 * a^3, and a ≤ 0.
-/
theorem cubic_polynomial_inequality (a b c : ℝ) : 
  (∃ x y z : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ 
               y^3 + a*y^2 + b*y + c = 0 ∧ 
               z^3 + a*z^2 + b*z + c = 0 ∧ 
               x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  6*a^3 + 10*(a^2 - 2*b)^(3/2) - 12*a*b ≥ 27*c ∧
  (6*a^3 + 10*(a^2 - 2*b)^(3/2) - 12*a*b = 27*c ↔ b = 0 ∧ c = -4/27 * a^3 ∧ a ≤ 0) :=
by sorry

end cubic_polynomial_inequality_l1245_124513


namespace find_A_value_l1245_124573

theorem find_A_value (A : Nat) : A < 10 → (691 - (A * 100 + 87) = 4) → A = 6 := by
  sorry

end find_A_value_l1245_124573


namespace jerry_butterflies_left_l1245_124565

/-- Given that Jerry had 93 butterflies initially and released 11 butterflies,
    prove that he now has 82 butterflies left. -/
theorem jerry_butterflies_left (initial : ℕ) (released : ℕ) (left : ℕ) : 
  initial = 93 → released = 11 → left = initial - released → left = 82 := by
  sorry

end jerry_butterflies_left_l1245_124565


namespace sum_of_digits_square_count_l1245_124582

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The count of valid numbers -/
def K : ℕ := sorry

theorem sum_of_digits_square_count :
  K % 1000 = 632 := by sorry

end sum_of_digits_square_count_l1245_124582


namespace anne_bottle_caps_l1245_124515

/-- Anne's initial bottle cap count -/
def initial_count : ℕ := 10

/-- Number of bottle caps Anne finds -/
def found_count : ℕ := 5

/-- Anne's final bottle cap count -/
def final_count : ℕ := initial_count + found_count

theorem anne_bottle_caps : final_count = 15 := by
  sorry

end anne_bottle_caps_l1245_124515


namespace inequality_system_integer_solutions_l1245_124587

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (5 * x - 1 > 3 * (x + 1) ∧ (1 + 2 * x) / 3 ≥ x - 1) ↔ (x = 3 ∨ x = 4) :=
by sorry

end inequality_system_integer_solutions_l1245_124587


namespace geometric_sequence_general_term_l1245_124519

/-- A geometric sequence {a_n} where a_3 = 2 and a_6 = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ a 3 = 2 ∧ a 6 = 16

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 2^(n - 2)

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : geometric_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = general_term n :=
sorry

end geometric_sequence_general_term_l1245_124519


namespace class_weight_problem_l1245_124501

theorem class_weight_problem (total_boys : Nat) (group_boys : Nat) (group_avg : Real) (total_avg : Real) :
  total_boys = 34 →
  group_boys = 26 →
  group_avg = 50.25 →
  total_avg = 49.05 →
  let remaining_boys := total_boys - group_boys
  let remaining_avg := (total_boys * total_avg - group_boys * group_avg) / remaining_boys
  remaining_avg = 45.15 := by
  sorry

end class_weight_problem_l1245_124501


namespace eagles_win_probability_l1245_124521

/-- The number of games in the series -/
def n : ℕ := 5

/-- The probability of winning a single game -/
def p : ℚ := 1/2

/-- The probability of winning exactly k games out of n -/
def prob_win (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- The probability of winning at least 3 games out of 5 -/
def prob_win_at_least_three : ℚ :=
  prob_win 3 + prob_win 4 + prob_win 5

theorem eagles_win_probability : prob_win_at_least_three = 1/2 := by
  sorry

end eagles_win_probability_l1245_124521


namespace inequality_solution_set_l1245_124553

theorem inequality_solution_set (x : ℝ) : 4 * x < 3 * x + 2 ↔ x < 2 := by
  sorry

end inequality_solution_set_l1245_124553


namespace max_A_at_375_l1245_124563

def A (k : ℕ) : ℝ := (Nat.choose 1500 k) * (0.3 ^ k)

theorem max_A_at_375 : 
  ∀ k : ℕ, k ≤ 1500 → A k ≤ A 375 :=
by sorry

end max_A_at_375_l1245_124563


namespace quadratic_equation_roots_l1245_124516

theorem quadratic_equation_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    2 * x₁^2 + 4 * m * x₁ + m = 0 ∧
    2 * x₂^2 + 4 * m * x₂ + m = 0 ∧
    x₁^2 + x₂^2 = 3/16) →
  m = -1/8 := by
sorry

end quadratic_equation_roots_l1245_124516


namespace num_ways_eq_1716_l1245_124581

/-- The number of distinct ways to choose 8 non-negative integers that sum to 6 -/
def num_ways : ℕ := Nat.choose 13 7

theorem num_ways_eq_1716 : num_ways = 1716 := by sorry

end num_ways_eq_1716_l1245_124581


namespace inequality_preservation_l1245_124514

theorem inequality_preservation (x y : ℝ) : x < y → 3 - x > 3 - y := by
  sorry

end inequality_preservation_l1245_124514


namespace sum_not_ending_in_seven_l1245_124549

theorem sum_not_ending_in_seven (n : ℕ) : ¬ (∃ k : ℕ, n * (n + 1) / 2 = 10 * k + 7) := by
  sorry

end sum_not_ending_in_seven_l1245_124549


namespace sum_of_A_and_B_l1245_124595

/-- The number of four-digit odd numbers divisible by 3 -/
def A : ℕ := sorry

/-- The number of four-digit multiples of 7 -/
def B : ℕ := sorry

/-- The sum of A and B is 2786 -/
theorem sum_of_A_and_B : A + B = 2786 := by sorry

end sum_of_A_and_B_l1245_124595


namespace symmetric_points_product_l1245_124551

/-- Given two points A and B symmetric about the origin, prove their coordinates' product -/
theorem symmetric_points_product (x y : ℝ) : 
  (2008 = -x ∧ y = 1) → x * y = -2008 := by
  sorry

end symmetric_points_product_l1245_124551


namespace first_month_sale_l1245_124510

def average_sale : ℝ := 5400
def num_months : ℕ := 6
def sale_month2 : ℝ := 5366
def sale_month3 : ℝ := 5808
def sale_month4 : ℝ := 5399
def sale_month5 : ℝ := 6124
def sale_month6 : ℝ := 4579

theorem first_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 5124 := by
sorry

end first_month_sale_l1245_124510


namespace repeating_decimal_problem_l1245_124586

/-- The number to be multiplied -/
def n : ℕ := 54

/-- The incorrect multiplier -/
def incorrect_multiplier : ℚ := 2.35

/-- The difference between correct and incorrect results -/
def difference : ℚ := 1.8

/-- The two-digit number formed by the repeating digits -/
def repeating_digits : ℕ := 35

/-- The correct multiplier as a rational number -/
def correct_multiplier : ℚ := 2 + repeating_digits / 99

theorem repeating_decimal_problem :
  n * correct_multiplier - n * incorrect_multiplier = difference :=
sorry

end repeating_decimal_problem_l1245_124586


namespace impossibility_of_simultaneous_inequalities_l1245_124546

theorem impossibility_of_simultaneous_inequalities 
  (a b c : Real) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  ¬(a * (1 - b) > 1/4 ∧ b * (1 - c) > 1/4 ∧ c * (1 - a) > 1/4) := by
sorry

end impossibility_of_simultaneous_inequalities_l1245_124546


namespace problem_solution_l1245_124529

def avg2 (a b : ℚ) : ℚ := (a + b) / 2

def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

theorem problem_solution : 
  avg3 (avg3 2 3 (-1)) (avg2 1 2) 1 = 23 / 18 := by
  sorry

end problem_solution_l1245_124529


namespace probability_four_twos_in_five_rolls_l1245_124500

theorem probability_four_twos_in_five_rolls (p : ℝ) :
  p = 1 / 8 →  -- probability of rolling a 2 on a fair 8-sided die
  (5 : ℝ) * p^4 * (1 - p) = 35 / 4096 := by
  sorry

end probability_four_twos_in_five_rolls_l1245_124500


namespace at_least_one_le_quarter_l1245_124505

theorem at_least_one_le_quarter (a b c : Real) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  (a * (1 - b) ≤ 1/4) ∨ (b * (1 - c) ≤ 1/4) ∨ (c * (1 - a) ≤ 1/4) := by
  sorry

end at_least_one_le_quarter_l1245_124505


namespace willy_crayon_count_l1245_124594

/-- Given that Lucy has 3,971 crayons and Willy has 1,121 more crayons than Lucy,
    prove that Willy has 5,092 crayons. -/
theorem willy_crayon_count (lucy_crayons : ℕ) (willy_extra_crayons : ℕ) 
    (h1 : lucy_crayons = 3971)
    (h2 : willy_extra_crayons = 1121) :
    lucy_crayons + willy_extra_crayons = 5092 := by
  sorry

end willy_crayon_count_l1245_124594


namespace billy_crayons_left_l1245_124511

/-- The number of crayons Billy has left after a hippopotamus eats some. -/
def crayons_left (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten

/-- Theorem stating that Billy has 163 crayons left after starting with 856 and a hippopotamus eating 693. -/
theorem billy_crayons_left : crayons_left 856 693 = 163 := by
  sorry

end billy_crayons_left_l1245_124511


namespace solution_set_absolute_value_equation_l1245_124588

theorem solution_set_absolute_value_equation (x : ℝ) :
  |1 - x| + |2*x - 1| = |3*x - 2| ↔ x ≤ 1/2 ∨ x ≥ 1 :=
sorry

end solution_set_absolute_value_equation_l1245_124588


namespace a_in_range_l1245_124585

/-- The function f(x) = -x^2 - 2ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 - 2*a*x

/-- The maximum value of f(x) on [0, 1] is a^2 -/
def max_value (a : ℝ) : Prop :=
  ∃ (x : ℝ), x ∈ Set.Icc 0 1 ∧ f a x = a^2 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 1 → f a y ≤ a^2

/-- If the maximum value of f(x) on [0, 1] is a^2, then a is in [-1, 0] -/
theorem a_in_range (a : ℝ) (h : max_value a) : a ∈ Set.Icc (-1) 0 := by
  sorry

end a_in_range_l1245_124585


namespace contrapositive_equivalence_contrapositive_true_negation_equivalence_inequality_condition_l1245_124583

-- Statement 1
theorem contrapositive_equivalence (x : ℝ) :
  (x^2 - 3*x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := by sorry

-- Statement 2
theorem contrapositive_true (m : ℝ) :
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) := by sorry

-- Statement 3
theorem negation_equivalence :
  (¬∃ x > 1, x^2 - 2*x - 3 = 0) ↔ (∀ x > 1, x^2 - 2*x - 3 ≠ 0) := by sorry

-- Statement 4
theorem inequality_condition (a : ℝ) :
  (∀ x, -2 < x ∧ x < -1 → (x + a)*(x + 1) < 0) → a > 2 := by sorry

end contrapositive_equivalence_contrapositive_true_negation_equivalence_inequality_condition_l1245_124583


namespace unique_x_with_three_prime_divisors_l1245_124590

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  11 ∣ x →
  (∀ s : ℕ, Prime s → s ∣ x → s = 11 ∨ s = 7 ∨ s = 101) →
  x = 7777 := by
sorry

end unique_x_with_three_prime_divisors_l1245_124590


namespace range_of_a_l1245_124577

def prop_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def prop_q (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) →
  (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) :=
sorry

end range_of_a_l1245_124577


namespace factorial_divides_power_difference_l1245_124518

theorem factorial_divides_power_difference (n : ℕ) : 
  (n.factorial : ℤ) ∣ (2^(2*n.factorial) - 2^n.factorial) :=
sorry

end factorial_divides_power_difference_l1245_124518


namespace divisibility_problem_l1245_124540

theorem divisibility_problem (m n : ℕ) (h : m * n ∣ m + n) :
  (Nat.Prime n → n ∣ m) ∧
  (∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q → ¬(n ∣ m)) := by
  sorry

end divisibility_problem_l1245_124540


namespace sally_score_l1245_124502

/-- Calculates the total score in a math competition given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - 0.25 * (incorrect : ℚ)

/-- Theorem: Sally's total score in the math competition is 15 points. -/
theorem sally_score :
  let correct : ℕ := 17
  let incorrect : ℕ := 8
  let unanswered : ℕ := 5
  calculate_score correct incorrect unanswered = 15 := by
  sorry

end sally_score_l1245_124502


namespace min_value_expression_min_value_achievable_l1245_124564

theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 1) :
  x^2 + 4*y^2 + 2*x*y ≥ 3/4 :=
sorry

theorem min_value_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x^2 + 4*y^2 + 2*x*y = 3/4 :=
sorry

end min_value_expression_min_value_achievable_l1245_124564


namespace equation_rewrite_l1245_124525

theorem equation_rewrite (a x y c : ℤ) :
  ∃ (m n p : ℕ), 
    (m = 4 ∧ n = 3 ∧ p = 4) ∧
    (a^8*x*y - a^7*y - a^6*x = a^5*(c^5 - 1)) ↔ 
    ((a^m*x - a^n)*(a^p*y - a^3) = a^5*c^5) := by
  sorry

end equation_rewrite_l1245_124525


namespace fourth_rectangle_perimeter_is_10_l1245_124572

/-- The perimeter of the fourth rectangle in a large rectangle cut into four smaller ones --/
def fourth_rectangle_perimeter (p1 p2 p3 : ℕ) : ℕ :=
  p1 + p2 - p3

/-- Theorem stating that the perimeter of the fourth rectangle is 10 --/
theorem fourth_rectangle_perimeter_is_10 :
  fourth_rectangle_perimeter 16 18 24 = 10 := by
  sorry

end fourth_rectangle_perimeter_is_10_l1245_124572


namespace birthday_age_multiple_l1245_124539

theorem birthday_age_multiple :
  let current_age : ℕ := 9
  let years_ago : ℕ := 6
  let age_then : ℕ := current_age - years_ago
  current_age / age_then = 3 :=
by sorry

end birthday_age_multiple_l1245_124539


namespace quarter_angle_tangent_line_through_point_line_with_y_intercept_l1245_124562

-- Define the original line
def original_line (x y : ℝ) : Prop := y = -Real.sqrt 3 * x + 1

-- Define the angle that is one fourth of the slope angle
def quarter_angle : ℝ := 30

-- Theorem 1: The tangent of the quarter angle is √3/3
theorem quarter_angle_tangent :
  Real.tan (quarter_angle * π / 180) = Real.sqrt 3 / 3 := by sorry

-- Theorem 2: Equation of the line passing through (√3, -1)
theorem line_through_point (x y : ℝ) :
  (Real.sqrt 3 * x - 3 * y - 6 = 0) ↔
  (y + 1 = (Real.sqrt 3 / 3) * (x - Real.sqrt 3)) := by sorry

-- Theorem 3: Equation of the line with y-intercept -5
theorem line_with_y_intercept (x y : ℝ) :
  (Real.sqrt 3 * x - 3 * y - 15 = 0) ↔
  (y = (Real.sqrt 3 / 3) * x - 5) := by sorry

end quarter_angle_tangent_line_through_point_line_with_y_intercept_l1245_124562


namespace cookie_box_cost_l1245_124557

def bracelet_cost : ℝ := 1
def bracelet_price : ℝ := 1.5
def num_bracelets : ℕ := 12
def money_left : ℝ := 3

theorem cookie_box_cost :
  let profit_per_bracelet := bracelet_price - bracelet_cost
  let total_profit := (num_bracelets : ℝ) * profit_per_bracelet
  total_profit - money_left = 3 := by sorry

end cookie_box_cost_l1245_124557


namespace midpoint_sum_equals_vertex_sum_l1245_124576

theorem midpoint_sum_equals_vertex_sum (a b c d : ℝ) :
  let vertices_sum := a + b + c + d
  let midpoints_sum := (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2
  midpoints_sum = vertices_sum :=
by sorry

end midpoint_sum_equals_vertex_sum_l1245_124576


namespace quadratic_roots_integer_l1245_124569

theorem quadratic_roots_integer (b c : ℤ) (k : ℤ) (h : b^2 - 4*c = k^2) :
  ∃ x1 x2 : ℤ, x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0 :=
sorry

end quadratic_roots_integer_l1245_124569


namespace sum_of_coefficients_l1245_124530

def sequence_v (n : ℕ) : ℚ :=
  sorry

theorem sum_of_coefficients :
  (∃ a b c : ℚ, ∀ n : ℕ, sequence_v n = a * n^2 + b * n + c) →
  (sequence_v 1 = 7) →
  (∀ n : ℕ, sequence_v (n + 1) - sequence_v n = 5 + 6 * (n - 1)) →
  (∃ a b c : ℚ, (∀ n : ℕ, sequence_v n = a * n^2 + b * n + c) ∧ a + b + c = 7) :=
by sorry


end sum_of_coefficients_l1245_124530


namespace walking_scenario_theorem_l1245_124599

/-- Represents the walking scenario with Yolanda, Bob, and Jim -/
structure WalkingScenario where
  total_distance : ℝ
  yolanda_speed : ℝ
  bob_speed_difference : ℝ
  jim_speed : ℝ
  yolanda_head_start : ℝ

/-- Calculates the distance Bob walked when he met Yolanda -/
def bob_distance_walked (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Calculates the point where Jim and Yolanda met, measured from point X -/
def jim_yolanda_meeting_point (scenario : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating the correct distances for Bob and Jim -/
theorem walking_scenario_theorem (scenario : WalkingScenario) 
  (h1 : scenario.total_distance = 80)
  (h2 : scenario.yolanda_speed = 4)
  (h3 : scenario.bob_speed_difference = 2)
  (h4 : scenario.jim_speed = 5)
  (h5 : scenario.yolanda_head_start = 1) :
  bob_distance_walked scenario = 45.6 ∧ 
  jim_yolanda_meeting_point scenario = 38 :=
by sorry

end walking_scenario_theorem_l1245_124599


namespace richard_remaining_distance_l1245_124532

/-- Calculates the remaining distance Richard has to walk to reach New York City. -/
def remaining_distance (total_distance day1_distance day2_fraction day2_reduction day3_distance : ℝ) : ℝ :=
  let day2_distance := day1_distance * day2_fraction - day2_reduction
  let distance_walked := day1_distance + day2_distance + day3_distance
  total_distance - distance_walked

/-- Theorem stating that Richard has 36 miles left to walk to reach New York City. -/
theorem richard_remaining_distance :
  remaining_distance 70 20 (1/2) 6 10 = 36 := by
  sorry

end richard_remaining_distance_l1245_124532


namespace minimum_triangle_area_l1245_124579

/-- Given a line passing through (2, 1) and intersecting the positive x and y axes at points A and B
    respectively, with O as the origin, the minimum area of triangle AOB is 4. -/
theorem minimum_triangle_area (k : ℝ) (h : k < 0) :
  let xA := 2 - 1 / k
  let yB := 1 - 2 * k
  let area := (1 / 2) * xA * yB
  4 ≤ area :=
by sorry

end minimum_triangle_area_l1245_124579


namespace total_amount_is_175_l1245_124574

/-- Represents the share of each person in rupees -/
structure Shares :=
  (first : ℝ)
  (second : ℝ)
  (third : ℝ)

/-- Calculates the total amount given the shares -/
def total_amount (s : Shares) : ℝ :=
  s.first + s.second + s.third

/-- Theorem: The total amount is 175 rupees -/
theorem total_amount_is_175 :
  ∃ (s : Shares),
    s.second = 45 ∧
    s.second = 0.45 * s.first ∧
    s.third = 0.30 * s.first ∧
    total_amount s = 175 :=
by
  sorry

end total_amount_is_175_l1245_124574


namespace max_product_value_l1245_124559

-- Define the functions f and h
def f : ℝ → ℝ := sorry
def h : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value (hf : Set.range f = Set.Icc (-3) 5) 
                          (hh : Set.range h = Set.Icc 0 4) : 
  ∃ x y : ℝ, f x * h y ≤ 20 ∧ ∃ a b : ℝ, f a * h b = 20 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end max_product_value_l1245_124559


namespace quadratic_inequality_solution_l1245_124506

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 - 2 * x + 1 > 7 ↔ x < -2/3 ∨ x > 3 := by
  sorry

end quadratic_inequality_solution_l1245_124506


namespace balance_sheet_equation_l1245_124571

/-- Given the equation 4m - t = 8000 where m = 4 and t = 4 + 100i, prove that t = -7988 - 100i. -/
theorem balance_sheet_equation (m t : ℂ) (h1 : 4 * m - t = 8000) (h2 : m = 4) (h3 : t = 4 + 100 * Complex.I) : 
  t = -7988 - 100 * Complex.I := by
  sorry

end balance_sheet_equation_l1245_124571


namespace quadratic_intersection_l1245_124552

theorem quadratic_intersection
  (a b c d h : ℝ)
  (h_a : a ≠ 0)
  (h_b : b ≠ 0)
  (h_h : h ≠ 0)
  (h_d : d ≠ c) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * (x - h)^2 + b * (x - h) + d
  ∃ x y : ℝ, f x = g x ∧ f x = y ∧ x = (d - c) / b ∧ y = a * ((d - c) / b)^2 + d :=
by sorry

end quadratic_intersection_l1245_124552


namespace f_is_linear_l1245_124543

/-- A function f: ℝ → ℝ is linear if there exist constants m and b such that 
    f(x) = mx + b for all x, where m ≠ 0 -/
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), m ≠ 0 ∧ ∀ x, f x = m * x + b

/-- The function f(x) = -8x -/
def f (x : ℝ) : ℝ := -8 * x

/-- Theorem: f(x) = -8x is a linear function -/
theorem f_is_linear : IsLinearFunction f := by
  sorry


end f_is_linear_l1245_124543


namespace min_value_theorem_l1245_124547

/-- Given positive real numbers a, b, c, and a function f with minimum value 2,
    prove that a + b + c = 2 and the minimum value of 1/a + 1/b + 1/c is 9/2 -/
theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hf : ∀ x, |x + a| + |x - b| + c ≥ 2) :
  (a + b + c = 2) ∧ (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 2 → 1/x + 1/y + 1/z ≥ 9/2) :=
by sorry

end min_value_theorem_l1245_124547


namespace infinite_set_equal_digit_sum_l1245_124575

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number contains zero in its decimal notation -/
def contains_zero (n : ℕ) : Prop := sorry

theorem infinite_set_equal_digit_sum (k : ℕ) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ t ∈ S, ¬contains_zero t ∧ sum_of_digits t = sum_of_digits (k * t) := by
  sorry

end infinite_set_equal_digit_sum_l1245_124575


namespace sequence_fifth_term_l1245_124589

theorem sequence_fifth_term (a : ℕ → ℤ) :
  (∀ n : ℕ, a n = 4 * n - 3) →
  a 5 = 17 := by
sorry

end sequence_fifth_term_l1245_124589


namespace investment_interest_l1245_124526

theorem investment_interest (total_investment : ℝ) (high_rate_investment : ℝ) (high_rate : ℝ) (low_rate : ℝ) :
  total_investment = 22000 →
  high_rate_investment = 7000 →
  high_rate = 0.18 →
  low_rate = 0.14 →
  let low_rate_investment := total_investment - high_rate_investment
  let high_rate_interest := high_rate_investment * high_rate
  let low_rate_interest := low_rate_investment * low_rate
  let total_interest := high_rate_interest + low_rate_interest
  total_interest = 3360 := by
sorry

end investment_interest_l1245_124526


namespace common_difference_is_fifteen_exists_valid_prism_l1245_124503

/-- Represents a rectangular prism with sides that are consecutive multiples of a certain number -/
structure RectangularPrism where
  base_number : ℕ
  common_difference : ℕ

/-- The base area of the rectangular prism -/
def base_area (prism : RectangularPrism) : ℕ :=
  prism.base_number * (prism.base_number + prism.common_difference)

/-- Theorem stating that for a rectangular prism with base area 450,
    the common difference between consecutive multiples is 15 -/
theorem common_difference_is_fifteen (prism : RectangularPrism) 
    (h : base_area prism = 450) : prism.common_difference = 15 := by
  sorry

/-- Proof of the existence of a rectangular prism satisfying the conditions -/
theorem exists_valid_prism : ∃ (prism : RectangularPrism), base_area prism = 450 ∧ prism.common_difference = 15 := by
  sorry

end common_difference_is_fifteen_exists_valid_prism_l1245_124503


namespace similar_triangles_side_length_l1245_124596

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
structure SimilarTriangles (P Q R X Y Z : ℝ × ℝ) : Prop where
  ratio_eq : (dist P Q) / (dist X Y) = (dist Q R) / (dist Y Z)

theorem similar_triangles_side_length 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles P Q R X Y Z) 
  (h_PQ : dist P Q = 9)
  (h_QR : dist Q R = 15)
  (h_YZ : dist Y Z = 30) :
  dist X Y = 18 := by
sorry

end similar_triangles_side_length_l1245_124596


namespace book_count_theorem_l1245_124567

/-- Represents the book collection of a person -/
structure BookCollection where
  initial : ℕ
  bought : ℕ
  lost : ℕ
  borrowed : ℕ

/-- Calculates the current number of books in a collection -/
def current_books (collection : BookCollection) : ℕ :=
  collection.initial - collection.lost

/-- Calculates the future number of books in a collection -/
def future_books (collection : BookCollection) : ℕ :=
  current_books collection + collection.bought + collection.borrowed

/-- Jason's book collection -/
def jason : BookCollection :=
  { initial := 18, bought := 8, lost := 0, borrowed := 0 }

/-- Mary's book collection -/
def mary : BookCollection :=
  { initial := 42, bought := 0, lost := 6, borrowed := 5 }

theorem book_count_theorem :
  (current_books jason + current_books mary = 54) ∧
  (future_books jason + future_books mary = 67) := by
  sorry

end book_count_theorem_l1245_124567


namespace waiter_tips_theorem_l1245_124593

/-- Calculates the total tips earned by a waiter --/
def total_tips (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) : ℕ :=
  (total_customers - non_tipping_customers) * tip_amount

/-- Proves that the waiter earned $15 in tips --/
theorem waiter_tips_theorem :
  total_tips 10 5 3 = 15 := by
  sorry

end waiter_tips_theorem_l1245_124593


namespace bakery_storage_ratio_l1245_124598

/-- Proves that the ratio of sugar to flour is 3:8 given the conditions in the bakery storage room --/
theorem bakery_storage_ratio : ∀ (flour baking_soda : ℕ),
  flour = 10 * baking_soda →
  flour = 8 * (baking_soda + 60) →
  (900 : ℕ) / flour = 3 / 8 :=
by
  sorry

#check bakery_storage_ratio

end bakery_storage_ratio_l1245_124598


namespace meal_serving_problem_l1245_124568

/-- The number of derangements of n elements -/
def subfactorial (n : ℕ) : ℕ := sorry

/-- The number of ways to serve meals to exactly three people correctly -/
def waysToServeThreeCorrectly (totalPeople : ℕ) (mealTypes : ℕ) (peoplePerMeal : ℕ) : ℕ :=
  Nat.choose totalPeople 3 * subfactorial (totalPeople - 3)

theorem meal_serving_problem :
  waysToServeThreeCorrectly 15 3 5 = 80157776755 := by sorry

end meal_serving_problem_l1245_124568


namespace red_envelope_prob_is_one_third_l1245_124597

def red_envelope_prob : ℚ :=
  let total_people : ℕ := 4
  let one_yuan_envelopes : ℕ := 3
  let five_yuan_envelopes : ℕ := 1
  let total_events : ℕ := Nat.choose total_people 2
  let favorable_events : ℕ := one_yuan_envelopes * five_yuan_envelopes
  favorable_events / total_events

theorem red_envelope_prob_is_one_third : 
  red_envelope_prob = 1/3 := by sorry

end red_envelope_prob_is_one_third_l1245_124597


namespace range_of_a_l1245_124536

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x + y + z = 1 → |a - 2| ≤ x^2 + 2*y^2 + 3*z^2) →
  16/11 ≤ a ∧ a ≤ 28/11 := by
sorry

end range_of_a_l1245_124536


namespace sara_apples_l1245_124566

theorem sara_apples (total : ℕ) (ali_factor : ℕ) (sara_apples : ℕ) : 
  total = 240 →
  ali_factor = 7 →
  total = sara_apples + ali_factor * sara_apples →
  sara_apples = 30 := by
sorry

end sara_apples_l1245_124566


namespace cot_15_plus_tan_45_l1245_124555

theorem cot_15_plus_tan_45 : Real.cos (15 * π / 180) / Real.sin (15 * π / 180) + Real.tan (45 * π / 180) = 3 + Real.sqrt 3 := by
  sorry

end cot_15_plus_tan_45_l1245_124555


namespace average_equation_l1245_124537

theorem average_equation (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 89 → a = 34 := by
  sorry

end average_equation_l1245_124537


namespace sector_arc_length_l1245_124534

/-- Given a circular sector with area 4 and central angle 2 radians, prove that the length of the arc is 4. -/
theorem sector_arc_length (S : ℝ) (θ : ℝ) (l : ℝ) : 
  S = 4 → θ = 2 → l = S * θ / 2 → l = 4 :=
by sorry

end sector_arc_length_l1245_124534


namespace teacher_age_l1245_124554

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students : ℝ) * student_avg_age + 36 = (num_students + 1 : ℝ) * total_avg_age :=
by sorry

end teacher_age_l1245_124554


namespace inverse_of_M_l1245_124507

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, -1]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![4, 1; 2, 3]
def M : Matrix (Fin 2) (Fin 2) ℚ := B * A

theorem inverse_of_M :
  M⁻¹ = !![3/10, -1/10; 1/5, -2/5] := by sorry

end inverse_of_M_l1245_124507


namespace smallest_four_digit_divisible_by_35_second_smallest_four_digit_divisible_by_35_l1245_124504

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_35 (n : ℕ) : Prop := n % 35 = 0

theorem smallest_four_digit_divisible_by_35 :
  (∀ n : ℕ, is_four_digit n ∧ divisible_by_35 n → 1005 ≤ n) ∧
  is_four_digit 1005 ∧ divisible_by_35 1005 :=
sorry

theorem second_smallest_four_digit_divisible_by_35 :
  (∀ n : ℕ, is_four_digit n ∧ divisible_by_35 n ∧ n ≠ 1005 → 1045 ≤ n) ∧
  is_four_digit 1045 ∧ divisible_by_35 1045 :=
sorry

end smallest_four_digit_divisible_by_35_second_smallest_four_digit_divisible_by_35_l1245_124504


namespace water_fountain_trips_l1245_124561

/-- The number of trips to the water fountain -/
def number_of_trips (total_distance : ℕ) (distance_to_fountain : ℕ) : ℕ :=
  total_distance / distance_to_fountain

theorem water_fountain_trips : 
  number_of_trips 120 30 = 4 := by
  sorry

end water_fountain_trips_l1245_124561


namespace pushups_percentage_l1245_124538

def jumping_jacks : ℕ := 12
def pushups : ℕ := 8
def situps : ℕ := 20

def total_exercises : ℕ := jumping_jacks + pushups + situps

def percentage_pushups : ℚ := (pushups : ℚ) / (total_exercises : ℚ) * 100

theorem pushups_percentage : percentage_pushups = 20 := by
  sorry

end pushups_percentage_l1245_124538


namespace kelly_games_left_l1245_124548

theorem kelly_games_left (initial_games : ℝ) (games_given_away : ℕ) : 
  initial_games = 121.0 →
  games_given_away = 99 →
  initial_games - games_given_away = 22.0 := by
  sorry

end kelly_games_left_l1245_124548


namespace three_power_plus_one_not_divisible_l1245_124545

theorem three_power_plus_one_not_divisible (n : ℕ) 
  (h_odd : Odd n) (h_gt_one : n > 1) : ¬(n ∣ 3^n + 1) := by
  sorry

end three_power_plus_one_not_divisible_l1245_124545


namespace fourth_root_equivalence_l1245_124520

theorem fourth_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x / x^(1/2))^(1/4) = x^(1/8) := by
  sorry

end fourth_root_equivalence_l1245_124520


namespace work_completion_time_l1245_124558

theorem work_completion_time (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  (1 / x = 1 / 30) →
  (1 / x + 1 / y = 1 / 18) →
  y = 45 := by
sorry

end work_completion_time_l1245_124558


namespace digit_placement_l1245_124512

theorem digit_placement (n : ℕ) (h : n < 10) :
  100 + 10 * n + 1 = 101 + 10 * n := by
  sorry

end digit_placement_l1245_124512


namespace f_min_at_three_l1245_124592

/-- The quadratic function f(x) = x^2 - 6x + 5 -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- Theorem stating that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_three : 
  ∀ x : ℝ, f x ≥ f 3 := by
sorry

end f_min_at_three_l1245_124592


namespace binomial_not_divisible_l1245_124580

theorem binomial_not_divisible (P n k : ℕ) (hP : P > 1) :
  ∃ i ∈ Finset.range (k + 1), ¬(P ∣ Nat.choose (n + i) k) := by
  sorry

end binomial_not_divisible_l1245_124580


namespace sqrt_greater_than_3x_iff_less_than_one_ninth_l1245_124535

theorem sqrt_greater_than_3x_iff_less_than_one_ninth 
  (x : ℝ) (hx : x > 0) : 
  Real.sqrt x > 3 * x ↔ x < 1/9 := by
  sorry

end sqrt_greater_than_3x_iff_less_than_one_ninth_l1245_124535


namespace calculation_proof_l1245_124541

theorem calculation_proof :
  ((-48) * 0.125 + 48 * (11/8) + (-48) * (5/4) = 0) ∧
  (|(-7/9)| / ((2/3) - (1/5)) - (1/3) * ((-4)^2) = -11/3) := by
  sorry

end calculation_proof_l1245_124541


namespace f_monotonicity_and_extrema_l1245_124544

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2*a - 1) * x - Real.log x

theorem f_monotonicity_and_extrema :
  (∀ x > 0, ∀ a : ℝ,
    (a = 1/2 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ∧
      (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
      (f a 1 = 1/2) ∧
      (∀ x ≠ 1, f a x > 1/2)) ∧
    (a ≤ 0 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) ∧
    (a > 0 →
      (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/(2*a) → f a x₁ > f a x₂) ∧
      (∀ x₁ x₂, 1/(2*a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))) :=
by sorry

end f_monotonicity_and_extrema_l1245_124544


namespace percentage_of_science_students_l1245_124584

theorem percentage_of_science_students (total_boys : ℕ) (school_A_percentage : ℚ) (non_science_boys : ℕ) : 
  total_boys = 250 →
  school_A_percentage = 1/5 →
  non_science_boys = 35 →
  (((school_A_percentage * total_boys) - non_science_boys) / (school_A_percentage * total_boys)) = 3/10 := by
  sorry

end percentage_of_science_students_l1245_124584


namespace sin_29pi_over_6_l1245_124528

theorem sin_29pi_over_6 : Real.sin (29 * π / 6) = 1 / 2 := by
  sorry

end sin_29pi_over_6_l1245_124528


namespace brick_length_calculation_l1245_124542

/-- Calculates the length of a brick given wall dimensions, partial brick dimensions, and number of bricks --/
theorem brick_length_calculation (wall_length wall_width wall_height brick_width brick_height num_bricks : ℝ) :
  wall_length = 800 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_width = 11.25 →
  brick_height = 6 →
  num_bricks = 3200 →
  (wall_length * wall_width * wall_height) / (num_bricks * brick_width * brick_height) = 50 := by
  sorry

#check brick_length_calculation

end brick_length_calculation_l1245_124542


namespace infinite_greater_than_index_l1245_124508

/-- A sequence of integers -/
def IntegerSequence := ℕ → ℤ

/-- Property: all elements in the sequence are pairwise distinct -/
def PairwiseDistinct (a : IntegerSequence) : Prop :=
  ∀ i j, i ≠ j → a i ≠ a j

/-- Property: all elements in the sequence are greater than 1 -/
def AllGreaterThanOne (a : IntegerSequence) : Prop :=
  ∀ k, a k > 1

/-- The main theorem -/
theorem infinite_greater_than_index
  (a : IntegerSequence)
  (h_distinct : PairwiseDistinct a)
  (h_greater : AllGreaterThanOne a) :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ (∀ k ∈ S, a k > k) :=
sorry

end infinite_greater_than_index_l1245_124508


namespace circle_symmetry_l1245_124517

/-- Given a circle C1 with equation (x+1)^2 + (y-1)^2 = 1, 
    prove that the circle C2 with equation (x-2)^2 + (y+2)^2 = 1 
    is symmetric to C1 with respect to the line x - y - 1 = 0 -/
theorem circle_symmetry (x y : ℝ) : 
  (∀ x y, (x + 1)^2 + (y - 1)^2 = 1 → 
    ∃ x' y', x' - y' = -(x - y) ∧ (x' + 1)^2 + (y' - 1)^2 = 1) → 
  (x - 2)^2 + (y + 2)^2 = 1 :=
by sorry

end circle_symmetry_l1245_124517


namespace quadratic_root_value_l1245_124578

theorem quadratic_root_value (c : ℝ) : 
  (∀ x : ℝ, (5/2 * x^2 + 17*x + c = 0) ↔ (x = (-17 + Real.sqrt 23) / 5 ∨ x = (-17 - Real.sqrt 23) / 5)) 
  → c = 26.6 := by
sorry

end quadratic_root_value_l1245_124578


namespace minimum_value_of_f_minimum_value_case1_minimum_value_case2_l1245_124560

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the theorem
theorem minimum_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) a, f x ≥ f a ∧ -2 < a ∧ a ≤ 1) ∨
  (∀ x ∈ Set.Icc (-2) a, f x ≥ -1 ∧ a > 1) := by
  sorry

-- Define helper theorems for each case
theorem minimum_value_case1 (a : ℝ) (h1 : -2 < a) (h2 : a ≤ 1) :
  ∀ x ∈ Set.Icc (-2) a, f x ≥ f a := by
  sorry

theorem minimum_value_case2 (a : ℝ) (h : a > 1) :
  ∀ x ∈ Set.Icc (-2) a, f x ≥ -1 := by
  sorry

end minimum_value_of_f_minimum_value_case1_minimum_value_case2_l1245_124560
