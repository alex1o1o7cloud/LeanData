import Mathlib

namespace train_speed_calculation_l1628_162841

/-- Proves that a train with given length, crossing a platform of given length in a specific time, has a specific speed. -/
theorem train_speed_calculation (train_length platform_length : ℝ) (crossing_time : ℝ) :
  train_length = 480 ∧ 
  platform_length = 620 ∧ 
  crossing_time = 71.99424046076314 →
  ∃ (speed : ℝ), abs (speed - 54.964) < 0.001 ∧ 
  speed = (train_length + platform_length) / crossing_time * 3.6 := by
  sorry

end train_speed_calculation_l1628_162841


namespace toby_breakfast_pb_servings_l1628_162829

/-- Calculates the number of peanut butter servings needed for a target calorie count -/
def peanut_butter_servings (target_calories : ℕ) (bread_calories : ℕ) (pb_calories_per_serving : ℕ) : ℕ :=
  ((target_calories - bread_calories) / pb_calories_per_serving)

/-- Proves that 2 servings of peanut butter are needed for Toby's breakfast -/
theorem toby_breakfast_pb_servings :
  peanut_butter_servings 500 100 200 = 2 := by
  sorry

#eval peanut_butter_servings 500 100 200

end toby_breakfast_pb_servings_l1628_162829


namespace cookies_remaining_cookies_remaining_result_l1628_162810

/-- Calculates the number of cookies remaining in Cristian's jar --/
theorem cookies_remaining (initial_white : ℕ) (black_white_diff : ℕ) : ℕ :=
  let initial_black := initial_white + black_white_diff
  let remaining_white := initial_white - (3 * initial_white / 4)
  let remaining_black := initial_black - (initial_black / 2)
  remaining_white + remaining_black

/-- Proves that the number of cookies remaining is 85 --/
theorem cookies_remaining_result : cookies_remaining 80 50 = 85 := by
  sorry

end cookies_remaining_cookies_remaining_result_l1628_162810


namespace point_outside_circle_l1628_162806

theorem point_outside_circle (m : ℝ) : 
  (1 : ℝ)^2 + (1 : ℝ)^2 + 4*m*1 - 2*1 + 5*m > 0 ∧ 
  ∃ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ 
  m > 1 ∨ (0 < m ∧ m < 1/4) := by
sorry

end point_outside_circle_l1628_162806


namespace quadratic_roots_implications_l1628_162859

theorem quadratic_roots_implications (a b c : ℝ) 
  (h_roots : ∃ (α β : ℝ), α > 0 ∧ β ≠ 0 ∧ 
    (∀ x : ℂ, x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 ↔ x = α + β * I ∨ x = α - β * I)) :
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (Real.sqrt a + Real.sqrt b > Real.sqrt c ∧
   Real.sqrt b + Real.sqrt c > Real.sqrt a ∧
   Real.sqrt c + Real.sqrt a > Real.sqrt b) := by
  sorry

end quadratic_roots_implications_l1628_162859


namespace absolute_value_sum_l1628_162852

theorem absolute_value_sum (a b : ℝ) : 
  (abs a = 3) → (abs b = 4) → (a * b < 0) → abs (a + b) = 1 := by
  sorry

end absolute_value_sum_l1628_162852


namespace median_invariant_after_remove_min_max_l1628_162878

/-- A function that returns the median of a list of real numbers -/
def median (l : List ℝ) : ℝ := sorry

/-- A function that removes the minimum and maximum elements from a list -/
def removeMinMax (l : List ℝ) : List ℝ := sorry

theorem median_invariant_after_remove_min_max (data : List ℝ) :
  data.length > 2 →
  data.Nodup →
  median data = median (removeMinMax data) :=
sorry

end median_invariant_after_remove_min_max_l1628_162878


namespace arithmetic_sequence_common_difference_l1628_162844

/-- 
Proves that in an arithmetic sequence with given conditions, 
the common difference is 3.
-/
theorem arithmetic_sequence_common_difference 
  (a : ℝ) (an : ℝ) (n : ℕ) (sum : ℝ) :
  a = 2 →
  an = 50 →
  sum = 442 →
  an = a + (n - 1) * (3 : ℝ) →
  sum = (n / 2) * (a + an) →
  (3 : ℝ) = (an - a) / (n - 1) := by
sorry

end arithmetic_sequence_common_difference_l1628_162844


namespace triangle_side_length_l1628_162858

theorem triangle_side_length (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angleB := Real.arccos ((BC^2 + (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))^2 - (Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2))^2) / (2 * BC * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)))
  let area := (1/2) * BC * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sin angleB
  BC = 1 → angleB = π/3 → area = Real.sqrt 3 → 
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = Real.sqrt 13 := by
sorry

end triangle_side_length_l1628_162858


namespace sin_A_value_l1628_162868

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- State the theorem
theorem sin_A_value (t : Triangle) 
  (h1 : t.a = 1) 
  (h2 : t.b = Real.sqrt 3) 
  (h3 : t.A + t.C = 2 * t.B) 
  (h4 : t.A + t.B + t.C = Real.pi) 
  (h5 : t.a / Real.sin t.A = t.b / Real.sin t.B) : 
  Real.sin t.A = 1 / 2 := by
  sorry

end sin_A_value_l1628_162868


namespace equation_solution_l1628_162860

-- Define the functions
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4
def g (x : ℝ) : ℝ := x^4 + 2*x^3 + x^2 + 11*x + 11
def h (x : ℝ) : ℝ := x + 1

-- Define the set of solutions
def solution_set : Set ℝ := {x | x = 1 ∨ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2}

-- State the theorem
theorem equation_solution :
  ∀ x ∈ solution_set, ∃ y, f y = g x ∧ y = h x :=
by sorry

end equation_solution_l1628_162860


namespace office_age_problem_l1628_162880

theorem office_age_problem (total_people : Nat) (group1_people : Nat) (group2_people : Nat)
  (total_avg_age : ℝ) (group1_avg_age : ℝ) (group2_avg_age : ℝ)
  (h1 : total_people = 16)
  (h2 : group1_people = 5)
  (h3 : group2_people = 9)
  (h4 : total_avg_age = 15)
  (h5 : group1_avg_age = 14)
  (h6 : group2_avg_age = 16) :
  (total_people : ℝ) * total_avg_age - 
  (group1_people : ℝ) * group1_avg_age - 
  (group2_people : ℝ) * group2_avg_age = 52 := by
sorry

end office_age_problem_l1628_162880


namespace green_balls_count_l1628_162809

theorem green_balls_count (blue_count : ℕ) (ratio_blue : ℕ) (ratio_green : ℕ) (green_count : ℕ) : 
  blue_count = 20 → 
  ratio_blue = 5 → 
  ratio_green = 3 → 
  blue_count * ratio_green = green_count * ratio_blue → 
  green_count = 12 := by
sorry

end green_balls_count_l1628_162809


namespace existence_of_special_number_l1628_162898

/-- A function that computes the sum of digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number contains only digits 2-9. -/
def contains_only_2_to_9 (n : ℕ) : Prop := sorry

/-- The number of digits in a natural number. -/
def num_digits (n : ℕ) : ℕ := sorry

theorem existence_of_special_number :
  ∃ N : ℕ, 
    num_digits N = 2020 ∧ 
    contains_only_2_to_9 N ∧ 
    N % sum_of_digits N = 0 := by
  sorry

end existence_of_special_number_l1628_162898


namespace complex_simplification_l1628_162867

theorem complex_simplification :
  ((2 + Complex.I) ^ 200) / ((2 - Complex.I) ^ 200) = Complex.exp (200 * Complex.I * Complex.arctan (4 / 3)) := by
  sorry

end complex_simplification_l1628_162867


namespace only_setC_is_pythagorean_triple_l1628_162833

-- Define the sets of numbers
def setA : List ℕ := [1, 2, 2]
def setB : List ℕ := [3^2, 4^2, 5^2]
def setC : List ℕ := [5, 12, 13]
def setD : List ℕ := [6, 6, 6]

-- Define a function to check if a list of three numbers forms a Pythagorean triple
def isPythagoreanTriple (triple : List ℕ) : Prop :=
  match triple with
  | [a, b, c] => a^2 + b^2 = c^2
  | _ => False

-- Theorem stating that only setC forms a Pythagorean triple
theorem only_setC_is_pythagorean_triple :
  ¬(isPythagoreanTriple setA) ∧
  ¬(isPythagoreanTriple setB) ∧
  (isPythagoreanTriple setC) ∧
  ¬(isPythagoreanTriple setD) :=
sorry

end only_setC_is_pythagorean_triple_l1628_162833


namespace solution_exists_l1628_162846

/-- The system of equations has at least one solution if and only if a is in the specified set -/
theorem solution_exists (a : ℝ) : 
  (∃ x y : ℝ, x - 1 = a * (y^3 - 1) ∧ 
              2 * x / (|y^3| + y^3) = Real.sqrt x ∧ 
              y > 0 ∧ x ≥ 0) ↔ 
  a < 0 ∨ (0 ≤ a ∧ a < 1) ∨ a > 1 :=
by sorry

end solution_exists_l1628_162846


namespace greatest_x_value_l1628_162849

theorem greatest_x_value (x : ℤ) : 
  (2.134 * (10 : ℝ) ^ (x : ℝ) < 21000) ↔ x ≤ 3 :=
sorry

end greatest_x_value_l1628_162849


namespace a_range_l1628_162845

/-- A quadratic function f(x) = x² + 2(a-1)x + 5 that is increasing on (4, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 5

/-- The property that f is increasing on (4, +∞) -/
def f_increasing (a : ℝ) : Prop :=
  ∀ x y, x > 4 → y > x → f a y > f a x

/-- The theorem stating the range of a -/
theorem a_range (a : ℝ) (h : f_increasing a) : a ≥ -3 :=
sorry

end a_range_l1628_162845


namespace largest_number_l1628_162811

theorem largest_number (π : ℝ) (h1 : π > 3) : π = max π (max (Real.sqrt 2) (max (-2) 3)) := by
  sorry

end largest_number_l1628_162811


namespace inequality_solution_l1628_162876

theorem inequality_solution (a : ℝ) : 
  (∃ x : ℝ, 2 * x - (1/3) * a ≤ 0 ∧ x ≤ 2) → a = 12 :=
by sorry

end inequality_solution_l1628_162876


namespace faster_train_speed_l1628_162870

theorem faster_train_speed 
  (train_length : ℝ) 
  (speed_difference : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 75) 
  (h2 : speed_difference = 36) 
  (h3 : passing_time = 54) : 
  ∃ (faster_speed : ℝ), faster_speed = 46 := by
  sorry

end faster_train_speed_l1628_162870


namespace solution_satisfies_equation_l1628_162818

open Real

noncomputable def y (x C₁ C₂ : ℝ) : ℝ :=
  C₁ * cos (2 * x) + C₂ * sin (2 * x) + (2 * cos (2 * x) + 8 * sin (2 * x)) * x + (1/2) * exp (2 * x)

theorem solution_satisfies_equation (x C₁ C₂ : ℝ) :
  (deriv^[2] (y C₁ C₂)) x + 4 * y C₁ C₂ x = -8 * sin (2 * x) + 32 * cos (2 * x) + 4 * exp (2 * x) := by
  sorry

end solution_satisfies_equation_l1628_162818


namespace largest_constant_inequality_l1628_162887

theorem largest_constant_inequality (x y z : ℝ) :
  ∃ (C : ℝ), C = Real.sqrt 2 ∧ 
  (∀ (x y z : ℝ), x^2 + y^2 + z^3 + 1 ≥ C * (x + y + z)) ∧
  (∀ (C' : ℝ), C' > C → ∃ (x y z : ℝ), x^2 + y^2 + z^3 + 1 < C' * (x + y + z)) :=
by sorry

end largest_constant_inequality_l1628_162887


namespace distance_AB_is_130_l1628_162885

-- Define the speeds of the three people
def speed_A : ℝ := 3
def speed_B : ℝ := 2
def speed_C : ℝ := 1

-- Define the initial distance traveled by A
def initial_distance_A : ℝ := 50

-- Define the distance between C and D
def distance_CD : ℝ := 12

-- Theorem statement
theorem distance_AB_is_130 :
  let total_distance := 4 * (speed_A + speed_B + speed_C) * distance_CD + initial_distance_A
  total_distance = 130 := by sorry

end distance_AB_is_130_l1628_162885


namespace blue_balls_count_l1628_162873

theorem blue_balls_count (total : ℕ) 
  (h_green : (1 : ℚ) / 4 * total = (total / 4 : ℕ))
  (h_blue : (1 : ℚ) / 8 * total = (total / 8 : ℕ))
  (h_yellow : (1 : ℚ) / 12 * total = (total / 12 : ℕ))
  (h_white : total - (total / 4 + total / 8 + total / 12) = 26) :
  total / 8 = 6 := by
sorry

end blue_balls_count_l1628_162873


namespace tan_sum_special_l1628_162807

theorem tan_sum_special : Real.tan (10 * π / 180) + Real.tan (50 * π / 180) + Real.sqrt 3 * Real.tan (10 * π / 180) * Real.tan (50 * π / 180) = Real.sqrt 3 := by
  sorry

end tan_sum_special_l1628_162807


namespace largest_prime_factor_of_expression_l1628_162804

theorem largest_prime_factor_of_expression : 
  let expr := 15^4 + 2*15^2 + 1 - 14^4
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ expr ∧ p = 211 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ expr → q ≤ p :=
by sorry

end largest_prime_factor_of_expression_l1628_162804


namespace greatest_common_multiple_8_12_under_90_l1628_162865

theorem greatest_common_multiple_8_12_under_90 : 
  ∃ (n : ℕ), n = 72 ∧ 
  (∀ m : ℕ, m < 90 → m % 8 = 0 → m % 12 = 0 → m ≤ n) ∧
  72 % 8 = 0 ∧ 72 % 12 = 0 ∧ 72 < 90 :=
by sorry

end greatest_common_multiple_8_12_under_90_l1628_162865


namespace birthday_stickers_proof_l1628_162825

/-- The number of stickers James had initially -/
def initial_stickers : ℕ := 39

/-- The number of stickers James had after his birthday -/
def final_stickers : ℕ := 61

/-- The number of stickers James got for his birthday -/
def birthday_stickers : ℕ := final_stickers - initial_stickers

theorem birthday_stickers_proof :
  birthday_stickers = final_stickers - initial_stickers :=
by sorry

end birthday_stickers_proof_l1628_162825


namespace unique_g_two_num_solutions_sum_solutions_final_result_l1628_162803

/-- A function satisfying the given property for all real x and y -/
def SatisfiesProperty (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g x * g y - g (x * y) = 2 * x + 2 * y

/-- The main theorem stating that g(2) = 3 is the only solution -/
theorem unique_g_two (g : ℝ → ℝ) (h : SatisfiesProperty g) : g 2 = 3 := by
  sorry

/-- The number of possible values for g(2) is 1 -/
theorem num_solutions (g : ℝ → ℝ) (h : SatisfiesProperty g) : 
  ∃! x : ℝ, g 2 = x := by
  sorry

/-- The sum of all possible values of g(2) is 3 -/
theorem sum_solutions (g : ℝ → ℝ) (h : SatisfiesProperty g) :
  ∃ x : ℝ, g 2 = x ∧ x = 3 := by
  sorry

/-- The product of the number of solutions and their sum is 3 -/
theorem final_result (g : ℝ → ℝ) (h : SatisfiesProperty g) :
  (∃! x : ℝ, g 2 = x) ∧ (∃ x : ℝ, g 2 = x ∧ x = 3) → 1 * 3 = 3 := by
  sorry

end unique_g_two_num_solutions_sum_solutions_final_result_l1628_162803


namespace triangle_inequality_l1628_162821

theorem triangle_inequality (a b c : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b)
  (perimeter : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2*(1 - a*b*c) := by
  sorry

end triangle_inequality_l1628_162821


namespace tan_alpha_plus_pi_fourth_l1628_162822

-- Define the angle α
def α : Real := sorry

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem tan_alpha_plus_pi_fourth (h : P.fst = -Real.tan α ∧ P.snd = Real.tan α * P.fst) :
  Real.tan (α + π/4) = -1/3 := by sorry

end tan_alpha_plus_pi_fourth_l1628_162822


namespace exponent_division_l1628_162851

theorem exponent_division (a : ℝ) : a^4 / a^2 = a^2 := by
  sorry

end exponent_division_l1628_162851


namespace coin_distribution_l1628_162853

theorem coin_distribution (x y k : ℕ) (hxy : x + y = 81) (hne : x ≠ y) 
  (hsq : x^2 - y^2 = k * (x - y)) : k = 81 := by
  sorry

end coin_distribution_l1628_162853


namespace square_root_squared_specific_case_l1628_162843

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n)^2 = n := by sorry

theorem specific_case : (Real.sqrt 987654)^2 = 987654 := by sorry

end square_root_squared_specific_case_l1628_162843


namespace magazine_purchase_combinations_l1628_162866

-- Define the number of magazines and their prices
def total_magazines : ℕ := 11
def magazines_2yuan : ℕ := 8
def magazines_1yuan : ℕ := 3
def total_money : ℕ := 10

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the theorem
theorem magazine_purchase_combinations : 
  (combination magazines_1yuan 2 * combination magazines_2yuan 4) + 
  (combination magazines_2yuan 5) = 266 := by
  sorry

#check magazine_purchase_combinations

end magazine_purchase_combinations_l1628_162866


namespace greatest_digit_sum_base_nine_l1628_162819

/-- 
Given a positive integer n less than 5000, returns the sum of digits
in its base-nine representation.
-/
def sumOfDigitsBaseNine (n : ℕ) : ℕ := sorry

/-- 
The greatest possible sum of the digits in the base-nine representation
of a positive integer less than 5000.
-/
def maxDigitSum : ℕ := 26

theorem greatest_digit_sum_base_nine :
  ∀ n : ℕ, n < 5000 → sumOfDigitsBaseNine n ≤ maxDigitSum :=
sorry

end greatest_digit_sum_base_nine_l1628_162819


namespace square_sum_from_difference_and_product_l1628_162812

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 118 := by
sorry

end square_sum_from_difference_and_product_l1628_162812


namespace mrs_petersons_change_l1628_162881

def change_calculation (num_tumblers : ℕ) (cost_per_tumbler : ℚ) (discount_rate : ℚ) (num_bills : ℕ) (bill_value : ℚ) : ℚ :=
  let total_cost_before_discount := num_tumblers * cost_per_tumbler
  let discount_amount := total_cost_before_discount * discount_rate
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let total_amount_paid := num_bills * bill_value
  total_amount_paid - total_cost_after_discount

theorem mrs_petersons_change :
  change_calculation 10 45 (1/10) 5 100 = 95 :=
by
  sorry

end mrs_petersons_change_l1628_162881


namespace number_of_observations_l1628_162815

theorem number_of_observations (initial_mean : ℝ) (wrong_obs : ℝ) (correct_obs : ℝ) (new_mean : ℝ)
  (h1 : initial_mean = 36)
  (h2 : wrong_obs = 23)
  (h3 : correct_obs = 45)
  (h4 : new_mean = 36.5) :
  ∃ (n : ℕ), n * initial_mean + (correct_obs - wrong_obs) = n * new_mean ∧ n = 44 := by
sorry

end number_of_observations_l1628_162815


namespace blocks_added_l1628_162801

/-- 
Given:
- initial_blocks: The initial number of blocks in Adolfo's tower
- final_blocks: The final number of blocks in Adolfo's tower

Prove that the number of blocks added is equal to the difference between 
the final and initial number of blocks.
-/
theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 35)
  (h2 : final_blocks = 65) : 
  final_blocks - initial_blocks = 30 := by
  sorry

end blocks_added_l1628_162801


namespace product_of_seven_consecutive_divisible_by_ten_l1628_162842

theorem product_of_seven_consecutive_divisible_by_ten (n : ℕ+) :
  10 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) := by
  sorry

end product_of_seven_consecutive_divisible_by_ten_l1628_162842


namespace min_people_for_tests_l1628_162861

/-- The minimum number of people required to achieve the given score ranges -/
def min_people (ranges : List ℕ) (min_range : ℕ) : ℕ :=
  if ranges.maximum = some min_range then 2 else 1

/-- Theorem: Given the conditions, at least 2 people took the tests -/
theorem min_people_for_tests : min_people [17, 28, 35, 45] 45 = 2 := by
  sorry

end min_people_for_tests_l1628_162861


namespace sufficient_not_necessary_l1628_162886

theorem sufficient_not_necessary (a b : ℝ) : 
  (∀ a b : ℝ, |a - b^2| + |b - a^2| ≤ 1 → (a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧
  (∃ a b : ℝ, (a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2 ∧ |a - b^2| + |b - a^2| > 1) :=
by sorry

end sufficient_not_necessary_l1628_162886


namespace trig_problem_l1628_162840

theorem trig_problem (A : Real) (h1 : 0 < A ∧ A < π) (h2 : Real.sin A + Real.cos A = 1/5) : 
  Real.sin A * Real.cos A = -12/25 ∧ Real.tan A = -4/3 := by
  sorry

end trig_problem_l1628_162840


namespace running_yardage_difference_l1628_162823

def player_yardage (total_yards pass_yards : ℕ) : ℕ :=
  total_yards - pass_yards

theorem running_yardage_difference (
  player_a_total player_a_pass player_b_total player_b_pass : ℕ
) (h1 : player_a_total = 150)
  (h2 : player_a_pass = 60)
  (h3 : player_b_total = 180)
  (h4 : player_b_pass = 80) :
  (player_yardage player_a_total player_a_pass : ℤ) - 
  (player_yardage player_b_total player_b_pass : ℤ) = -10 :=
by
  sorry

end running_yardage_difference_l1628_162823


namespace joan_gave_25_marbles_l1628_162875

/-- The number of yellow marbles Joan gave Sam -/
def marbles_from_joan (initial_yellow : ℝ) (final_yellow : ℕ) : ℝ :=
  final_yellow - initial_yellow

theorem joan_gave_25_marbles :
  let initial_yellow : ℝ := 86.0
  let final_yellow : ℕ := 111
  marbles_from_joan initial_yellow final_yellow = 25 := by
  sorry

end joan_gave_25_marbles_l1628_162875


namespace congruence_problem_l1628_162884

theorem congruence_problem : ∃! n : ℕ, n ≤ 14 ∧ n ≡ 8657 [ZMOD 15] ∧ n = 2 := by
  sorry

end congruence_problem_l1628_162884


namespace quadratic_inequality_solution_l1628_162824

-- Define the quadratic function
def f (a b x : ℝ) := a * x^2 - x - b

-- Define the solution set condition
def solution_set_condition (a b : ℝ) :=
  ∀ x, f a b x > 0 ↔ (x > 2 ∨ x < -1)

-- Theorem statement
theorem quadratic_inequality_solution :
  ∀ a b : ℝ, solution_set_condition a b →
    (a = 1 ∧ b = 2) ∧
    (∀ c : ℝ,
      (c > 1 → ∀ x, x^2 - (c+1)*x + c < 0 ↔ 1 < x ∧ x < c) ∧
      (c = 1 → ∀ x, ¬(x^2 - (c+1)*x + c < 0)) ∧
      (c < 1 → ∀ x, x^2 - (c+1)*x + c < 0 ↔ c < x ∧ x < 1)) :=
by sorry


end quadratic_inequality_solution_l1628_162824


namespace cookie_pie_slices_left_l1628_162832

theorem cookie_pie_slices_left (num_pies : ℕ) (slices_per_pie : ℕ) (num_people : ℕ) : 
  num_pies = 5 → 
  slices_per_pie = 12 → 
  num_people = 33 → 
  num_pies * slices_per_pie - num_people = 27 := by
sorry

end cookie_pie_slices_left_l1628_162832


namespace rhombus_side_length_l1628_162805

-- Define the rhombus ABCD
def Rhombus (A B C D : Point) : Prop := sorry

-- Define the pyramid SABCD
def Pyramid (S A B C D : Point) : Prop := sorry

-- Define the inclination of lateral faces
def LateralFacesInclined (S A B C D : Point) (angle : ℝ) : Prop := sorry

-- Define midpoints
def Midpoint (M A B : Point) : Prop := sorry

-- Define the rectangular parallelepiped
def RectangularParallelepiped (M N K L F P R Q : Point) : Prop := sorry

-- Define the intersection points
def IntersectionPoints (S A B C D M N K L F P R Q : Point) : Prop := sorry

-- Define the volume of a polyhedron
def PolyhedronVolume (M N K L F P R Q : Point) : ℝ := sorry

-- Define the radius of an inscribed circle
def InscribedCircleRadius (A B C D : Point) : ℝ := sorry

-- Define the side length of a rhombus
def RhombusSideLength (A B C D : Point) : ℝ := sorry

theorem rhombus_side_length 
  (A B C D S M N K L F P R Q : Point) :
  Rhombus A B C D →
  Pyramid S A B C D →
  LateralFacesInclined S A B C D (60 * π / 180) →
  Midpoint M A B ∧ Midpoint N B C ∧ Midpoint K C D ∧ Midpoint L D A →
  RectangularParallelepiped M N K L F P R Q →
  IntersectionPoints S A B C D M N K L F P R Q →
  PolyhedronVolume M N K L F P R Q = 12 * Real.sqrt 3 →
  InscribedCircleRadius A B C D = 2.4 →
  RhombusSideLength A B C D = 5 := by
  sorry

end rhombus_side_length_l1628_162805


namespace josies_remaining_money_l1628_162834

/-- Given an initial amount of money and the costs of items,
    calculate the remaining amount after purchasing the items. -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Prove that given an initial amount of $50, after spending $9 each on two items
    and $25 on another item, the remaining amount is $7. -/
theorem josies_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end josies_remaining_money_l1628_162834


namespace smallest_prime_ten_less_than_perfect_square_l1628_162839

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_ten_less_than_perfect_square :
  (∀ p : ℕ, p < 71 → (is_prime p → ¬∃ n : ℕ, is_perfect_square (p + 10))) ∧
  (is_prime 71 ∧ ∃ n : ℕ, is_perfect_square (71 + 10)) :=
sorry

end smallest_prime_ten_less_than_perfect_square_l1628_162839


namespace two_digit_S_equals_50_l1628_162889

/-- R(n) is the sum of remainders when n is divided by 2, 3, 4, 5, and 6 -/
def R (n : ℕ) : ℕ :=
  n % 2 + n % 3 + n % 4 + n % 5 + n % 6

/-- S(n) is defined as R(n) + R(n+2) -/
def S (n : ℕ) : ℕ :=
  R n + R (n + 2)

/-- A two-digit number is between 10 and 99, inclusive -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- There are exactly 2 two-digit integers n such that S(n) = 50 -/
theorem two_digit_S_equals_50 :
  ∃! (count : ℕ), count = (Finset.filter (fun n => S n = 50) (Finset.range 90)).card ∧ count = 2 :=
sorry

end two_digit_S_equals_50_l1628_162889


namespace total_fish_cost_l1628_162871

def fish_cost : ℕ := 4
def dog_fish : ℕ := 40
def cat_fish : ℕ := dog_fish / 2

theorem total_fish_cost : dog_fish * fish_cost + cat_fish * fish_cost = 240 := by
  sorry

end total_fish_cost_l1628_162871


namespace steve_take_home_pay_l1628_162808

/-- Calculates the take-home pay given annual salary and deductions --/
def takeHomePay (annualSalary : ℝ) (taxRate : ℝ) (healthcareRate : ℝ) (unionDues : ℝ) : ℝ :=
  annualSalary - (annualSalary * taxRate + annualSalary * healthcareRate + unionDues)

/-- Proves that Steve's take-home pay is $27,200 --/
theorem steve_take_home_pay :
  takeHomePay 40000 0.20 0.10 800 = 27200 := by
  sorry

end steve_take_home_pay_l1628_162808


namespace factorial_300_trailing_zeros_l1628_162890

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: 300! has 74 trailing zeros -/
theorem factorial_300_trailing_zeros :
  trailingZeros 300 = 74 := by
  sorry

end factorial_300_trailing_zeros_l1628_162890


namespace garage_sale_pricing_l1628_162826

theorem garage_sale_pricing (total_items : ℕ) (highest_rank : ℕ) (lowest_rank : ℕ) 
  (h1 : total_items = 36)
  (h2 : highest_rank = 15)
  (h3 : lowest_rank + highest_rank = total_items + 1) : 
  lowest_rank = 22 := by
sorry

end garage_sale_pricing_l1628_162826


namespace cube_with_holes_surface_area_l1628_162854

/-- The total surface area of a cube with holes -/
def total_surface_area (cube_edge : ℝ) (hole_side : ℝ) : ℝ :=
  let original_surface_area := 6 * cube_edge^2
  let hole_area := 6 * hole_side^2
  let new_exposed_area := 6 * 4 * hole_side^2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem: The total surface area of a cube with edge length 3 and square holes of side 1 is 72 -/
theorem cube_with_holes_surface_area :
  total_surface_area 3 1 = 72 := by
  sorry

end cube_with_holes_surface_area_l1628_162854


namespace excluded_students_average_mark_l1628_162831

/-- Given a class of students with their exam marks, this theorem proves
    the average mark of excluded students based on the given conditions. -/
theorem excluded_students_average_mark
  (total_students : ℕ)
  (all_average : ℝ)
  (excluded_count : ℕ)
  (remaining_average : ℝ)
  (h1 : total_students = 10)
  (h2 : all_average = 70)
  (h3 : excluded_count = 5)
  (h4 : remaining_average = 90) :
  (total_students * all_average - (total_students - excluded_count) * remaining_average) / excluded_count = 50 := by
  sorry

end excluded_students_average_mark_l1628_162831


namespace equality_of_products_l1628_162855

theorem equality_of_products (a b c d x y z q : ℝ) 
  (h1 : a ^ x = c ^ q) (h2 : c ^ q = b) 
  (h3 : c ^ y = a ^ z) (h4 : a ^ z = d) : 
  x * y = q * z := by
  sorry

end equality_of_products_l1628_162855


namespace distance_sum_is_48_l1628_162877

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)
  (bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 14)
  (ca_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 15)

-- Define points Q and R
def Q (t : Triangle) : ℝ × ℝ := sorry
def R (t : Triangle) : ℝ × ℝ := sorry

-- Define the right angle condition
def is_right_angle (A B C : ℝ × ℝ) : Prop := sorry

-- Define similarity of triangles
def are_similar (t1 t2 t3 : Triangle) : Prop := sorry

-- Define the distance from a point to a line
def distance_to_line (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem distance_sum_is_48 (t : Triangle) 
  (h1 : is_right_angle (Q t) (t.C) (t.B))
  (h2 : is_right_angle (R t) (t.B) (t.C))
  (P1 P2 : ℝ × ℝ)
  (h3 : are_similar 
    ⟨P1, Q t, R t, sorry, sorry, sorry⟩ 
    ⟨P2, Q t, R t, sorry, sorry, sorry⟩ 
    t) :
  distance_to_line P1 t.B t.C + distance_to_line P2 t.B t.C = 48 := by
  sorry

end distance_sum_is_48_l1628_162877


namespace trig_expression_equals_negative_four_l1628_162817

theorem trig_expression_equals_negative_four :
  (Real.sqrt 3 / Real.cos (10 * π / 180)) - (1 / Real.sin (10 * π / 180)) = -4 := by
  sorry

end trig_expression_equals_negative_four_l1628_162817


namespace correct_operation_l1628_162827

theorem correct_operation (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end correct_operation_l1628_162827


namespace seashell_ratio_l1628_162830

theorem seashell_ratio : 
  let henry_shells : ℕ := 11
  let paul_shells : ℕ := 24
  let initial_total : ℕ := 59
  let final_total : ℕ := 53
  let leo_initial : ℕ := initial_total - henry_shells - paul_shells
  let leo_gave_away : ℕ := initial_total - final_total
  (leo_gave_away : ℚ) / leo_initial = 1 / 4 := by
  sorry

end seashell_ratio_l1628_162830


namespace planet_surface_area_unchanged_l1628_162835

theorem planet_surface_area_unchanged 
  (planet_diameter : ℝ) 
  (explosion_radius : ℝ) 
  (h1 : planet_diameter = 10000) 
  (h2 : explosion_radius = 5000) :
  let planet_radius : ℝ := planet_diameter / 2
  let initial_surface_area : ℝ := 4 * Real.pi * planet_radius ^ 2
  let new_surface_area : ℝ := initial_surface_area
  new_surface_area = 100000000 * Real.pi := by sorry

end planet_surface_area_unchanged_l1628_162835


namespace actual_average_height_l1628_162883

/-- Calculates the actual average height of students given initial incorrect data and correction --/
theorem actual_average_height
  (num_students : ℕ)
  (initial_average : ℝ)
  (incorrect_height : ℝ)
  (actual_height : ℝ)
  (h_num_students : num_students = 20)
  (h_initial_average : initial_average = 175)
  (h_incorrect_height : incorrect_height = 151)
  (h_actual_height : actual_height = 136) :
  (num_students * initial_average - (incorrect_height - actual_height)) / num_students = 174.25 := by
  sorry

end actual_average_height_l1628_162883


namespace sum_of_reciprocals_of_quadratic_roots_l1628_162816

theorem sum_of_reciprocals_of_quadratic_roots :
  ∀ (p q : ℝ), 
    p^2 - 11*p + 6 = 0 →
    q^2 - 11*q + 6 = 0 →
    p ≠ 0 →
    q ≠ 0 →
    1/p + 1/q = 11/6 := by
  sorry

end sum_of_reciprocals_of_quadratic_roots_l1628_162816


namespace system_solvable_l1628_162869

/-- The system of equations has a real solution if and only if m ≠ 3/2 -/
theorem system_solvable (m : ℝ) : 
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 3/2 := by
sorry

end system_solvable_l1628_162869


namespace cos_greater_when_sin_greater_in_second_quadrant_l1628_162802

theorem cos_greater_when_sin_greater_in_second_quadrant 
  (α β : Real) 
  (h1 : π/2 < α ∧ α < π) 
  (h2 : π/2 < β ∧ β < π) 
  (h3 : Real.sin α > Real.sin β) : 
  Real.cos α > Real.cos β := by
sorry

end cos_greater_when_sin_greater_in_second_quadrant_l1628_162802


namespace paint_cost_per_quart_l1628_162828

/-- The cost of paint per quart given specific conditions -/
theorem paint_cost_per_quart : 
  ∀ (coverage_per_quart : ℝ) (cube_edge_length : ℝ) (cost_to_paint_cube : ℝ),
  coverage_per_quart = 1200 →
  cube_edge_length = 10 →
  cost_to_paint_cube = 1.6 →
  (∃ (cost_per_quart : ℝ),
    cost_per_quart = 3.2 ∧
    cost_per_quart * (6 * cube_edge_length^2 / coverage_per_quart) = cost_to_paint_cube) :=
by sorry


end paint_cost_per_quart_l1628_162828


namespace simplify_expression_l1628_162891

theorem simplify_expression : (81 ^ (1/4) - (33/4) ^ (1/2)) ^ 2 = (69 - 12 * 33 ^ (1/2)) / 4 := by
  sorry

end simplify_expression_l1628_162891


namespace calculate_female_students_l1628_162895

/-- Given a population of students and a sample, calculate the number of female students in the population -/
theorem calculate_female_students 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (male_in_sample : ℕ) 
  (h1 : total_population = 2000) 
  (h2 : sample_size = 200) 
  (h3 : male_in_sample = 103) :
  (total_population - (male_in_sample * (total_population / sample_size))) = 970 := by
  sorry

#check calculate_female_students

end calculate_female_students_l1628_162895


namespace consecutive_even_sum_46_l1628_162813

theorem consecutive_even_sum_46 (n m : ℤ) : 
  (Even n) → (Even m) → (m = n + 2) → (n + m = 46) → (m = 24) := by
  sorry

end consecutive_even_sum_46_l1628_162813


namespace base_8_4532_equals_2394_l1628_162882

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_8_4532_equals_2394 :
  base_8_to_10 [2, 3, 5, 4] = 2394 := by
  sorry

end base_8_4532_equals_2394_l1628_162882


namespace adam_figurines_count_l1628_162899

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of basswood blocks Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ :=
  basswood_blocks * basswood_figurines +
  butternut_blocks * butternut_figurines +
  aspen_blocks * aspen_figurines

theorem adam_figurines_count :
  total_figurines = 245 := by
  sorry

end adam_figurines_count_l1628_162899


namespace laundry_time_l1628_162848

def total_time : ℕ := 120
def bathroom_time : ℕ := 15
def room_time : ℕ := 35
def homework_time : ℕ := 40

theorem laundry_time : 
  ∃ (laundry_time : ℕ), 
    laundry_time + bathroom_time + room_time + homework_time = total_time ∧ 
    laundry_time = 30 :=
by sorry

end laundry_time_l1628_162848


namespace total_weight_theorem_l1628_162863

/-- The total weight of three balls -/
def total_weight (blue_weight brown_weight green_weight : ℝ) : ℝ :=
  blue_weight + brown_weight + green_weight

/-- Theorem: The total weight of the three balls is 9.12 + x -/
theorem total_weight_theorem (x : ℝ) :
  total_weight 6 3.12 x = 9.12 + x := by
  sorry

end total_weight_theorem_l1628_162863


namespace three_digit_geometric_progression_l1628_162800

theorem three_digit_geometric_progression :
  ∀ a b c : ℕ,
  100 ≤ 100 * a + 10 * b + c ∧ 100 * a + 10 * b + c < 1000 →
  (∃ r : ℚ,
    (100 * b + 10 * c + a : ℚ) = r * (100 * a + 10 * b + c : ℚ) ∧
    (100 * c + 10 * a + b : ℚ) = r * (100 * b + 10 * c + a : ℚ)) →
  ((a = b ∧ b = c ∧ 1 ≤ a ∧ a ≤ 9) ∨
   (a = 2 ∧ b = 4 ∧ c = 3) ∨
   (a = 4 ∧ b = 8 ∧ c = 6)) :=
by sorry

end three_digit_geometric_progression_l1628_162800


namespace water_bottle_calculation_l1628_162892

/-- Given an initial number of bottles, calculate the final number after removing some and adding more. -/
def final_bottles (initial remove add : ℕ) : ℕ :=
  initial - remove + add

/-- Theorem: Given 14 initial bottles, removing 8 and adding 45 results in 51 bottles. -/
theorem water_bottle_calculation :
  final_bottles 14 8 45 = 51 := by
  sorry

end water_bottle_calculation_l1628_162892


namespace simplest_common_denominator_example_l1628_162879

/-- The simplest common denominator of two fractions -/
def simplestCommonDenominator (f1 f2 : ℚ) : ℤ :=
  sorry

/-- Theorem: The simplest common denominator of 1/(m^2-9) and 1/(2m+6) is 2(m+3)(m-3) -/
theorem simplest_common_denominator_example (m : ℚ) :
  simplestCommonDenominator (1 / (m^2 - 9)) (1 / (2*m + 6)) = 2 * (m + 3) * (m - 3) :=
by sorry

end simplest_common_denominator_example_l1628_162879


namespace cube_equation_solution_l1628_162894

theorem cube_equation_solution :
  ∃! x : ℝ, (8 - x)^3 = x^3 ∧ x = 8 := by sorry

end cube_equation_solution_l1628_162894


namespace greater_number_sum_and_difference_l1628_162856

theorem greater_number_sum_and_difference (x y : ℝ) : 
  x + y = 30 → x - y = 6 → x > y → x = 18 := by sorry

end greater_number_sum_and_difference_l1628_162856


namespace percentage_women_non_union_l1628_162836

-- Define the total number of employees
variable (E : ℝ)
-- Assume E is positive
variable (hE : E > 0)

-- Define the percentage of unionized employees
def unionized_percent : ℝ := 0.60

-- Define the percentage of men among unionized employees
def men_in_union_percent : ℝ := 0.70

-- Define the percentage of women among non-union employees
def women_non_union_percent : ℝ := 0.65

-- Theorem to prove
theorem percentage_women_non_union :
  women_non_union_percent = 0.65 := by
  sorry

end percentage_women_non_union_l1628_162836


namespace complement_P_intersect_Q_l1628_162888

universe u

def U : Set Nat := {1, 2, 3, 4}
def P : Set Nat := {2, 3, 4}
def Q : Set Nat := {1, 2}

theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = {1} := by sorry

end complement_P_intersect_Q_l1628_162888


namespace discount_age_limit_l1628_162837

/-- Represents the age limit for the discount at an amusement park. -/
def age_limit : ℕ := 10

/-- Represents the regular ticket cost. -/
def regular_ticket_cost : ℕ := 109

/-- Represents the discount amount for children. -/
def child_discount : ℕ := 5

/-- Represents the number of adults in the family. -/
def num_adults : ℕ := 2

/-- Represents the number of children in the family. -/
def num_children : ℕ := 2

/-- Represents the ages of the children in the family. -/
def children_ages : List ℕ := [6, 10]

/-- Represents the amount paid by the family. -/
def amount_paid : ℕ := 500

/-- Represents the change received by the family. -/
def change_received : ℕ := 74

/-- Theorem stating that the age limit for the discount is 10 years old. -/
theorem discount_age_limit : 
  (∀ (age : ℕ), age ∈ children_ages → age ≤ age_limit) ∧
  (amount_paid - change_received = 
    num_adults * regular_ticket_cost + 
    num_children * (regular_ticket_cost - child_discount)) →
  age_limit = 10 := by
  sorry

end discount_age_limit_l1628_162837


namespace intersection_of_A_and_B_l1628_162872

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l1628_162872


namespace complex_power_magnitude_l1628_162862

theorem complex_power_magnitude (z : ℂ) : z = (1 / 2 : ℂ) + (Complex.I * (Real.sqrt 3 / 2)) → Complex.abs (z ^ 5) = 1 := by
  sorry

end complex_power_magnitude_l1628_162862


namespace balance_theorem_l1628_162874

/-- Represents the balance between shapes -/
structure Balance where
  triangle : ℚ
  diamond : ℚ
  circle : ℚ

/-- First balance equation: 5 triangles + 2 diamonds = 12 circles -/
def balance1 : Balance := { triangle := 5, diamond := 2, circle := 12 }

/-- Second balance equation: 1 triangle = 1 diamond + 3 circles -/
def balance2 : Balance := { triangle := 1, diamond := 1, circle := 3 }

/-- The balance we want to prove: 4 diamonds = 12/7 circles -/
def target_balance : Balance := { triangle := 0, diamond := 4, circle := 12/7 }

/-- Checks if two balances are equivalent -/
def is_equivalent (b1 b2 : Balance) : Prop :=
  b1.triangle / b2.triangle = b1.diamond / b2.diamond ∧
  b1.triangle / b2.triangle = b1.circle / b2.circle

/-- The main theorem to prove -/
theorem balance_theorem (b1 b2 : Balance) (h1 : is_equivalent b1 balance1) 
    (h2 : is_equivalent b2 balance2) : 
  is_equivalent target_balance { triangle := 0, diamond := 1, circle := 3/7 } := by
  sorry

end balance_theorem_l1628_162874


namespace find_number_l1628_162850

theorem find_number : ∃ N : ℕ, N = (555 + 445) * (2 * (555 - 445)) + 70 ∧ N = 220070 := by
  sorry

end find_number_l1628_162850


namespace base_value_l1628_162896

/-- A triangle with specific side length properties -/
structure SpecificTriangle where
  left : ℝ
  right : ℝ
  base : ℝ
  sum_of_sides : left + right + base = 50
  right_longer : right = left + 2
  left_value : left = 12

theorem base_value (t : SpecificTriangle) : t.base = 24 := by
  sorry

end base_value_l1628_162896


namespace expected_value_biased_die_l1628_162857

/-- A biased die with six faces and specified winning conditions -/
structure BiasedDie where
  /-- The probability of rolling each number is 1/6 -/
  prob : Fin 6 → ℚ
  prob_eq : ∀ i, prob i = 1/6
  /-- The winnings for each roll -/
  winnings : Fin 6 → ℚ
  /-- Rolling 1 or 2 wins $5 -/
  win_12 : winnings 0 = 5 ∧ winnings 1 = 5
  /-- Rolling 3 or 4 wins $0 -/
  win_34 : winnings 2 = 0 ∧ winnings 3 = 0
  /-- Rolling 5 or 6 loses $4 -/
  lose_56 : winnings 4 = -4 ∧ winnings 5 = -4

/-- The expected value of winnings after one roll of the biased die is 1/3 -/
theorem expected_value_biased_die (d : BiasedDie) : 
  (Finset.univ.sum fun i => d.prob i * d.winnings i) = 1/3 := by
  sorry

end expected_value_biased_die_l1628_162857


namespace system_equation_ratio_l1628_162820

theorem system_equation_ratio (x y a b : ℝ) : 
  x ≠ 0 → 
  y ≠ 0 → 
  b ≠ 0 → 
  8 * x - 6 * y = a → 
  12 * y - 18 * x = b → 
  a / b = -1 / 2 := by
sorry

end system_equation_ratio_l1628_162820


namespace valid_lineup_count_l1628_162864

def team_size : ℕ := 15
def lineup_size : ℕ := 6

def cannot_play_together (p1 p2 : ℕ) : Prop := p1 ≠ p2

def excludes_player (p1 p2 : ℕ) : Prop := p1 ≠ p2

def valid_lineup (lineup : Finset ℕ) : Prop :=
  lineup.card = lineup_size ∧
  (∀ p ∈ lineup, p ≤ team_size) ∧
  ¬(1 ∈ lineup ∧ 2 ∈ lineup) ∧
  (1 ∈ lineup → 3 ∉ lineup)

def count_valid_lineups : ℕ := sorry

theorem valid_lineup_count :
  count_valid_lineups = 3795 := by sorry

end valid_lineup_count_l1628_162864


namespace danny_found_58_new_caps_l1628_162897

/-- Represents the number of bottle caps Danny has at different stages -/
structure BottleCaps where
  initial : ℕ
  thrown_away : ℕ
  final : ℕ

/-- Calculates the number of new bottle caps Danny found -/
def new_bottle_caps (bc : BottleCaps) : ℕ :=
  bc.final - (bc.initial - bc.thrown_away)

/-- Theorem stating that Danny found 58 new bottle caps -/
theorem danny_found_58_new_caps : 
  ∀ (bc : BottleCaps), 
  bc.initial = 69 → bc.thrown_away = 60 → bc.final = 67 → 
  new_bottle_caps bc = 58 := by
  sorry

end danny_found_58_new_caps_l1628_162897


namespace existence_of_special_numbers_l1628_162814

theorem existence_of_special_numbers : ∃ (a : Fin 15 → ℕ),
  (∀ i, ∃ k, a i = 35 * k) ∧
  (∀ i j, i ≠ j → ¬(a i ∣ a j)) ∧
  (∀ i j, (a i)^6 ∣ (a j)^5) := by
  sorry

end existence_of_special_numbers_l1628_162814


namespace difference_equals_negative_two_hundred_l1628_162838

/-- Given that the average of (a+d) and (b+d) is 80, and the average of (b+d) and (c+d) is 180,
    prove that a - c = -200 -/
theorem difference_equals_negative_two_hundred
  (a b c d : ℝ)
  (h1 : ((a + d) + (b + d)) / 2 = 80)
  (h2 : ((b + d) + (c + d)) / 2 = 180) :
  a - c = -200 := by sorry

end difference_equals_negative_two_hundred_l1628_162838


namespace smallest_root_of_unity_order_l1628_162893

theorem smallest_root_of_unity_order (z : ℂ) : 
  (∃ (n : ℕ), n > 0 ∧ (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^n = 1) ∧ 
   (∀ m : ℕ, m > 0 → (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^m = 1) → m ≥ n)) → 
  (∃ (n : ℕ), n = 18 ∧ n > 0 ∧ (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^n = 1) ∧ 
   (∀ m : ℕ, m > 0 → (∀ w : ℂ, w^6 - w^3 + 1 = 0 → w^m = 1) → m ≥ n)) :=
sorry

end smallest_root_of_unity_order_l1628_162893


namespace x_value_l1628_162847

theorem x_value (x : Real) : 
  Real.sin (π / 2 - x) = -Real.sqrt 3 / 2 → 
  π < x → 
  x < 2 * π → 
  x = 7 * π / 6 := by
sorry

end x_value_l1628_162847
