import Mathlib

namespace bob_puppy_savings_l378_37836

/-- The minimum number of additional weeks Bob must win first place to buy a puppy -/
def minimum_additional_weeks (initial_weeks : ℕ) (prize_per_week : ℕ) (puppy_cost : ℕ) : ℕ :=
  let initial_earnings := initial_weeks * prize_per_week
  let remaining_cost := puppy_cost - initial_earnings
  (remaining_cost + prize_per_week - 1) / prize_per_week

theorem bob_puppy_savings : minimum_additional_weeks 2 100 1000 = 8 := by
  sorry

end bob_puppy_savings_l378_37836


namespace stone_width_is_five_dm_l378_37892

/-- Proves that the width of stones used to pave a hall is 5 decimeters -/
theorem stone_width_is_five_dm (hall_length : ℝ) (hall_width : ℝ) 
  (stone_length : ℝ) (num_stones : ℕ) :
  hall_length = 36 →
  hall_width = 15 →
  stone_length = 0.4 →
  num_stones = 2700 →
  ∃ (stone_width : ℝ),
    stone_width = 0.5 ∧
    hall_length * hall_width * 100 = num_stones * stone_length * stone_width :=
by sorry

end stone_width_is_five_dm_l378_37892


namespace smallest_number_divisible_l378_37889

theorem smallest_number_divisible (n : ℕ) : n = 32127 ↔ 
  (∀ m : ℕ, m < n → ¬(((m + 3) % 510 = 0) ∧ ((m + 3) % 4590 = 0) ∧ ((m + 3) % 105 = 0))) ∧
  ((n + 3) % 510 = 0) ∧ ((n + 3) % 4590 = 0) ∧ ((n + 3) % 105 = 0) :=
by sorry

end smallest_number_divisible_l378_37889


namespace combustible_ice_volume_scientific_notation_l378_37893

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ coefficient
  h2 : coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem combustible_ice_volume_scientific_notation :
  toScientificNotation 19400000000 = ScientificNotation.mk 1.94 10 (by norm_num) (by norm_num) :=
sorry

end combustible_ice_volume_scientific_notation_l378_37893


namespace eight_digit_factorization_comparison_l378_37831

theorem eight_digit_factorization_comparison :
  let total_eight_digit_numbers := 99999999 - 10000000 + 1
  let four_digit_numbers := 9999 - 1000 + 1
  let products_of_four_digit_numbers := four_digit_numbers.choose 2 + four_digit_numbers
  total_eight_digit_numbers - products_of_four_digit_numbers > products_of_four_digit_numbers := by
  sorry

end eight_digit_factorization_comparison_l378_37831


namespace smallest_divisible_by_5_13_7_l378_37829

theorem smallest_divisible_by_5_13_7 : ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n → n ≥ 455 := by
  sorry

end smallest_divisible_by_5_13_7_l378_37829


namespace expression_value_l378_37853

theorem expression_value : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 := by
  sorry

end expression_value_l378_37853


namespace sum_of_squares_16_to_30_l378_37885

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 :=
by
  sorry

end sum_of_squares_16_to_30_l378_37885


namespace annie_cookies_l378_37882

/-- The number of cookies Annie ate over three days -/
def total_cookies (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem: Annie ate 29 cookies over three days -/
theorem annie_cookies : ∃ (monday tuesday wednesday : ℕ),
  monday = 5 ∧
  tuesday = 2 * monday ∧
  wednesday = tuesday + (tuesday * 2 / 5) ∧
  total_cookies monday tuesday wednesday = 29 := by
  sorry

end annie_cookies_l378_37882


namespace diagonal_cells_in_rectangle_diagonal_cells_199_991_l378_37805

theorem diagonal_cells_in_rectangle : ℕ → ℕ → ℕ
  | m, n => m + n - Nat.gcd m n

theorem diagonal_cells_199_991 :
  diagonal_cells_in_rectangle 199 991 = 1189 := by
  sorry

end diagonal_cells_in_rectangle_diagonal_cells_199_991_l378_37805


namespace probability_two_red_marbles_l378_37875

/-- The probability of selecting two red marbles without replacement from a bag containing 2 red marbles and 3 green marbles. -/
theorem probability_two_red_marbles (red : ℕ) (green : ℕ) (total : ℕ) :
  red = 2 →
  green = 3 →
  total = red + green →
  (red / total) * ((red - 1) / (total - 1)) = 1 / 10 :=
by sorry

end probability_two_red_marbles_l378_37875


namespace magical_card_stack_l378_37878

/-- 
Given a stack of 2n cards numbered 1 to 2n, with the top n cards forming pile A 
and the rest forming pile B, prove that when restacked by alternating from 
piles B and A, the total number of cards where card 161 retains its original 
position is 482.
-/
theorem magical_card_stack (n : ℕ) : 
  (∃ (total : ℕ), 
    total = 2 * n ∧ 
    161 ≤ n ∧ 
    (∀ (k : ℕ), k ≤ total → k = 161 → (k - 1) / 2 = (n - 161))) → 
  2 * n = 482 :=
by sorry

end magical_card_stack_l378_37878


namespace company_average_salary_l378_37810

/-- Calculate the average salary for a company given the number of managers and associates, and their respective average salaries. -/
theorem company_average_salary
  (num_managers : ℕ)
  (num_associates : ℕ)
  (avg_salary_managers : ℝ)
  (avg_salary_associates : ℝ)
  (h_managers : num_managers = 15)
  (h_associates : num_associates = 75)
  (h_salary_managers : avg_salary_managers = 90000)
  (h_salary_associates : avg_salary_associates = 30000) :
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  let total_employees := num_managers + num_associates
  total_salary / total_employees = 40000 := by
  sorry

#check company_average_salary

end company_average_salary_l378_37810


namespace sphere_radii_problem_l378_37883

theorem sphere_radii_problem (r₁ r₂ r₃ : ℝ) : 
  -- Three spheres touch each other externally
  2 * Real.sqrt (r₁ * r₂) = 2 ∧
  2 * Real.sqrt (r₁ * r₃) = Real.sqrt 3 ∧
  2 * Real.sqrt (r₂ * r₃) = 1 ∧
  -- The spheres touch a plane at the vertices of a right triangle
  -- One leg of the triangle has length 1
  -- The angle opposite to the leg of length 1 is 30°
  -- (These conditions are implicitly satisfied by the equations above)
  -- The radii are positive
  r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0
  →
  -- The radii of the spheres are √3, 1/√3, and √3/4
  (r₁ = Real.sqrt 3 ∧ r₂ = 1 / Real.sqrt 3 ∧ r₃ = Real.sqrt 3 / 4) ∨
  (r₁ = Real.sqrt 3 ∧ r₂ = Real.sqrt 3 / 4 ∧ r₃ = 1 / Real.sqrt 3) ∨
  (r₁ = 1 / Real.sqrt 3 ∧ r₂ = Real.sqrt 3 ∧ r₃ = Real.sqrt 3 / 4) ∨
  (r₁ = 1 / Real.sqrt 3 ∧ r₂ = Real.sqrt 3 / 4 ∧ r₃ = Real.sqrt 3) ∨
  (r₁ = Real.sqrt 3 / 4 ∧ r₂ = Real.sqrt 3 ∧ r₃ = 1 / Real.sqrt 3) ∨
  (r₁ = Real.sqrt 3 / 4 ∧ r₂ = 1 / Real.sqrt 3 ∧ r₃ = Real.sqrt 3) :=
by sorry

end sphere_radii_problem_l378_37883


namespace fraction_of_a_equal_half_b_l378_37800

/-- Given two amounts a and b, where their sum is 1210 and b is 484,
    prove that the fraction of a's amount equal to half of b's amount is 1/3 -/
theorem fraction_of_a_equal_half_b (a b : ℕ) : 
  a + b = 1210 → b = 484 → ∃ f : ℚ, f * a = (1 / 2) * b ∧ f = 1 / 3 := by
  sorry

end fraction_of_a_equal_half_b_l378_37800


namespace student_B_more_stable_l378_37858

/-- Represents a student's performance metrics -/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if the first student has more stable performance than the second -/
def more_stable (s1 s2 : StudentPerformance) : Prop :=
  s1.average_score = s2.average_score ∧ s1.variance < s2.variance

/-- The performance metrics for student A -/
def student_A : StudentPerformance :=
  { average_score := 82
    variance := 245 }

/-- The performance metrics for student B -/
def student_B : StudentPerformance :=
  { average_score := 82
    variance := 190 }

/-- Theorem stating that student B has more stable performance than student A -/
theorem student_B_more_stable : more_stable student_B student_A := by
  sorry

end student_B_more_stable_l378_37858


namespace focus_coincidence_l378_37816

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := (5*x^2)/3 - (5*y^2)/2 = 1

-- Define the focus of a parabola
def parabola_focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Define the right focus of a hyperbola
def hyperbola_right_focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Theorem statement
theorem focus_coincidence :
  ∀ (x y : ℝ), parabola_focus x y ↔ hyperbola_right_focus x y :=
sorry

end focus_coincidence_l378_37816


namespace product_of_roots_l378_37807

theorem product_of_roots (x : ℝ) : 
  (x^3 - 15*x^2 + 75*x - 125 = 0) → 
  (∃ a b c : ℝ, x^3 - 15*x^2 + 75*x - 125 = (x - a) * (x - b) * (x - c) ∧ a * b * c = 125) :=
by sorry

end product_of_roots_l378_37807


namespace inequality_proof_l378_37830

theorem inequality_proof (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a * x + b * y) * (b * x + a * y) ≥ x * y := by
  sorry

end inequality_proof_l378_37830


namespace find_n_l378_37899

theorem find_n : ∃ n : ℤ, (15 : ℝ) ^ (2 * n) = (1 / 15 : ℝ) ^ (3 * n - 30) → n = 6 := by
  sorry

end find_n_l378_37899


namespace log_ratio_sixteen_four_l378_37813

theorem log_ratio_sixteen_four : (Real.log 16) / (Real.log 4) = 2 := by
  sorry

end log_ratio_sixteen_four_l378_37813


namespace inequality_problem_l378_37844

theorem inequality_problem (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  (a - d > b - c) ∧ (a / d > b / c) ∧ (a * c > b * d) ∧ ¬(a + d > b + c) := by
  sorry

end inequality_problem_l378_37844


namespace f_g_f_3_equals_186_l378_37804

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x + 2
def g (x : ℝ) : ℝ := 3 * x + 4

-- Theorem statement
theorem f_g_f_3_equals_186 : f (g (f 3)) = 186 := by
  sorry

end f_g_f_3_equals_186_l378_37804


namespace rectangle_perimeter_l378_37809

theorem rectangle_perimeter (x y z : ℝ) : 
  x + y + z = 75 →
  x > 0 → y > 0 → z > 0 →
  2 * (x + 75) = (2 * (y + 75) + 2 * (z + 75)) / 2 →
  2 * (x + 75) = 20 * 10 :=
by
  sorry

end rectangle_perimeter_l378_37809


namespace savings_distribution_l378_37856

/-- Represents the savings and debt problem of Tamara, Nora, and Lulu -/
theorem savings_distribution (debt : ℕ) (lulu_savings : ℕ) : 
  debt = 40 →
  lulu_savings = 6 →
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  let remainder := total_savings - debt
  remainder / 3 = 2 := by sorry

end savings_distribution_l378_37856


namespace teachers_survey_result_l378_37820

def teachers_survey (total : ℕ) (high_bp : ℕ) (stress : ℕ) (both : ℕ) : Prop :=
  let neither := total - (high_bp + stress - both)
  (neither : ℚ) / total * 100 = 20

theorem teachers_survey_result : teachers_survey 150 90 60 30 := by
  sorry

end teachers_survey_result_l378_37820


namespace binary_101101_equals_octal_55_l378_37864

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_equals_octal_55 : 
  decimal_to_octal (binary_to_decimal binary_101101) = [5, 5] := by
  sorry

end binary_101101_equals_octal_55_l378_37864


namespace remainder_2468135792_mod_101_l378_37833

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 47 := by
  sorry

end remainder_2468135792_mod_101_l378_37833


namespace probability_theorem_l378_37837

def total_candidates : ℕ := 9
def boys : ℕ := 5
def girls : ℕ := 4
def volunteers : ℕ := 4

def probability_1girl_3boys : ℚ := 20 / 63

def P (n : ℕ) : ℚ := 
  (Nat.choose boys n * Nat.choose girls (volunteers - n)) / Nat.choose total_candidates volunteers

theorem probability_theorem :
  (probability_1girl_3boys = P 3) ∧
  (∀ n : ℕ, P n ≥ 3/4 → n ≤ 2) ∧
  (P 2 ≥ 3/4) :=
sorry

end probability_theorem_l378_37837


namespace marsha_second_package_distance_l378_37894

/-- Represents the distance Marsha drives for her second package delivery -/
def second_package_distance : ℝ := 28

/-- Represents Marsha's total payment for the day -/
def total_payment : ℝ := 104

/-- Represents Marsha's payment per mile -/
def payment_per_mile : ℝ := 2

/-- Represents the distance Marsha drives for her first package delivery -/
def first_package_distance : ℝ := 10

theorem marsha_second_package_distance :
  second_package_distance = 28 ∧
  total_payment = payment_per_mile * (first_package_distance + second_package_distance + second_package_distance / 2) :=
by sorry

end marsha_second_package_distance_l378_37894


namespace trig_inequality_l378_37843

theorem trig_inequality (x : ℝ) : 1 ≤ Real.sin x ^ 10 + 10 * Real.sin x ^ 2 * Real.cos x ^ 2 + Real.cos x ^ 10 ∧ 
  Real.sin x ^ 10 + 10 * Real.sin x ^ 2 * Real.cos x ^ 2 + Real.cos x ^ 10 ≤ 41 / 16 := by
  sorry

end trig_inequality_l378_37843


namespace binomial_10_3_l378_37819

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_3_l378_37819


namespace sqrt_difference_equals_neg_six_sqrt_three_l378_37845

theorem sqrt_difference_equals_neg_six_sqrt_three :
  Real.sqrt ((5 - 3 * Real.sqrt 3)^2) - Real.sqrt ((5 + 3 * Real.sqrt 3)^2) = -6 * Real.sqrt 3 := by
  sorry

end sqrt_difference_equals_neg_six_sqrt_three_l378_37845


namespace abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one_l378_37870

theorem abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one :
  (∀ x : ℝ, |x - 1| > 2 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ |x - 1| ≤ 2) := by
sorry

end abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one_l378_37870


namespace special_function_sum_l378_37840

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- Main theorem -/
theorem special_function_sum (f : ℝ → ℝ) 
  (h_odd : IsOdd f)
  (h_prop : ∀ x, f (2 + x) + f (2 - x) = 0)
  (h_f1 : f 1 = 9) :
  f 2010 + f 2011 + f 2012 = -9 := by
  sorry

end special_function_sum_l378_37840


namespace f_positive_iff_x_gt_one_l378_37812

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (deriv f) x > f x)
variable (h2 : f 1 = 0)

-- State the theorem
theorem f_positive_iff_x_gt_one :
  (∀ x, f x > 0 ↔ x > 1) :=
sorry

end f_positive_iff_x_gt_one_l378_37812


namespace midpoint_parallelogram_area_ratio_l378_37808

/-- Given a parallelogram, the area of the parallelogram formed by joining its midpoints is 1/4 of the original area -/
theorem midpoint_parallelogram_area_ratio (P : ℝ) (h : P > 0) :
  ∃ (smaller_area : ℝ), smaller_area = P / 4 := by
  sorry

end midpoint_parallelogram_area_ratio_l378_37808


namespace tenth_pebble_count_l378_37884

def pebble_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | n + 4 => pebble_sequence (n + 3) + (3 * (n + 4) - 2)

theorem tenth_pebble_count : pebble_sequence 9 = 145 := by
  sorry

end tenth_pebble_count_l378_37884


namespace smallest_n_satisfying_inequality_l378_37827

theorem smallest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + (n : ℚ)/8 > 1 ↔ n ≥ 7 :=
sorry

end smallest_n_satisfying_inequality_l378_37827


namespace line_through_point_with_opposite_intercepts_l378_37847

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passesThrough (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has opposite intercepts
def hasOppositeIntercepts (l : Line) : Prop :=
  (l.a ≠ 0 ∧ l.b ≠ 0) ∧ (l.c / l.a) * (l.c / l.b) < 0

-- Theorem statement
theorem line_through_point_with_opposite_intercepts :
  ∀ (l : Line),
    passesThrough l {x := 2, y := 3} →
    hasOppositeIntercepts l →
    (∃ (k : ℝ), l.a = k ∧ l.b = -k ∧ l.c = k) ∨
    (l.a = 3 ∧ l.b = -2 ∧ l.c = 0) :=
sorry

end line_through_point_with_opposite_intercepts_l378_37847


namespace median_mode_difference_l378_37887

def data : List ℕ := [12, 13, 14, 15, 15, 21, 21, 21, 32, 32, 38, 39, 40, 41, 42, 43, 53, 58, 59]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference : |median data - mode data| = 11 := by sorry

end median_mode_difference_l378_37887


namespace total_students_l378_37802

theorem total_students (french : ℕ) (spanish : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 5)
  (h2 : spanish = 10)
  (h3 : both = 4)
  (h4 : neither = 13) :
  french + spanish + both + neither = 32 := by
  sorry

end total_students_l378_37802


namespace expression_simplification_l378_37854

theorem expression_simplification 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hc : c > 0) 
  (hd : 2 * (a - b^2)^2 + (2 * b * Real.sqrt (2 * a))^2 ≠ 0) :
  (Real.sqrt 3 * (a - b^2) + Real.sqrt 3 * b * (8 * b^3)^(1/3)) / 
  Real.sqrt (2 * (a - b^2)^2 + (2 * b * Real.sqrt (2 * a))^2) * 
  (Real.sqrt (2 * a) - Real.sqrt (2 * c)) / 
  (Real.sqrt (3 / a) - Real.sqrt (3 / c)) = 
  -Real.sqrt (a * c) := by
  sorry

end expression_simplification_l378_37854


namespace correct_calculation_l378_37832

theorem correct_calculation (x : ℤ) : x - 6 = 51 → 6 * x = 342 := by
  sorry

end correct_calculation_l378_37832


namespace hyperbola_a_value_l378_37806

/-- The value of 'a' for a hyperbola with equation x²/a² - y² = 1, a > 0, and eccentricity √5 -/
theorem hyperbola_a_value (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x y : ℝ, x^2 / a^2 - y^2 = 1) 
  (h3 : ∃ c : ℝ, c / a = Real.sqrt 5) : a = 1/2 := by
  sorry

end hyperbola_a_value_l378_37806


namespace rectangle_circle_union_area_l378_37828

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 8
  let circle_radius : ℝ := 8
  let rectangle_area := rectangle_length * rectangle_width
  let circle_area := π * circle_radius^2
  let overlap_area := (1/4) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 48 * π := by
  sorry

end rectangle_circle_union_area_l378_37828


namespace division_problem_l378_37874

theorem division_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 199 →
  divisor = 18 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 11 := by
  sorry

end division_problem_l378_37874


namespace no_six_odd_reciprocals_sum_to_one_l378_37868

theorem no_six_odd_reciprocals_sum_to_one :
  ¬ ∃ (a b c d e f : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
    Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧
    1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f = 1 := by
  sorry


end no_six_odd_reciprocals_sum_to_one_l378_37868


namespace max_cubes_in_box_l378_37867

/-- The maximum number of 27 cubic centimetre cubes that can fit in a rectangular box -/
def max_cubes (l w h : ℕ) (cube_volume : ℕ) : ℕ :=
  (l * w * h) / cube_volume

/-- Theorem: The maximum number of 27 cubic centimetre cubes that can fit in a 
    rectangular box measuring 8 cm x 9 cm x 12 cm is 32 -/
theorem max_cubes_in_box : max_cubes 8 9 12 27 = 32 := by
  sorry

end max_cubes_in_box_l378_37867


namespace percentage_problem_l378_37859

theorem percentage_problem (N P : ℝ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 := by
  sorry

end percentage_problem_l378_37859


namespace hyperbola_vertex_distance_l378_37839

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 144 - y^2 / 49 = 1

-- State the theorem
theorem hyperbola_vertex_distance :
  ∃ (x y : ℝ), hyperbola x y → 
    (let vertex_distance := 2 * (Real.sqrt 144);
     vertex_distance = 24) :=
by sorry

end hyperbola_vertex_distance_l378_37839


namespace log_ratio_squared_l378_37846

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1)
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
  (h2 : x * y = 243) :
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
sorry

end log_ratio_squared_l378_37846


namespace division_remainder_l378_37895

theorem division_remainder : 
  let dividend : ℕ := 23
  let divisor : ℕ := 5
  let quotient : ℕ := 4
  dividend % divisor = 3 := by
  sorry

end division_remainder_l378_37895


namespace solution_part1_solution_part2_l378_37865

/-- The system of equations -/
def system_equations (x y m : ℝ) : Prop :=
  2 * x - y = m ∧ 3 * x + 2 * y = m + 7

/-- Point in second quadrant with given distances -/
def point_conditions (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0 ∧ y = 3 ∧ x = -2

theorem solution_part1 :
  ∃ x y : ℝ, system_equations x y 0 ∧ x = 1 ∧ y = 2 := by sorry

theorem solution_part2 :
  ∀ x y : ℝ, system_equations x y (-7) ∧ point_conditions x y := by sorry

end solution_part1_solution_part2_l378_37865


namespace third_month_sale_l378_37852

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_4 : ℕ := 7230
def sales_5 : ℕ := 6562
def sales_6 : ℕ := 7991
def average_sale : ℕ := 7000
def num_months : ℕ := 6

theorem third_month_sale :
  ∃ (sales_3 : ℕ),
    sales_3 = num_months * average_sale - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6855 := by
  sorry

end third_month_sale_l378_37852


namespace max_value_of_f_l378_37835

/-- The quadratic function f(x) = -3x^2 + 18x - 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 5

theorem max_value_of_f :
  ∃ (M : ℝ), M = 22 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l378_37835


namespace work_completion_time_l378_37866

/-- The time (in days) it takes for person a to complete the work alone -/
def time_a : ℝ := 90

/-- The time (in days) it takes for person b to complete the work alone -/
def time_b : ℝ := 45

/-- The time (in days) it takes for persons a, b, and c working together to complete the work -/
def time_together : ℝ := 5

/-- The time (in days) it takes for person c to complete the work alone -/
def time_c : ℝ := 6

/-- The theorem stating that given the work times for a, b, and the group, 
    the time for c to complete the work alone is 6 days -/
theorem work_completion_time :
  (1 / time_a + 1 / time_b + 1 / time_c = 1 / time_together) := by
  sorry

end work_completion_time_l378_37866


namespace circle_diameter_ratio_l378_37891

theorem circle_diameter_ratio (C D : Real) (h1 : D = 20) 
  (h2 : C > 0 ∧ C < D) (h3 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 5) : 
  C = 10 * Real.sqrt 3 / 3 := by
sorry

end circle_diameter_ratio_l378_37891


namespace domain_exact_domain_contains_l378_37814

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + x - a

-- Theorem 1: When the domain is exactly (-2, 3), a = -6
theorem domain_exact (a : ℝ) : 
  (∀ x, -2 < x ∧ x < 3 ↔ f a x > 0) → a = -6 :=
sorry

-- Theorem 2: When the domain contains (-2, 3), a ≤ -6
theorem domain_contains (a : ℝ) :
  (∀ x, -2 < x ∧ x < 3 → f a x > 0) → a ≤ -6 :=
sorry

end domain_exact_domain_contains_l378_37814


namespace rectangle_perimeter_problem_l378_37822

theorem rectangle_perimeter_problem : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    a > 0 ∧ 
    b > 0 ∧ 
    (a * b : ℕ) = 2 * (2 * a + 2 * b) ∧ 
    2 * (a + b) = 36 := by
  sorry

end rectangle_perimeter_problem_l378_37822


namespace archer_weekly_spending_is_1056_l378_37811

/-- The archer's weekly spending on arrows -/
def archer_weekly_spending (shots_per_day : ℕ) (days_per_week : ℕ) 
  (recovery_rate : ℚ) (arrow_cost : ℚ) (team_payment_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let unrecovered_arrows := total_shots * (1 - recovery_rate)
  let total_cost := unrecovered_arrows * arrow_cost
  total_cost * (1 - team_payment_rate)

/-- Theorem: The archer spends $1056 on arrows per week -/
theorem archer_weekly_spending_is_1056 :
  archer_weekly_spending 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end archer_weekly_spending_is_1056_l378_37811


namespace probability_x_plus_y_leq_6_l378_37838

/-- The probability that x + y ≤ 6 when (x, y) is randomly selected from a rectangle where 0 ≤ x ≤ 4 and 0 ≤ y ≤ 5 -/
theorem probability_x_plus_y_leq_6 :
  let rectangle_area : ℝ := 4 * 5
  let favorable_area : ℝ := 15
  favorable_area / rectangle_area = 3 / 4 := by sorry

end probability_x_plus_y_leq_6_l378_37838


namespace drums_per_day_l378_37873

/-- Given that 266 pickers fill 90 drums in 5 days, prove that the number of drums filled per day is 18. -/
theorem drums_per_day (pickers : ℕ) (total_drums : ℕ) (days : ℕ) 
  (h1 : pickers = 266) 
  (h2 : total_drums = 90) 
  (h3 : days = 5) : 
  total_drums / days = 18 := by
  sorry

end drums_per_day_l378_37873


namespace M_intersect_N_is_empty_l378_37855

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (2*x - x^2)}

-- Theorem statement
theorem M_intersect_N_is_empty : M ∩ N = ∅ := by
  sorry

end M_intersect_N_is_empty_l378_37855


namespace march_first_is_tuesday_l378_37834

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Given that March 15 is a Tuesday, prove that March 1 is also a Tuesday -/
theorem march_first_is_tuesday (march15 : MarchDate) 
  (h15 : march15.day = 15 ∧ march15.dayOfWeek = DayOfWeek.Tuesday) :
  ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Tuesday :=
sorry

end march_first_is_tuesday_l378_37834


namespace arithmetic_sequence_problem_l378_37886

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  (∃ (a : ℕ → ℝ), 
    (∀ n, a n = arithmetic_sequence a₁ d n) ∧
    ((Real.sin (a 3))^2 * (Real.cos (a 6))^2 - (Real.sin (a 6))^2 * (Real.cos (a 3))^2) / 
      Real.sin (a 4 + a 5) = 1 ∧
    d ∈ Set.Ioo (-1 : ℝ) 0 ∧
    (∀ n : ℕ, n ≠ 9 → 
      (n * a₁ + n * (n - 1) / 2 * d) ≤ (9 * a₁ + 9 * 8 / 2 * d))) →
  a₁ = 17 * Real.pi / 12 ∧ a₁ ∈ Set.Ioo (4 * Real.pi / 3) (3 * Real.pi / 2) :=
by sorry

end arithmetic_sequence_problem_l378_37886


namespace greatest_common_divisor_of_sum_l378_37896

-- Define an arithmetic sequence of positive integers
def arithmetic_sequence (a₁ : ℕ+) (d : ℕ) : ℕ → ℕ+
  | 0 => a₁
  | n + 1 => ⟨(arithmetic_sequence a₁ d n).val + d, by sorry⟩

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic (a₁ : ℕ+) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁.val + (n - 1) * d) / 2

-- Theorem statement
theorem greatest_common_divisor_of_sum (a₁ : ℕ+) (d : ℕ) :
  6 = Nat.gcd (sum_arithmetic a₁ d 12) (Nat.gcd (sum_arithmetic (⟨a₁.val + 1, by sorry⟩) d 12)
    (sum_arithmetic (⟨a₁.val + 2, by sorry⟩) d 12)) :=
by sorry

end greatest_common_divisor_of_sum_l378_37896


namespace jumper_cost_l378_37803

def initial_amount : ℕ := 26
def tshirt_cost : ℕ := 4
def heels_cost : ℕ := 5
def remaining_amount : ℕ := 8

theorem jumper_cost :
  initial_amount - tshirt_cost - heels_cost - remaining_amount = 9 :=
by sorry

end jumper_cost_l378_37803


namespace unique_triple_l378_37818

theorem unique_triple : ∃! (a b c : ℤ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a + b = c ∧ 
  b * c = a ∧ 
  a = -4 ∧ b = 2 ∧ c = -2 := by sorry

end unique_triple_l378_37818


namespace smallest_m_inequality_l378_37881

theorem smallest_m_inequality (a b c : ℝ) :
  ∃ (M : ℝ), M = (9 * Real.sqrt 2) / 32 ∧
  |a*b*(a^2 - b^2) + b*c*(b^2 - c^2) + c*a*(c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2 ∧
  ∀ (N : ℝ), (∀ (x y z : ℝ), |x*y*(x^2 - y^2) + y*z*(y^2 - z^2) + z*x*(z^2 - x^2)| ≤ N * (x^2 + y^2 + z^2)^2) → M ≤ N :=
by sorry

end smallest_m_inequality_l378_37881


namespace plumber_distribution_l378_37862

/-- The number of ways to distribute n plumbers to k residences,
    where all plumbers are assigned, each plumber goes to only one residence,
    and each residence has at least one plumber. -/
def distributionSchemes (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 5 plumbers to 4 residences
    results in 240 different distribution schemes. -/
theorem plumber_distribution :
  distributionSchemes 5 4 = 240 := by sorry

end plumber_distribution_l378_37862


namespace num_distinguishable_triangles_eq_960_l378_37815

/-- Represents the number of colors available for the small triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles needed to form a large triangle -/
def triangles_per_large : ℕ := 4

/-- Represents the number of corner triangles in a large triangle -/
def num_corners : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  ((num_colors + 
    num_colors * (num_colors - 1) + 
    choose num_colors num_corners) * num_colors)

/-- The main theorem stating the number of distinguishable large triangles -/
theorem num_distinguishable_triangles_eq_960 :
  num_distinguishable_triangles = 960 := by sorry

end num_distinguishable_triangles_eq_960_l378_37815


namespace bernie_selection_probability_l378_37857

theorem bernie_selection_probability 
  (p_carol : ℝ) 
  (p_both : ℝ) 
  (h1 : p_carol = 4/5)
  (h2 : p_both = 0.48)
  (h3 : p_both = p_carol * p_bernie)
  : p_bernie = 3/5 :=
by
  sorry

end bernie_selection_probability_l378_37857


namespace steak_cost_calculation_l378_37851

/-- Calculate the total cost of steaks with a buy two get one free offer and a discount --/
theorem steak_cost_calculation (price_per_pound : ℝ) (pounds_bought : ℝ) (discount_rate : ℝ) : 
  price_per_pound = 15 →
  pounds_bought = 24 →
  discount_rate = 0.1 →
  (pounds_bought * price_per_pound) * (1 - discount_rate) = 324 := by
sorry

end steak_cost_calculation_l378_37851


namespace unique_triangle_solution_l378_37890

/-- Represents the assignment of numbers to letters in the triangle puzzle -/
structure TriangleAssignment where
  A : Nat
  B : Nat
  C : Nat
  D : Nat
  E : Nat
  F : Nat

/-- The set of numbers used in the puzzle -/
def puzzleNumbers : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- Checks if the given assignment satisfies all conditions of the puzzle -/
def isValidAssignment (assignment : TriangleAssignment) : Prop :=
  assignment.A ∈ puzzleNumbers ∧
  assignment.B ∈ puzzleNumbers ∧
  assignment.C ∈ puzzleNumbers ∧
  assignment.D ∈ puzzleNumbers ∧
  assignment.E ∈ puzzleNumbers ∧
  assignment.F ∈ puzzleNumbers ∧
  assignment.D + assignment.E + assignment.B = 14 ∧
  assignment.A + assignment.C = 3 ∧
  assignment.A ≠ assignment.B ∧ assignment.A ≠ assignment.C ∧ assignment.A ≠ assignment.D ∧
  assignment.A ≠ assignment.E ∧ assignment.A ≠ assignment.F ∧
  assignment.B ≠ assignment.C ∧ assignment.B ≠ assignment.D ∧ assignment.B ≠ assignment.E ∧
  assignment.B ≠ assignment.F ∧
  assignment.C ≠ assignment.D ∧ assignment.C ≠ assignment.E ∧ assignment.C ≠ assignment.F ∧
  assignment.D ≠ assignment.E ∧ assignment.D ≠ assignment.F ∧
  assignment.E ≠ assignment.F

/-- The unique solution to the triangle puzzle -/
def triangleSolution : TriangleAssignment :=
  { A := 1, B := 3, C := 2, D := 5, E := 6, F := 4 }

/-- Theorem stating that the triangleSolution is the only valid assignment -/
theorem unique_triangle_solution :
  ∀ assignment : TriangleAssignment,
    isValidAssignment assignment → assignment = triangleSolution := by
  sorry

end unique_triangle_solution_l378_37890


namespace rachel_final_lives_l378_37824

/-- Calculates the total number of lives after losing and gaining lives in a video game. -/
def totalLives (initialLives livesLost livesGained : ℕ) : ℕ :=
  initialLives - livesLost + livesGained

/-- Proves that given the initial conditions, Rachel ends up with 32 lives. -/
theorem rachel_final_lives :
  totalLives 10 4 26 = 32 := by
  sorry

#eval totalLives 10 4 26

end rachel_final_lives_l378_37824


namespace cupcakes_per_package_l378_37872

theorem cupcakes_per_package 
  (initial_cupcakes : ℕ) 
  (eaten_cupcakes : ℕ) 
  (num_packages : ℕ) 
  (h1 : initial_cupcakes = 20)
  (h2 : eaten_cupcakes = 11)
  (h3 : num_packages = 3)
  (h4 : eaten_cupcakes < initial_cupcakes) :
  (initial_cupcakes - eaten_cupcakes) / num_packages = 3 :=
by sorry

end cupcakes_per_package_l378_37872


namespace number_problem_l378_37801

theorem number_problem : ∃ x : ℝ, x^2 + 75 = (x - 20)^2 ∧ x = 8.125 := by
  sorry

end number_problem_l378_37801


namespace ducks_in_lake_l378_37897

theorem ducks_in_lake (initial_ducks additional_ducks : ℕ) 
  (h1 : initial_ducks = 13)
  (h2 : additional_ducks = 20) :
  initial_ducks + additional_ducks = 33 := by
  sorry

end ducks_in_lake_l378_37897


namespace exists_non_acute_triangle_with_two_acute_angles_l378_37898

-- Define what an acute angle is
def is_acute_angle (angle : Real) : Prop := 0 < angle ∧ angle < Real.pi / 2

-- Define what a right angle is
def is_right_angle (angle : Real) : Prop := angle = Real.pi / 2

-- Define a triangle structure
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  sum_angles : angle1 + angle2 + angle3 = Real.pi

-- Define what an acute triangle is
def is_acute_triangle (t : Triangle) : Prop :=
  is_acute_angle t.angle1 ∧ is_acute_angle t.angle2 ∧ is_acute_angle t.angle3

-- Theorem statement
theorem exists_non_acute_triangle_with_two_acute_angles :
  ∃ (t : Triangle), (is_acute_angle t.angle1 ∧ is_acute_angle t.angle2) ∧ ¬is_acute_triangle t :=
sorry

end exists_non_acute_triangle_with_two_acute_angles_l378_37898


namespace point_2023_coordinates_l378_37850

/-- Defines the x-coordinate of the nth point in the sequence -/
def x_coord (n : ℕ) : ℤ := 2 * n - 1

/-- Defines the y-coordinate of the nth point in the sequence -/
def y_coord (n : ℕ) : ℤ := (-1 : ℤ) ^ (n - 1) * 2 ^ n

/-- Theorem stating the coordinates of the 2023rd point -/
theorem point_2023_coordinates :
  (x_coord 2023, y_coord 2023) = (4045, 2 ^ 2023) := by
  sorry

end point_2023_coordinates_l378_37850


namespace prime_factors_equation_l378_37879

/-- Given an expression (4^x) * (7^5) * (11^2) with 29 prime factors, prove x = 11 -/
theorem prime_factors_equation (x : ℕ) : 
  (2 * x + 5 + 2 = 29) → x = 11 := by sorry

end prime_factors_equation_l378_37879


namespace line_intersection_x_axis_l378_37817

/-- Given a line y = ax + b passing through points (0, 2) and (-3, 0),
    prove that the solution to ax + b = 0 is x = -3. -/
theorem line_intersection_x_axis 
  (a b : ℝ) 
  (h1 : 2 = a * 0 + b) 
  (h2 : 0 = a * (-3) + b) : 
  ∀ x, a * x + b = 0 ↔ x = -3 :=
sorry

end line_intersection_x_axis_l378_37817


namespace layla_earnings_l378_37863

/-- Calculates the total earnings from babysitting given the hourly rates and hours worked for three families. -/
def total_earnings (rate1 rate2 rate3 : ℕ) (hours1 hours2 hours3 : ℕ) : ℕ :=
  rate1 * hours1 + rate2 * hours2 + rate3 * hours3

/-- Proves that Layla's total earnings from babysitting equal $273 given the specified rates and hours. -/
theorem layla_earnings : total_earnings 15 18 20 7 6 3 = 273 := by
  sorry

#eval total_earnings 15 18 20 7 6 3

end layla_earnings_l378_37863


namespace geese_count_l378_37888

/-- The number of ducks in the marsh -/
def num_ducks : ℝ := 37.0

/-- The difference between the number of geese and ducks -/
def geese_duck_difference : ℕ := 21

/-- The number of geese in the marsh -/
def num_geese : ℝ := num_ducks + geese_duck_difference

theorem geese_count : num_geese = 58 := by
  sorry

end geese_count_l378_37888


namespace all_T_divisible_by_4_l378_37877

/-- The set of all numbers which are the sum of the squares of four consecutive integers
    added to the sum of the integers themselves. -/
def T : Set ℤ :=
  {x | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n-1) + n + (n+1) + (n+2)}

/-- All members of set T are divisible by 4. -/
theorem all_T_divisible_by_4 : ∀ x ∈ T, 4 ∣ x := by sorry

end all_T_divisible_by_4_l378_37877


namespace triangle_area_OAB_l378_37848

/-- Given a line passing through (0, -2) that intersects the parabola y² = 16x at points A and B,
    where the y-coordinates of A and B satisfy y₁² - y₂² = 1, 
    prove that the area of triangle OAB (where O is the origin) is 1/16. -/
theorem triangle_area_OAB : 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (∃ m : ℝ, x₁ = m * y₁ + 2 * m ∧ x₂ = m * y₂ + 2 * m) → -- Line equation
    y₁^2 = 16 * x₁ →                                      -- A satisfies parabola equation
    y₂^2 = 16 * x₂ →                                      -- B satisfies parabola equation
    y₁^2 - y₂^2 = 1 →                                     -- Given condition
    (1/2 : ℝ) * |x₁ * y₂ - x₂ * y₁| = 1/16 :=             -- Area of triangle OAB
by sorry

end triangle_area_OAB_l378_37848


namespace final_value_exceeds_initial_l378_37841

theorem final_value_exceeds_initial (p q r M : ℝ) 
  (hp : p > 0) (hq : 0 < q ∧ q < 100) (hr : 0 < r ∧ r < 100) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) * (1 + r / 100) > M ↔ 
  p > (100 * (q - r + q * r / 100)) / (100 - q + r + q * r / 100) :=
by sorry

end final_value_exceeds_initial_l378_37841


namespace existence_of_greater_indices_l378_37871

theorem existence_of_greater_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p > q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end existence_of_greater_indices_l378_37871


namespace farm_width_is_15km_l378_37825

/-- A rectangular farm with given properties has a width of 15 kilometers. -/
theorem farm_width_is_15km (length width : ℝ) : 
  length > 0 →
  width > 0 →
  2 * (length + width) = 46 →
  width = length + 7 →
  width = 15 := by
sorry

end farm_width_is_15km_l378_37825


namespace probability_four_green_marbles_l378_37880

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 8

/-- The number of purple marbles in the bag -/
def purple_marbles : ℕ := 4

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of draws -/
def num_draws : ℕ := 8

/-- The number of green marbles we want to draw -/
def target_green : ℕ := 4

/-- The probability of drawing exactly 'target_green' green marbles in 'num_draws' draws -/
def probability_exact_green : ℚ :=
  (Nat.choose num_draws target_green : ℚ) *
  (green_marbles ^ target_green * purple_marbles ^ (num_draws - target_green)) /
  (total_marbles ^ num_draws)

theorem probability_four_green_marbles :
  probability_exact_green = 1120 / 6561 :=
sorry

end probability_four_green_marbles_l378_37880


namespace school_election_votes_l378_37823

theorem school_election_votes (randy_votes shaun_votes eliot_votes : ℕ) : 
  randy_votes = 16 →
  shaun_votes = 5 * randy_votes →
  eliot_votes = 2 * shaun_votes →
  eliot_votes = 160 := by
  sorry

end school_election_votes_l378_37823


namespace cubic_root_magnitude_l378_37821

theorem cubic_root_magnitude (q : ℝ) (r₁ r₂ r₃ : ℝ) : 
  r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ →
  r₁^3 + q*r₁^2 + 6*r₁ + 9 = 0 →
  r₂^3 + q*r₂^2 + 6*r₂ + 9 = 0 →
  r₃^3 + q*r₃^2 + 6*r₃ + 9 = 0 →
  (q^2 * 6^2 - 4 * 6^3 - 4*q^3 * 9 - 27 * 9^2 + 18 * q * 6 * (-9)) ≠ 0 →
  max (|r₁|) (max (|r₂|) (|r₃|)) > 2 := by
sorry

end cubic_root_magnitude_l378_37821


namespace intersection_polar_radius_l378_37876

-- Define the line l
def line_l (x : ℝ) : ℝ := x + 1

-- Define the curve C in polar form
def curve_C (ρ θ : ℝ) : Prop := ρ * (Real.sin θ)^2 - 4 * Real.cos θ = 0 ∧ ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem intersection_polar_radius :
  ∃ (x y ρ θ : ℝ),
    y = line_l x ∧
    curve_C ρ θ ∧
    x = ρ * Real.cos θ ∧
    y = ρ * Real.sin θ ∧
    ρ = Real.sqrt 5 :=
by sorry

end intersection_polar_radius_l378_37876


namespace ducks_theorem_l378_37869

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_theorem : ducks_remaining 320 = 140 := by
  sorry

end ducks_theorem_l378_37869


namespace function_equality_l378_37861

theorem function_equality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f (f x + f y)) = f x + y) →
  (∀ x : ℝ, f x = x) :=
by sorry

end function_equality_l378_37861


namespace six_times_two_minus_three_l378_37860

theorem six_times_two_minus_three : 6 * 2 - 3 = 9 := by
  sorry

end six_times_two_minus_three_l378_37860


namespace no_integer_roots_l378_37849

theorem no_integer_roots : ¬∃ (x : ℤ), x^3 - 4*x^2 - 11*x + 20 = 0 := by
  sorry

end no_integer_roots_l378_37849


namespace circle_area_irrational_if_rational_diameter_l378_37826

theorem circle_area_irrational_if_rational_diameter :
  ∀ d : ℚ, d > 0 → ∃ A : ℝ, A = π * (d / 2)^2 ∧ Irrational A := by
  sorry

end circle_area_irrational_if_rational_diameter_l378_37826


namespace positive_solution_x_l378_37842

theorem positive_solution_x (x y z : ℝ) : 
  x > 0 →
  x * y + 3 * x + 4 * y + 10 = 30 →
  y * z + 4 * y + 2 * z + 8 = 6 →
  x * z + 4 * x + 3 * z + 12 = 30 →
  x = 3 := by
sorry

end positive_solution_x_l378_37842
