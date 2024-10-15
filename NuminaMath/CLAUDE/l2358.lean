import Mathlib

namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l2358_235884

theorem unique_congruence_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -1872 [ZMOD 9] ∧ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l2358_235884


namespace NUMINAMATH_CALUDE_fraction_exponent_product_l2358_235876

theorem fraction_exponent_product : 
  (8 / 9 : ℚ) ^ 3 * (1 / 3 : ℚ) ^ 3 = 512 / 19683 := by sorry

end NUMINAMATH_CALUDE_fraction_exponent_product_l2358_235876


namespace NUMINAMATH_CALUDE_polygon_reassembly_l2358_235893

-- Define a polygon as a set of points in 2D space
def Polygon : Type := Set (ℝ × ℝ)

-- Define the area of a polygon
def area (P : Polygon) : ℝ := sorry

-- Define a function to represent cutting and reassembling a polygon
def can_reassemble (P Q : Polygon) : Prop := sorry

-- Define a rectangle with one side of length 1
def rectangle_with_unit_side (R : Polygon) : Prop := sorry

theorem polygon_reassembly (P Q : Polygon) (h : area P = area Q) :
  (∃ R : Polygon, can_reassemble P R ∧ rectangle_with_unit_side R) ∧
  can_reassemble P Q := by sorry

end NUMINAMATH_CALUDE_polygon_reassembly_l2358_235893


namespace NUMINAMATH_CALUDE_digits_of_product_l2358_235812

theorem digits_of_product : ∃ n : ℕ, n > 0 ∧ 10^(n-1) ≤ 2^15 * 5^10 * 3 ∧ 2^15 * 5^10 * 3 < 10^n ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_product_l2358_235812


namespace NUMINAMATH_CALUDE_domain_f_minus_one_l2358_235853

-- Define a real-valued function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_f_minus_one (h : ∀ x, f (x + 1) ∈ domain_f_plus_one ↔ x ∈ domain_f_plus_one) :
  ∀ x, f (x - 1) ∈ Set.Icc 0 5 ↔ x ∈ Set.Icc 0 5 :=
sorry

end NUMINAMATH_CALUDE_domain_f_minus_one_l2358_235853


namespace NUMINAMATH_CALUDE_divisibility_by_20p_l2358_235840

theorem divisibility_by_20p (p : ℕ) (h_prime : Prime p) (h_odd : Odd p) :
  ∃ k : ℤ, (⌊(Real.sqrt 5 + 2)^p - 2^(p+1)⌋ : ℤ) = 20 * p * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_20p_l2358_235840


namespace NUMINAMATH_CALUDE_shirt_price_change_l2358_235820

theorem shirt_price_change (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.84 * P → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_change_l2358_235820


namespace NUMINAMATH_CALUDE_xy_value_l2358_235824

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2358_235824


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l2358_235825

theorem consecutive_page_numbers (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 15300 → n + (n + 1) = 247 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l2358_235825


namespace NUMINAMATH_CALUDE_felicity_gas_usage_l2358_235821

theorem felicity_gas_usage (adhira : ℝ) 
  (h1 : 4 * adhira - 5 + adhira = 30) : 
  4 * adhira - 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_felicity_gas_usage_l2358_235821


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_100_l2358_235817

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_greater_than_100 : 
  (¬ ∃ n : ℕ, 2^n > 100) ↔ (∀ n : ℕ, 2^n ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_100_l2358_235817


namespace NUMINAMATH_CALUDE_find_y_l2358_235800

theorem find_y (a b y : ℝ) (ha : a > 0) (hb : b > 0) (hy : y > 0) 
  (h : (2 * a) ^ (4 * b) = a ^ b * y ^ (3 * b)) : 
  y = 2 ^ (4 / 3) * a :=
sorry

end NUMINAMATH_CALUDE_find_y_l2358_235800


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2358_235818

theorem sqrt_expression_equality (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - x^3) / (3 * x^3))^2) = (x^3 - 1 + Real.sqrt (x^6 - 2*x^3 + 10)) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2358_235818


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2358_235863

theorem perfect_square_condition (a : ℕ+) 
  (h : ∀ n : ℕ, ∃ d : ℕ, d ≠ 1 ∧ d % n = 1 ∧ (n^2 * a.val - 1) % d = 0) : 
  ∃ k : ℕ, a.val = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2358_235863


namespace NUMINAMATH_CALUDE_tims_score_is_56_l2358_235868

/-- The sum of the first n even numbers -/
def sum_first_n_even (n : ℕ) : ℕ :=
  (2 * n * (n + 1)) / 2

/-- Tim's math score -/
def tims_score : ℕ := sum_first_n_even 7

theorem tims_score_is_56 : tims_score = 56 := by
  sorry

end NUMINAMATH_CALUDE_tims_score_is_56_l2358_235868


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_2023_l2358_235806

theorem tens_digit_of_13_pow_2023 :
  ∃ k : ℕ, 13^2023 = 100 * k + 97 :=
by
  -- We assume 13^20 ≡ 1 (mod 100) as a hypothesis
  have h1 : ∃ m : ℕ, 13^20 = 100 * m + 1 := sorry
  
  -- We use the division algorithm to write 2023 = 20q + r
  have h2 : ∃ q r : ℕ, 2023 = 20 * q + r ∧ r < 20 := sorry
  
  -- We prove that r = 3
  have h3 : ∃ q : ℕ, 2023 = 20 * q + 3 := sorry
  
  -- We prove that 13^3 ≡ 97 (mod 100)
  have h4 : ∃ n : ℕ, 13^3 = 100 * n + 97 := sorry
  
  -- Main proof
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_2023_l2358_235806


namespace NUMINAMATH_CALUDE_becketts_age_l2358_235867

theorem becketts_age (beckett olaf shannen jack : ℕ) 
  (h1 : beckett = olaf - 3)
  (h2 : shannen = olaf - 2)
  (h3 : jack = 2 * shannen + 5)
  (h4 : beckett + olaf + shannen + jack = 71) :
  beckett = 12 := by
  sorry

end NUMINAMATH_CALUDE_becketts_age_l2358_235867


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2358_235802

theorem rationalize_denominator :
  ∃ (A B : ℚ) (C : ℕ),
    (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C ∧
    A = 11 / 4 ∧
    B = 5 / 4 ∧
    C = 5 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2358_235802


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l2358_235852

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x + a)(x - 4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x + a)(x - 4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four :
  ∀ a : ℝ, IsEven (f a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l2358_235852


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2358_235855

/-- Given that (x^2 - 1) + (x^2 + 3x + 2)i is a purely imaginary number, prove that x = 1 -/
theorem purely_imaginary_complex_number (x : ℝ) : 
  (x^2 - 1 : ℂ) + (x^2 + 3*x + 2 : ℂ)*I = (0 : ℂ) + (y : ℝ)*I ∧ y ≠ 0 → x = 1 := by
  sorry


end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2358_235855


namespace NUMINAMATH_CALUDE_third_one_is_13th_a_2015_is_31_l2358_235845

-- Define the sequence a_n
def a : ℕ → ℚ
| 0 => 0  -- We start from index 1, so define 0 as a placeholder
| n => 
  let k := (n.sqrt + 1) / 2  -- Calculate which group the term belongs to
  let m := n - (k - 1) * k   -- Calculate position within the group
  m.succ / (k + 1 - m)       -- Return the fraction

-- Third term equal to 1
theorem third_one_is_13th : ∃ n₁ n₂ : ℕ, n₁ < n₂ ∧ n₂ < 13 ∧ a n₁ = 1 ∧ a n₂ = 1 ∧ a 13 = 1 :=
sorry

-- 2015th term
theorem a_2015_is_31 : a 2015 = 31 :=
sorry

end NUMINAMATH_CALUDE_third_one_is_13th_a_2015_is_31_l2358_235845


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_l2358_235857

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 3| + |y + 2| = 0 → (y - x = -5 ∧ x * y = -6) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_l2358_235857


namespace NUMINAMATH_CALUDE_unique_prime_twice_squares_l2358_235866

theorem unique_prime_twice_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x : ℕ), p + 1 = 2 * x^2) ∧ 
    (∃ (y : ℕ), p^2 + 1 = 2 * y^2) ∧ 
    p = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_twice_squares_l2358_235866


namespace NUMINAMATH_CALUDE_horner_rule_v2_equals_14_l2358_235898

def f (x : ℝ) : ℝ := 2*x^4 + 3*x^3 + 5*x - 4

def horner_v2 (a b c d e x : ℝ) : ℝ := ((a * x + b) * x + c) * x + d

theorem horner_rule_v2_equals_14 : 
  horner_v2 2 3 0 5 (-4) 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_v2_equals_14_l2358_235898


namespace NUMINAMATH_CALUDE_towel_average_price_l2358_235854

theorem towel_average_price :
  let towel_group1 : ℕ := 3
  let price1 : ℕ := 100
  let towel_group2 : ℕ := 5
  let price2 : ℕ := 150
  let towel_group3 : ℕ := 2
  let price3 : ℕ := 400
  let total_towels := towel_group1 + towel_group2 + towel_group3
  let total_cost := towel_group1 * price1 + towel_group2 * price2 + towel_group3 * price3
  (total_cost : ℚ) / (total_towels : ℚ) = 185 := by
  sorry

end NUMINAMATH_CALUDE_towel_average_price_l2358_235854


namespace NUMINAMATH_CALUDE_pairwise_sum_problem_l2358_235879

/-- Given four numbers that when added pairwise result in specific sums, 
    prove the remaining sums and possible sets of numbers -/
theorem pairwise_sum_problem (a b c d : ℝ) : 
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ 
  d + c = 20 ∧ d + b = 16 ∧ 
  ((d + a = 13 ∧ c + b = 9) ∨ (d + a = 9 ∧ c + b = 13)) →
  (a + b = 2 ∧ a + c = 6) ∧
  ((a = -0.5 ∧ b = 2.5 ∧ c = 6.5 ∧ d = 13.5) ∨
   (a = -2.5 ∧ b = 4.5 ∧ c = 8.5 ∧ d = 11.5)) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_sum_problem_l2358_235879


namespace NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l2358_235865

/-- The minimum distance between a point on y = e^x and a point on y = x is √2/2 -/
theorem min_distance_exp_curve_to_line : 
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
    ∀ (p q : ℝ × ℝ), p.2 = Real.exp p.1 → q.2 = q.1 → 
      dist p q ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l2358_235865


namespace NUMINAMATH_CALUDE_combined_weight_l2358_235886

/-- The weight of a peach in grams -/
def peach_weight : ℝ := sorry

/-- The weight of a bun in grams -/
def bun_weight : ℝ := sorry

/-- Condition 1: One peach weighs the same as 2 buns plus 40 grams -/
axiom condition1 : peach_weight = 2 * bun_weight + 40

/-- Condition 2: One peach plus 80 grams weighs the same as one bun plus 200 grams -/
axiom condition2 : peach_weight + 80 = bun_weight + 200

/-- Theorem: The combined weight of 1 peach and 1 bun is 280 grams -/
theorem combined_weight : peach_weight + bun_weight = 280 := by sorry

end NUMINAMATH_CALUDE_combined_weight_l2358_235886


namespace NUMINAMATH_CALUDE_anya_balloons_l2358_235847

theorem anya_balloons (total : ℕ) (colors : ℕ) (anya_fraction : ℚ) 
  (h1 : total = 672) 
  (h2 : colors = 4) 
  (h3 : anya_fraction = 1/2) : 
  (total / colors) * anya_fraction = 84 := by
  sorry

end NUMINAMATH_CALUDE_anya_balloons_l2358_235847


namespace NUMINAMATH_CALUDE_sign_up_options_count_l2358_235860

/-- The number of students. -/
def num_students : ℕ := 5

/-- The number of teams. -/
def num_teams : ℕ := 3

/-- The total number of sign-up options. -/
def total_options : ℕ := num_teams ^ num_students

/-- 
Theorem: Given 5 students and 3 teams, where each student must choose exactly one team,
the total number of possible sign-up combinations is 3^5.
-/
theorem sign_up_options_count :
  total_options = 243 := by sorry

end NUMINAMATH_CALUDE_sign_up_options_count_l2358_235860


namespace NUMINAMATH_CALUDE_rectangle_side_length_l2358_235887

/-- Given a rectangle with area 9a^2 - 6ab + 3a and one side length 3a, 
    the other side length is 3a - 2b + 1 -/
theorem rectangle_side_length (a b : ℝ) : 
  let area := 9*a^2 - 6*a*b + 3*a
  let side1 := 3*a
  let side2 := 3*a - 2*b + 1
  area = side1 * side2 := by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l2358_235887


namespace NUMINAMATH_CALUDE_partner_a_contribution_l2358_235809

/-- A business partnership where two partners contribute capital for different durations and share profits proportionally. -/
structure BusinessPartnership where
  /-- Duration (in months) that Partner A's capital is used -/
  duration_a : ℕ
  /-- Duration (in months) that Partner B's capital is used -/
  duration_b : ℕ
  /-- Fraction of profit received by Partner B -/
  profit_share_b : ℚ
  /-- Fraction of capital contributed by Partner A -/
  capital_fraction_a : ℚ

/-- Theorem stating that under given conditions, Partner A's capital contribution is 1/4 -/
theorem partner_a_contribution
  (bp : BusinessPartnership)
  (h1 : bp.duration_a = 15)
  (h2 : bp.duration_b = 10)
  (h3 : bp.profit_share_b = 2/3)
  : bp.capital_fraction_a = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_partner_a_contribution_l2358_235809


namespace NUMINAMATH_CALUDE_power_equation_solution_l2358_235801

theorem power_equation_solution (n : ℕ) : (3^n)^2 = 3^16 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2358_235801


namespace NUMINAMATH_CALUDE_distance_traveled_l2358_235873

/-- Given a speed of 20 km/hr and a travel time of 2.5 hours, prove that the distance traveled is 50 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 20)
  (h2 : time = 2.5)
  (h3 : distance = speed * time) : 
  distance = 50 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l2358_235873


namespace NUMINAMATH_CALUDE_cosine_equality_l2358_235834

theorem cosine_equality (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (830 * π / 180) → n = 70 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l2358_235834


namespace NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2358_235848

/-- Represents a circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem: The y-intercept of the common external tangent with positive slope for two given circles --/
theorem common_external_tangent_y_intercept 
  (c1 : Circle) 
  (c2 : Circle) 
  (h1 : c1.center = (5, -2)) 
  (h2 : c1.radius = 5)
  (h3 : c2.center = (20, 6))
  (h4 : c2.radius = 12) :
  ∃ (m b : ℝ), m > 0 ∧ b = -2100/161 ∧ 
  (∀ (x y : ℝ), y = m * x + b ↔ 
    (y - c1.center.2)^2 + (x - c1.center.1)^2 = (c1.radius + c2.radius)^2 ∧
    (y - c2.center.2)^2 + (x - c2.center.1)^2 = (c1.radius + c2.radius)^2) :=
sorry

end NUMINAMATH_CALUDE_common_external_tangent_y_intercept_l2358_235848


namespace NUMINAMATH_CALUDE_wendy_trip_miles_l2358_235871

def three_day_trip (day1_miles day2_miles total_miles : ℕ) : Prop :=
  ∃ day3_miles : ℕ, day1_miles + day2_miles + day3_miles = total_miles

theorem wendy_trip_miles :
  three_day_trip 125 223 493 →
  ∃ day3_miles : ℕ, day3_miles = 145 :=
by sorry

end NUMINAMATH_CALUDE_wendy_trip_miles_l2358_235871


namespace NUMINAMATH_CALUDE_parallelepiped_coverage_l2358_235827

/-- Represents a parallelepiped -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a square -/
structure Square where
  side : ℕ

/-- Calculates the surface area of a parallelepiped -/
def surface_area (p : Parallelepiped) : ℕ :=
  2 * (p.length * p.width + p.length * p.height + p.width * p.height)

/-- Calculates the area of a square -/
def square_area (s : Square) : ℕ :=
  s.side * s.side

/-- Theorem stating that a 1x1x4 parallelepiped can be covered by two 4x4 squares and one 1x1 square -/
theorem parallelepiped_coverage :
  ∃ (p : Parallelepiped) (s1 s2 s3 : Square),
    p.length = 1 ∧ p.width = 1 ∧ p.height = 4 ∧
    s1.side = 4 ∧ s2.side = 4 ∧ s3.side = 1 ∧
    surface_area p = square_area s1 + square_area s2 + square_area s3 :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_coverage_l2358_235827


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l2358_235816

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 20)
  (h2 : average_speed = 40) : 
  ∃ (speed_second_hour : ℝ), speed_second_hour = 60 := by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l2358_235816


namespace NUMINAMATH_CALUDE_number_of_combinations_max_probability_sums_l2358_235896

-- Define the structure of a box
structure Box :=
  (ball1 : Nat)
  (ball2 : Nat)

-- Define the set of boxes
def boxes : List Box := [
  { ball1 := 1, ball2 := 2 },
  { ball1 := 1, ball2 := 2 },
  { ball1 := 1, ball2 := 2 }
]

-- Define a combination of drawn balls
def Combination := Nat × Nat × Nat

-- Function to generate all possible combinations
def generateCombinations (boxes : List Box) : List Combination := sorry

-- Function to calculate the sum of a combination
def sumCombination (c : Combination) : Nat := sorry

-- Function to count occurrences of a sum
def countSum (sum : Nat) (combinations : List Combination) : Nat := sorry

-- Theorem: The number of possible combinations is 8
theorem number_of_combinations :
  (generateCombinations boxes).length = 8 := by sorry

-- Theorem: The sums 4 and 5 have the highest probability
theorem max_probability_sums (combinations : List Combination := generateCombinations boxes) :
  ∀ (s : Nat), s ≠ 4 ∧ s ≠ 5 →
    countSum s combinations ≤ countSum 4 combinations ∧
    countSum s combinations ≤ countSum 5 combinations := by sorry

end NUMINAMATH_CALUDE_number_of_combinations_max_probability_sums_l2358_235896


namespace NUMINAMATH_CALUDE_evaluate_f_l2358_235869

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

-- State the theorem
theorem evaluate_f : 3 * f 2 - 2 * f (-2) = -31 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l2358_235869


namespace NUMINAMATH_CALUDE_jessica_bank_account_l2358_235831

theorem jessica_bank_account (B : ℝ) : 
  B > 0 →
  (3/5) * B = B - 200 →
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    (3/5) * B + (x/y) * ((3/5) * B) = 450 →
    x/y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_bank_account_l2358_235831


namespace NUMINAMATH_CALUDE_tennis_tournament_result_l2358_235810

/-- Represents the number of participants with k points after m rounds in a tournament with 2^n participants. -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * Nat.choose m k

/-- The rules of the tennis tournament. -/
structure TournamentRules where
  participants : ℕ
  victoryPoints : ℕ
  lossPoints : ℕ
  additionalPointRule : Bool
  pairingRule : Bool

/-- The specific tournament in question. -/
def tennisTournament : TournamentRules where
  participants := 256  -- Including two fictitious participants
  victoryPoints := 1
  lossPoints := 0
  additionalPointRule := true
  pairingRule := true

/-- The theorem to be proved. -/
theorem tennis_tournament_result (t : TournamentRules) (h : t = tennisTournament) :
  f 8 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_result_l2358_235810


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2358_235826

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ x^10 + 7*x^9 + 14*x^8 + 1729*x^7 - 1379*x^6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l2358_235826


namespace NUMINAMATH_CALUDE_expression_evaluation_l2358_235880

/-- Given a = -2 and b = -1/2, prove that the expression 3(2a²-4ab)-[a²-3(4a+ab)] evaluates to -13 -/
theorem expression_evaluation (a b : ℚ) (h1 : a = -2) (h2 : b = -1/2) :
  3 * (2 * a^2 - 4 * a * b) - (a^2 - 3 * (4 * a + a * b)) = -13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2358_235880


namespace NUMINAMATH_CALUDE_cubic_equation_real_root_l2358_235842

theorem cubic_equation_real_root (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 3) * (x + 2) := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_real_root_l2358_235842


namespace NUMINAMATH_CALUDE_sum_of_digits_1_to_10000_l2358_235822

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sequence_sum_of_digits (n : Nat) : Nat :=
  (List.range n).map sum_of_digits |> List.sum

theorem sum_of_digits_1_to_10000 :
  sequence_sum_of_digits 10000 = 180001 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_1_to_10000_l2358_235822


namespace NUMINAMATH_CALUDE_min_sum_of_product_l2358_235828

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : 
  ∀ x y : ℤ, x * y = 144 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l2358_235828


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l2358_235875

theorem product_of_roots_plus_one (a b c : ℝ) : 
  (a^3 - 15*a^2 + 20*a - 8 = 0) → 
  (b^3 - 15*b^2 + 20*b - 8 = 0) → 
  (c^3 - 15*c^2 + 20*c - 8 = 0) → 
  (1 + a) * (1 + b) * (1 + c) = 28 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l2358_235875


namespace NUMINAMATH_CALUDE_collinear_points_sum_l2358_235814

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define collinearity for four points in 3D space
def collinear (p q r s : Point3D) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), 
    q.x - p.x = t₁ * (r.x - p.x) ∧
    q.y - p.y = t₁ * (r.y - p.y) ∧
    q.z - p.z = t₁ * (r.z - p.z) ∧
    s.x - p.x = t₂ * (r.x - p.x) ∧
    s.y - p.y = t₂ * (r.y - p.y) ∧
    s.z - p.z = t₂ * (r.z - p.z) ∧
    t₃ * (q.x - p.x) = s.x - p.x ∧
    t₃ * (q.y - p.y) = s.y - p.y ∧
    t₃ * (q.z - p.z) = s.z - p.z

theorem collinear_points_sum (a b : ℝ) : 
  collinear 
    (Point3D.mk 2 a b) 
    (Point3D.mk a 3 b) 
    (Point3D.mk a b 4) 
    (Point3D.mk 5 b a) → 
  a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l2358_235814


namespace NUMINAMATH_CALUDE_pizza_toppings_l2358_235839

/-- Represents a pizza with a given number of slices and topping distribution -/
structure Pizza where
  total_slices : ℕ
  pepperoni_slices : ℕ
  mushroom_slices : ℕ
  both_toppings : ℕ
  pepperoni_only : ℕ
  mushroom_only : ℕ
  h_total : total_slices = pepperoni_only + mushroom_only + both_toppings
  h_pepperoni : pepperoni_slices = pepperoni_only + both_toppings
  h_mushroom : mushroom_slices = mushroom_only + both_toppings

/-- Theorem stating that a pizza with the given conditions has 2 slices with both toppings -/
theorem pizza_toppings (p : Pizza) 
  (h_total : p.total_slices = 18)
  (h_pep : p.pepperoni_slices = 10)
  (h_mush : p.mushroom_slices = 10) :
  p.both_toppings = 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2358_235839


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2358_235850

theorem arithmetic_sequence_problem (d a_n n : ℤ) (h1 : d = 2) (h2 : n = 15) (h3 : a_n = -10) :
  ∃ (a_1 S_n : ℤ),
    a_1 = -38 ∧
    S_n = -360 ∧
    a_n = a_1 + (n - 1) * d ∧
    S_n = n * (a_1 + a_n) / 2 :=
by sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2358_235850


namespace NUMINAMATH_CALUDE_nancys_payment_is_384_l2358_235858

/-- Nancy's annual payment for her daughter's car insurance -/
def nancys_annual_payment (monthly_cost : ℝ) (nancys_percentage : ℝ) : ℝ :=
  monthly_cost * nancys_percentage * 12

/-- Theorem: Nancy's annual payment for her daughter's car insurance is $384 -/
theorem nancys_payment_is_384 :
  nancys_annual_payment 80 0.4 = 384 := by
  sorry

end NUMINAMATH_CALUDE_nancys_payment_is_384_l2358_235858


namespace NUMINAMATH_CALUDE_polynomial_value_relation_l2358_235897

theorem polynomial_value_relation (m n : ℝ) : 
  -m^2 + 3*n = 2 → m^2 - 3*n - 1 = -3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_relation_l2358_235897


namespace NUMINAMATH_CALUDE_total_distance_calculation_l2358_235803

/-- Represents the problem of calculating the total distance traveled by a person
    given specific conditions. -/
theorem total_distance_calculation (d : ℝ) : 
  (d / 6 + d / 12 + d / 18 + d / 24 + d / 30 = 17 / 60) → 
  (5 * d = 425 / 114) := by
  sorry

#check total_distance_calculation

end NUMINAMATH_CALUDE_total_distance_calculation_l2358_235803


namespace NUMINAMATH_CALUDE_jennas_eel_length_l2358_235838

theorem jennas_eel_length (j b : ℝ) (h1 : j = b / 3) (h2 : j + b = 64) :
  j = 16 := by
  sorry

end NUMINAMATH_CALUDE_jennas_eel_length_l2358_235838


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l2358_235889

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < 0 ∧ 0 < x₂ ∧ y₁ = 3 / x₁ ∧ y₂ = 3 / x₂ → y₁ < 0 ∧ 0 < y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l2358_235889


namespace NUMINAMATH_CALUDE_linda_current_age_l2358_235899

/-- Represents the ages of Sarah, Jake, and Linda -/
structure Ages where
  sarah : ℚ
  jake : ℚ
  linda : ℚ

/-- The conditions of the problem -/
def age_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 11
  (ages.sarah + ages.jake + ages.linda) / 3 = 11 ∧
  -- Five years ago, Linda was the same age as Sarah is now
  ages.linda - 5 = ages.sarah ∧
  -- In 4 years, Jake's age will be 3/4 of Sarah's age at that time
  ages.jake + 4 = 3 / 4 * (ages.sarah + 4)

/-- The theorem stating Linda's current age -/
theorem linda_current_age (ages : Ages) (h : age_conditions ages) : 
  ages.linda = 14 := by
  sorry

end NUMINAMATH_CALUDE_linda_current_age_l2358_235899


namespace NUMINAMATH_CALUDE_probability_play_exactly_one_l2358_235870

def total_people : ℕ := 800
def play_at_least_one_ratio : ℚ := 1 / 5
def play_two_or_more : ℕ := 64

theorem probability_play_exactly_one (total_people : ℕ) (play_at_least_one_ratio : ℚ) (play_two_or_more : ℕ) :
  (play_at_least_one_ratio * total_people - play_two_or_more : ℚ) / total_people = 12 / 100 :=
by sorry

end NUMINAMATH_CALUDE_probability_play_exactly_one_l2358_235870


namespace NUMINAMATH_CALUDE_base_number_problem_l2358_235843

theorem base_number_problem (x : ℝ) (k : ℕ) 
  (h1 : x^k = 5) 
  (h2 : x^(2*k + 2) = 400) : 
  x = 5 := by sorry

end NUMINAMATH_CALUDE_base_number_problem_l2358_235843


namespace NUMINAMATH_CALUDE_equation_roots_range_l2358_235835

theorem equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 2*k*x₁^2 + (8*k+1)*x₁ = -8*k ∧ 2*k*x₂^2 + (8*k+1)*x₂ = -8*k) →
  (k > -1/16 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_range_l2358_235835


namespace NUMINAMATH_CALUDE_alex_original_seat_l2358_235882

/-- Represents a seat in the movie theater --/
inductive Seat
| one | two | three | four | five | six | seven

/-- Represents the possible movements of friends --/
inductive Movement
| left : ℕ → Movement
| right : ℕ → Movement
| switch : Movement
| none : Movement

/-- Represents a friend in the theater --/
structure Friend :=
  (name : String)
  (initial_seat : Seat)
  (movement : Movement)

/-- The state of the theater --/
structure TheaterState :=
  (friends : List Friend)
  (alex_initial : Seat)
  (alex_final : Seat)

def is_end_seat (s : Seat) : Prop :=
  s = Seat.one ∨ s = Seat.seven

def move_left (s : Seat) (n : ℕ) : Seat :=
  match s, n with
  | Seat.one, _ => Seat.one
  | Seat.two, 1 => Seat.one
  | Seat.three, 1 => Seat.two
  | Seat.three, 2 => Seat.one
  | Seat.four, 1 => Seat.three
  | Seat.four, 2 => Seat.two
  | Seat.four, 3 => Seat.one
  | Seat.five, 1 => Seat.four
  | Seat.five, 2 => Seat.three
  | Seat.five, 3 => Seat.two
  | Seat.five, 4 => Seat.one
  | Seat.six, 1 => Seat.five
  | Seat.six, 2 => Seat.four
  | Seat.six, 3 => Seat.three
  | Seat.six, 4 => Seat.two
  | Seat.six, 5 => Seat.one
  | Seat.seven, 1 => Seat.six
  | Seat.seven, 2 => Seat.five
  | Seat.seven, 3 => Seat.four
  | Seat.seven, 4 => Seat.three
  | Seat.seven, 5 => Seat.two
  | Seat.seven, 6 => Seat.one
  | s, _ => s

theorem alex_original_seat (state : TheaterState) :
  state.friends = [
    ⟨"Bob", Seat.three, Movement.right 3⟩,
    ⟨"Cara", Seat.five, Movement.left 2⟩,
    ⟨"Dana", Seat.four, Movement.switch⟩,
    ⟨"Eve", Seat.two, Movement.switch⟩,
    ⟨"Fiona", Seat.six, Movement.right 1⟩,
    ⟨"Greg", Seat.seven, Movement.none⟩
  ] →
  is_end_seat state.alex_final →
  state.alex_initial = Seat.three :=
by sorry


end NUMINAMATH_CALUDE_alex_original_seat_l2358_235882


namespace NUMINAMATH_CALUDE_gcd_g_x_l2358_235877

def g (x : ℤ) : ℤ := (4*x+5)*(5*x+2)*(11*x+8)*(3*x+7)

theorem gcd_g_x (x : ℤ) (h : 2520 ∣ x) : Int.gcd (g x) x = 280 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_l2358_235877


namespace NUMINAMATH_CALUDE_circle_through_points_center_on_line_l2358_235862

/-- A circle passing through two points with its center on a given line -/
theorem circle_through_points_center_on_line (A B O : ℝ × ℝ) (r : ℝ) :
  A = (1, -1) →
  B = (-1, 1) →
  O.1 + O.2 = 2 →
  r = 2 →
  ∀ (x y : ℝ), (x - O.1)^2 + (y - O.2)^2 = r^2 ↔
    ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) ∨
    (x - O.1)^2 + (y - O.2)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_through_points_center_on_line_l2358_235862


namespace NUMINAMATH_CALUDE_sqrt_pi_squared_minus_6pi_plus_9_l2358_235885

theorem sqrt_pi_squared_minus_6pi_plus_9 : 
  Real.sqrt (π^2 - 6*π + 9) = π - 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pi_squared_minus_6pi_plus_9_l2358_235885


namespace NUMINAMATH_CALUDE_no_real_m_for_equal_roots_l2358_235861

/-- The equation whose roots we're analyzing -/
def equation (x m : ℝ) : Prop :=
  (3 * x^2 * (x - 2) - (2*m + 3)) / ((x - 2) * (m - 2)) = 2 * x^2 / m

/-- Theorem stating that there are no real values of m for which the roots of the equation are equal -/
theorem no_real_m_for_equal_roots :
  ¬ ∃ (m : ℝ), ∃ (x : ℝ), ∀ (y : ℝ), equation y m → y = x :=
sorry

end NUMINAMATH_CALUDE_no_real_m_for_equal_roots_l2358_235861


namespace NUMINAMATH_CALUDE_wolf_hunger_theorem_l2358_235804

/-- Represents the satiety value of a food item -/
structure SatietyValue (α : Type) where
  value : ℝ

/-- Represents the satiety state of the wolf -/
inductive SatietyState
  | Hunger
  | Satisfied
  | Overeating

/-- The satiety value of a piglet -/
def piglet_satiety : SatietyValue ℝ := ⟨1⟩

/-- The satiety value of a kid -/
def kid_satiety : SatietyValue ℝ := ⟨1⟩

/-- Calculates the total satiety value of a meal -/
def meal_satiety (piglets kids : ℕ) : ℝ :=
  (piglets : ℝ) * piglet_satiety.value + (kids : ℝ) * kid_satiety.value

/-- Determines the satiety state based on the meal satiety -/
def get_satiety_state (meal : ℝ) : SatietyState := sorry

/-- The theorem to be proved -/
theorem wolf_hunger_theorem :
  (get_satiety_state (meal_satiety 3 7) = SatietyState.Hunger) →
  (get_satiety_state (meal_satiety 7 1) = SatietyState.Overeating) →
  (get_satiety_state (meal_satiety 0 11) = SatietyState.Hunger) :=
by sorry

end NUMINAMATH_CALUDE_wolf_hunger_theorem_l2358_235804


namespace NUMINAMATH_CALUDE_f_inequality_l2358_235856

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the conditions
axiom period : ∀ x : ℝ, f (x + 3) = f (x - 3)
axiom even_shifted : ∀ x : ℝ, f (x + 3) = f (-x + 3)
axiom decreasing : ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 3 → f y < f x

-- State the theorem
theorem f_inequality : f 3.5 < f (-4.5) ∧ f (-4.5) < f 12.5 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2358_235856


namespace NUMINAMATH_CALUDE_shaded_area_proof_l2358_235819

/-- Given a grid and two right triangles, prove the area of the smaller triangle -/
theorem shaded_area_proof (grid_width grid_height : ℕ) 
  (large_triangle_base large_triangle_height : ℕ)
  (small_triangle_base small_triangle_height : ℕ) :
  grid_width = 15 →
  grid_height = 5 →
  large_triangle_base = grid_width →
  large_triangle_height = grid_height - 1 →
  small_triangle_base = 12 →
  small_triangle_height = 3 →
  (small_triangle_base * small_triangle_height) / 2 = 18 := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_proof_l2358_235819


namespace NUMINAMATH_CALUDE_tim_bodyguard_cost_l2358_235836

def bodyguards_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

theorem tim_bodyguard_cost :
  bodyguards_cost 2 20 8 7 = 2240 :=
by sorry

end NUMINAMATH_CALUDE_tim_bodyguard_cost_l2358_235836


namespace NUMINAMATH_CALUDE_faiths_weekly_earnings_l2358_235881

/-- Calculates the total weekly earnings for Faith given her work conditions --/
def total_weekly_earnings (
  hourly_wage : ℝ)
  (regular_hours_per_day : ℝ)
  (regular_days_per_week : ℝ)
  (overtime_hours_per_day : ℝ)
  (overtime_days_per_week : ℝ)
  (overtime_rate_multiplier : ℝ)
  (commission_rate : ℝ)
  (total_sales : ℝ) : ℝ :=
  let regular_earnings := hourly_wage * regular_hours_per_day * regular_days_per_week
  let overtime_earnings := hourly_wage * overtime_rate_multiplier * overtime_hours_per_day * overtime_days_per_week
  let commission := commission_rate * total_sales
  regular_earnings + overtime_earnings + commission

/-- Theorem stating that Faith's total weekly earnings are $1,062.50 --/
theorem faiths_weekly_earnings :
  total_weekly_earnings 13.5 8 5 2 5 1.5 0.1 3200 = 1062.5 := by
  sorry

end NUMINAMATH_CALUDE_faiths_weekly_earnings_l2358_235881


namespace NUMINAMATH_CALUDE_sandy_marks_problem_l2358_235841

theorem sandy_marks_problem (marks_per_correct : ℕ) (total_attempts : ℕ) (total_marks : ℕ) (correct_sums : ℕ) :
  marks_per_correct = 3 →
  total_attempts = 30 →
  total_marks = 65 →
  correct_sums = 25 →
  (marks_per_correct * correct_sums - total_marks) / (total_attempts - correct_sums) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sandy_marks_problem_l2358_235841


namespace NUMINAMATH_CALUDE_star_value_l2358_235813

def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 15) (h2 : a * b = 56) :
  star a b = 15 / 56 := by
sorry

end NUMINAMATH_CALUDE_star_value_l2358_235813


namespace NUMINAMATH_CALUDE_m_range_proof_l2358_235851

/-- Definition of p -/
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

/-- Definition of q -/
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

/-- ¬p is a sufficient but not necessary condition for ¬q -/
def not_p_sufficient_not_necessary_for_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

/-- The range of values for m -/
def m_range (m : ℝ) : Prop := 0 < m ∧ m ≤ 3

/-- Main theorem: Given the conditions, prove the range of m -/
theorem m_range_proof :
  ∀ m, not_p_sufficient_not_necessary_for_not_q m → m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_range_proof_l2358_235851


namespace NUMINAMATH_CALUDE_ordered_pair_solution_l2358_235833

theorem ordered_pair_solution :
  ∀ (c d : ℤ),
  Real.sqrt (16 - 12 * Real.cos (30 * π / 180)) = c + d * (1 / Real.cos (30 * π / 180)) →
  c = 4 ∧ d = -1 := by
sorry

end NUMINAMATH_CALUDE_ordered_pair_solution_l2358_235833


namespace NUMINAMATH_CALUDE_johns_number_l2358_235864

theorem johns_number (n : ℕ) : 
  (125 ∣ n) ∧ 
  (180 ∣ n) ∧ 
  1000 ≤ n ∧ 
  n ≤ 3000 ∧
  (∀ m : ℕ, (125 ∣ m) ∧ (180 ∣ m) ∧ 1000 ≤ m ∧ m ≤ 3000 → n ≤ m) →
  n = 1800 := by
sorry

end NUMINAMATH_CALUDE_johns_number_l2358_235864


namespace NUMINAMATH_CALUDE_envelope_addressing_equation_l2358_235837

theorem envelope_addressing_equation : 
  ∀ (x : ℝ), 
  (∃ (machine1_time machine2_time combined_time : ℝ),
    machine1_time = 12 ∧ 
    combined_time = 4 ∧
    machine2_time = x ∧
    (1 / machine1_time + 1 / machine2_time = 1 / combined_time)) ↔
  (1 / 12 + 1 / x = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_envelope_addressing_equation_l2358_235837


namespace NUMINAMATH_CALUDE_sequence_2009th_term_l2358_235895

theorem sequence_2009th_term :
  let sequence : ℕ → ℕ := fun n => 2^(n - 1)
  sequence 2009 = 2^2008 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2009th_term_l2358_235895


namespace NUMINAMATH_CALUDE_exists_b_for_even_f_l2358_235849

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = 2x^2 - bx where b is a real number -/
def f (b : ℝ) : ℝ → ℝ := λ x ↦ 2 * x^2 - b * x

/-- There exists a real number b such that f(x) = 2x^2 - bx is an even function -/
theorem exists_b_for_even_f : ∃ b : ℝ, IsEven (f b) := by
  sorry

end NUMINAMATH_CALUDE_exists_b_for_even_f_l2358_235849


namespace NUMINAMATH_CALUDE_unique_root_of_sin_plus_constant_l2358_235829

theorem unique_root_of_sin_plus_constant :
  ∃! x : ℝ, x = Real.sin x + 1993 := by sorry

end NUMINAMATH_CALUDE_unique_root_of_sin_plus_constant_l2358_235829


namespace NUMINAMATH_CALUDE_instagram_followers_after_year_l2358_235807

/-- Calculates the final number of followers for an Instagram influencer after a year --/
theorem instagram_followers_after_year 
  (initial_followers : ℕ) 
  (new_followers_per_day : ℕ) 
  (days_in_year : ℕ) 
  (unfollowers : ℕ) 
  (h1 : initial_followers = 100000)
  (h2 : new_followers_per_day = 1000)
  (h3 : days_in_year = 365)
  (h4 : unfollowers = 20000) :
  initial_followers + (new_followers_per_day * days_in_year) - unfollowers = 445000 :=
by sorry

end NUMINAMATH_CALUDE_instagram_followers_after_year_l2358_235807


namespace NUMINAMATH_CALUDE_apple_cost_l2358_235872

theorem apple_cost (initial_apples : ℕ) (initial_oranges : ℕ) (orange_cost : ℚ) 
  (final_apples : ℕ) (final_oranges : ℕ) (total_earnings : ℚ) :
  initial_apples = 50 →
  initial_oranges = 40 →
  orange_cost = 1/2 →
  final_apples = 10 →
  final_oranges = 6 →
  total_earnings = 49 →
  ∃ (apple_cost : ℚ), apple_cost = 4/5 := by
  sorry

#check apple_cost

end NUMINAMATH_CALUDE_apple_cost_l2358_235872


namespace NUMINAMATH_CALUDE_original_savings_l2358_235883

def lindas_savings : ℝ → Prop := λ s =>
  (3/4 * s + 1/4 * s = s) ∧  -- Total spending equals savings
  (1/4 * s = 200)            -- TV cost is 1/4 of savings and equals $200

theorem original_savings : ∃ s : ℝ, lindas_savings s ∧ s = 800 := by
  sorry

end NUMINAMATH_CALUDE_original_savings_l2358_235883


namespace NUMINAMATH_CALUDE_circle_radius_when_area_is_250_percent_of_circumference_l2358_235844

theorem circle_radius_when_area_is_250_percent_of_circumference (r : ℝ) : 
  r > 0 → π * r^2 = 2.5 * (2 * π * r) → r = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_when_area_is_250_percent_of_circumference_l2358_235844


namespace NUMINAMATH_CALUDE_johns_money_l2358_235891

/-- The problem of determining John's money given the conditions --/
theorem johns_money (total money : ℕ) (ali nada john : ℕ) : 
  total = 67 →
  ali = nada - 5 →
  john = 4 * nada →
  total = ali + nada + john →
  john = 48 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_l2358_235891


namespace NUMINAMATH_CALUDE_black_queen_thought_l2358_235859

-- Define the possible states for each character
inductive State
  | Asleep
  | Awake

-- Define the characters
structure Character where
  name : String
  state : State
  thought : State

-- Define the perverse judgment property
def perverseJudgment (c : Character) : Prop :=
  (c.state = State.Asleep ∧ c.thought = State.Awake) ∨
  (c.state = State.Awake ∧ c.thought = State.Asleep)

-- Define the rational judgment property
def rationalJudgment (c : Character) : Prop :=
  (c.state = State.Asleep ∧ c.thought = State.Asleep) ∨
  (c.state = State.Awake ∧ c.thought = State.Awake)

-- Theorem statement
theorem black_queen_thought (blackKing blackQueen : Character) :
  blackKing.name = "Black King" →
  blackQueen.name = "Black Queen" →
  blackKing.thought = State.Asleep →
  (perverseJudgment blackKing ∨ rationalJudgment blackKing) →
  (perverseJudgment blackQueen ∨ rationalJudgment blackQueen) →
  blackQueen.thought = State.Asleep :=
by
  sorry

end NUMINAMATH_CALUDE_black_queen_thought_l2358_235859


namespace NUMINAMATH_CALUDE_base_r_transaction_l2358_235805

/-- Converts a number from base r to base 10 -/
def to_base_10 (digits : List Nat) (r : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * r ^ i) 0

/-- The problem statement -/
theorem base_r_transaction (r : Nat) : r > 1 →
  (to_base_10 [0, 6, 5] r) + (to_base_10 [0, 2, 4] r) = (to_base_10 [0, 0, 1, 1] r) ↔ r = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_r_transaction_l2358_235805


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2358_235888

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ m : ℝ, ∃ x : ℝ, x^2 + x + m = 0) ↔ 
  (∃ m : ℝ, ∀ x : ℝ, x^2 + x + m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2358_235888


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2358_235832

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_first_term : a 1 = 2) 
  (h_sum_condition : a 2 + a 4 = a 6) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) - a n = d) ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2358_235832


namespace NUMINAMATH_CALUDE_lemonade_problem_l2358_235846

theorem lemonade_problem (V : ℝ) 
  (h1 : V > 0)
  (h2 : V / 10 = V - 2 * (V / 5))
  (h3 : V / 8 = V - 2 * (V / 5 + V / 20))
  (h4 : V / 3 = V - 2 * (V / 5 + V / 20 + 5 * V / 12)) :
  V / 6 = V - (V / 3) / 2 := by sorry

end NUMINAMATH_CALUDE_lemonade_problem_l2358_235846


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l2358_235808

theorem power_of_three_mod_five : 3^304 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l2358_235808


namespace NUMINAMATH_CALUDE_hyperbola_tangent_line_l2358_235811

/-- The equation of a tangent line to a hyperbola -/
theorem hyperbola_tangent_line (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : x₀^2 / a^2 - y₀^2 / b^2 = 1) :
  ∃ (x y : ℝ → ℝ), ∀ t, 
    (x t)^2 / a^2 - (y t)^2 / b^2 = 1 ∧ 
    x 0 = x₀ ∧ 
    y 0 = y₀ ∧
    (∀ s, x₀ * (x s) / a^2 - y₀ * (y s) / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_line_l2358_235811


namespace NUMINAMATH_CALUDE_thomas_monthly_earnings_l2358_235823

/-- Calculates Thomas's total earnings for one month --/
def thomasEarnings (initialWage : ℝ) (weeklyIncrease : ℝ) (overtimeHours : ℕ) (overtimeRate : ℝ) (deduction : ℝ) : ℝ :=
  let week1 := initialWage
  let week2 := initialWage * (1 + weeklyIncrease)
  let week3 := week2 * (1 + weeklyIncrease)
  let week4 := week3 * (1 + weeklyIncrease)
  let overtimePay := (overtimeHours : ℝ) * overtimeRate
  week1 + week2 + week3 + week4 + overtimePay - deduction

/-- Theorem stating that Thomas's earnings for the month equal $19,761.07 --/
theorem thomas_monthly_earnings :
  thomasEarnings 4550 0.05 10 25 100 = 19761.07 := by
  sorry

end NUMINAMATH_CALUDE_thomas_monthly_earnings_l2358_235823


namespace NUMINAMATH_CALUDE_circle_tangent_line_l2358_235890

/-- A circle in polar coordinates with equation ρ = 2cosθ -/
def Circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- A line in polar coordinates with equation 3ρcosθ + 4ρsinθ + a = 0 -/
def Line (ρ θ a : ℝ) : Prop := 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0

/-- The circle is tangent to the line -/
def IsTangent (a : ℝ) : Prop :=
  ∃! (ρ θ : ℝ), Circle ρ θ ∧ Line ρ θ a

theorem circle_tangent_line (a : ℝ) :
  IsTangent a ↔ (a = -8 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_line_l2358_235890


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l2358_235830

/-- The equation of the first circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 25 = 0

/-- The equation of the second circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 3*y - 10 = 0

/-- The equation of the potential common chord -/
def common_chord (x y : ℝ) : Prop := 4*x - 3*y - 15 = 0

/-- Theorem stating that the given equation represents the common chord of the two circles -/
theorem common_chord_of_circles :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l2358_235830


namespace NUMINAMATH_CALUDE_vieta_sum_product_l2358_235874

theorem vieta_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x ≠ y) →
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 9 ∧ x * y = 15) →
  p + q = 72 :=
by sorry

end NUMINAMATH_CALUDE_vieta_sum_product_l2358_235874


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2358_235894

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∀ x, equation x = 0) → sum_of_roots = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l2358_235894


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l2358_235815

/-- The number of houses in Lincoln County after a housing boom -/
def houses_after_boom (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem: The number of houses in Lincoln County after the housing boom is 118558 -/
theorem lincoln_county_houses :
  houses_after_boom 20817 97741 = 118558 := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l2358_235815


namespace NUMINAMATH_CALUDE_sum_not_prime_l2358_235878

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : 
  ¬(Nat.Prime (a + b + c + d)) :=
by sorry

end NUMINAMATH_CALUDE_sum_not_prime_l2358_235878


namespace NUMINAMATH_CALUDE_area_of_EFGH_l2358_235892

-- Define the properties of the smaller rectangles
def short_side : ℝ := 4
def long_side : ℝ := 2 * short_side

-- Define the properties of the larger rectangle EFGH
def EFGH_width : ℝ := 4 * long_side
def EFGH_length : ℝ := short_side

-- State the theorem
theorem area_of_EFGH : EFGH_width * EFGH_length = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_of_EFGH_l2358_235892
